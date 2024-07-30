import torch
import torch.nn as nn
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from itertools import permutations
import torch.nn.functional as F
import pdb
EPS = 1e-8

class PESQ(nn.Module):
    def __init__(self):
        super(PESQ,self).__init__()
        self.EPS = 1e-8
    def forward(self, source, estimate_source):
        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]

        pesq = PerceptualEvaluationSpeechQuality(16000,'wb')

        return pesq(estimate_source,source)

class SI_SNR(nn.Module):
    def __init__(self):
        super(SI_SNR, self).__init__()
        self.EPS = 1e-8
        self.BCE = torch.nn.BCELoss()

    def forward(self, source, estimate_source,compare_a=None,compare_v=None):

        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]

        # step 1: Zero-mean norm
        source = source - torch.mean(source, dim=-1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

        # step 2: Cal si_snr
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + self.EPS
        proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + self.EPS)
        sisnr = 10 * torch.log10(ratio + self.EPS)

        loss = 0 - torch.mean(sisnr)

        if compare_a != None:
            B,C = compare_a.shape
            compare_v = torch.zeros((B,C))
            mix_num = source.shape[0]
            compare_v[:,:mix_num]=1
            compare_v = compare_v.to(compare_a.device)
            loss_attractor = self.BCE(compare_a,compare_v)
            loss = loss + loss_attractor

        return loss

class MuSE_loss(nn.Module):
    def __init__(self):
        super(MuSE_loss, self).__init__()
        self.si_snr_loss = SI_SNR()
        self.speaker_loss = nn.CrossEntropyLoss()

    def forward(self, tgt_wav, pred_wav, tgt_spk, pred_spk):
        si_snr = self.si_snr_loss(tgt_wav, pred_wav)
        ce = self.speaker_loss(pred_spk[0], tgt_spk) + self.speaker_loss(pred_spk[1], tgt_spk) + self.speaker_loss(pred_spk[2], tgt_spk) + self.speaker_loss(pred_spk[3], tgt_spk)
        return {'si_snr': si_snr, 'ce': ce} 

class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        Torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)

        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
        Arguments
        ---------
        tensor : torch.Tensor
            Tensor to reorder given the optimal permutation, of shape
            [batch, ..., sources].
        p : list of tuples
            List of optimal permutations, e.g. for batch=2 and n_sources=3
            [(0, 1, 2), (0, 2, 1].

        Returns
        -------
        reordered : torch.Tensor
            Reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor, device=tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss : torch.Tensor
                Permutation invariant loss for current examples, tensor of
                shape [batch]

            perms : list
                List of indexes for optimal permutation of the inputs over
                sources.
                e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms

def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    mask = source.new_ones(source.size()[:-1]).unsqueeze(-1).transpose(1, -2)
    B = source.size(-2)
    for i in range(B):
        mask[source_lengths[i] :, i] = 0
    return mask.transpose(-2, 1)

def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)


class SI_SNR_PIT(nn.Module):
    """This function wraps si_snr calculation with the speechbrain pit-wrapper.

    Arguments:
    ---------
    source: [B, T, C],
        Where B is the batch size, T is the length of the sources, C is
        the number of sources the ordering is made so that this loss is
        compatible with the class PitWrapper.

    estimate_source: [B, T, C]
        The estimated source.

    Example:
    ---------
    >>> x = torch.arange(600).reshape(3, 100, 2)
    >>> xhat = x[:, :, (1, 0)]
    >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    >>> print(si_snr)
    tensor([135.2284, 135.2284, 135.2284])
    """
    def __init__(self):
        super(SI_SNR_PIT,self).__init__()
        self.EPS = 1e-8
        self.BCE = torch.nn.BCELoss()

    def forward(self, source, estimate_source,compare_a,compare_v):

        pit_si_snr = PitWrapper(cal_si_snr)
        loss, perms = pit_si_snr(estimate_source,source)

        #############################################################
        loss = torch.mean(loss)
        re_pred = pit_si_snr.reorder_tensor(estimate_source,perms)

        # if compare_a != None:
        #     T,B,N = compare_a.shape
        #     compare_a = F.normalize(compare_a,dim=-1)
        #     compare_v = F.normalize(compare_v,dim=-1)
        #     compare_a = compare_a.transpose(0,1) #(B,T,N)
        #     compare_v = compare_v.permute(1,2,0) #(B,N,T)
        #     mask = torch.eye(T,T).to(compare_a.device)
        #     mask = mask.unsqueeze(0).repeat(B,1,1) #(B,T,T)
        #     logits = torch.exp(torch.bmm(compare_a,compare_v)) #(B,T,T)
        #     loss_contra_1 = -torch.mean(torch.log((torch.sum(logits*mask,dim=-1)+self.EPS)/(torch.sum(logits*(1-mask),dim=-1) + self.EPS)))
        #     loss_contra_2 = -torch.mean(torch.log((torch.sum(logits*mask,dim=-2)+self.EPS)/(torch.sum(logits*(1-mask),dim=-2) + self.EPS )))         
        #     loss = loss + loss_contra_1 + loss_contra_2
        if compare_a != None:
            B,C = compare_a.shape
            compare_v = torch.ones((B,C))
            compare_v[:,-1]=0
            compare_v = compare_v.to(compare_a.device)
            loss_attractor = self.BCE(compare_a,compare_v)
            loss = loss + loss_attractor
            #print("loss_attractor",loss_attractor)
        #############################################################

        return loss,re_pred




if __name__ == '__main__':
    x = torch.randn(3,10,2)
    y = torch.randn(3,10,2)
    gg = SI_SNR()
    pdb.set_trace()
    cal = SI_SNR_PIT()
    loss = cal(x,y)


