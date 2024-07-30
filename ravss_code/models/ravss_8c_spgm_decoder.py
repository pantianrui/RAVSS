import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, List
from .utils import PositionalEncoding, PositionalEncodingPermute2D
import pdb

EPS = 1e-8

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm,
              self).__init__(dim,
                             elementwise_affine=elementwise_affine,
                             eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512,
                           512,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=512,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class CrossTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 depth=4,
                 dropout=0.1,
                 dim_feedforward=2048,
                 activation=F.relu):
        super(CrossTransformer, self).__init__()
        self.transformer_layers = []
        for _ in range(depth):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(d_model,
                                           nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=True))
        self.transformer_layers = nn.Sequential(*self.transformer_layers)

    def forward(self,audio):
        x = audio
        x = self.transformer_layers(x)
        return x


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        use_positional_encoding=True,
        norm_before=True,
    ):
        super(CrossTransformerBlock, self).__init__()

        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = CrossTransformer(d_model,
                                    nhead,
                                    dim_feedforward=d_ffn,
                                    depth=num_layers,
                                    dropout=dropout,
                                    activation=activation)

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, dropout=0.0)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        if self.use_positional_encoding:
            x = self.pos_enc(x)
            x = self.mdl(x)
        else:
            x = self.mdl(x)
        return x.permute(1, 0, 2)


class SBTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=d_ffn,
                                                     dropout=dropout,
                                                     activation=activation,
                                                     norm_first=norm_before),
            num_layers=num_layers)
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        x = x.permute(1, 0, 2)
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            x = self.mdl(pos_enc)
        else:
            x = self.mdl(x)
        return x.permute(1, 0, 2)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, v_memory, 
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, v_memory, 
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(v_memory, pos),
                                   value=v_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, v_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, v_memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, v_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, v_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, v_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class Query_Decoder(nn.Module):
    def __init__(
        self,
        num_decoder_layers,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        activation="relu",
        normalize_before=True,
    ):
        super(Query_Decoder,self).__init__()
        ve_blocks = []
        for _ in range(5):
            ve_blocks += [VisualConv1D()]
        ve_blocks += [nn.Conv1d(512, 256, 1)]
        self.visual_conv = nn.Sequential(*ve_blocks)
        self.query_embed = nn.Embedding(2,256)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=False) 

        self.linear_a = nn.Linear(d_model,d_model)
        self.linear_b = nn.Linear(d_model,d_model)

    def forward(self,x,memory,video):
        B,_,_ = memory.shape
        memory = memory.permute(2,0,1).contiguous() #(1999,B,256)
        v = self.visual_conv(video)
        v = v.permute(2,0,1).contiguous() #(25,2*B,256) #ABCD
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) #(2,2,256)
        query = query.reshape(1,B*2,-1) #(1,B*2,N) #ZXZX

        memory = memory.unsqueeze(2).repeat(1,1,2,1) #(1999,B,2,256)
        memory = memory.reshape(-1,B*2,256) #(1999,B*2,256) #(AB)(AB)(CD)(CD)

        out_query = self.decoder(query,memory,v)
        out_query = out_query.reshape(B,2,-1) #(B,2,N) x.shape (B,N,K,S)

        out = self.linear_a(out_query).unsqueeze(3).unsqueeze(4) * x.unsqueeze(1) + self.linear_b(out_query).unsqueeze(3).unsqueeze(4)
        
        #(B,2,N,K,S)
        return out 

class SPGM(nn.Module):
    def __init__(self,):
        super(SPGM,self).__init__()
        self.linear_s = nn.Linear(256,256)
        self.linear_w = nn.Linear(256,256)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        global_pool = x[:,-1:,:,:] #(B,1,S,N)
        global_pool = torch.mean(global_pool,dim=-2) 
        global_pool = global_pool.unsqueeze(-2) #(B,1,1,N)
        inter = self.sigmoid(self.linear_s(global_pool)) * x + self.linear_w(global_pool) * x

        return inter

class Cross_Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Cross_Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        #self.inter_mdl = inter_mdl
        self.spgm = nn.ModuleList([copy.deepcopy(SPGM()) for i in range(2)])
        self.skip_around_intra = skip_around_intra
        self.pos2d = PositionalEncodingPermute2D(256)

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        pe = self.pos2d(x)
        x = x + pe
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, N]

        intra = self.intra_mdl(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)

        #inter = self.inter_mdl(inter)
            
        # [B, K, S, N]
        inter = inter.view(B, K, S, N)

        for layer in self.spgm:
            inter = layer(inter)

        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Cross_Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=160,
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Cross_Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Cross_Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                    )))

        self.query_decoder = Query_Decoder(4,256,8,1024)

        # self.conv2d = nn.Conv2d(out_channels,
        #                         out_channels * num_spks,
        #                         kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x, video):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        video = video.transpose(1, 2) 
        #video.shape (1,25,512)
        #print("video.shape",video.shape) video.shape torch.Size([2, 72, 1024])

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(
                1, -1) + x * (x.size(1)**0.5)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        # x = self.conv2d(x)
        ############################ 把这个映射换成用视觉特征做query的后处理 #############################
        memory = self._over_add(x, gap) #(B,N,L)
        x = self.query_decoder(x, memory,video)
        ############################################################################################
        B, _, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (torch.cat([input1, input2], dim=3).view(B, N, -1,
                                                         K).transpose(2, 3))

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Cross_Sepformer(nn.Module):
    def __init__(self,
                 IntraSeparator,
                 InterSeparator,
                 kernel_size=16,
                 N_encoder_out=256,
                 num_spks=2):
        super(Cross_Sepformer, self).__init__()

        self.AudioEncoder = Encoder(kernel_size=kernel_size,
                                    out_channels=N_encoder_out)

        self.AudioDecoder = Decoder(in_channels=N_encoder_out,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    stride=kernel_size // 2,
                                    bias=False)
        self.Separator = Cross_Dual_Path_Model(num_spks=num_spks,
                                               in_channels=N_encoder_out,
                                               out_channels=N_encoder_out,
                                               num_layers=2,
                                               K=160,
                                               intra_model=IntraSeparator,
                                               inter_model=InterSeparator,
                                               norm='ln',
                                               skip_around_intra=True)
        self.num_spks = num_spks

    def forward(self, mix, video):

        #print("mix.shape",mix.shape) #(B,L)
        ###################################################################
        #mix = mix.transpose(1,2)
        ###################################################################
        mix_w = self.AudioEncoder(mix)

        est_mask = self.Separator(mix_w,video)
        mix_w = torch.stack([mix_w] * self.num_spks) #(1,2,256,1999)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.AudioDecoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        ##################################################################
        B,L,spks = est_source.shape
        est_source = est_source.transpose(1,2)
        est_source = est_source.reshape(-1,L)
        est_source = est_source.unsqueeze(-1)
        #T_origin = mix.size(2)
        T_origin = mix.size(1)
        ##################################################################

        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.permute(0, 2, 1).squeeze(1)


def Cross_Sepformer_warpper(kernel_size=16, N_encoder_out=256, num_spks=2):

    InterSeparator = CrossTransformerBlock(num_layers=8,
                                           d_model=N_encoder_out,
                                           nhead=8,
                                           d_ffn=1024,
                                           dropout=0,
                                           use_positional_encoding=False,
                                           norm_before=True)
    IntraSeparator = SBTransformerBlock(num_layers=8,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)
    return Cross_Sepformer(IntraSeparator,
                           InterSeparator,
                           kernel_size=kernel_size,
                           N_encoder_out=N_encoder_out,
                           num_spks=num_spks)


if __name__ == '__main__':
    model = Cross_Sepformer_warpper()
    print(model(torch.randn(2,16000,8), torch.randn(4, 25, 512)).shape)