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

class SPKTransformerBlock(nn.Module):
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
        super(SPKTransformerBlock, self).__init__()
        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")      

        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = activation

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_attention(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        num_spks,B,N = tgt.shape
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = tgt
        tgt2 = self.norm2(tgt2)
        tgt2 = tgt2.transpose(0,1) #(B,num_spks,N)
        tgt2 = tgt2.reshape(1,B*num_spks,-1) #(1,B*num_spks,N)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.reshape(B,num_spks,-1)
        tgt2 = tgt2.transpose(0,1) #(num_spks,B,N)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, speaker, v_memory,            
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        speaker = speaker.permute(1,0,2) #(S,BCK,N)
        v_memory = v_memory.permute(1,0,2) #(S,BCK,N) 
        tgt = v_memory     
        tgt2 = self.norm1(tgt)
        speaker = self.norm2(speaker)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(speaker, pos),
                                   value=speaker, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)

        tgt = tgt + speaker 
        
        return tgt.permute(1,2,0) #(BCK,N,S)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        # self.norm5 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        ######################################################
        # self.linear_a = nn.Linear(d_model, d_model)
        # self.linear_v = nn.Linear(d_model, d_model)
        ######################################################

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

    def forward_pre(self, tgt, query_a, query_v, memory, v_memory, 
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        #########################################################################
        # tmp_a = query_a
        # query_a = self.norm3(query_a) #(T,B,N)
        # query_a = self.multihead_attn_1(query=self.with_pos_embed(query_a, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # query_a = query_a + tmp_a

        # T,B,_ = query_a.shape
        # num_spks = query_v.shape[0] // T
        # query_v = query_v.reshape(T,num_spks,B,-1)
        # query_v = query_v.transpose(1,2) #(T,B,2,N)
        # query_v = query_v.reshape(T,B*num_spks,-1)

        # tmp_v = query_v
        # query_v = self.norm4(query_v) #(8,B*num_spks,N)
        # query_v = self.multihead_attn_2(query=self.with_pos_embed(query_v, query_pos),
        #                            key=self.with_pos_embed(v_memory, pos),
        #                            value=v_memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0] 
        # query_v = query_v + tmp_v
        
        # query_v = query_v.reshape(T,B,num_spks,-1)
        # query_v = query_v.transpose(1,2) #(T,2,B,N)
        # query_v = query_v.reshape(T*num_spks,B,-1) #(T*2,B,N)

        # deep_query = self.linear_a(query_a).unsqueeze(1) * self.linear_v(query_v).unsqueeze(0) #(T,T*2,B,N)
        # deep_query = deep_query.reshape(T*T,num_spks,B,-1) #(T*T,2,B,N)
        # deep_query = deep_query.transpose(1,2).reshape(T*T,B*num_spks,-1) #(T*T,B*2,N)

        # _,B,_ = tgt.shape
        # tgt2 = self.norm5(tgt) #(2,B,N) # v_memory (25,2*B,N)
        # tgt2 = tgt2.transpose(0,1)  #(B,2,N)
        # tgt2 = tgt2.reshape(1,B*num_spks,-1) #(1,B*2,N)
        # tgt2 = self.multihead_attn_3(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(deep_query, pos),
        #                            value=deep_query, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # tgt2 = tgt2.reshape(B,num_spks,-1)
        # tgt2 = tgt2.transpose(0,1) #(2,B,N)
        # tgt = tgt + self.dropout3(tgt2)
        #########################################################################
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = tgt
        tgt2 = self.norm3(tgt2) #(8,B*num_spks,N)
        tgt2 = self.multihead_attn_1(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0] 
        tgt = tgt + self.dropout3(tgt2)


        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt,query_a,query_v

    def forward(self, tgt, query_embed_a, query_embed_v, memory, v_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, query_embed_a, query_embed_v, memory, v_memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, v_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, query_embed_a, query_embed_v, memory, v_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i in range(self.num_layers):
            output, query_embed_a, query_embed_v = self.layers(output, query_embed_a, query_embed_v, memory, v_memory, tgt_mask=tgt_mask,
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

        self.query_embed = nn.Embedding(5,d_model)
        # self.query_embed_a = nn.Embedding(32,256)
        # self.query_embed_v = nn.Embedding(32*num_spks,256) #16 * num_spks

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=False) 

        self.linear_a = nn.Linear(d_model,d_model)
        self.linear_b = nn.Linear(d_model,d_model)
        #self.num_spks = num_spks 
        self.linear_down = nn.Linear(d_model,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,memory,v_memory,num_spks):
        B,_,_ = memory.shape
        memory = memory.permute(2,0,1).contiguous() #(1999,B,256)

        query = self.query_embed.weight.unsqueeze(1).repeat(1,B,1) #(2,B,N)
        # query_embed_a = self.query_embed_a.weight.unsqueeze(1).repeat(1,B,1) #(16,B,N)
        # query_embed_v = self.query_embed_v.weight.unsqueeze(1).repeat(1,B,1) #(16*num_spks,B,N)
        query_embed_a,query_embed_v = None,None
        ###############   for causal version of self-attention   ###############
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(query.shape[0])
        tgt_mask = tgt_mask.to(x.device)
        out_query = self.decoder(query,query_embed_a,query_embed_v,memory,v_memory,tgt_mask) #(2,B,N)
        #######################################################################################
        #out_query = out_query.reshape(B,2,-1) #(B,2,N) x.shape (B,N,K,S)
        out_query = out_query.transpose(0,1) #(B,2,N)
        #######################################################################################

        out = self.linear_a(out_query[:,:num_spks,:]).unsqueeze(3).unsqueeze(4) * x.unsqueeze(1) + self.linear_b(out_query[:,:num_spks,:]).unsqueeze(3).unsqueeze(4)
        
        out_attractor = self.sigmoid(self.linear_down(out_query).squeeze(-1))
        #(B,2,N,K,S)
        return out, out_attractor

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
        self.inter_mdl = inter_mdl
        #self.spgm = nn.ModuleList([copy.deepcopy(SPGM()) for i in range(2)])
        self.skip_around_intra = skip_around_intra
        #self.pos2d = PositionalEncodingPermute2D(out_channels)

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
        # pe = self.pos2d(x)
        # x = x + pe
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

        inter = self.inter_mdl(inter)
            
        # [B, K, S, N]
        inter = inter.view(B, K, S, N)

        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + x

        return out


class Cross_Triple_Computation_Block(nn.Module):
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
        num_spks,
        intra_mdl,
        inter_mdl,
        speaker_cross_mdl,
        speaker_self_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Cross_Triple_Computation_Block, self).__init__()

        self.num_spks = num_spks
        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.speaker_cross_mdl = speaker_cross_mdl
        self.speaker_self_mdl = speaker_self_mdl
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)
            self.speaker_norm = select_norm(norm, out_channels, 4)
            self.speaker_self_norm = select_norm(norm, out_channels, 4)

    def forward(self, i, x, v_memory, mix_num):
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
        # if i==0:
        #     x = x.unsqueeze(1).repeat(1,mix_num,1,1,1) #(B,C,N,K,S)

        B, num_spks, N, K, S = x.shape
        tmp_b = x
        x = x.reshape(-1,N,K,S) #(B*num_spks,N,K,S)
        # intra RNN
        # [BCS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * num_spks * S, K, N)
        # [BCS, K, N]
        intra = self.intra_mdl(intra)
        # [BC, S, K, N]
        intra = intra.view(B * num_spks, S, K, N)
        # [BC, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)
        # [BC, N, K, S]
        if self.skip_around_intra:
            intra = intra + x
        
        speaker = intra #(BC,N,K,S)
        speaker = speaker.permute(0, 2, 3, 1).contiguous().view(B * num_spks * K, S, N) #(BCK,S,N)
        v_memory = v_memory.unsqueeze(-2).repeat(1,1,K,1) #(BC,N,S)->(BC,N,K,S)
        v_memory = v_memory.permute(0, 2, 3, 1).contiguous().view(B * num_spks * K, S, N) #(BCK,S,N)
        speaker = self.speaker_cross_mdl(speaker,v_memory) #(BCK,N,S)
        if self.norm is not None:
            speaker = self.speaker_norm(speaker)

        # inter RNN
        # [BCK, S, N]
        inter = speaker.transpose(1,2) #(BCK,S,N)
        inter = self.inter_mdl(inter)
        # [BC, K, S, N]
        inter = inter.view(B * num_spks, K, S, N)
        # [BC, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        inter = inter.view(B,num_spks,N,K,S)
     
        # [BKS, num_spks, N]
        speaker_self = inter.permute(0,3,4,1,2).contiguous().view(B * K * S, num_spks, N)
        speaker_self = self.speaker_self_mdl(speaker_self)
        speaker_self = speaker_self.view(B,K,S,num_spks,N)
        speaker_self = speaker_self.permute(0,3,4,1,2).contiguous().view(B*num_spks,N,K,S) #(B * num_spks,N,K,S)
        if self.norm is not None:
            speaker_self = self.speaker_self_norm(speaker_self)
        out = speaker_self.view(B,num_spks,N,K,S)
        out = out + tmp_b

        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Cross_Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model_1,
        inter_model_1,
        intra_model,
        inter_model,
        speaker_cross_model,
        speaker_self_model,
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
        #self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)

        ve_blocks = []
        for _ in range(5):
            ve_blocks += [VisualConv1D()]
        ve_blocks += [nn.Conv1d(512, 256, 1)]
        self.visual_conv = nn.Sequential(*ve_blocks)

        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        self.dual_mdl = nn.ModuleList([])
        for i in range(1):
            self.dual_mdl.append(
                copy.deepcopy(
                    Cross_Dual_Computation_Block(
                        intra_model_1,
                        inter_model_1,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                    )))

        self.triple_mdl = nn.ModuleList([])
        for i in range(self.num_layers):
            self.triple_mdl.append(
                copy.deepcopy(
                    Cross_Triple_Computation_Block(
                        num_spks,
                        intra_model,
                        inter_model,
                        speaker_cross_model,
                        speaker_self_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                    )))

        self.query_decoder = Query_Decoder(2,out_channels,8,1024)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.prelu_2 = nn.PReLU()
        self.activation = nn.ReLU()
        # # gated output layer
        self.output = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1),
                                    nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1), nn.Sigmoid())

    def forward(self, x, video, mix_num):
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
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(1):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        video = video.transpose(1, 2) 
        v = self.visual_conv(video)
        # ############################ 把这个映射换成用视觉特征做query的后处理 #############################
        memory = self._over_add(x, gap) #(B,N,L)
        x,out_attractor = self.query_decoder(x, memory,v,mix_num)
        # ############################################################################################
        B, _, N, K, S = x.shape


        v = F.pad(v, (0, x.shape[-1] - v.shape[-1]), mode='replicate')  #(B*num_spks,N,S)
        #############################################################################################
        for i in range(self.num_layers):
            x = self.triple_mdl[i](i,x,v,mix_num)
        x = self.prelu_2(x)
        #############################################################################################

        B, _, N, K, S = x.shape
        # [B*spks, N, K, S]
        x = x.view(B * mix_num, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [B*spks, N, L]
        x = self.end_conv1x1(x)
        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, mix_num, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x,out_attractor

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
                 IntraSeparator_1,
                 InterSeparator_1,
                 IntraSeparator,
                 InterSeparator,
                 SpeakerSeparator_Cross,
                 SpeakerSeparator_Self,
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
                                               num_layers=5,
                                               K=160,
                                               intra_model_1=IntraSeparator_1,
                                               inter_model_1=InterSeparator_1,
                                               intra_model=IntraSeparator,
                                               inter_model=InterSeparator,
                                               speaker_cross_model=SpeakerSeparator_Cross,
                                               speaker_self_model=SpeakerSeparator_Self,
                                               norm='ln',
                                               skip_around_intra=True)

        #self.num_spks = num_spks

    def forward(self, mix, video, mix_num):

        #print("mix.shape",mix.shape) #(B,L)
        ###################################################################
        #mix = mix.transpose(1,2)
        ###################################################################
        mix_w = self.AudioEncoder(mix)

        est_mask,out_attractor = self.Separator(mix_w,video,mix_num)
        mix_w = torch.stack([mix_w] * mix_num) #(1,2,256,1999)
        sep_h = mix_w * est_mask #(num_spks,B,N,L)
        
        # Decoding
        est_source = torch.cat(
            [
                self.AudioDecoder(sep_h[i]).unsqueeze(-1)
                for i in range(mix_num)
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


        compare_a,compare_v = out_attractor,None

        return est_source.permute(0, 2, 1).squeeze(1),compare_a,compare_v


def Cross_Sepformer_warpper(kernel_size=16, N_encoder_out=256, num_spks=1):

    IntraSeparator_1 = SBTransformerBlock(num_layers=2,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)
    InterSeparator_1 = CrossTransformerBlock(num_layers=2,
                                           d_model=N_encoder_out,
                                           nhead=8,
                                           d_ffn=1024,
                                           dropout=0,
                                           use_positional_encoding=False,
                                           norm_before=True)

    IntraSeparator = SBTransformerBlock(num_layers=2,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)
    InterSeparator = CrossTransformerBlock(num_layers=2,
                                           d_model=N_encoder_out,
                                           nhead=8,
                                           d_ffn=1024,
                                           dropout=0,
                                           use_positional_encoding=False,
                                           norm_before=True)
    SpeakerSeparator_Cross = SPKTransformerBlock(num_layers=1,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=False,
                                        norm_before=True)   

    SpeakerSeparator_Self = SBTransformerBlock(num_layers=1,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)  

    return Cross_Sepformer(IntraSeparator_1,
                           InterSeparator_1,
                           IntraSeparator,
                           InterSeparator,
                           SpeakerSeparator_Cross,
                           SpeakerSeparator_Self,
                           kernel_size=kernel_size,
                           N_encoder_out=N_encoder_out,
                           num_spks=num_spks)


if __name__ == '__main__':
    model = Cross_Sepformer_warpper()
    print(model(torch.randn(2,16000), torch.randn(4, 25, 512)).shape)