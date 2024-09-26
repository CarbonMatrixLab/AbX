import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as LayerNorm

from einops import rearrange
from abx.common import residue_constants

class Linear_common(nn.Linear):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)

        assert init in ['gate', 'final', 'attn', 'relu', 'linear']

        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            # GlorotUniform
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            # Relu, He
            # linear, Le cun
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')

        if bias:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)


def Linear(input_dim, output_dim, init, bias=True, config=None):
    assert init in ['gate', 'final', 'attn', 'relu', 'linear']  
    if config is not None:
        return Linear_common(input_dim, output_dim, init, bias)            
    else:
        return Linear_common(input_dim, output_dim, init, bias)
    

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        if broadcast_dim is not None:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1
            with torch.no_grad():
                scale = 1. / (1. - rate)
                keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
                keep = torch.bernoulli(keep_rate)
            return scale * keep * tensor
        else:
            return F.dropout(tensor, rate)
    else:
        return tensor

def pseudo_beta_fn_v2(aatype, all_atom_positions, all_atom_masks=None):
    """all_atom_positions is in atom37 format"""

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
        
    N = all_atom_positions[..., n_idx, :]
    CA = all_atom_positions[..., ca_idx, :]
    C = all_atom_positions[..., c_idx, :]
    
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    
    if all_atom_masks is not None:
        CB_mask = torch.all(
                torch.stack([all_atom_masks[...,n_idx], all_atom_masks[...,ca_idx], all_atom_masks[...,c_idx]], dim=-1), dim=-1)
        return CB, CB_mask

    return CB

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    pseudo_beta = torch.where(
        #torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        is_gly[...,None],
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx].to(dtype=torch.float32),
            all_atom_masks[..., cb_idx].to(dtype=torch.float32))
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    breaks = torch.linspace(min_bin, max_bin, steps=num_bins-1, device=positions.device)
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, 'b l c -> b l () c') -
            rearrange(positions, 'b l c -> b () l c')),
        dim=-1,
        keepdims=True)

    true_bins = torch.sum(dist2 > sq_breaks, axis=-1).long()

    return true_bins
