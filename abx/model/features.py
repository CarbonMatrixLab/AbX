import os
import functools
from inspect import isfunction

import torch
from torch.nn import functional as F
from einops import rearrange
import random

from abx.utils import default
from abx.model.utils import batched_select
from abx.model.r3 import rigids_to_tensor7, rigids_op
from abx.utils import default,uniform_sample
from abx.common import residue_constants
from abx.model.common_modules import pseudo_beta_fn
from abx.common import geometry
from abx.data.utils import apply_patch_to_tensor, _mask_select
from diffuser.full_diffuser import FullDiffuser
import pdb
import logging

logger = logging.getLogger(__name__)
_feats_fn = {}

def linsapce(t, num_steps):
    delta_mat = torch.tile(torch.arange(1, num_steps+1, device=t.device)[None, :], (t.shape[0], 1))
    delta_t = t / num_steps
    return delta_mat * delta_t[:, None]

def make_one_antibody_seq(seq, heavy_length, light_length):
    heavy_seq = seq[:heavy_length]
    light_seq = seq[heavy_length: light_length+heavy_length]
    heavy_seq = torch.clamp(heavy_seq, min=0, max=20)
    light_seq = torch.clamp(light_seq, min=0, max=20)

    str_heavy_seq_ = ''.join([residue_constants.restypes[index] for index in heavy_seq])
    str_light_seq_ = ''.join([residue_constants.restypes[index] for index in light_seq])
    return str_heavy_seq_, str_light_seq_

def take1st(fn):
    """Supply all arguments but the first."""

    @functools.wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feats_fn
    _feats_fn[fn.__name__] = fc

    return fc

@take1st
def make_restype_atom_constants(batch, is_training=False):
    device = batch['seq'].device

    batch['atom14_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), batch['seq'])
    batch['atom14_atom_is_ambiguous'] = batched_select(torch.tensor(residue_constants.restype_atom14_is_ambiguous, device=device), batch['seq'])
    
    if 'residx_atom37_to_atom14' not in batch:
        batch['residx_atom37_to_atom14'] = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), batch['seq'])

    if 'atom37_atom_exists' not in batch:
        batch['atom37_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=device), batch['seq'])
    
    return batch


@take1st
def make_atom14_alt_gt_positions(batch, is_training=True):
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    device = batch['seq'].device

    restype_atom_swap_index = batched_select(torch.tensor(residue_constants.restype_ambiguous_atoms_swap_index, device=device), batch['seq'])
    batch['atom14_alt_gt_positions'] = batched_select(batch['atom14_gt_positions'], restype_atom_swap_index, batch_dims=2)
    batch['atom14_alt_gt_exists'] = batched_select(batch['atom14_gt_exists'], restype_atom_swap_index, batch_dims=2)

    return batch

@take1st
def make_pseudo_beta(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch['pseudo_beta'], batch['pseudo_beta_mask'] = pseudo_beta_fn(
            batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists'])

    return batch

@take1st
def make_gt_frames(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)

    batch.update(
            geometry.atom37_to_frames(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))
    
    return batch

@take1st
def make_calpha3_frames(batch, is_training=True):
    calpha_pos = batch['atom37_gt_positions'][:,:,1]
    calpha_mask = batch['atom37_gt_exists'][:,:,1]
    
    batch.update(geometry.calpha3_to_frames(calpha_pos, calpha_mask))

    return batch

@take1st
def make_torsion_angles(batch, is_training=True):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch.update(
            geometry.atom37_to_torsion_angles(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))

    return batch

def make_atom37_positions(batch, is_Normalize=True, scale_factor=1.0):
    device = batch['seq'].device
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    
    batch['atom37_gt_positions'] = batched_select(batch['atom14_gt_positions'], batch['residx_atom37_to_atom14'], batch_dims=2)
    batch['atom37_gt_exists'] = torch.logical_and(
            batched_select(batch['atom14_gt_exists'], batch['residx_atom37_to_atom14'], batch_dims=2),
            batch['atom37_atom_exists'])
    
    return batch


@take1st
def make_diffuser_features(batch, generate_area, diff_conf, shrink_limit=1,extend_limit=2, is_training=True):
    full_diffuser = FullDiffuser.get(diff_conf)

    device = batch['seq'].device
    anchor_flag = batch['anchor_flag'].int()
    antibody_len = anchor_flag.shape[1]
    batch_size = batch['seq'].shape[0]
    gt_bb_rigid = rigids_op(batch['rigidgroups_gt_frames'], lambda x: x[:,:,0])
    rigids_0 = rigids_to_tensor7(gt_bb_rigid)
    seq_0 = batch['seq']

    if generate_area == 'cdr':
        cdr_all = anchor_flag[anchor_flag > 0].unique().tolist()
        if is_training == False:
            cdrs_to_mask = cdr_all
        else:
            num_cdrs_to_mask = random.randint(1, len(cdr_all))
            random.shuffle(cdr_all)
            cdrs_to_mask = cdr_all[:num_cdrs_to_mask]

    elif generate_area in residue_constants.cdr_str_to_enum:
        cdrs_to_mask = [residue_constants.cdr_str_to_enum[generate_area]]

    diffused_mask = torch.zeros_like(batch['mask'], dtype=torch.int32, device=device)
    antibody_struc_loss_mask = torch.zeros_like(batch['anchor_flag'], dtype=torch.int32, device=device)
    struc_loss_mask = batch['mask'].type(torch.int32)

    for cdr in cdrs_to_mask:
        indices = torch.nonzero(anchor_flag == cdr).tolist()
        for i in range(0, len(indices)-1, 2):
            right = indices[i][1]
            left = indices[i+1][1]
            if is_training:
                right = max(0, right-random.randint(-shrink_limit, extend_limit))
                left = min(left+random.randint(-shrink_limit, extend_limit), diffused_mask.shape[1]-1)
            diffused_mask[indices[i][0], right+1: left-1] = 1  
            antibody_struc_loss_mask[indices[i][0], max(right-1,0): min(left+1, diffused_mask.shape[1]-1)] = 1

    struc_loss_mask[:,:antibody_len] = antibody_struc_loss_mask
    fixed_mask = 1 - diffused_mask

    if is_training:
        t = uniform_sample(batch_size, 0.01, 1.0).to(device=device)
        diff_feats_t = FullDiffuser.forward_marginal(
            full_diffuser,
            rigids_0=rigids_0,
            seq_0 = seq_0,
            t = t,
            diffuse_mask=diffused_mask
        )
            
    else:
        inference_step = diff_conf['inference_step']
        if 'opt_step' not in diff_conf:
            t = torch.ones((batch_size,),device=device,dtype=torch.float32)
            diff_feats_t = FullDiffuser.sample_ref(
                full_diffuser,
                n_samples=rigids_0.shape[:2],
                impute_rigids=rigids_0,
                impute_seq=seq_0,
                diffuse_mask=diffused_mask
            )
        else:
            opt_step = diff_conf['opt_step']
            time = opt_step / inference_step
            t = torch.full((batch_size,), fill_value = time, device=device,dtype=torch.float32)
            diff_feats_t = FullDiffuser.forward_marginal(
                full_diffuser,
                rigids_0=rigids_0,
                seq_0 = seq_0,
                t = t,
                diffuse_mask=diffused_mask
            )

    batch.update(diff_feats_t)
    batch.update(
        t = t,
        struc_loss_mask = struc_loss_mask,
        fixed_mask = fixed_mask,
        rigids_0 = rigids_0
    )
    return batch


@take1st
def make_to_device(protein, fields, device, is_training=True):
    if isfunction(device):
        device = device()
    for k in fields:
        if k in protein:
            protein[k] = protein[k].to(device)
    return protein

@take1st
def make_selection(protein, fields, is_training=True):
    return {k: protein[k] for k in fields}


class FeatureBuilder:
    def __init__(self, config, is_training=True):
        self.config = config
        self.training = is_training

    def build(self, protein):
        for fn, kwargs in default(self.config, []):
            f = _feats_fn[fn](is_training=self.training, **kwargs)
            # print(f"f: {f}")
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
