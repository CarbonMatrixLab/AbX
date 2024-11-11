import torch
import math
from torch import nn
from torch.nn import functional as F
import random
from abx.model.folding import InvariantPointAttention as IPA
import functools as fn
from abx.model.seqformer import EmbeddingAndSeqformer
from abx.utils import *
from einops import rearrange
import logging

Tensor = torch.Tensor
from abx.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)
from abx.model.sidechain import TorsionModule
from abx.model.utils import batched_select
from abx.model import r3
from abx.model import quat_affine
from abx.model.common_modules import(
        Linear,
        LayerNorm)
from abx.model.sidechain import MultiRigidSidechain
import pdb


logger = logging.getLogger()

class IpaScore(nn.Module):
    """
    Modified from Structure Module of AlphaFold2
    Return:
        rot_score: [B, N, N, 3]
        trans_score: [B, N, 3]
        final_atom_positions: [B, N, 37, 3]
        final_atom14_positions: [B, N, 14, 3]
        final_affines: ([B, N, 3, 3], [B, N, 3])
        rigids: [B, N, 7]
        traj: list of intermediate status of rigids len(traj) = num_layer
    """
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel, diffuser):
        super(IpaScore,self).__init__()        
        c = config.IPA
        self.score_network_conf = config
        self._embed_conf = config.embed
        self.config = c
        self.diffuser = diffuser
        num_pair_channel = num_in_pair_channel

        self.num_in_seq_channel = self._embed_conf.index_embed_size
        self.num_in_pair_channel =  2 * self.num_in_seq_channel
        self.num_in_seq_channel += num_in_seq_channel
        self.num_in_pair_channel += num_in_pair_channel

        self.proj_init_seq_act = Linear(self.num_in_seq_channel, c.num_channel, init='linear')
        self.proj_init_pair_act = Linear(self.num_in_pair_channel, num_pair_channel, init='linear')

        self.init_seq_layer_norm = LayerNorm(c.num_channel)
        self.init_pair_layer_norm = LayerNorm(num_pair_channel)

        self.proj_seq = Linear(c.num_channel, c.num_channel, init='linear')

        self.attention_module = IPA(c, num_pair_channel)
        self.attention_layer_norm = LayerNorm(c.num_channel)

        transition_moduel = []
        for k in range(c.num_layer_in_transition):
            is_last = (k == c.num_layer_in_transition - 1)
            transition_moduel.append(
                    Linear(c.num_channel, c.num_channel, init='linear' if is_last else 'final'))
            if not is_last:
                transition_moduel.append(nn.ReLU())
        self.transition_module = nn.Sequential(*transition_moduel)
        self.transition_layer_norm = LayerNorm(c.num_channel)

        self.affine_update = Linear(c.num_channel, 6, init='final')
        self.sidechain_module = MultiRigidSidechain(c, num_in_seq_channel)
    
    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, representations, batch):
        c = self.config
        b, n, device = batch['seq_t'].shape[0], batch['anchor_flag'].shape[1] ,batch['seq_t'].device
        seq_act, static_pair_act = representations['seq'], representations['pair']
        seq = batch['seq_t']
        node_mask = batch['mask'].type(torch.float32)
        fixed_mask = batch['fixed_mask']
        init_rigids = batch['rigids_t'].type(torch.float32)

        # seq = torch.cat((seq, batch['antigen_seq']), dim=-1)
        # node_mask = torch.cat((node_mask, batch['antigen_mask'].type(torch.float32)), dim=-1)
        # fixed_mask = torch.cat((fixed_mask, torch.ones_like(batch['antigen_seq'], dtype=torch.int32, device=device)), dim=-1)

        # antigen_bb_rigid = r3.rigids_op(batch['antigen_rigidgroups_gt_frames'], lambda x: x[:,:,0])

        # antigen_rigids = r3.rigids_to_tensor7(antigen_bb_rigid)
        # init_rigids = torch.cat((
        #     init_rigids,
        #     antigen_rigids
        # ), dim=1)

        init_trans = init_rigids[..., 4:]
        init_quats = init_rigids[..., :4]

        delta_quat, _ = quat_affine.make_identity(out_shape=(b, seq.shape[1]), device=seq_act.device)

        curr_rigids = init_rigids
        curr_quats = curr_rigids[..., :4]
        curr_trans = curr_rigids[..., 4:]
        curr_rots = quat_affine.quat_to_rot(curr_quats)

        curr_trans = curr_trans / c.position_scale
        
        # Main Trunk
        seq_act = self.proj_init_seq_act(seq_act)
        static_pair_act = self.proj_init_pair_act(static_pair_act)
        seq_act = self.init_seq_layer_norm(seq_act)
        static_pair_act = self.init_pair_layer_norm(static_pair_act)

        initial_seq_act = seq_act
        seq_act = self.proj_seq(seq_act)
        outputs = dict(traj = [], sidechains=[])

        for fold_it in range(c.num_layer):
            is_last = (fold_it == (c.num_layer - 1))
            seq_act = seq_act + self.attention_module(inputs_1d = seq_act, inputs_2d = static_pair_act, mask = node_mask, in_rigids=(curr_rots, curr_trans))
        
            seq_act = F.dropout(seq_act, p = c.dropout, training=self.training)
            seq_act = self.attention_layer_norm(seq_act)
            
            seq_act = seq_act + self.transition_module(seq_act)
            seq_act = F.dropout(seq_act, p = c.dropout, training=self.training)
            seq_act = self.transition_layer_norm(seq_act)

            quaternion_update, translation_update = self.affine_update(seq_act).chunk(2, dim = -1)
            delta_quat = quat_affine.quat_precompose_vec(delta_quat, quaternion_update)
            curr_quats= quat_affine.quat_precompose_vec(curr_quats, quaternion_update)
            curr_trans = r3.rigids_mul_vecs((curr_rots, curr_trans), translation_update)

            curr_quats = self._apply_mask(
                curr_quats, init_quats, 1-fixed_mask[...,None]
            )
            curr_trans = self._apply_mask(
                curr_trans, init_trans / c.position_scale, 1-fixed_mask[...,None]
            )

            curr_rots = quat_affine.quat_to_rot(curr_quats)

            outputs['traj'].append((curr_rots, curr_trans * c.position_scale ))

            if self.training or is_last:
                sidechains = self.sidechain_module(
                        seq,
                        (curr_rots, curr_trans * c.position_scale),
                        [seq_act, initial_seq_act], batch, compute_atom_pos=is_last)
                
                outputs['sidechains'].append(sidechains)

            if not is_last:
                curr_rots = curr_rots.detach()
                curr_quats = curr_quats.detach()


        curr_quats_= quat_affine.quat_multiply(init_quats, delta_quat)
        curr_quats_ = self._apply_mask(
                curr_quats_, init_quats, 1-fixed_mask[...,None]
            )
        # quat_ = curr_quats_[:, :n,...]
        # trans = curr_trans[:, :n,...]

        rot_score = self.diffuser.calc_quat_score(
            init_quats,
            curr_quats_,
            batch['t']
        )
        # rot_score = rot_score[:, :n, ...]

        trans_score = self.diffuser.calc_trans_score(
            init_trans,
            curr_trans * c.position_scale,
            batch['t']
        )
        # trans_score = trans_score[:, :n, ...]
        
        outputs['trans_score'] = trans_score
        outputs['rot_score'] = rot_score
        outputs['representations'] = {'structure_module': seq_act}

        rigids = torch.zeros_like(init_rigids, device=curr_rots.device, dtype=curr_rots.dtype)
        rigids[..., :4] = curr_quats_
        rigids[..., 4:] = curr_trans * c.position_scale
        outputs['rigids'] = rigids

        return outputs







