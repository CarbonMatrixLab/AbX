import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from abx.model.utils import batched_select
from abx.model import r3
from abx.model import quat_affine
from abx.model.common_modules import(
        Linear,
        LayerNorm)
from abx.model.seqformer import SpatialDepthWiseInception
from abx.model.sidechain import MultiRigidSidechain

logger = logging.getLogger(__name__)

def print_grad(grad, name):
    print(f"Gradient of {name}: {grad}")

class InvariantPointAttention(nn.Module):
    def __init__(self, config, num_in_pair_channel, dist_epsilon = 1e-8):
        super().__init__()

        c = config

        bias=True
        self.proj_q_scalar = Linear(c.num_channel, c.num_head * c.num_scalar_qk, init='attn', bias=bias)
        self.proj_kv_scalar = Linear(c.num_channel, c.num_head * (c.num_scalar_v + c.num_scalar_qk), init='attn', bias=bias)

        self.proj_q_point_local = Linear(c.num_channel, 3 * c.num_head * c.num_point_qk, init='attn', bias=bias)
        self.proj_kv_point_local = Linear(c.num_channel, 3 * c.num_head * (c.num_point_v + c.num_point_qk), init='attn', bias=bias)

        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='attn', bias=bias)

        point_weight_init_value = torch.log(torch.exp(torch.full((c.num_head,), 1.)) - 1.)
        self.trainable_point_weights = nn.Parameter(point_weight_init_value)

        self.final_proj = Linear(
                c.num_head * (c.num_scalar_v + num_in_pair_channel + c.num_point_v * (3 + 1)), c.num_channel, init='final')

        self.config = c
        self.dist_epsilon = dist_epsilon
        
    def forward(self, inputs_1d, inputs_2d, mask, in_rigids):
        batch_size, num_residues, _ = inputs_1d.shape

        c = self.config

        num_head = c.num_head
        num_scalar_qk = c.num_scalar_qk
        num_point_qk = c.num_point_qk
        num_scalar_v = c.num_scalar_v
        num_point_v = c.num_point_v
        
        # hyper settings
        scalar_variance = max(num_scalar_qk, 1) * 1.
        point_variance = max(num_point_qk, 1) * 9. / 2

        num_logit_terms = 3

        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))
        
        # calculation begins
        q_scalar = self.proj_q_scalar(inputs_1d)
        q_scalar = rearrange(q_scalar, 'b l (h c) -> b h l c', h = num_head)

        kv_scalar = self.proj_kv_scalar(inputs_1d)
        # print(f"IPA proj_q_scalar layer weights: {self.proj_q_scalar.weight.grad}")
        # print(f"IPA proj_kv_scalar layer weights: {self.proj_kv_scalar.bias.grad}")

        kv_scalar = rearrange(kv_scalar, 'b l (h c) -> b h l c', h = num_head)
        k_scalar, v_scalar = torch.split(kv_scalar, [num_scalar_qk, num_scalar_v], dim=-1)

        attn_qk_scalar = torch.einsum('b h i c, b h j c -> b h i j', q_scalar * scalar_weights, k_scalar)

        q_point_local = self.proj_q_point_local(inputs_1d)
        q_point_local = rearrange(q_point_local, 'b l (r n) -> b l n r', r=3)

        kv_point_local = self.proj_kv_point_local(inputs_1d)

        kv_point_local = rearrange(kv_point_local, 'b l (r n) -> b l n r', r=3)

        # to global
        q_point_global = r3.rigids_apply(in_rigids, q_point_local)
        kv_point_global = r3.rigids_apply(in_rigids, kv_point_local)
        q_point_global = rearrange(q_point_global, 'b l (h n) r -> b l h n r', h=num_head)
        kv_point_global = rearrange(kv_point_global, 'b l (h n) r -> b l h n r', h=num_head)
        k_point_global, v_point_global = torch.split(kv_point_global, [num_point_qk, num_point_v], dim=-2)
        
        dist2 = torch.sum(torch.square(rearrange(q_point_global, 'b i h n r -> b i () h n r') - rearrange(k_point_global, 'b j h n r -> b () j h n r')), dim=[-1,-2])
        point_weights = -0.5 * point_weights * F.softplus(self.trainable_point_weights)
        attn_qk_point = rearrange(point_weights * dist2, 'b i j h -> b h i j')

        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.proj_pair(inputs_2d)
        attention_2d = attention_2d_weights * rearrange(attention_2d, 'b i j h -> b h i j')

        attn_logits = attn_logits + attention_2d

        mask_2d = rearrange(mask, 'b l -> b l ()') * rearrange(mask, 'b l -> b () l')
        mask_2d = rearrange(mask_2d, 'b i j -> b () i  j')
        mask_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(~mask_2d.bool(), mask_value)

        attn = F.softmax(attn_logits, dim=-1)

        # results on scalar
        result_scalar = torch.matmul(attn, v_scalar)
        result_scalar = rearrange(result_scalar, 'b h l c -> b l (h c)')
        output_features = [result_scalar]

        # results on points
        result_point_global = torch.einsum('b h i j, b j h n r -> b h i n r', attn, v_point_global)
        result_point_global = rearrange(result_point_global, 'b h l n r -> b l (h n) r')
        result_point_local = r3.rigids_apply(r3.invert_rigids(in_rigids), result_point_global)
        output_features.append(rearrange(result_point_local, 'b l n r -> b l (r n)'))
        output_features.append(torch.sqrt(torch.sum(torch.square(result_point_local), dim=-1) + self.dist_epsilon))

        # results on input 2d
        result_attention_over_2d = torch.einsum('b h i j, b i j c -> b h i c', attn, inputs_2d)
        result_attention_over_2d = rearrange(result_attention_over_2d, 'b h l c -> b l (h c)')
        output_features.append(result_attention_over_2d)
        
        final_act = torch.cat(output_features, dim=-1)

        return self.final_proj(final_act)

class StructureModule(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()

        c = config
        
        self.proj_init_seq_act = Linear(num_in_seq_channel, c.num_channel, init='linear')
        self.proj_init_pair_act = Linear(num_in_pair_channel, num_in_pair_channel, init='linear')

        self.init_seq_layer_norm = LayerNorm(c.num_channel)
        self.init_pair_layer_norm = LayerNorm(num_in_pair_channel)

        self.proj_seq = Linear(c.num_channel, c.num_channel, init='linear')

        self.attention_module = InvariantPointAttention(c, num_in_pair_channel)
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

        self.sidechain_module = MultiRigidSidechain(config, num_in_seq_channel)

        self.config = c

    def forward(self, representations, batch):
        c = self.config
        b, n, device = *batch['seq'].shape[:2], batch['seq'].device

        seq_act, static_pair_act = representations['seq'], representations['pair']
        
        seq_act = self.proj_init_seq_act(seq_act)
        static_pair_act = self.proj_init_pair_act(static_pair_act)
        seq_act = self.init_seq_layer_norm(seq_act)
        static_pair_act = self.init_pair_layer_norm(static_pair_act)

        initial_seq_act = seq_act
        seq_act = self.proj_seq(seq_act)

        outputs = dict(traj = [], sidechains=[])

        quaternions, translations = quat_affine.make_identity(out_shape=(b, n), device=seq_act.device)
        rotations = quat_affine.quat_to_rot(quaternions)

        for fold_it in range(c.num_layer):
            is_last = (fold_it == (c.num_layer - 1))
            seq_act = seq_act + self.attention_module(inputs_1d = seq_act, inputs_2d = static_pair_act, mask = batch['mask'], in_rigids=(rotations, translations))
            seq_act = F.dropout(seq_act, p = c.dropout, training=self.training)
            seq_act = self.attention_layer_norm(seq_act)
            
            seq_act = seq_act + self.transition_module(seq_act)
            seq_act = F.dropout(seq_act, p = c.dropout, training=self.training)
            seq_act = self.transition_layer_norm(seq_act)
            
            quaternion_update, translation_update = self.affine_update(seq_act).chunk(2, dim = -1)
            
            quaternions = quat_affine.quat_precompose_vec(quaternions, quaternion_update)
            translations = r3.rigids_mul_vecs((rotations, translations), translation_update)
            rotations = quat_affine.quat_to_rot(quaternions)
            
            outputs['traj'].append((rotations, translations * c.position_scale))

            if self.training or is_last:
                sidechains = self.sidechain_module(
                        batch['seq'],
                        (rotations, translations * c.position_scale),
                        [seq_act, initial_seq_act], compute_atom_pos=is_last)

                outputs['sidechains'].append(sidechains)

            if not is_last:
                rotations = rotations.detach()

        outputs['representations'] = {'structure_module': seq_act}

        outputs['final_atom14_positions'] = outputs['sidechains'][-1]['atom_pos']

        outputs['final_atom_positions'] = batched_select(outputs['final_atom14_positions'], batch['residx_atom37_to_atom14'], batch_dims=2)
        outputs['final_affines'] = outputs['traj'][-1]
        ca = outputs['traj'][-1][1]

        return outputs
