import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from abx.model.common_modules import(
        Linear,
        LayerNorm,
        apply_dropout,)
from abx.common import residue_constants

from abx.model.encoder import ResidueEmbedding, PairEmbedding, ESMEmbedding
import pdb
import logging
import functools as fn
import math

logger = logging.getLogger(__name__)

def pair_concat(pair_1, pair_2):
    assert pair_1.shape[0] == pair_2.shape[0] and pair_1.shape[-1] == pair_2.shape[-1]
    assert pair_1.device == pair_2.device
    device = pair_1.device
    batch_size = pair_1.shape[0]
    channel = pair_1.shape[-1]

    length_1 = pair_1.shape[1]
    length_2 = pair_2.shape[1]
    concat_dim1 = torch.cat(
        (
        pair_1, 
        torch.zeros((batch_size, length_2, length_1, channel), device=device)
        ), dim=1)
    
    concat_dim2 = torch.cat(
        (
        torch.zeros((batch_size, length_1, length_2, channel), device=device), 
        pair_2
        ), dim=1)
    pair_all = torch.cat([concat_dim1, concat_dim2], dim=2)
    return pair_all



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    From Fairseq.Build sinusoidal embeddings.This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # Zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):
    """
    Evoformer for ESMFold and t embedding
    Returns:
        node_embed: [B, N,  seq_channel + index_embed_size]
        edge_embed: [B, N, N, pair_channel + 2*index_embed_size]
    """
    def __init__(self, model_conf):
        super(Embedder,self).__init__()
        self._embed_conf = model_conf
        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        
        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim = index_embed_size
        )


    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])


    def forward(self, seq_act, pair_act, batch):
        num_batch, num_res = batch['seq'].shape
        # node_feats = []
        t = batch['t'] 
        fixed_mask = batch['fixed_mask']

        # Set time step to epsilon=1e-5 for fixed residues
        # Embedding times t
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1,num_res,1)
        ) # [B, N, index_embed_size]

        # Evoformer embedding
        pair_act = pair_act.reshape([num_batch, num_res**2, -1])

        seq_feats = [seq_act]
        pair_feats = [pair_act]
        seq_feats.append(prot_t_embed)
        pair_feats.append(self._cross_concat(prot_t_embed, num_batch, num_res))

        
        pair_feats = torch.cat(pair_feats, dim=-1).float()
        seq_feats = torch.cat(seq_feats, dim=-1).float()
        pair_feats = pair_feats.reshape([num_batch, num_res, num_res, -1])

        return seq_feats, pair_feats
    


class EmbeddingAndSeqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.num_token = residue_constants.restype_num + 3
        self.num_region = residue_constants.num_ab_regions + 1

        self.proj_aa_type = nn.Embedding(self.num_token, c.seq_channel, padding_idx=20)
        self.encode_residue_emb = ResidueEmbedding(c)
        self.encode_pair_emb = PairEmbedding(c)

        self.aa_proj = nn.Sequential(
            LayerNorm(c.seq_channel),
            Linear(c.seq_channel, c.seq_channel, init='linear', bias=True),
            nn.ReLU(),
            Linear(c.seq_channel, c.seq_channel, init='linear', bias=True),
            #nn.Dropout(p=c.esm.dropout_rate),
        )

        if c.esm.enabled:
            self.encode_esm_emb = ESMEmbedding(c)

            esm_embed_weights = torch.zeros((c.esm.num_layers + 1,))
            self.esm_embed_weights = nn.Parameter(esm_embed_weights)

            self.proj_esm_embed = nn.Sequential(
                LayerNorm(c.esm.embed_channel),
                Linear(c.esm.embed_channel, c.seq_channel, init='linear', bias=True),
                nn.ReLU(),
                Linear(c.seq_channel, c.seq_channel, init='linear', bias=True),
                )
        
        self.proj_rel_pos = torch.nn.Embedding(c.max_relative_feature * 2 + 2, c.pair_channel)

        if c.recycle_features:
            self.prev_seq_norm = LayerNorm(c.seq_channel+c.index_embed_size)
            self.prev_pair_norm = LayerNorm(c.pair_channel+2*c.index_embed_size)

        if c.recycle_pos:
            self.proj_prev_pos = nn.Embedding(c.prev_pos.num_bins, c.pair_channel+2*c.index_embed_size)

        self.seqformer = Seqformer(c)
        self.t_embeder = Embedder(c)

        self.config = config

    def forward(self, batch):
        c = self.config

        seq, mask, seq_pos= batch['seq_t'], batch['mask'], batch['residx']
        antibody_len = batch['anchor_flag'].shape[1]

        # Antibody Encoder Information
        antibody_seq = seq[:, :antibody_len]
        antibody_seq_pos = seq_pos[:, :antibody_len]
        antibody_offset = rearrange(antibody_seq_pos, 'b l -> b () l') - rearrange(antibody_seq_pos, 'b l -> b l ()')
        antibody_rel_pos = torch.clip(antibody_offset + c.max_relative_feature, min=0, max=2*c.max_relative_feature) + 1

        antibody_seq_act = self.proj_aa_type(antibody_seq.long())
        antibody_pair_act = self.proj_rel_pos(antibody_rel_pos)

        if c.esm.enabled:
            layer_weights = F.softmax(self.esm_embed_weights, dim=-1)
            
            antibody_esm_embed = self.encode_esm_emb(batch).to(dtype=layer_weights.dtype)
            antibody_esm_embed = torch.einsum('b l c n, n -> b l c', antibody_esm_embed, layer_weights)
            antibody_esm_embed = self.proj_esm_embed(antibody_esm_embed)
            antibody_seq_act = antibody_seq_act + antibody_esm_embed

        # Antigen Encoder Information
        antigen_seq = batch['seq'][:,antibody_len:]
        antigen_seq_pos = batch['residx'][:,antibody_len:]
        antigen_offset = rearrange(antigen_seq_pos, 'b l -> b () l') - rearrange(antigen_seq_pos, 'b l -> b l ()')
        antigen_rel_pos = torch.clip(antigen_offset + c.max_relative_feature, min=0, max=2*c.max_relative_feature) + 1

        antigen_seq_embed = self.proj_aa_type(antigen_seq)
        antigen_seq_act = self.aa_proj(antigen_seq_embed)
        antigen_pair_act = self.proj_rel_pos(antigen_rel_pos)


        # Encoder Information
        seq_act = torch.cat((antibody_seq_act, antigen_seq_act), dim=1)
        pair_act = pair_concat(antibody_pair_act, antigen_pair_act)

        encoder_seq_act =  self.encode_residue_emb(batch)
        encoder_pair_act = self.encode_pair_emb(batch)
        
        seq_act = seq_act + encoder_seq_act
        pair_act = pair_act + encoder_pair_act

        seq_act, pair_act = self.t_embeder(seq_act, pair_act, batch)

        if c.recycle_features:
            if 'prev_seq' in batch:
                seq_act = seq_act + self.prev_seq_norm(batch['prev_seq'])
            if 'prev_pair' in batch:
                pair_act = pair_act + self.prev_pair_norm(batch['prev_pair'])

        if c.recycle_pos and 'prev_pos' in batch:
            pair_act = pair_act + self.proj_prev_pos(batch['prev_pos'])
        seq_act, pair_act = self.seqformer(seq_act, pair_act, mask=mask, is_recycling=batch['is_recycling'])

        return seq_act, pair_act

class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,split_first=True, gating=True,inp_kernels=None, config=None):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim, self.value_dim = key_dim, value_dim

        self.num_head = num_head
        
        self.split_first = split_first

        if self.split_first:
            self.proj_q = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_k = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_v = Linear(input_dim, value_dim, init='attn', bias=False, config=config)
        else:
            assert (key_dim == value_dim)
            self.proj_in = Linear(input_dim, key_dim * 3, init='attn', bias=False, config=config)
        
        self.gating = gating
        if gating:
            self.gate= Linear(input_dim, value_dim, init='gate', config=config)

        self.proj_out = Linear(value_dim, output_dim, init='final', config=config)
         
        self.inp_kernels = inp_kernels
        if inp_kernels:
            self.inp_q = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_k = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_v = SpatialDepthWiseInception(value_dim // num_head, inp_kernels)

    def forward(self, q_data, k_data=None, bias=None, k_mask=None):
        """
        Arguments:
            q_data: (batch_size, N_seqs, N_queries, q_channel)
            k_data: (batch_size, N_seqs, N_keys, k_channel)
            k_mask: (batch_size, N_seqs, N_keys)
            bias  : (batch_size, N_queries, N_keys). shared by all seqs
        Returns:
            (b s l c)
        """
        key_dim, value_dim = self.key_dim // self.num_head, self.value_dim // self.num_head
        
        if self.split_first:
            assert (k_data is not None)
            q = self.proj_q(q_data) 
            k = self.proj_k(k_data)
            v = self.proj_v(k_data)
            q, k, v = map(lambda t: rearrange(t, 'b s l (h d) -> b s h l d', h = self.num_head), (q, k, v))
        else:
            assert (k_data is None)
            t = rearrange(self.proj_in(q_data), "... l (h d) -> ... h l d", h=self.num_head)
            q, k, v = torch.chunk(t, 3, dim=-1)
        
        if self.inp_kernels:
            q, k, v = map(lambda t: rearrange(t, 'b s h l d-> b (s h) l d'), (q, k, v))
            q = self.inp_q(q)
            k = self.inp_k(k)
            v = self.inp_v(v)
            q, k, v = map(lambda t: rearrange(t, 'b (s h) l d-> b s h l d', h = self.num_head), (q, k, v))
        
        q = q* key_dim**(-0.5)

        logits = torch.einsum('... h q d, ... h k d -> ... h q k', q, k)

        if bias is not None:
            logits = logits + rearrange(bias,  'b h q k -> b () h q k')

        if k_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            k_mask = rearrange(k_mask, 'b s k -> b s () () k')
            logits = logits.masked_fill(~k_mask.bool(), mask_value)

        weights = F.softmax(logits, dim = -1)
        weighted_avg = torch.einsum('b s h q k, b s h k d -> b s h q d', weights, v)
        weighted_avg = rearrange(weighted_avg, 'b s h q d -> b s q (h d)')
        
        if self.gating:
            gate_values = torch.sigmoid(self.gate(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.proj_out(weighted_avg)

        return output

class SeqAttentionWithPairBias(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.seq_norm = LayerNorm(num_in_seq_channel)
        self.pair_norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)

        self.attn = Attention(
                input_dim=num_in_seq_channel,
                key_dim=num_in_seq_channel,
                value_dim=num_in_seq_channel,
                output_dim=num_in_seq_channel,
                num_head=c.num_head,
                split_first=False,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, seq_act, pair_act, mask):
        """
        Arguments:
            seq_act: (b l c)
            pair_act: (b l l c)
            mask: (b l), padding mask
        Returns:
            (b l c)
        """
        mask = rearrange(mask, 'b l -> b () l')
        seq_act = self.seq_norm(seq_act)
        
        pair_act = self.pair_norm(pair_act)
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        
        seq_act = rearrange(seq_act, 'b l c -> b () l c')
        seq_act = self.attn(q_data=seq_act, bias=bias, k_mask=mask)
        seq_act = rearrange(seq_act, 'b s l c -> (b s) l c')
        return seq_act

class Transition(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        intermediate_channel = num_in_channel * c.num_intermediate_factor
        self.transition = nn.Sequential(
                LayerNorm(num_in_channel),
                Linear(num_in_channel, intermediate_channel, init='linear', config=LoRA_conf),
                nn.ReLU(),
                Linear(intermediate_channel, num_in_channel, init='final', config=LoRA_conf),
                )

    def forward(self, act, mask):
        return self.transition(act)

# AF2 and ESM-FOLD have different implementations
# Here we just follow ESMFOLD
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)

        self.out_proj = Linear(2 * c.num_outer_channel, num_out_channel, init='final', config=LoRA_conf)

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)
        
        prod = left_act[:, None, :, :] * right_act[:, :, None, :]
        diff = left_act[:, None, :, :] - right_act[:, :, None, :]

        act = torch.cat([prod, diff], dim=-1)
        act = self.out_proj(act)

        return act

class TriangleMultiplication(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)

        self.left_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)

        self.final_norm = LayerNorm(c.num_intermediate_channel)
        
        if c.gating:
            self.left_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.right_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.final_gate = Linear(num_in_channel, num_in_channel, init='gate', config=LoRA_conf)
        
        self.proj_out = Linear(c.num_intermediate_channel, num_in_channel, init='final', config=LoRA_conf)

        
        if c.inp_kernels:
            self.inp_left = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)
            self.inp_right = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)

        self.config = c

    def forward(self, act, mask):
        """
        act: (b l l c)
        mask: (b l)
        """
        c = self.config

        #pair_mask = rearrange(mask, 'b l -> b l () ()') * rearrange(mask, 'b l -> b () l ()')
        pair_mask = mask[:,:,None,None] * mask[:,None,:,None]
        
        act = self.norm(act)

        input_act = act

        left_proj_act = self.left_proj(act)
        right_proj_act = self.right_proj(act)
        
        if c.inp_kernels:
            if c.orientation == 'per_row':
                equation = 'b i j (h d) -> b (i h) j d'
            else:
                equation = 'b i j (h d) -> b (j h) i d'

            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))

            left_proj_act = self.inp_left(left_proj_act)
            right_proj_act = self.inp_right(right_proj_act)
            
            if c.orientation == 'per_row':
                equation = 'b (i h) j d -> b i j (h d)'
            else:
                equation = 'b (j h) i d -> b i j (h d)'
            
            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))
        
        left_proj_act = pair_mask * left_proj_act
        right_proj_act = pair_mask * right_proj_act
        
        if c.gating:
            left_gate_values = torch.sigmoid(self.left_gate(act))
            right_gate_values = torch.sigmoid(self.right_gate(act))

            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        if c.orientation == 'per_row':
            act = torch.einsum('b i k c, b j k c -> b i j c', left_proj_act, right_proj_act)
        elif c.orientation == 'per_column':
            act = torch.einsum('b k i c, b k j c -> b i j c', left_proj_act, right_proj_act)
        else:
            raise NotImplementedError(f'{self.orientation} not Implemented')

        act = self.final_norm(act)
        act = self.proj_out(act)
        
        if c.gating:
            gate_values = torch.sigmoid(self.final_gate(input_act))
            act = act * gate_values

        return act

class TriangleAttention(nn.Module):
    def __init__(self, config, num_in_pair_channel):
        super().__init__()
        c = config

        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None

        self.norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)
        self.attn = Attention(
                input_dim=num_in_pair_channel,
                key_dim=num_in_pair_channel,
                value_dim=num_in_pair_channel,
                output_dim=num_in_pair_channel,
                num_head=c.num_head,
                gating=c.gating,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, pair_act, seq_mask):
        '''
        pair_act: (b l l c)
        seq_mask: (b l)
        '''
        c = self.config
        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        pair_act = self.norm(pair_act)
        seq_mask = rearrange(seq_mask, 'b l -> b () l')

        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')

        pair_act = self.attn(q_data=pair_act, k_data=pair_act, bias=bias, k_mask=seq_mask)

        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        return pair_act

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)
        
        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act
        
        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)
        
        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)
        
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask), c.triangle_multiplication_incoming)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)
        
        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([SeqformerIteration(c.seqformer, c.seq_channel+c.index_embed_size, c.pair_channel+2*c.index_embed_size) for _ in range(c.seqformer_num_block)])

    def forward(self, seq_act, pair_act, mask, is_recycling=True):
        for it, block in enumerate(self.blocks):
            block_fn = functools.partial(block, seq_mask=mask)
            if self.training and not is_recycling and it > 0:
                seq_act, pair_act = checkpoint(block_fn, seq_act, pair_act)
                # for name, param in block.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}")
                #         print("Parameter value: ", param.data)
                #         print("Parameter gradient: ", param.grad)
                # print(f"seq_act: {seq_act}")
                # print(f"pair_act: {pair_act}")
                #seq_act, pair_act = block_fn(seq_act, pair_act)
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)
        return seq_act, pair_act

class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                kernel_size=(kernel_size,),
                # padding=(kernel_size - 1,),
                padding=kernel_size//2,
                groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        #if self.kernel_size>1:
        #    x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x

class SpatialDepthWiseInception(nn.Module):
    def __init__(self, head_dim, kernels):
        super().__init__()
       
        assert len(kernels) > 1 and  kernels[0] == 1

        self.convs = torch.nn.ModuleList([SpatialDepthWiseConvolution(head_dim, kernel_size=k) for k in kernels[1:]])
        self.kernels = kernels
    def forward(self, x):
        # x: (batch, num_heads, len, head_dim)
        
        assert x.shape[1] % len(self.kernels) == 0
        group_num_head = x.shape[1] // len(self.kernels)
        
        outputs = [x[:,:group_num_head]]

        for i, conv in enumerate(self.convs):
            outputs.append(conv(x[:,group_num_head*(i+1):group_num_head*(i+2)]))

        outputs = torch.cat(outputs, dim=1)

        return outputs
