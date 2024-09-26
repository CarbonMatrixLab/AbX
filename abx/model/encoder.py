import torch
import torch.nn as nn
import torch.nn.functional as F

from abx.common import residue_constants
from abx.data.esm import ESMEmbeddingExtractor
from abx.data.utils import pad_for_batch
from abx.model.utils import batched_select
from einops import rearrange

from esm.pretrained import load_model_and_alphabet_local

from abx.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions,
        Linear,
        LayerNorm,
        apply_dropout,)
import pdb

class ESMEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.esm.esm_embed

        self.sep_pad_num = self.config.sep_pad_num
        self.return_attnw = self.config.return_attnw
        self.repr_layer = self.config.repr_layer

        self.model_path = self.config.model_path
        self.model, self.alphabet = load_model_and_alphabet_local(self.model_path)
        self.model.requires_grad_(False)
        self.model.half()
        self.batch_converter = self.alphabet.get_batch_converter()

    
    def _make_one_antibody_seq(self, seq, heavy_length, light_length):
        heavy_seq = seq[:heavy_length]
        light_seq = seq[heavy_length: light_length+heavy_length]
        # heavy_seq = torch.clamp(heavy_seq, min=0, max=19)
        # light_seq = torch.clamp(light_seq, min=0, max=19)

        str_heavy_seq_ = ''.join([residue_constants.restypes_with_x[index] for index in heavy_seq])
        str_light_seq_ = ''.join([residue_constants.restypes_with_x[index] for index in light_seq])
        return str_heavy_seq_, str_light_seq_
    
    def extract(self, label_seqs, device, linker_mask=None):

        max_len = max([len(s) for l, s in label_seqs])

        if type(self.repr_layer) is int:
            self.repr_layer = [self.repr_layer]

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter(label_seqs)
            batch_tokens = batch_tokens.to(device=device)
            
            if linker_mask is not None:
                batch_tokens = torch.where(linker_mask, self.alphabet.padding_idx, batch_tokens)
            
            results = self.model(batch_tokens, repr_layers=self.repr_layer, need_head_weights=self.return_attnw)
            single = [results['representations'][r][:,1 : 1 + max_len] for r in self.repr_layer]

            ret = dict(single=single)

            # if self.return_attnw:
            #     atten = rearrange(results['attentions'][:, :, :, 1:1+max_len, 1:1+max_len], 'b l d i j -> b i j (l d)')
            #     ret['pair'] = (atten + rearrange(atten, 'b i j h -> b j i h')) * 0.5
        
        return ret
    
    def forward(self, batch):
        seq_t, antibody_len = batch['seq_t'], batch['anchor_flag'].shape[1]
        antibody_seq_t = batch['seq_t'][:,:antibody_len]

        str_seq_t = [self._make_one_antibody_seq(antibody_seq_t[k], len(x), len(y)) for k, (x,y) in enumerate(zip(batch['str_heavy_seq'],batch['str_light_seq']))]

        batch['str_heavy_seq_t'], batch['str_light_seq_t'] = zip(*str_seq_t)
        
        def _one(key, linker_mask=None):
            data_in = list(zip(batch['name'], batch[key]))
            data_out = self.extract(data_in, linker_mask=linker_mask, device=seq_t.device)
            return data_out

        lengths = (len(h) + len(l) for h, l in zip(batch['str_heavy_seq_t'], batch['str_light_seq_t']))
        batch_length =  max(lengths)
        if self.sep_pad_num == 0:
            heavy_embed = _one('str_heavy_seq_t')
            light_embed = _one('str_light_seq_t')

            embed = [torch.cat([heavy_embed['single'][k,:len(x)], light_embed['single'][k,:len(y)]], dim=0) for k, (x,y) in enumerate(zip(batch['str_heavy_seq'],batch['str_light_seq']))]
        else:
            batch['sep_pad_seq'] = [h + 'G' * self.sep_pad_num + l for h, l in zip(batch['str_heavy_seq_t'], batch['str_light_seq_t'])]

            embed = _one('sep_pad_seq', linker_mask=None)

            if len(embed['single']) == 1:
                embed = embed['single'][0]
            else:
                embed = torch.stack(embed['single'], dim=-1)
            
            embed = [torch.cat([
                embed[k,:len(x)],
                embed[k,len(x)+self.sep_pad_num:len(x)+self.sep_pad_num+len(y)]],
                dim=0) for k, (x,y) in enumerate(zip(batch['str_heavy_seq_t'],batch['str_light_seq_t']))]


            esm_embed = pad_for_batch(embed, batch_length, dtype='ebd')
            
            # # FIXED ME
            # if self.return_attnw:
            #     embed = [torch.cat([
            #         F.pad(light_embed['pair'][k,:len(x), :len(x)], (0, 0, 0, len(y)), value=0.),
            #         F.pad(heavy_embed['pair'][k,:len(y), :len(y)], (0, 0, len(x), 0), value=0.)], dim=0) for k, (x,y) in enumerate(zip(batch['st_light_seq'],batch['str_heavy_seq']))]


            #     esm_pair_embed = pad_for_batch(embed, batch_length, dtype='pair')
            #     return esm_embed, esm_pair_embed
                

            return esm_embed

class ResidueEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.seq_channel
        self.max_aa_types = residue_constants.restype_num
        self.aatype_embed = nn.Embedding(self.max_aa_types+3, feat_dim)
        self.cdr_embed = nn.Embedding(residue_constants.num_ab_regions+1, feat_dim)
        
        self.coordinate_embed = nn.Sequential(
            Linear(14*3 + 7*2, feat_dim, init='linear', bias=True), 
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),             
        )

        infeat_dim = feat_dim * 3 + 2
        self.mlp = nn.Sequential(
            Linear(infeat_dim, feat_dim*2, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim*2, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
        )

    def forward(self, batch):
        """
        Args:
            aa:         (N, L).
            residx:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            fragment_type:  (N, L).
        """
        mask, fixed_mask = batch['mask'], batch['fixed_mask']
        mask = torch.logical_and(mask, fixed_mask)
        N, L = mask.shape

        # Amino acid, Chain id, residue number and cdr definition
        aa, chain_ids, residx, cdr_def, coords, torsion_angle = batch['seq_t'], batch['chain_id'], batch['residx'], batch['cdr_def'], batch['atom14_gt_positions'], batch['torsion_angles_sin_cos']

        aa_feat = self.aatype_embed(aa.long()) # (N, L, feat)
        aa_feat = aa_feat * mask[:, :, None]
        cdr_feat = self.cdr_embed(cdr_def)
        # Coordinates and torsion angles
        coord_feat = self.coordinate_embed(torch.cat((coords.reshape(N, L, -1), torsion_angle.reshape(N, L, -1)), dim=-1))
        out_feat = self.mlp(torch.cat([aa_feat, chain_ids[..., None], residx[..., None], cdr_feat, coord_feat], dim=-1)) # (N, L, F)

        out_feat = out_feat * mask[:, :, None]
        return out_feat



class PairEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.pair_channel
        self.dgram_config = config.prev_pos
        self.num_bins = self.dgram_config.num_bins
        self.max_num_atoms = 14
        self.max_aa_types = residue_constants.restype_num + 3
        self.max_relpos = 32
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*self.max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, self.max_num_atoms*self.max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            Linear(self.max_num_atoms*self.max_num_atoms, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
        )

        self.dgram_embed = nn.Embedding(self.num_bins, feat_dim)

        infeat_dim = feat_dim * 4
        self.out_mlp = nn.Sequential(
            Linear(infeat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
        )

    def forward(self, batch):
        """
        Args:
            aa: (N, L).
            residx: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """

        mask, fixed_mask = batch['mask'], batch['fixed_mask']
        mask = torch.logical_and(mask, fixed_mask)
        mask_pair = mask[:, :, None] * mask[:, None, :]
        N, L = mask.shape

        # Amino acid, Chain id, residue number 
        aa, chain_ids, residx, coords, coords_mask = batch['seq_t'], batch['chain_id'], batch['residx'], batch['atom14_gt_positions'], batch['atom14_gt_exists']
        mask_atoms = coords_mask[..., residue_constants.atom_order['CA']]

        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair.long())
    
        # Relative sequential positions
        same_chain = (chain_ids[:, :, None] == chain_ids[:, None, :])
        relpos = torch.clamp(
            residx[:,:,None] - residx[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # Distances
        distance = ((torch.linalg.norm(
            coords[:,:,None,:,None] - coords[:,None,:,None,:],
            dim = -1, ord = 2,
        ))/10).reshape(N, L, L, -1) # (N, L, L, A*A)
        distance_coef = F.softplus(self.aapair_to_distcoef(aa_pair.long()))    # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * distance_coef * distance**2)

        mask_atom_pair = mask_atoms[:,:,None,None]*mask_atoms[:,None,:,None]
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Dgram
        peusdo_beta = pseudo_beta_fn_v2(aa, coords)
        disto_bins = dgram_from_positions(peusdo_beta, **self.dgram_config)

        feat_dgram = self.dgram_embed(disto_bins)

        
        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dgram], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all

