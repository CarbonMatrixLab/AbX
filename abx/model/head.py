import sys
import functools
import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from abx.common import residue_constants

from abx.model import atom, quat_affine
from abx.model import atom as functional
from abx.utils import *
from abx.model.common_modules import(
        Linear,
        LayerNorm,
        pseudo_beta_fn)
from abx.model.utils import squared_difference, plddt, batched_select
from abx.model.score_network import IpaScore
import pdb

logger = logging.getLogger(__name__)

class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config

        self.breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1)
        self.proj = Linear(num_in_channel+2*c.index_embed_size, c.num_bins, init='final')

        self.config = config

    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.proj(x)
        logits = (x + rearrange(x, 'b i j c -> b j i c')) * 0.5
        breaks = self.breaks.to(logits.device)  
        return dict(logits=logits, breaks=breaks)

    
class DiffusionHead(nn.Module):
    """Head to Diffusion 3d struct.
    """
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel, diffuser):
        super().__init__()
        self.ScoreNetwork = IpaScore(config, num_in_seq_channel, num_in_pair_channel, diffuser)

        self.config = config

    def forward(self, headers, representations, batch):
        return self.ScoreNetwork(representations, batch)


class MetricDict(dict):
    def __add__(self, o):
        n = MetricDict(**self)
        for k in o:
            if k in n:
                n[k] = n[k] + o[k]
            else:
                n[k] = o[k]
        return n

    def __mul__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] * o
        return n

    def __truediv__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] / o
        return n

class MetricDictHead(nn.Module):
    """Head to calculate metrics
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):
        metrics = MetricDict()
        if 'distogram' in headers:
            assert 'logits' in headers['distogram'] and 'breaks' in headers['distogram']
            logits, breaks = headers['distogram']['logits'], headers['distogram']['breaks']
            positions = batch['pseudo_beta']
            # positions = torch.cat((positions, batch['antigen_pseudo_beta']), dim=1)
            mask = batch['pseudo_beta_mask']
            # mask = torch.cat((mask, batch['antigen_pseudo_beta_mask']), dim=-1)
            cutoff = self.config.get('contact_cutoff', 8.0)
            t =  torch.sum(breaks <= cutoff)
            pred = F.softmax(logits, dim=-1)
            pred = torch.sum(pred[...,:t+1], dim=-1)
            #truth = torch.cdist(positions, positions, p=2)
            truth = torch.sqrt(torch.sum(squared_difference(positions[:,:,None], positions[:,None]), dim=-1))
            precision_list = contact_precision(
                    pred, truth, mask=mask,
                    ratios=self.config.get('contact_ratios'),
                    ranges=self.config.get('contact_ranges'),
                    cutoff=cutoff)
            metrics['contact'] = MetricDict()
            for (i, j), ratio, precision in precision_list:
                i, j = default(i, 0), default(j, 'inf')
                metrics['contact'][f'[{i},{j})_{ratio}'] = precision
        return dict(loss=metrics) if metrics else None

class TMscoreHead(nn.Module):
    """Head to predict TM-score.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):
        c = self.config
        # only for CA atom
        if 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch:
            preds, labels = headers['folding']['final_atom_positions'][...,1,:].detach(), batch['atom14_gt_positions'][...,1,:].detach()
            gt_mask = batch['atom14_gt_exists'][...,1]

            tmscore = 0.
            for b in range(preds.shape[0]):
                mask = gt_mask[b]
                pred_aligned, label_aligned = Kabsch(
                        rearrange(preds[b][mask], 'c d -> d c'),
                        rearrange(labels[b][mask], 'c d -> d c'))

                tmscore += TMscore(pred_aligned[None,:,:], label_aligned[None,:,:], L=torch.sum(mask).item())

            return dict(loss = tmscore / preds.shape[0])
        return None
    
class SequenceHead(nn.Module):
    """Head to Diffusion 3d struct.
    """
    def __init__(self, config, num_res=20):
        super().__init__()
        c = config
        dim = c.num_channel
        hidden_dim = c.num_hidden_channel
        self.net = nn.Sequential(
            LayerNorm(dim),
            Linear(dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, num_res, init='relu', bias=True),

        )
        self.config = config

    def forward(self, headers, representations, batch):
        assert 'folding' in headers
        act = headers['folding']['representations']['structure_module']
        logits = self.net(act)
        p_0t = F.softmax(logits, dim=-1)
        seq_0 = torch.max(p_0t, dim=-1)[1]
        fixed_mask = batch['fixed_mask']
        seq_0 = seq_0 * (1 - fixed_mask) + batch['seq_t'] * fixed_mask
        
        # Compute the final atom positions
        angles = headers['folding']['sidechains'][-1]['angles_sin_cos']
        rigids = headers['folding']['rigids']
        rots = quat_affine.quat_to_rot(rigids[...,:4])
        trans = rigids[..., 4:]
        backb_to_global = (rots, trans)

        # (N, 8)
        all_frames_to_global = atom.torsion_angles_to_frames(seq_0, backb_to_global, angles)
        # (N, 14)
        pred_positions = atom.frames_and_literature_positions_to_atom14_pos(seq_0, all_frames_to_global)
        # (N, 37)
        final_atom_positions = batched_select(pred_positions, batch['residx_atom37_to_atom14'], batch_dims=2)
        
        atom14_atom_exists = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=seq_0.device), seq_0)
        atom37_atom_exists = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=seq_0.device), seq_0)


        headers['folding'].update(
            final_atom14_positions = pred_positions,
            final_atom_positions = final_atom_positions,
            atom14_atom_exists = atom14_atom_exists,
            atom37_atom_exists = atom37_atom_exists,
        )

        headers['folding']['sidechains'][-1].update(
            atom_pos = pred_positions,
            frames = all_frames_to_global
        )
        # pdb.set_trace()
        return dict(logits=logits, seq_0=seq_0)
    

class PredictedLDDTHead(nn.Module):
    def __init__(self, config, bins=50):
        super().__init__()
        c = config
        dim = c.num_channel
        hidden_dim = c.num_hidden_channel
        self.net = nn.Sequential(
            LayerNorm(dim),
            Linear(dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, bins, init='relu', bias=True),

        )
        self.config = config

    def forward(self, headers, representations, batch):
        assert 'folding' in headers
        act = headers['folding']['representations']['structure_module']
        logits = self.net(act)
        pLDDT = plddt(logits)
        return dict(logits=logits, pLDDT=pLDDT)

class HeaderBuilder:
    @staticmethod
    def build(config, seq_channel, pair_channel, parent, diffuser=None):
        head_factory = OrderedDict(
                diffusion_module = functools.partial(DiffusionHead, num_in_seq_channel=seq_channel,
                num_in_pair_channel=pair_channel,diffuser=diffuser),
                sequence_module = SequenceHead,
                distogram = functools.partial(DistogramHead, num_in_channel=pair_channel),
                metric = MetricDictHead,
                tmscore = TMscoreHead,
                predicted_lddt = PredictedLDDTHead)
        
        def gen():
            for head_name, h in head_factory.items():

                if head_name not in config:
                    continue
                head_config = config[head_name]
                
                head = h(config=head_config)

                if isinstance(parent, nn.Module):
                    parent.add_module(head_name, head)
                
                if head_name == 'diffusion_module':
                    head_name = 'folding'
                yield head_name, head, head_config

        return list(gen())
