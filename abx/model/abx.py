import functools
import logging
import random

import torch
from torch import nn

from abx.common import residue_constants
from abx.model.seqformer import EmbeddingAndSeqformer
from abx.model.head import HeaderBuilder
from abx.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)
import pdb
logger = logging.getLogger(__name__)

def get_prev(batch, value, config):
    prev_peusdo_beta = pseudo_beta_fn_v2(batch['seq'], value['heads']['folding']['final_atom_positions'])
    prev_disto_bins = dgram_from_positions(prev_peusdo_beta, **config.embeddings_and_seqformer.prev_pos)
    
    new_prev = {
        'prev_pos': prev_disto_bins.detach(),
        'prev_seq': value['representations']['seq'].detach(),
        'prev_pair': value['representations']['pair'].detach()
    }
    return new_prev

class ScoreNetworkIteration(nn.Module):
    def __init__(self, model_conf, diffuser) -> None:
        super(ScoreNetworkIteration, self).__init__()
        self._model_conf = model_conf

        self.seqformer = EmbeddingAndSeqformer(self._model_conf.embeddings_and_seqformer)
        self.diffuser = diffuser
        self.heads = HeaderBuilder.build(
                self._model_conf.heads,
                seq_channel=self._model_conf.embeddings_and_seqformer.seq_channel,
                pair_channel=self._model_conf.embeddings_and_seqformer.pair_channel,
                parent=self,
                diffuser=diffuser)
    
    def forward(self, batch, compute_loss=False):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch
        Returns:
            model_out: dictionary of model outputs.
        """

        seq_act, pair_act = self.seqformer(batch)
        representations = {'pair': pair_act, 'seq': seq_act}
        ret = {}
        
        # Evoformer Embedding
        ret['representations'] = representations
        ret['heads'] = {}

        for name, module, options in self.heads:
            if compute_loss or name == 'folding' or name == 'sequence_module':
                value = module(ret['heads'], representations, batch)
                if value is not None:
                    ret['heads'][name] = value

        return ret


class ScoreNetwork(nn.Module):
    def __init__(self, model_conf, diffuser) -> None:
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf
        self.num_in_seq_channel = self._model_conf.embeddings_and_seqformer.seq_channel
        self.num_in_pair_channel = self._model_conf.embeddings_and_seqformer.pair_channel
        self.index_embed_size = self._model_conf.embeddings_and_seqformer.index_embed_size
        self.impl = ScoreNetworkIteration(model_conf, diffuser)

    def forward(self, input_feats, compute_loss=True):
        batch_size, num_res, device = *input_feats['seq'].shape[:2], input_feats['seq'].device
        
        if 'prev_seq' not in input_feats:
            prev = {
                'prev_pos': torch.zeros([batch_size, num_res, num_res], device=device, dtype=torch.int64),
                'prev_seq': torch.zeros([batch_size, num_res, self.num_in_seq_channel+self.index_embed_size], device=device),
                'prev_pair': torch.zeros([batch_size, num_res, num_res, self.num_in_pair_channel+2*self.index_embed_size], device=device)
            }
            input_feats.update(prev)

        if self.training:
            num_recycle = random.randint(0, self._model_conf.num_recycle)
        else:
            num_recycle = self._model_conf.num_recycle
        # pdb.set_trace()

        with torch.no_grad():
            input_feats.update(is_recycling=True)
            for i in range(num_recycle):
                ret = self.impl(input_feats, compute_loss=False)
                prev = get_prev(input_feats, ret, self._model_conf)
                if 'sequence_module' in ret['heads']:
                    input_feats.update(seq_t = ret['heads']['sequence_module']['seq_0'])
                input_feats.update(prev)

        input_feats.update(is_recycling=False)
        ret = self.impl(input_feats, compute_loss=compute_loss)

        return ret
        

