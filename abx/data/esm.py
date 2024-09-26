import torch
from einops import rearrange

from esm.pretrained import load_model_and_alphabet_local

# adapted from https://github.com/facebookresearch/esm
_extractor_dict = {}

class ESMEmbeddingExtractor:
    def __init__(self, model_path):
        self.model, self.alphabet = load_model_and_alphabet_local(model_path)
        self.model.requires_grad_(False)
        self.model.half()

        self.batch_converter = self.alphabet.get_batch_converter()

    def extract(self, label_seqs, repr_layer=None, return_attnw=False, device=None, linker_mask=None):
        device = label_seqs.device if device is None else device

        max_len = max([len(s) for l, s in label_seqs])

        if type(repr_layer) is int:
            repr_layer = [repr_layer]

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter(label_seqs)
            batch_tokens = batch_tokens.to(device=device)
            
            if linker_mask is not None:
                batch_tokens = torch.where(linker_mask, self.alphabet.padding_idx, batch_tokens)
            
            results = self.model(batch_tokens, repr_layers=repr_layer, need_head_weights=return_attnw)
            single = [results['representations'][r][:,1 : 1 + max_len] for r in repr_layer]

            ret = dict(single=single)

            if return_attnw:
                atten = rearrange(results['attentions'][:, :, :, 1:1+max_len, 1:1+max_len], 'b l d i j -> b i j (l d)')
                ret['pair'] = (atten + rearrange(atten, 'b i j h -> b j i h')) * 0.5
        
        return ret

    @staticmethod
    def get(model_path, device=None):
        global _extractor_dict

        if model_path not in _extractor_dict:
            obj = ESMEmbeddingExtractor(model_path)
            if device is not None:
                obj.model.to(device=device)
            _extractor_dict[model_path] = obj
        return _extractor_dict[model_path]
