import os
import functools
import logging
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from abx.common import residue_constants
from abx.common.utils import str_seq_to_index as AF_str_seq_to_index
from abx.model.features import FeatureBuilder
from abx.data.utils import pad_for_batch

import pdb
from abx.data.utils import process_pdb


logger = logging.getLogger(__file__)

def str_seq_to_index(x):
    return AF_str_seq_to_index(x)

def continuous_flag_to_range(flag):
    # pdb.set_trace()
    first = (torch.arange(0, flag.shape[0])[flag]).min(dim=0).values.item()
    last = (torch.arange(0, flag.shape[0])[flag]).max(dim=0).values.item()
    return first, last

def patch_idx(a, b, mask_a, mask_b, distance_threshold=16.0):
    assert len(a.shape) == 3 and len(b.shape) == 3
    diff = a[:, None, :, None, :] - b[None, :, None, :, :]
    mask = mask_a[:, None, :, None] * mask_b[None, :, None, :]
    distance = torch.where(mask, torch.norm(diff, dim=-1), torch.tensor(1e+10, device=a.device))
    distance = distance.reshape(a.shape[0], b.shape[0], -1).min(dim=2)[0]
    min_distance = distance.min(dim=1)[0]
    patch_idx = torch.nonzero(min_distance < distance_threshold).reshape(-1).cpu().numpy()
    expanded_patch_idx = [i for j in patch_idx for i in range(j-5, j+5)]
    expanded_patch_idx = sorted(list(set(expanded_patch_idx)))
    return expanded_patch_idx



class Cluster(object):
    def __init__(self, names):
        self.names = names
        self.idx = 0
        assert len(names) > 0

    def get_next(self):
        idx = random.randrange(len(self.names))
        # item = self.names[self.idx]
        # self.idx += 1
        # if self.idx == len(self.names):
        #     self.idx = 0
        item = self.names[idx]
        return item

    def __expr__(self):
        return self.names[self.idx]

    def __str__(self):
        return self.names[self.idx]

def parse_cluster(file_name):
    ret = []
    with open(file_name) as f:
        for line in f:
            items = line.strip().split()
            ret.append(Cluster(names=items))
    return ret

class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, rank, word_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.word_size = word_size

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.word_size == self.rank:
                yield sample
    
    def collate_fn(self, *args, **kwargs):
        return self.dataset.collate_fn(*args, **kwargs)


class IgStructureDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, name_idx, max_antigen_seq_len=32, reduce_num=None, is_cluster_idx=False, is_training=False):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.name_idx = name_idx
        self.max_antigen_seq_len = max_antigen_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx
        self.is_training = is_training

        logger.info(f'dataset size= {len(name_idx)} max_antigen_seq_len= {max_antigen_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')
        self.epoch_count = 0
    
    def __len__(self):
        return len(self.name_idx)
    
    def _get_name_idx(self):
        if self.reduce_num is None:
            return self.name_idx

        random.seed(2022 + self.epoch_count)
        random.shuffle(self.name_idx)
        self.epoch_count += 1
        logging.info(f'data: epoch_count={self.epoch_count} reduce_num={self.reduce_num} all={len(self.name_idx)} ex={",".join([str(x) for x in self.name_idx[:4]])}')
        
        return self.name_idx[:self.reduce_num]

    def __iter__(self):
        name_idx = self._get_name_idx()
        for item in name_idx:
            if self.is_cluster_idx:
                name = item.get_next()
            else:
                name = item
            ret = self.get_structure_label_npz(name)
            if ret:
                antigen_len = len(ret.get('antigen_str_seq', ''))
                
                if antigen_len > self.max_antigen_seq_len:
                    start, end = sample_with_struc(ret['antigen_atom14_gt_exists'][:,1], antigen_len, 32)
                    for k, v in ret.items():
                        if 'antigen' in k and 'origin' not in k:
                            ret[k] = v[start:end]
                yield ret

    def get_structure_label_npz(self, name, scale_factor=1.0):
        num_atoms = 14
        struc = np.load(os.path.join(self.data_dir, name + '.npz'))

        # Antibody Feature
        antibody_coords = torch.from_numpy(struc.get('antibody_coords', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antibody_coord_mask = torch.from_numpy(struc.get('antibody_coord_mask', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antibody_cdr_def = torch.from_numpy(struc.get('antibody_cdr_def', np.zeros((0,), dtype=np.int32)))
        antibody_chain_ids = torch.from_numpy(struc.get('antibody_chain_ids', np.zeros((0,), dtype=np.int32)))
        antibody_residx = torch.from_numpy(struc.get('antibody_residx', np.zeros((0,), dtype=np.int32)))

        antibody_str_seq = str(struc.get('antibody_str_seq', ''))
        heavy_len = len(antibody_chain_ids[antibody_chain_ids==0])
        str_heavy_seq = antibody_str_seq[:heavy_len]
        str_light_seq = antibody_str_seq[heavy_len:]

        heavy_seq = torch.tensor(str_seq_to_index(str_heavy_seq), dtype=torch.int64)
        light_seq = torch.tensor(str_seq_to_index(str_light_seq), dtype=torch.int64)
        antibody_seq = torch.cat([heavy_seq, light_seq], dim=-1)
        antibody_mask = torch.ones_like(antibody_chain_ids, dtype=torch.bool)

        # Antigen Feature
        antigen_coords = torch.from_numpy(struc.get('antigen_coords', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antigen_coord_mask = torch.from_numpy(struc.get('antigen_coord_mask', np.zeros((0, num_atoms), dtype=np.bool_)))
        antigen_str_seq = str(struc.get('antigen_str_seq', ''))
        antigen_seq = torch.tensor(str_seq_to_index(antigen_str_seq), dtype=torch.int64)
        antigen_chain_ids = torch.from_numpy(struc.get('antigen_chain_ids', np.zeros((0,), dtype=np.int32)))
        antigen_residx = torch.from_numpy(struc.get('antigen_residx', np.zeros((0,), dtype=np.int32)))
        antigen_mask = torch.ones(len(antigen_str_seq), dtype=torch.bool)
        antigen_cdr_def = torch.from_numpy(struc.get('antigen_cdr_def', np.zeros((0,), dtype=np.int32)))
        
        # Normalized
        ca_idx = residue_constants.atom_order['CA']
        antibody_bb_mask = antibody_coord_mask[:,ca_idx]
        antibody_bb_pos = antibody_coords[:,ca_idx]
        antibody_bb_center = torch.sum(antibody_bb_pos, axis=0)/ (torch.sum(antibody_bb_mask, axis=0, keepdim=True) + 1e-5)
        # pdb.set_trace()
        antibody_centered_pos = antibody_coords - antibody_bb_center[None, None, :]
        antibody_scaled_pos = antibody_centered_pos / scale_factor
        antibody_coords = antibody_scaled_pos * antibody_coord_mask[..., None] 

        antigen_centered_pos = antigen_coords - antibody_bb_center[None, None, :]
        antigen_scaled_pos = antigen_centered_pos / scale_factor
        antigen_coords = antigen_scaled_pos * antigen_coord_mask[..., None]

        ret = dict( name=name,
                    # Antibody Features
                    antibody_seq=antibody_seq,
                    antibody_residx=antibody_residx,
                    antibody_mask = antibody_mask,
                    str_heavy_seq = str_heavy_seq, str_light_seq=str_light_seq,
                    antibody_atom14_gt_positions=antibody_coords, antibody_atom14_gt_exists=antibody_coord_mask,
                    antibody_cdr_def = antibody_cdr_def,
                    antibody_chain_ids = antibody_chain_ids,
                    # bb_center = antibody_bb_center,
                    # Antigen Features
                    antigen_atom14_gt_positions = antigen_coords,
                    antigen_atom14_gt_exists = antigen_coord_mask,
                    antigen_str_seq = antigen_str_seq,
                    antigen_seq = antigen_seq,
                    antigen_mask = antigen_mask, 
                    antigen_chain_ids = antigen_chain_ids,
                    antigen_residx = antigen_residx,
                    antigen_cdr_def = antigen_cdr_def)
        
        # Crop Antigen in Interface
        ret = Patch_Around_Anchor(ret, is_training=self.is_training)
        
        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 
            # Antibody Features
            'antibody_mask', 'antibody_seq', 'str_heavy_seq', 'str_light_seq', 
            'antibody_atom14_gt_positions', 'antibody_atom14_gt_exists',
            'antibody_cdr_def', 'antibody_chain_ids', 'antibody_residx', 'anchor_flag',
            # Antigen Features
            'antigen_mask', 'antigen_seq', 'antigen_str_seq', 
            'antigen_atom14_gt_positions', 'antigen_atom14_gt_exists',
            'antigen_cdr_def', 'antigen_chain_ids', 'antigen_residx')
        name, \
        antibody_mask, antibody_seq, str_heavy_seq, str_light_seq, antibody_atom14_gt_positions, antibody_atom14_gt_exists, antibody_cdr_def, antibody_chain_ids, antibody_residx, anchor_flag, \
        antigen_mask, antigen_seq, antigen_str_seq, antigen_atom14_gt_positions, antigen_atom14_gt_exists, antigen_cdr_def, antigen_chain_ids, antigen_residx = list(zip(*[[b[k] for k in fields] for b in batch]))
        

        # Padded Antibody Features
        max_full_len = max(tuple(len(a) + len(b) for a, b in zip(str_heavy_seq, str_light_seq)))
        padded_antibody_seqs = pad_for_batch(antibody_seq, max_full_len, 'seq')
        padded_antibody_masks = pad_for_batch(antibody_mask, max_full_len, 'msk')
        padded_antibody_atom14_gt_positions = pad_for_batch(antibody_atom14_gt_positions, max_full_len, 'crd')
        padded_antibody_atom14_gt_existss = pad_for_batch(antibody_atom14_gt_exists, max_full_len, 'crd_msk')
        padded_antibody_cdr_def = pad_for_batch(antibody_cdr_def, max_full_len, 'msk')
        padded_antibody_chain_id = pad_for_batch(antibody_chain_ids, max_full_len, 'msk')
        padded_antibody_residx = pad_for_batch(antibody_residx, max_full_len, 'msk')
        padded_anchor_flag = pad_for_batch(anchor_flag, max_full_len, 'msk')


        # Padded Antigen Features
        max_antigen_len = max(tuple(len(s) for s in antigen_str_seq))
        padded_antigen_seqs = pad_for_batch(antigen_seq, max_antigen_len, 'seq')
        padded_antigen_atom14_gt_positions = pad_for_batch(antigen_atom14_gt_positions, max_antigen_len, 'crd')
        padded_antigen_atom14_gt_exists = pad_for_batch(antigen_atom14_gt_exists, max_antigen_len, 'crd_msk')
        padded_antigen_chain_ids = pad_for_batch(antigen_chain_ids, max_antigen_len, 'msk')
        padded_antigen_residx = pad_for_batch(antigen_residx, max_antigen_len, 'msk')
        padded_antigen_mask = pad_for_batch(antigen_mask, max_antigen_len, 'msk')
        padded_antigen_cdr_def = pad_for_batch(antigen_cdr_def, max_antigen_len, 'msk')

        padded_masks = torch.cat([padded_antibody_masks, padded_antigen_mask], dim=1)
        padded_atom14_gt_exists = torch.cat([padded_antibody_atom14_gt_existss, padded_antigen_atom14_gt_exists], dim=1)
        padded_atom14_gt_positions = torch.cat([padded_antibody_atom14_gt_positions, padded_antigen_atom14_gt_positions], dim=1)
        padded_chain_ids = torch.cat([padded_antibody_chain_id, padded_antigen_chain_ids], dim=1)
        padded_residx = torch.cat([padded_antibody_residx, padded_antigen_residx], dim=1)
        padded_cdr_def = torch.cat([padded_antibody_cdr_def, padded_antigen_cdr_def], dim=1)
        padded_seqs = torch.cat([padded_antibody_seqs, padded_antigen_seqs], dim=1)
        ret = dict(
		        name=name,
                seq=padded_seqs,
                mask=padded_masks,
                str_heavy_seq=str_heavy_seq,
                str_light_seq=str_light_seq,
                atom14_gt_positions=padded_atom14_gt_positions,
                atom14_gt_exists=padded_atom14_gt_exists,
                cdr_def=padded_cdr_def,
                chain_id=padded_chain_ids,
                residx = padded_residx,
                anchor_flag = padded_anchor_flag,
                )
        if self.is_training is False:
            fields = ('antigen_origin_str_seq', 'antigen_origin_atom14_gt_positions', 'antigen_origin_atom14_gt_exists', 'antigen_origin_chain_ids', 'antigen_origin_residx')
            antigen_origin_str_seq, antigen_origin_atom14_gt_positions, antigen_origin_atom14_gt_exists, antigen_origin_chain_ids, antigen_origin_residx = list(zip(*[[b[k] for k in fields] for b in batch]))

            antigen_origin_atom14_gt_exists = [mask.cpu().numpy() for mask in antigen_origin_atom14_gt_exists]
            antigen_origin_atom14_gt_positions = [coords.cpu().numpy() for coords in antigen_origin_atom14_gt_positions]
            antigen_origin_chain_ids = [chain_ids.cpu().numpy() for chain_ids in antigen_origin_chain_ids]
            antigen_origin_residx = [residx.cpu().numpy() for residx in antigen_origin_residx]
            
            ret.update(
                antigen_origin_str_seq = antigen_origin_str_seq,
                antigen_origin_atom14_gt_positions = antigen_origin_atom14_gt_positions,
                antigen_origin_atom14_gt_exists = antigen_origin_atom14_gt_exists,
                antigen_origin_chain_ids = antigen_origin_chain_ids,
                antigen_origin_residx = antigen_origin_residx,
            )

        if feat_builder:
            ret = feat_builder.build(ret)
        
        return ret
    

class IgStructureData(torch.utils.data.IterableDataset):
    def __init__(self, pdb_file, max_antigen_seq_len=32, reduce_num=None, is_cluster_idx=False, is_training=False):
        super().__init__()
        self.pdb_file = pathlib.Path(pdb_file)
        self.pdb_name = (pdb_file.split('/')[-1]).split('.')[0]
        self.code = self.pdb_name.split('_')[0]
        self.chain_ids = self.pdb_name.split('_')[1:]
        self.max_antigen_seq_len = max_antigen_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx
        self.is_training = is_training
        self.ret = process_pdb(self.code,self.chain_ids, pdb_file)
        logger.info(f'Target antibody-antigen complex {pdb_file} max_antigen_seq_len= {max_antigen_seq_len}')
        self.epoch_count = 0
    
    def __len__(self):
        return 1
    

    def __iter__(self):
        ret = self.get_structure_label_npz()
        antigen_len = len(ret.get('antigen_str_seq', ''))
        
        if antigen_len > self.max_antigen_seq_len:
            start, end = sample_with_struc(ret['antigen_atom14_gt_exists'][:,1], antigen_len, 32)
            for k, v in ret.items():
                if 'antigen' in k and 'origin' not in k:
                    ret[k] = v[start:end]
        yield ret

    def get_structure_label_npz(self, scale_factor=1.0):
        num_atoms = 14
        # struc = np.load(os.path.join(self.data_dir, name + '.npz'))
        struc = self.ret
        name = self.pdb_name
        # Antibody Feature
        antibody_coords = torch.from_numpy(struc.get('antibody_coords', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antibody_coord_mask = torch.from_numpy(struc.get('antibody_coord_mask', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antibody_cdr_def = torch.from_numpy(struc.get('antibody_cdr_def', np.zeros((0,), dtype=np.int32)))
        antibody_chain_ids = torch.from_numpy(struc.get('antibody_chain_ids', np.zeros((0,), dtype=np.int32)))
        antibody_residx = torch.from_numpy(struc.get('antibody_residx', np.zeros((0,), dtype=np.int32)))

        antibody_str_seq = str(struc.get('antibody_str_seq', ''))
        heavy_len = len(antibody_chain_ids[antibody_chain_ids==0])
        str_heavy_seq = antibody_str_seq[:heavy_len]
        str_light_seq = antibody_str_seq[heavy_len:]

        heavy_seq = torch.tensor(str_seq_to_index(str_heavy_seq), dtype=torch.int64)
        light_seq = torch.tensor(str_seq_to_index(str_light_seq), dtype=torch.int64)
        antibody_seq = torch.cat([heavy_seq, light_seq], dim=-1)
        antibody_mask = torch.ones_like(antibody_chain_ids, dtype=torch.bool)

        # Antigen Feature
        antigen_coords = torch.from_numpy(struc.get('antigen_coords', np.zeros((0, num_atoms, 3), dtype=np.float32)))
        antigen_coord_mask = torch.from_numpy(struc.get('antigen_coord_mask', np.zeros((0, num_atoms), dtype=np.bool_)))
        antigen_str_seq = str(struc.get('antigen_str_seq', ''))
        antigen_seq = torch.tensor(str_seq_to_index(antigen_str_seq), dtype=torch.int64)
        antigen_chain_ids = torch.from_numpy(struc.get('antigen_chain_ids', np.zeros((0,), dtype=np.int32)))
        antigen_residx = torch.from_numpy(struc.get('antigen_residx', np.zeros((0,), dtype=np.int32)))
        antigen_mask = torch.ones(len(antigen_str_seq), dtype=torch.bool)
        antigen_cdr_def = torch.from_numpy(struc.get('antigen_cdr_def', np.zeros((0,), dtype=np.int32)))
        
        # Normalized
        ca_idx = residue_constants.atom_order['CA']
        antibody_bb_mask = antibody_coord_mask[:,ca_idx]
        antibody_bb_pos = antibody_coords[:,ca_idx]
        antibody_bb_center = torch.sum(antibody_bb_pos, axis=0)/ (torch.sum(antibody_bb_mask, axis=0, keepdim=True) + 1e-5)
        # pdb.set_trace()
        antibody_centered_pos = antibody_coords - antibody_bb_center[None, None, :]
        antibody_scaled_pos = antibody_centered_pos / scale_factor
        antibody_coords = antibody_scaled_pos * antibody_coord_mask[..., None] 

        antigen_centered_pos = antigen_coords - antibody_bb_center[None, None, :]
        antigen_scaled_pos = antigen_centered_pos / scale_factor
        antigen_coords = antigen_scaled_pos * antigen_coord_mask[..., None]

        ret = dict( name=name,
                    # Antibody Features
                    antibody_seq=antibody_seq,
                    antibody_residx=antibody_residx,
                    antibody_mask = antibody_mask,
                    str_heavy_seq = str_heavy_seq, str_light_seq=str_light_seq,
                    antibody_atom14_gt_positions=antibody_coords, antibody_atom14_gt_exists=antibody_coord_mask,
                    antibody_cdr_def = antibody_cdr_def,
                    antibody_chain_ids = antibody_chain_ids,
                    # bb_center = antibody_bb_center,
                    # Antigen Features
                    antigen_atom14_gt_positions = antigen_coords,
                    antigen_atom14_gt_exists = antigen_coord_mask,
                    antigen_str_seq = antigen_str_seq,
                    antigen_seq = antigen_seq,
                    antigen_mask = antigen_mask, 
                    antigen_chain_ids = antigen_chain_ids,
                    antigen_residx = antigen_residx,
                    antigen_cdr_def = antigen_cdr_def)
        
        # Crop Antigen in Interface
        ret = Patch_Around_Anchor(ret, is_training=self.is_training)
        
        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 
            # Antibody Features
            'antibody_mask', 'antibody_seq', 'str_heavy_seq', 'str_light_seq', 
            'antibody_atom14_gt_positions', 'antibody_atom14_gt_exists',
            'antibody_cdr_def', 'antibody_chain_ids', 'antibody_residx', 'anchor_flag',
            # Antigen Features
            'antigen_mask', 'antigen_seq', 'antigen_str_seq', 
            'antigen_atom14_gt_positions', 'antigen_atom14_gt_exists',
            'antigen_cdr_def', 'antigen_chain_ids', 'antigen_residx')
        name, \
        antibody_mask, antibody_seq, str_heavy_seq, str_light_seq, antibody_atom14_gt_positions, antibody_atom14_gt_exists, antibody_cdr_def, antibody_chain_ids, antibody_residx, anchor_flag, \
        antigen_mask, antigen_seq, antigen_str_seq, antigen_atom14_gt_positions, antigen_atom14_gt_exists, antigen_cdr_def, antigen_chain_ids, antigen_residx = list(zip(*[[b[k] for k in fields] for b in batch]))
        

        # Padded Antibody Features
        max_full_len = max(tuple(len(a) + len(b) for a, b in zip(str_heavy_seq, str_light_seq)))
        padded_antibody_seqs = pad_for_batch(antibody_seq, max_full_len, 'seq')
        padded_antibody_masks = pad_for_batch(antibody_mask, max_full_len, 'msk')
        padded_antibody_atom14_gt_positions = pad_for_batch(antibody_atom14_gt_positions, max_full_len, 'crd')
        padded_antibody_atom14_gt_existss = pad_for_batch(antibody_atom14_gt_exists, max_full_len, 'crd_msk')
        padded_antibody_cdr_def = pad_for_batch(antibody_cdr_def, max_full_len, 'msk')
        padded_antibody_chain_id = pad_for_batch(antibody_chain_ids, max_full_len, 'msk')
        padded_antibody_residx = pad_for_batch(antibody_residx, max_full_len, 'msk')
        padded_anchor_flag = pad_for_batch(anchor_flag, max_full_len, 'msk')


        # Padded Antigen Features
        max_antigen_len = max(tuple(len(s) for s in antigen_str_seq))
        padded_antigen_seqs = pad_for_batch(antigen_seq, max_antigen_len, 'seq')
        padded_antigen_atom14_gt_positions = pad_for_batch(antigen_atom14_gt_positions, max_antigen_len, 'crd')
        padded_antigen_atom14_gt_exists = pad_for_batch(antigen_atom14_gt_exists, max_antigen_len, 'crd_msk')
        padded_antigen_chain_ids = pad_for_batch(antigen_chain_ids, max_antigen_len, 'msk')
        padded_antigen_residx = pad_for_batch(antigen_residx, max_antigen_len, 'msk')
        padded_antigen_mask = pad_for_batch(antigen_mask, max_antigen_len, 'msk')
        padded_antigen_cdr_def = pad_for_batch(antigen_cdr_def, max_antigen_len, 'msk')

        padded_masks = torch.cat([padded_antibody_masks, padded_antigen_mask], dim=1)
        padded_atom14_gt_exists = torch.cat([padded_antibody_atom14_gt_existss, padded_antigen_atom14_gt_exists], dim=1)
        padded_atom14_gt_positions = torch.cat([padded_antibody_atom14_gt_positions, padded_antigen_atom14_gt_positions], dim=1)
        padded_chain_ids = torch.cat([padded_antibody_chain_id, padded_antigen_chain_ids], dim=1)
        padded_residx = torch.cat([padded_antibody_residx, padded_antigen_residx], dim=1)
        padded_cdr_def = torch.cat([padded_antibody_cdr_def, padded_antigen_cdr_def], dim=1)
        padded_seqs = torch.cat([padded_antibody_seqs, padded_antigen_seqs], dim=1)
        ret = dict(
		        name=name,
                seq=padded_seqs,
                mask=padded_masks,
                str_heavy_seq=str_heavy_seq,
                str_light_seq=str_light_seq,
                atom14_gt_positions=padded_atom14_gt_positions,
                atom14_gt_exists=padded_atom14_gt_exists,
                cdr_def=padded_cdr_def,
                chain_id=padded_chain_ids,
                residx = padded_residx,
                anchor_flag = padded_anchor_flag,
                )
        if self.is_training is False:
            fields = ('antigen_origin_str_seq', 'antigen_origin_atom14_gt_positions', 'antigen_origin_atom14_gt_exists', 'antigen_origin_chain_ids', 'antigen_origin_residx')
            antigen_origin_str_seq, antigen_origin_atom14_gt_positions, antigen_origin_atom14_gt_exists, antigen_origin_chain_ids, antigen_origin_residx = list(zip(*[[b[k] for k in fields] for b in batch]))

            antigen_origin_atom14_gt_exists = [mask.cpu().numpy() for mask in antigen_origin_atom14_gt_exists]
            antigen_origin_atom14_gt_positions = [coords.cpu().numpy() for coords in antigen_origin_atom14_gt_positions]
            antigen_origin_chain_ids = [chain_ids.cpu().numpy() for chain_ids in antigen_origin_chain_ids]
            antigen_origin_residx = [residx.cpu().numpy() for residx in antigen_origin_residx]
            
            ret.update(
                antigen_origin_str_seq = antigen_origin_str_seq,
                antigen_origin_atom14_gt_positions = antigen_origin_atom14_gt_positions,
                antigen_origin_atom14_gt_exists = antigen_origin_atom14_gt_exists,
                antigen_origin_chain_ids = antigen_origin_chain_ids,
                antigen_origin_residx = antigen_origin_residx,
            )

        if feat_builder:
            ret = feat_builder.build(ret)
        
        return ret




def sample_with_struc(struc_mask, str_len, max_antigen_seq_len):
    num_struc = torch.sum(struc_mask)
    if num_struc > 0 and num_struc < str_len:
        struc_start, struc_end = 0, str_len
        while struc_start < str_len and struc_mask[struc_start] == False:
            struc_start += 1
        while struc_end > 0 and struc_mask[struc_end - 1] == False:
            struc_end -= 1
        if struc_end - struc_start > max_antigen_seq_len:
            start = random.randint(struc_start, struc_end - max_antigen_seq_len)
            end = start + max_antigen_seq_len
        else:
            extra = max_antigen_seq_len - (struc_end - struc_start)
            left_extra = struc_start - extra // 2 - 10
            right_extra = struc_end + extra // 2 + 10
            start = random.randint(left_extra, right_extra)
            end = start + max_antigen_seq_len
            if start < 0:
                start = 0
                end = start + max_antigen_seq_len
            elif end > str_len:
                end = str_len
                start = end - max_antigen_seq_len
    else:
        start = random.randint(0, str_len - max_antigen_seq_len)
        end = start + max_antigen_seq_len
    return start, end

def Patch_Around_Anchor(data, distance_threshold=16.0, is_training=False):
    anchor_flag = torch.zeros_like(data['antibody_cdr_def'])

    idx = []
    for sele in ['H1','H2','H3','L1','L2','L3']:
        cdr_to_mask_flag = (data['antibody_cdr_def'] == residue_constants.cdr_str_to_enum[sele])
        if cdr_to_mask_flag.any() == True:
            cdr_fist, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
            left_idx = max(0, cdr_fist - 1)
            right_idx = min(cdr_last + 1, data['antibody_seq'].shape[0]-1)
            anchor_flag[left_idx] = residue_constants.cdr_str_to_enum[sele]
            anchor_flag[right_idx] = residue_constants.cdr_str_to_enum[sele]
            anchor_pos = data['antibody_atom14_gt_positions'][[left_idx, right_idx]]
            anchor_mask = data['antibody_atom14_gt_exists'][[left_idx, right_idx]]
            antigen_pos = data['antigen_atom14_gt_positions']
            antigen_mask = data['antigen_atom14_gt_exists']
            idx_element = patch_idx(antigen_pos, anchor_pos, antigen_mask, anchor_mask, distance_threshold=distance_threshold)
            idx.extend(idx_element)
    
    mask = data['antigen_atom14_gt_positions'][:,residue_constants.atom_order['CA']]
    mask_idx = torch.nonzero(mask).reshape(-1).cpu().numpy().tolist()
    antigen_idx = sorted(list(set(idx).intersection(set(mask_idx))))
    antigen_anchor_coords = data['antigen_atom14_gt_positions'][antigen_idx]
    antigen_anchor_coords_mask = data['antigen_atom14_gt_exists'][antigen_idx]
    antigen_anchor_residx = data['antigen_residx'][antigen_idx]
    antigen_anchor_chain_ids = data['antigen_chain_ids'][antigen_idx]
    antigen_anchor_str_seq = [data['antigen_str_seq'][idx] for idx in antigen_idx]
    antigen_anchor_str_seq = ''.join(antigen_anchor_str_seq) 
    antigen_anchor_seq = data['antigen_seq'][antigen_idx]
    antigen_anchor_cdr_def = data['antigen_cdr_def'][antigen_idx]
    antigen_anchor_mask = data['antigen_mask'][antigen_idx]

    data.update(
        anchor_flag = anchor_flag,
        antigen_atom14_gt_positions = antigen_anchor_coords,
        antigen_atom14_gt_exists = antigen_anchor_coords_mask,
        antigen_residx = antigen_anchor_residx,
        antigen_chain_ids = antigen_anchor_chain_ids,
        antigen_str_seq = antigen_anchor_str_seq,
        antigen_seq = antigen_anchor_seq,
        antigen_cdr_def = antigen_anchor_cdr_def,
        antigen_mask = antigen_anchor_mask
    )
    if is_training is False:
        data.update(
            antigen_origin_atom14_gt_positions = data['antigen_atom14_gt_positions'],
            antigen_origin_atom14_gt_exists = data['antigen_atom14_gt_exists'],
            antigen_origin_str_seq = data['antigen_str_seq'],
            antigen_origin_residx = data['antigen_residx'],
            antigen_origin_chain_ids = data['antigen_chain_ids'],
        )
    if len(antigen_idx) > 0: 
        return data
    else: 
        return None


def load(data_dir, name_idx,
        feats=None, 
        is_training=True,
        max_antigen_seq_len=32, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        **kwargs):

    dataset = IgStructureDataset(data_dir, name_idx, max_antigen_seq_len=max_antigen_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx, is_training=is_training)

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, prefetch_factor=2, **kwargs)


def load_single(pdb_file, 
        feats=None, 
        is_training=True,
        max_antigen_seq_len=32, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        **kwargs):

    dataset = IgStructureData(pdb_file,  max_antigen_seq_len=max_antigen_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx, is_training=is_training)
    
    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, is_training=is_training))

    return torch.utils.data.DataLoader(dataset, prefetch_factor=2, **kwargs)
