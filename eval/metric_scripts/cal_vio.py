import os
import argparse
import functools
import multiprocessing as mp
import logging
import itertools
import json
import pandas as pd
import traceback
import pickle
import numpy as np
import pdb
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBParser import PDBParser

import torch
import torch.nn.functional as F

from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions

from abx.model.utils import (
        l2_normalize,
        squared_difference,
        batched_select,
        lddt)

def between_residue_bond_loss(
    pred_atom_positions, pred_atom_mask,
    chain_id, aatype,t=None, t_filter=None,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0):
    
    assert len(pred_atom_positions.shape) == 4
    assert len(pred_atom_mask.shape) == 3
    assert len(chain_id.shape) == 2
    assert len(aatype.shape) == 2

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:, :-1, 1]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]         # (N - 1)
    this_c_pos = pred_atom_positions[:, :-1, 2]   # (N - 1, 3)
    this_c_mask = pred_atom_mask[:, :-1, 2]          # (N - 1)
    next_n_pos = pred_atom_positions[:, 1:, 0]    # (N - 1, 3)
    next_n_mask = pred_atom_mask[:, 1:, 0]           # (N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1]   # (N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]          # (N - 1)
    
    has_no_gap_mask = torch.eq(chain_id[:, 1:], chain_id[:, :-1])

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        1e-6 + torch.sum(squared_difference(this_c_pos, next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = torch.eq(aatype[:,1:], residue_constants.resname_to_idx['PRO']).to(dtype=torch.float32)
    gt_length = (
        (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
        + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = (
        (1. - next_is_proline) *
        residue_constants.between_res_bond_length_stddev_c_n[0] +
        next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = torch.sqrt(1e-6 +
                                     torch.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = F.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev)

    mask = this_c_mask * next_n_mask * has_no_gap_mask
    if t is not None and t_filter is not None:
        mask *= (t < t_filter)[:,None]
    c_n_loss = torch.sum(mask * c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    c_ca_unit_vec = l2_normalize(this_ca_pos - this_c_pos)
    c_n_unit_vec = l2_normalize(next_n_pos - this_c_pos)
    n_ca_unit_vec = l2_normalize(next_ca_pos - next_n_pos)

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = F.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    if t is not None and t_filter is not None:
        mask *= (t < t_filter)[:,None]
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
                                    (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = F.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    if t is not None and t_filter is not None:
        mask *= (t < t_filter)[:,None]
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))
    
    return c_n_violation_mask



def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    
    df = pd.read_csv(path, sep='\t')
    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    
    logger.info(f'all pairs: {df.shape[0]}')
    
    df = df.fillna({'Hchain':'', 'Lchain':''})
    df = df[df['Hchain'] != '']
    logger.info(f'number of H chains: {df.shape[0]}')

    df = df[df['model'] == 0]
    
    logger.info(f'number of model 0: {df.shape[0]}')

    df = df[df['antigen_chain'].notna()]
    df = df[df['antigen_type'].str.contains('protein|peptide')]

    logger.info(f'number of antigen: {df.shape[0]}')

   
    for code, r in df.groupby(by='pdb'):
        chain_list = list(zip(r['Hchain'], r['Lchain'], r['antigen_chain']))
        yield (code, chain_list)

def continuous_flag_to_range(flag):
    first = (np.arange(0, flag.shape[0])[flag]).min().item()
    last = (np.arange(0, flag.shape[0])[flag]).max().item()
    return first, last

def Patch_idx(a, b, mask_a, mask_b):
    assert len(a.shape) == 3 and len(b.shape) == 3
    diff = a[:, np.newaxis, :, np.newaxis, :] - b[np.newaxis, :, np.newaxis, :, :]
    mask = mask_a[:, np.newaxis, :, np.newaxis] * mask_b[np.newaxis, :, np.newaxis, :]
    distance = np.where(mask, np.linalg.norm(diff, axis=-1), 1e+10)
    distance = np.min(distance.reshape(a.shape[0], b.shape[0], -1), axis=(-1,-2))
    patch_idx = np.argwhere(distance < 32).reshape(-1)
    expanded_patch_idx = [i for j in patch_idx for i in range(j-5, j+5)]
    expanded_patch_idx = sorted(list(set(expanded_patch_idx)))
    logger.info(f"Antigen idx length@ {len(expanded_patch_idx)}")
    return expanded_patch_idx


def make_feature(str_seq, seq2struc, structure):
    n = len(str_seq)
    assert n > 0
    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=bool)
    
    for seq_idx, residue_at_position in seq2struc.items():
        if not residue_at_position.is_missing and residue_at_position.hetflag == ' ':
            residue_id = (residue_at_position.hetflag,
                    residue_at_position.position.residue_number,
                    residue_at_position.position.insertion_code)
            
            residue = structure[residue_id]

            if residue.resname not in residue_constants.restype_name_to_atom14_names:
                continue
            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[seq_idx, atom14idx] = atom.get_coord()
                coord_mask[seq_idx, atom14idx]= True
    
    feature = dict(str_seq=str_seq,
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_antigen(features):
    for i, data in enumerate(features):
        chain_id = np.full((len(data['str_seq'])), i+1)
        residx = np.arange(0,len(data['str_seq']))
        data.update(dict(residx=residx))
        data.update(dict(chain_id=chain_id))
    chain_ids = np.concatenate([data['chain_id'] for data in features],axis=0)   
    residx = np.concatenate([data['residx'] for data in features],axis=0)         
    str_seq = ''.join([data['str_seq'] for data in features])
    coord_mask = np.concatenate([data['coord_mask'] for data in features],axis=0)
    coords = np.concatenate([data['coords'] for data in features],axis=0)
    features = dict(antigen_str_seq=str_seq,
            antigen_coords=coords,
            antigen_coord_mask=coord_mask,
            antigen_chain_ids=chain_ids,
            antigen_residx=residx)
    return features

# def center(data, origin):
    

def PatchAroundAnchor(data, antigen_feature):
    cdr_str_to_enum = {
        'H1': 1,
        'H2': 3,
        'H3': 5,
        'L1': 8,
        'L2': 10,
        'L3': 12,
    }
    def anchor_flag_generate(data, antigen_feature):
        heavy_cdr_flag = data['heavy_cdr_def']
        heavy_anchor_flag = np.zeros((len(data['heavy_str_seq'])))

        if 'light_cdr_def' in data:
            light_cdr_flag = data['light_cdr_def']
            light_anchor_flag = np.zeros((len(data['light_str_seq'])))

        idx = []
        for sele in cdr_str_to_enum.keys():
            if sele in ['H1', 'H2', 'H3']:
                cdr_to_mask_flag = (heavy_cdr_flag == cdr_str_to_enum[sele])
                cdr_fist, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
                left_idx = max(0, cdr_fist - 1)
                right_idx = min(cdr_last + 1, len(data['heavy_str_seq'])-1)
                heavy_anchor_flag[left_idx] = cdr_str_to_enum[sele]
                heavy_anchor_flag[right_idx] = cdr_str_to_enum[sele]
                anchor_pos = data['heavy_coords'][[left_idx, right_idx]]
                anchor_mask = data['heavy_coord_mask'][[left_idx, right_idx]]
                antigen_pos = antigen_feature['antigen_coords']
                antigen_mask = antigen_feature['antigen_coord_mask']
                init_patch_idx = Patch_idx(antigen_pos, anchor_pos, antigen_mask, anchor_mask)
                idx.extend(init_patch_idx)
            
            elif sele in ['L1', 'L2', 'L3'] and 'light_cdr_def' in data:
                cdr_to_mask_flag = (light_cdr_flag == cdr_str_to_enum[sele])
                cdr_fist, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
                left_idx = max(0, cdr_fist - 1)
                right_idx = min(cdr_last + 1, len(data['light_str_seq'])-1)
                light_anchor_flag[left_idx] = cdr_str_to_enum[sele]
                light_anchor_flag[right_idx] = cdr_str_to_enum[sele]
                anchor_pos = data['light_coords'][[left_idx, right_idx]]
                anchor_mask = data['light_coord_mask'][[left_idx, right_idx]]
                antigen_pos = antigen_feature['antigen_coords']
                antigen_mask = antigen_feature['antigen_coord_mask']
                init_patch_idx = Patch_idx(antigen_pos, anchor_pos, antigen_mask, anchor_mask)
                idx.extend(init_patch_idx)
            else:
                light_anchor_flag = None
        
        mask = antigen_feature['antigen_coord_mask'][...,residue_constants.atom_order['CA']]
        mask_idx = np.argwhere(mask).reshape(-1).tolist()
        antigen_idx = sorted(list(set(idx).intersection(set(mask_idx))))
        antigen_anchor_coords = antigen_feature['antigen_coords'][antigen_idx]
        antigen_anchor_coords_mask = antigen_feature['antigen_coord_mask'][antigen_idx]
        antigen_anchor_residx = antigen_feature['antigen_residx'][antigen_idx]
        antigen_anchor_chain_ids = antigen_feature['antigen_chain_ids'][antigen_idx]
        antigen_anchor_str_seq = [antigen_feature['antigen_str_seq'][idx] for idx in antigen_idx]
        antigen_anchor_str_seq = ''.join(antigen_anchor_str_seq) 
        try:       
            assert len(antigen_idx) > 0
        except:
            logger.warning('No neighboring Antigen')
            raise ValueError('No neighboring Antigen')
        antigen_anchor_idx = np.array(antigen_idx)

        return heavy_anchor_flag, light_anchor_flag, antigen_anchor_coords, antigen_anchor_coords_mask, antigen_anchor_str_seq, antigen_anchor_residx, antigen_anchor_chain_ids, antigen_anchor_idx
    
    heavy_anchor_flag, light_anchor_flag, antigen_anchor_coords, antigen_anchor_coords_mask, antigen_anchor_str_seq, antigen_anchor_residx, antigen_anchor_chain_ids, antigen_anchor_idx = anchor_flag_generate(data, antigen_feature)
    if light_anchor_flag is not None:
        data.update(dict(heavy_anchor_flag = heavy_anchor_flag, 
                        light_anchor_flag = light_anchor_flag,
                        antigen_coords = antigen_anchor_coords,
                        antigen_coord_mask = antigen_anchor_coords_mask,
                        antigen_str_seq = antigen_anchor_str_seq,
                        antigen_residx = antigen_anchor_residx ,
                        antigen_chain_ids = antigen_anchor_chain_ids
                        ))
    else:
        data.update(dict(heavy_anchor_flag = heavy_anchor_flag, 
                antigen_coords = antigen_anchor_coords,
                antigen_coord_mask = antigen_anchor_coords_mask,
                antigen_str_seq = antigen_anchor_str_seq,
                antigen_residx = antigen_anchor_residx ,
                antigen_chain_ids = antigen_anchor_chain_ids
                ))
    


def make_npz(heavy_data, light_data, antigen_data):
    def _make_domain(feature, chain_id):
        allow = ['H'] if chain_id == 'H' else ['K', 'L']

        anarci_res = renumber_ab_seq(feature['str_seq'], allow=allow, scheme='imgt')
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])
        # print(f"anarci_res: {anarci_res}")
        assert domain_numbering is not None
        
        cdr_def = get_ab_regions(domain_numbering, chain_id=chain_id)
        
        updated_feature = {k : v[domain_start:domain_end] for k, v in feature.items()}
        domain_numbering = ','.join([''.join([str(xx) for xx in x]).strip() for x in domain_numbering])

        updated_feature.update(dict(cdr_def=cdr_def, numbering=domain_numbering))
        
        prefix = 'heavy' if chain_id == 'H' else 'light'

        return {f'{prefix}_{k}' : v for k, v in updated_feature.items()}
    
    feature = {}
    
    if heavy_data:
        str_seq, seq2struc, struc = map(heavy_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        heavy_feature = make_feature(str_seq, seq2struc, struc)
        heavy_feature = _make_domain(heavy_feature, 'H')
        feature.update(heavy_feature)
    
    if light_data:
        str_seq, seq2struc, struc = map(light_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        light_feature = make_feature(str_seq, seq2struc, struc)
        light_feature = _make_domain(light_feature, 'L')
        feature.update(light_feature)
    
    antigen_features = []
    for i, antigen_item in enumerate(antigen_data):
        str_seq, seq2struc, struc = map(antigen_item.get, ['str_seq', 'seqres_to_structure', 'struc'])
        antigen_feature = make_feature(str_seq, seq2struc, struc)
        antigen_features.append(antigen_feature)
    antigen_feature = make_antigen(antigen_features)

    PatchAroundAnchor(feature, antigen_feature)
    return feature

def save_feature(feature, code, heavy_chain_id, light_chain_id, antigen_chain_ids, output_dir):
    antigen_chain_id = ''.join(antigen_chain_ids)
    np.savez(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}.npz'), **feature)
    with open(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}.fasta'), 'w') as fw:
        if 'heavy_numbering' in feature:
            fw.write(f'>{code}_H {feature["heavy_numbering"]}\n{feature["heavy_str_seq"]}\n')
        if 'light_numbering' in feature:
            fw.write(f'>{code}_L {feature["light_numbering"]}\n{feature["light_str_seq"]}\n')
        if 'antigen_str_seq' in feature:
            antigen_res = ','.join(map(str, feature['antigen_residx']))
            fw.write(f'>{code}_Antigen {antigen_res}\n{feature["antigen_str_seq"]}\n')
    return

def save_header(header, file_path):
    with open(file_path, 'w') as fw:
        json.dump(header, fw)

def process(code, chain_ids, args):
    logger.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    pdb_file = os.path.join(args.mmcif_dir, f'{code}.cif')
    try:
        parsing_result = PDBParser(QUIET=1)
        model = parser.get_structure('pdb', pdb_file)[0]
    except PDBConstructionException as e:
        logger.warning('mmcif_parse: %s {%s}', pdb_file, str(e))
    except Exception as e:
        logger.warning('mmcif_parse: %s {%s}', pdb_file, str(e))
        raise Exception('...') from e
    # if not parsing_result.mmcif_object:
    #     return
    
    # save_header(parsing_result.mmcif_object.header, 
    #         os.path.join(args.output_dir, f'{code}.json'))

    # struc = parsing_result.mmcif_object.structure 
    struc = model
    def _parse_chain_id(heavy_chain_id, light_chain_id):
        if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
            heavy_chain_id = heavy_chain_id.upper()
        elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
            light_chain_id = light_chain_id.upper()
        return heavy_chain_id, light_chain_id

    for orig_heavy_chain_id, orig_light_chain_id, orig_antigen_chain_id in chain_ids:
        antigen_chain_ids = orig_antigen_chain_id.split('|')
        antigen_chain_ids = [s.replace(" ", "") for s in antigen_chain_ids]
        
        heavy_chain_id, light_chain_id = _parse_chain_id(orig_heavy_chain_id, orig_light_chain_id)

        # if ((heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres) or
        #     (light_chain_id and light_chain_id not in parsing_result.mmcif_object.chain_to_seqres)):
        #     logger.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
        #     continue
        
        # flag = 0
        # for antigen_chain_id in antigen_chain_ids:
        #     if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
        #         logger.warning(f"antigen id: {antigen_chain_id} not exist")
        #         flag += 1
        #         continue
        # if flag > 0:
        #     continue

        if heavy_chain_id:
            heavy_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[heavy_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[heavy_chain_id],
                    struc = struc[heavy_chain_id])
        else:
            heavy_data = None
        
        if light_chain_id:
            light_data = dict(
                    str_seq = parsing_result.mmcif_object.chain_to_seqres[light_chain_id],
                    seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[light_chain_id],
                    struc = struc[light_chain_id])
        else:
            light_data = None
        
        antigen_data = []
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                continue
            antigen_data.append(dict(
                str_seq = parsing_result.mmcif_object.chain_to_seqres[antigen_chain_id],
                seqres_to_structure = parsing_result.mmcif_object.seqres_to_structure[antigen_chain_id],
                struc = struc[antigen_chain_id])
            )

        try:
            feature = make_npz(heavy_data, light_data, antigen_data)
            save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, antigen_chain_ids, args.output_dir)
            logger.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logger.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')

def main(args):
    func = functools.partial(process, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.summary_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--summary_file', type=str, required=True)
    parser.add_argument('--mmcif_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output_dir,'make_energy_data.log')
    logger.info(f"log_file: {log_file}")
    handler_test = logging.FileHandler(log_file) # stdout to file
    handler_control = logging.StreamHandler()    # stdout to console

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_test.setFormatter(formatter)
    handler_control.setFormatter(formatter)
    logger.setLevel('DEBUG')           #设置了这个才会把debug以上的输出到控制台
    logger.addHandler(handler_test)    #添加handler
    logger.addHandler(handler_control)
    
    main(args)
