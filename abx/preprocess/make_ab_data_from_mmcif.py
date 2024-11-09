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

from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions
from Bio.SeqUtils import seq1

def parse_list(path):
    with open(path) as f:
        names = [n.strip() for n in f]
    
    df = pd.read_csv(path, sep='\t')
    df = df[df['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    
    logging.info(f'all pairs: {df.shape[0]}')
    
    df = df.fillna({'Hchain':'', 'Lchain':''})
    df = df[df['Hchain'] != '']
    logging.info(f'number of H chains: {df.shape[0]}')

    df = df[df['model'] == 0]
    
    logging.info(f'number of model 0: {df.shape[0]}')

    df = df[df['antigen_chain'].notna()]
    df = df[df['antigen_type'].str.contains('protein|peptide')]

    logging.info(f'number of antigen: {df.shape[0]}')

   
    for code, r in df.groupby(by='pdb'):
        chain_list = list(zip(r['Hchain'], r['Lchain'], r['antigen_chain']))
        yield (code, chain_list)


def make_chain_feature(chain):    
    residues = list(chain.get_residues())
    residues = [r for r in residues if r.get_resname() in residue_constants.restype_3to1.keys()]
    
    str_seq = [seq1(r.get_resname()) for r in residues if r.get_resname() in residue_constants.restype_3to1.keys()]
    N = len(str_seq)
    
    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    # str_seq_new = []
    for jj, residue in enumerate(residues):
        if residue.get_resname() in residue_constants.restype_3to1.keys():
            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]            
            for atom in residue.get_atoms():
                if atom.id not in res_atom14_list:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[jj, atom14idx] = atom.get_coord()
                coord_mask[jj, atom14idx]= True


    feature = dict(
            str_seq=''.join(str_seq),
            coords=coords,
            coord_mask=coord_mask)
    return feature


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

def merge_chains(features):
    
    for i, data in enumerate(features):
        if 'cdr_def' in data:
            chain_flag = 0
            prefix = 'antibody'
        else:
            chain_flag = 2 
            prefix = 'antigen'
        chain_id = np.full((len(data['str_seq'])), i+chain_flag)
        residx = np.arange(0,len(data['str_seq']))
        
        if prefix == 'antibody' and i > 0:
            residx += residue_constants.residue_chain_index_offset
        if prefix == 'antigen':
            cdr_def = np.full_like(chain_id, fill_value=14)
            data.update(dict(cdr_def=cdr_def))
        data.update(dict(residx=residx))
        data.update(dict(chain_id=chain_id))
    chain_ids = np.concatenate([data['chain_id'] for data in features],axis=0) 
    residx = np.concatenate([data['residx'] for data in features],axis=0)         
    str_seq = ''.join([data['str_seq'] for data in features])
    coord_mask = np.concatenate([data['coord_mask'] for data in features],axis=0)
    coords = np.concatenate([data['coords'] for data in features],axis=0)
    cdr_def = np.concatenate([data['cdr_def']for data in features],axis=0)

    merge_features = dict(str_seq=str_seq,
            coords=coords,
            coord_mask=coord_mask,
            chain_ids=chain_ids,
            residx=residx,
            cdr_def=cdr_def)
    merge_features = {f'{prefix}_{k}' : v for k, v in merge_features.items()}
    
    return merge_features

def make_pdb_npz(struc, chain_ids, heavy_chain_id, light_chain_id, antigen_chain_id):
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
        
        return updated_feature
    all_chain = list(struc.get_chains())

    # 查找特定链的信息
    def _get_chain_info(chain_id):
        for chain in all_chain:
            if chain.id == chain_id:
                return chain
        return None
    
    antibody_feature = []
    features = {}
    if heavy_chain_id:
        
        heavy_feature = make_chain_feature(_get_chain_info(heavy_chain_id))
        heavy_feature = _make_domain(heavy_feature, 'H')
        antibody_feature.append(heavy_feature)
    
    if light_chain_id:
        light_feature = make_chain_feature(_get_chain_info(light_chain_id))
        light_feature = _make_domain(light_feature, 'L')
        antibody_feature.append(light_feature)
    features.update(merge_chains(antibody_feature))

    if antigen_chain_id:
        antigen_features = []
        for i, antigen_chain in enumerate(antigen_chain_id):
            if antigen_chain not in chain_ids:
                continue
            antigen_feature = make_chain_feature(_get_chain_info(antigen_chain))
            antigen_features.append(antigen_feature)
        features.update(merge_chains(antigen_features))
    return features


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
        
        return updated_feature
    
    antibody_feature = []
    features = {}

    if heavy_data:
        str_seq, seq2struc, struc = map(heavy_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        heavy_feature = make_feature(str_seq, seq2struc, struc)
        heavy_feature = _make_domain(heavy_feature, 'H')
        antibody_feature.append(heavy_feature)
    
    if light_data:
        str_seq, seq2struc, struc = map(light_data.get, ['str_seq', 'seqres_to_structure', 'struc'])
        light_feature = make_feature(str_seq, seq2struc, struc)
        light_feature = _make_domain(light_feature, 'L')
        antibody_feature.append(light_feature)
    features.update(merge_chains(antibody_feature))

    if antigen_data:
        antigen_features = []
        for i, antigen_item in enumerate(antigen_data):
            str_seq, seq2struc, struc = map(antigen_item.get, ['str_seq', 'seqres_to_structure', 'struc'])
            antigen_feature = make_feature(str_seq, seq2struc, struc)
            antigen_features.append(antigen_feature)
        features.update(merge_chains(antigen_features))

    return features

def save_feature(feature, code, heavy_chain_id, light_chain_id, antigen_chain_ids, output_dir):
    antigen_chain_id = ''.join(antigen_chain_ids)
    np.savez(os.path.join(output_dir, f'{code}_{heavy_chain_id}_{light_chain_id}_{antigen_chain_id}.npz'), **feature)
    return

def save_header(header, file_path):
    with open(file_path, 'w') as fw:
        json.dump(header, fw)

def process(code, chain_ids, args):
    logging.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    mmcif_file = os.path.join(args.data_dir, f'{code}.cif')
    try:
        parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e
    if not parsing_result.mmcif_object:
        return
    
    # save_header(parsing_result.mmcif_object.header, 
    #         os.path.join(args.output_dir, f'{code}.json'))

    struc = parsing_result.mmcif_object.structure 
    
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

        if ((heavy_chain_id and heavy_chain_id not in parsing_result.mmcif_object.chain_to_seqres) or
            (light_chain_id and light_chain_id not in parsing_result.mmcif_object.chain_to_seqres)):
            logging.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
            continue
        
        flag = 0
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in parsing_result.mmcif_object.chain_to_seqres:
                logging.warning(f"antigen id: {antigen_chain_id} not exist")
                flag += 1
                continue
        if flag > 0:
            continue

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
            logging.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logging.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')


def process_pdb(code, chain_ids, args):
    logging.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    mmcif_file = os.path.join(args.data_dir, f'{code}.pdb')
    try:
        parser = PDBParser()
        struc = parser.get_structure('model', mmcif_file)
        # parsing_result = mmcif_parse(file_id=code, mmcif_file=mmcif_file)
    except PDBConstructionException as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
    except Exception as e:
        logging.warning('mmcif_parse: %s {%s}', mmcif_file, str(e))
        raise Exception('...') from e
    pdb_chain_id = list((struc.get_chains()))
    pdb_chain_id = [id.id for id in pdb_chain_id]
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

        if ((heavy_chain_id and heavy_chain_id not in pdb_chain_id) or
            (light_chain_id and light_chain_id not in pdb_chain_id)):
            logging.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
            continue
        
        flag = 0
        for antigen_chain_id in antigen_chain_ids:
            if antigen_chain_id not in pdb_chain_id:
                logging.warning(f"antigen id: {antigen_chain_id} not exist")
                flag += 1
                continue
        if flag > 0:
            continue
        
        try:
            feature = make_pdb_npz(struc, pdb_chain_id, heavy_chain_id, light_chain_id, antigen_chain_ids)
            save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, antigen_chain_ids, args.output_dir)
            logging.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        except Exception as e:
            traceback.print_exc()
            logging.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--summary_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--data_mode', type=str, required=True, help=['pdb', 'mmcif'])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.data_mode == 'mmcif':
        func = functools.partial(process, args=args)
    else:
        func = functools.partial(process_pdb, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.summary_file))


if __name__ == '__main__':
    
    
    main()
