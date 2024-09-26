import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from abx.common.ab_utils import calc_ab_metrics
from abx.common import residue_constants

from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
import logging
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions


from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

#Core Includesfrom pyrosetta import create_score_function


init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
    -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 2 -no_fconfig')


def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']


def InterfaceEnergy(pdb_file):
    if '@' in pdb_file.split('/')[-1]:
        pdb_name = (pdb_file.split('/')[-1]).split('@')[0]
        code, heavy_chain_id, light_chain_id, antigen_chain_ids = pdb_name.split('_')
    else:
        pdb_name = (pdb_file.split('/')[-1]).split('.pdb')[0]
        code, heavy_chain_id, light_chain_id, antigen_chain_ids = pdb_name.split('_')
    parser = PDBParser()
    struc = parser.get_structure(code, pdb_file)[0]
    antibody_chains = [heavy_chain_id, light_chain_id]
    antigen_chain_ids = list(antigen_chain_ids)
    antigen_chains = set()
    for chain in struc:
        if chain.id in antibody_chains:
            continue
        if chain.id in antigen_chain_ids:
            antigen_chains.add(chain.id)
    antigen_chains = ''.join(antigen_chains)
    antibody_chains = ''.join(antibody_chains)
    interface = f"{antibody_chains}_{antigen_chains}"

    dG = pyrosetta_interface_energy(pdb_file, interface)
    return dG

def cdr_numbering(heavy_str, light_str):
    
    def _make_cdr(str, chain_id):
        allow = ['H'] if chain_id == 'H' else ['K', 'L']
        anarci_res = renumber_ab_seq(str, allow=allow, scheme='imgt')
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])
        # print(f"anarci_res: {anarci_res}")
        assert domain_numbering is not None
        
        cdr_def = get_ab_regions(domain_numbering, chain_id=chain_id)
        
        # updated_feature.update(dict(cdr_def=cdr_def, numbering=domain_numbering))
        return cdr_def
    heavy_cdr = _make_cdr(heavy_str, 'H')
    light_cdr = _make_cdr(light_str, 'L')
    cdr_def = np.concatenate([heavy_cdr, light_cdr], axis=0)
    return cdr_def

def make_coords(pdb_file):
    parser = PDBParser(QUIET=1)
    model = parser.get_structure('pdb', pdb_file)[0]
    pid, heavy_chain_id, light_chain_id, antigen_chain_id = pdb_file.split('/')[-1].split('.')[0].split('_')
    residues = list(model[heavy_chain_id].get_residues()) + list(model[light_chain_id].get_residues())
    heavy_residues = list(model[heavy_chain_id].get_residues())
    light_residues = list(model[light_chain_id].get_residues())


    residue_single_letter_codes = [seq1(residue.get_resname()) for residue in residues]
    heavy_single_letter_codes = [seq1(residue.get_resname()) for residue in heavy_residues]
    light_single_letter_codes = [seq1(residue.get_resname()) for residue in light_residues]

    str_seq = ''.join(residue_single_letter_codes)
    heavy_str_seq = ''.join(heavy_single_letter_codes)
    light_str_seq = ''.join(light_single_letter_codes)

    coords = np.zeros((len(residues), 3))
    for i, r in enumerate(residues):
        coords[i] = r['CA'].get_coord()

    return coords, str_seq, heavy_str_seq, light_str_seq

def eval_metric(pred_file, reference_data, args):
    # print(f"pred_file: {pred_file}")
    if '@' in pred_file.split('/')[-1]:
        pdb_name = (pred_file.split('/')[-1]).split('@')[0]
        code, heavy_chain_id, light_chain_id, antigen_chain_ids = pdb_name.split('_')
    else:
        pdb_name = (pred_file.split('/')[-1]).split('.pdb')[0]
        code, heavy_chain_id, light_chain_id, antigen_chain_ids = pdb_name.split('_')

    reference_data = reference_data[f'{pdb_name}']

    # print(f"ref_data: {reference_data}")
    if args.energy:
        # fields = 'cdr_def, coords, str_seq, dG'
        # # name = pred_file.split('/')[-1]
        # cdr_def, gt_ab_ca, gt_ab_str_seq, gt_dG = map(reference_data.get, fields)
        fields = 'cdr_def, coords, str_seq'
        cdr_def, gt_ab_ca, gt_ab_str_seq = reference_data['cdr_def'], reference_data['coords'], reference_data['str_seq']
    else:
        fields = 'cdr_def, coords, str_seq'
        cdr_def, gt_ab_ca, gt_ab_str_seq = reference_data['cdr_def'], reference_data['coords'], reference_data['str_seq']
    # gt_ca, gt_str_seq, gt_heavy_str_seq, gt_light_str_seq = make_coords(reference_file)
    pred_ab_ca, pred_ab_str_seq, pred_heavy_str_seq, pred_light_str_seq= make_coords(pred_file)
    assert (gt_ab_ca.shape[0] == pred_ab_ca.shape[0] and gt_ab_ca.shape[0] == cdr_def.shape[0])
    ab_metrics = calc_ab_metrics(gt_ab_ca, pred_ab_ca, cdr_def, gt_ab_str_seq, pred_ab_str_seq)
    ab_metrics.update(
        {
            'code': pdb_name,
            'file_path': pred_file
        }
    )
    if args.energy:
        file_dir_path = os.path.join(*pred_file.split('/')[:-1])
        pdb_name = pred_file.split('/')[-1]
        if '@' in pdb_name:
            # code, heavy_chain_id, light_chain_id,antigen_chain_ids = (pdb_name.split('@')[0]).split('_')
            output = '.'.join(pdb_name.split('.')[:-1])
        else:
            # code, heavy_chain_id, light_chain_id,antigen_chain_ids = (pdb_name.split('.')[0]).split('_')
            output = pdb_name.split('.')[0]

        relaxed_file = os.path.join(file_dir_path, f"{output}_relaxed.pdb")
        pred_dG = InterfaceEnergy(pred_file)
        ab_metrics.update(
            {
            # 'dG_ref': gt_dG,
            'dG_gen': pred_dG,
            # 'ddG': pred_dG - gt_dG
            }
        )
    # print(f"{pred_file}: {ab_metrics}")
    return ab_metrics
