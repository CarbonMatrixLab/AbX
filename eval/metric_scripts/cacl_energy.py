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
from Bio.PDB import PDBParser
from Bio import PDB
from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions
import random
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

#Core Includes
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.kinematics import FoldTree
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.simple_metrics import metrics
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core import select
from pyrosetta.rosetta.core.select.movemap import *
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, AndResidueSelector, OrResidueSelector
from pyrosetta import create_score_function

#Protocol Includes
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols import relax as rel
from pyrosetta.rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from pyrosetta.rosetta.protocols.antibody import *
from pyrosetta.rosetta.protocols.loops import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
    -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 2 -no_fconfig')

have_name = list(pd.read_csv('./na.idx', names=['name'], header=None)['name'])

def parse_list(name_idx, pdb_dir):
    names = list(pd.read_csv(name_idx, names=['name'], header=None)['name'])
    names = names[::-1]
    random.shuffle(names)
    
    for name in names:
        if name in have_name:
            continue
        code, heavy_chain, light_chain, antigen_chain = name.split('_')
        def _parse_chain_id(heavy_chain_id, light_chain_id):
            if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
                heavy_chain_id = heavy_chain_id.upper()
            elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
                light_chain_id = light_chain_id.upper()
            return heavy_chain_id, light_chain_id
        heavy_chain_, light_chain_ = _parse_chain_id(heavy_chain, light_chain)
        pdb_file = os.path.join(pdb_dir, f'{code}_{heavy_chain_}_{light_chain_}_{antigen_chain}' ,f'{code}_{heavy_chain_}{light_chain_}{antigen_chain}_ab_ag.pdb')
        yield pdb_file 

def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(pyrosetta.create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']


def InterfaceEnergy(origin_pdb_file):
    logger.info(f"Calculate Rosetta Interface Energy for {origin_pdb_file}")
    code, heavy_chain_id, light_chain_id, antigen_chain_ids = (origin_pdb_file.split('/')[-2]).split('_')
    parser = PDBParser()
    struc = parser.get_structure(code, origin_pdb_file)[0]
    antibody_chains = [heavy_chain_id, light_chain_id]
    antigen_chain_ids = antigen_chain_ids.split()
    antigen_chains = set()
    for chain in struc:
        if chain.id in antibody_chains:
            continue
        if chain.id in antigen_chain_ids:
            antigen_chains.add(chain.id)
    antigen_chains = ''.join(antigen_chains)
    antibody_chains = ''.join(antibody_chains)
    interface = f"{antibody_chains}_{antigen_chains}"

    dG_ref = pyrosetta_interface_energy(origin_pdb_file, interface)
    return dG_ref

def process(pdb_file, args):
    dG_ref = InterfaceEnergy(pdb_file)
    logger.info(f"{pdb_file.split('/')[-2]}@dG_wild: {dG_ref}")
    return dG_ref

def main(args):
    with mp.Pool(args.cpus) as p:
        p.starmap(process, [(item, args) for item in parse_list(args.name_idx, args.pdb_dir)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--pdb_dir', type=str, required=True)
    parser.add_argument('-n', '--name_idx', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output_dir,'native_energy.log')
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

