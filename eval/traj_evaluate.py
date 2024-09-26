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
from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB
from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions

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

def parse_list(name_idx, pdb_dir, pred_pdb_dir, output_dir):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])
    # logger.info(f"names: {names}")
    for name in names:
        code, heavy_chain, light_chain, antigen_chain= name.split('_')
        def _parse_chain_id(heavy_chain_id, light_chain_id):
            if heavy_chain_id.islower() and heavy_chain_id.upper() == light_chain_id:
                heavy_chain_id = heavy_chain_id.upper()
            elif light_chain_id.islower() and light_chain_id.upper() == heavy_chain_id:
                light_chain_id = light_chain_id.upper()
            return heavy_chain_id, light_chain_id
        heavy_chain_, light_chain_ = _parse_chain_id(heavy_chain, light_chain)
        pdb_file = os.path.join(pdb_dir, f'{code}_{heavy_chain_}_{light_chain_}_{antigen_chain}' ,f'{code}_{heavy_chain_}{light_chain_}{antigen_chain}_ab_ag.pdb')
        reverse_steps = np.linspace(0.01, 1.0, 100)[::-1]
        pid = f'{code}_{heavy_chain_}_{light_chain_}_{antigen_chain}'
        for step in reverse_steps:
            if step > 0.01:    
                pred_pdb_file = os.path.join(pred_pdb_dir, f"{pid}@{step:.3f}.pdb")
            else:
                pred_pdb_file = os.path.join(pred_pdb_dir, f"{pid}.pdb")
            yield pdb_file, pred_pdb_file, output_dir
        # step = 0.20
        # pred_pdb_file = os.path.join(pred_pdb_dir, f"{pid}@{step:.3f}.pdb")
        # yield pdb_file, pred_pdb_file, output_dir

def find_all_indices(input_list, value_to_find):
    return [index for index, value in enumerate(input_list) if value == value_to_find]


def get_seqres_from_pdb(structure, chain_id):
    """Extract SEQRES sequence for a specific chain from a PDB file."""
    # parser = PDBParser(QUIET=True)
    # structure = parser.get_structure('structure', pdb_filename)
    model = structure[0]  # Assuming only one model in the PDB
    chain = model[chain_id]
    seq = ""
    for res in chain.get_unpacked_list():
        resname = res.get_resname().strip()
        if resname in three_to_one:
            seq += three_to_one[resname]
    # print(seq)

    return seq



def AddAntigen(structure, pred_structure, antigen_chain_ids):
    chain_to_add = None
    for antigen_chain_id in antigen_chain_ids:
        for chain in structure[0]:  # 假设我们关注第一个model
            if chain.id == antigen_chain_id:
                pred_structure[0].add(chain)

        # 保存新的结构为新的PDB文件
    return pred_structure
    
class ResidueSelect(Select):
    def __init__(self, chain_id, residue_indices):
        self.chain_id = chain_id
        self.residue_indices = residue_indices
    def accept_residue(self, residue):
        if residue.get_parent().id == self.chain_id and residue.id[1] in self.residue_indices:
            # 保留N, CA, C, O原子
            for atom in residue.get_atoms():
                if atom.get_name() not in ["N", "CA", "C", "O"]:
                    residue.detach_child(atom.id)
            return 1
        return 1
    
class CombinedResidueSelect(Select):
    def __init__(self, select1, select2):
        self.select1 = select1
        self.select2 = select2

    def accept_residue(self, residue):
        # 如果任一选择器接受该残基，则该残基被接受
        return self.select1.accept_residue(residue) and self.select2.accept_residue(residue)


def make_one_full_antibody(code, origin_antibody, pred_antibody, output_dir):
    def _make_domain(feature, chain_id):
        allow = ['H'] if chain_id == 'H' else ['K', 'L']
        anarci_res = renumber_ab_seq(feature['str_seq'], allow=allow, scheme='imgt')
        # print(f"str_seq: {feature['str_seq']}, anarci_res: {anarci_res}")
        domain_numbering, domain_start, domain_end = map(anarci_res.get, ['domain_numbering', 'start', 'end'])
        assert domain_numbering is not None
        
        cdr_def = get_ab_regions(domain_numbering, chain_id=chain_id)
        CDR_indices = dict()
        if allow == ['H']:
            CDR_H1_indice = find_all_indices(cdr_def, 1)
            CDR_H2_indice = find_all_indices(cdr_def, 3)
            CDR_H3_indice = find_all_indices(cdr_def, 5)
            CDR_indices.update(
                CDR_H1 = [min(CDR_H1_indice)+1, max(CDR_H1_indice)+1],
                CDR_H2 = [min(CDR_H2_indice)+1, max(CDR_H2_indice)+1],
                CDR_H3 = [min(CDR_H3_indice)+1, max(CDR_H3_indice)+1],
            )
        if allow == ['K', 'L']:
            CDR_L1_indice = find_all_indices(cdr_def, 8)
            CDR_L2_indice = find_all_indices(cdr_def, 10)
            CDR_L3_indice = find_all_indices(cdr_def, 12)
            CDR_indices.update(
                CDR_L1 = [min(CDR_L1_indice)+1, max(CDR_L1_indice)+1],
                CDR_L2 = [min(CDR_L2_indice)+1, max(CDR_L2_indice)+1],
                CDR_L3 = [min(CDR_L3_indice)+1, max(CDR_L3_indice)+1],
            )
            
        return domain_start, domain_end, CDR_indices
    
    output_dir = os.path.join(output_dir, 'before_relax')
    os.makedirs(output_dir, exist_ok=True)
    try:
        parser = PDBParser()
        struc = parser.get_structure(code, origin_antibody)
        pred_struc = parser.get_structure(code, pred_antibody)
    except:
        raise ValueError('PDB_parse: %s', pred_antibody)
    heavy_chain_id, light_chain_id,antigen_chain_ids = ((pred_antibody.split('/')[-1]).split('.')[0]).split('@')[0].split('_')[1:4]
    antigen_chain_ids = list(antigen_chain_ids)
    if heavy_chain_id:
        heavy_str_seq = get_seqres_from_pdb(struc, heavy_chain_id)
        heavy_data = dict(
            str_seq = heavy_str_seq,
        )
    else:
        heavy_data = None
    if light_chain_id:
        light_str_seq = get_seqres_from_pdb(struc, light_chain_id)
        light_data = dict(
            str_seq = light_str_seq,
        )
    else:
        light_data = None

    CDR_indice = dict()
    if heavy_data:
        heavy_domain_start, heavy_domain_end, heavy_cdr_indice = _make_domain(heavy_data, 'H')
        CDR_indice.update(
            heavy_cdr_indice,
        )
    if light_data:
        light_domain_start, light_domain_end, light_cdr_indice = _make_domain(light_data, 'L')
        CDR_indice.update(
            light_cdr_indice,
        )
    heavy_indices = []
    light_indices = []
    for CDR_name, indice in CDR_indice.items():
        if 'H' in CDR_name:
            heavy_indices = heavy_indices + list(range(indice[0], indice[1]+1))
        elif 'L' in CDR_name:
            light_indices = light_indices + list(range(indice[0], indice[1]+1))

    select = ResidueSelect(heavy_chain_id, heavy_indices)
    if light_data:
        select2 = ResidueSelect(light_chain_id, light_indices)
        select = CombinedResidueSelect(select, select2)
    
    struc = AddAntigen(struc, pred_struc, antigen_chain_ids)
    io = PDB.PDBIO()
    io.set_structure(struc)

    name = (pred_antibody.split('/')[-1]).split('.pdb')[0]
    io.save(f'{output_dir}/{name}_before_relax.pdb', select=select)
    return CDR_indice


def make_full_antibody(pdb_file, pred_pdb_file, output_dir):
    code, heavy_chain_id, light_chain_id, antigen_chain_ids = (((pred_pdb_file.split('/'))[-1]).split('.'))[0].split('@')[0].split('_')
    logger.info(f"code: {code}, heavy_chain_id: {heavy_chain_id}, light_chain_id: {light_chain_id}, antigen_chain_ids: {antigen_chain_ids}")
    try:
        CDR_indice = make_one_full_antibody(code, pdb_file, pred_pdb_file, output_dir)
        return CDR_indice
    except:
        logger.error(f"pdb_file: {pred_pdb_file} error!")

def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(pyrosetta.create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']


def InterfaceEnergy(origin_pdb_file, pred_pdb_file):
    logger.info(f"Calculate Rosetta Interface Energy for {pred_pdb_file}")
    code, heavy_chain_id, light_chain_id, antigen_chain_ids = (pred_pdb_file.split('/')[-1]).split('@')[0].split('_')
    parser = PDBParser()
    struc = parser.get_structure(code, origin_pdb_file)[0]
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
    print(f"interface: {interface}")
    dG_gen = pyrosetta_interface_energy(pred_pdb_file, interface)
    dG_ref = pyrosetta_interface_energy(origin_pdb_file, interface)
    return dG_gen, dG_ref

def Rosetta_packer(pdb_file, cdr_dict, output_dir):
    output_dir = os.path.join(output_dir, 'after_relax')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Rosetta processing {pdb_file} for Relax')
    pose = pose_from_pdb(pdb_file)
    scorefxn = create_score_function('ref2015')
    pack_mover = PackRotamersMover()
    code, heavy_chain_id, light_chain_id, antigen_chain_ids= ((pdb_file.split('/')[-1]).split('.pdb')[0]).split('@')[0].split('_')

    output_a = ((pdb_file.split('/')[-1]).split('.pdb')[0]).split('_')
    output_file_a = '_'.join(output_a[0:4])

    assert len(cdr_dict) == 6 if light_chain_id else 3
    original_pose = pose.clone()
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking()) 
    tf.push_back(operation.PreventRepacking()) 
    flexible_dict = dict()
    count = 0
    for cdr_name, indice in cdr_dict.items():
        if count < 3:
            flexible_dict.update(
                {cdr_name: [(heavy_chain_id, indice[0]), (heavy_chain_id, indice[1])]}
            )
        else:
            flexible_dict.update(
                {cdr_name: [(light_chain_id, indice[0]), (light_chain_id, indice[1])]}
            ) 
        count += 1       
    gen_selector = selections.ResidueIndexSelector('1')
    for cdr_name, indice in flexible_dict.items():
        flexible_residue_first = indice[0]
        flexible_residue_last = indice[1]
        gen_selector1 = selections.ResidueIndexSelector()
        gen_selector1.set_index_range(
            pose.pdb_info().pdb2pose(*flexible_residue_first), 
            pose.pdb_info().pdb2pose(*flexible_residue_last), 
        )
        gen_selector = OrResidueSelector(gen_selector, gen_selector1)
    nbr_selector = selections.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(gen_selector)
    nbr_selector.set_include_focus_in_subset(True)
    subset_selector = nbr_selector
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(
        prevent_repacking_rlt, 
        subset_selector,
        flip_subset=True,
    )
    tf.push_back(prevent_subset_repacking)
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    movemap = MoveMapFactory()
    movemap.add_bb_action(move_map_action.mm_enable, gen_selector)  # 允许第i个残基的骨架移动
    movemap.add_chi_action(move_map_action.mm_enable, subset_selector) # 允许第i个残基的侧链移动
    mm = movemap.create_movemap_from_pose(pose)
    fastrelax = FastRelax()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mm) #使用默认的Movemap()
    fastrelax.set_task_factory(tf)
    fastrelax.apply(pose)

    pose.dump_pdb(f'{output_dir}/{output_file_a}_relaxed.pdb')


def process(pdb_file, pred_pdb_file, output_dir, args):
    cdr_dict = make_full_antibody(pdb_file, pred_pdb_file, output_dir)
    before_relax_pdb_file = os.path.join(output_dir, 'before_relax', (pred_pdb_file.split('/')[-1]).split('.pdb')[0] + '_before_relax.pdb')
    try:
        Rosetta_packer(before_relax_pdb_file, cdr_dict, output_dir)
        relax_pdb_file = os.path.join(output_dir, 'after_relax', (pred_pdb_file.split('/')[-1]).split('.pdb')[0] + '_relaxed.pdb')
        dG_gen, dG_ref = InterfaceEnergy(pdb_file, relax_pdb_file)
        logger.info(f"{pred_pdb_file}@dG_wild: {dG_ref:.3f}")
        logger.info(f"{pred_pdb_file}@dG_design: {dG_gen:.3f}")
        logger.info(f"{pred_pdb_file}@ddG: {(dG_gen - dG_ref):.3f}")
        # return dG_gen, dG_ref
    except:
        ValueError(f"PDB file: {pred_pdb_file} Energy Error!")
    

def main(args):
    func = functools.partial(process, args=args)
    
    with mp.Pool(args.cpus) as p:
        p.starmap(func, parse_list(args.name_idx, args.pdb_dir, args.pred_antibody, args.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--pdb_dir', type=str, required=True)
    parser.add_argument('-n', '--name_idx', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-p', '--pred_antibody', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.output_dir,'relax.log')
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

