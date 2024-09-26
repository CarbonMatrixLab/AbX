from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB

from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

#Core Includesfrom pyrosetta import create_score_function

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
        code, heavy_chain_id, light_chain_id, antigen_chain_ids= (pdb_name.split('/')[-1]).split('.pdb')[0].split('_')
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