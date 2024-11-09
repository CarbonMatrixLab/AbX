
import logging
import os

import numpy as np

from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO

import torch
from torch.nn import functional as F

from abx.common import protein, residue_constants
from abx.utils import exists

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBParser import PDBParser

from abx.common import residue_constants
from abx.data.mmcif_parsing import parse as mmcif_parse
from abx.preprocess.numbering import renumber_ab_seq, get_ab_regions
from Bio.SeqUtils import seq1
import pdb
import traceback
from abx.preprocess.make_ab_data_from_mmcif import *
logger = logging.getLogger()


def process_pdb(code, chain_ids, pdb_file):
    logging.info(f'processing {code}, {",".join(["_".join(x) for x in chain_ids])}')
    mmcif_file = pdb_file
    try:
        parser = PDBParser()
        struc = parser.get_structure('model', mmcif_file)[0]
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

    orig_heavy_chain_id, orig_light_chain_id, orig_antigen_chain_id = chain_ids
    antigen_chain_ids = orig_antigen_chain_id.split('|')
    antigen_chain_ids = [s.replace(" ", "") for s in antigen_chain_ids]
    
    heavy_chain_id, light_chain_id = _parse_chain_id(orig_heavy_chain_id, orig_light_chain_id)

    if ((heavy_chain_id and heavy_chain_id not in pdb_chain_id) or
        (light_chain_id and light_chain_id not in pdb_chain_id)):
        logging.warning(f'{code} {heavy_chain_id} {light_chain_id}: chain ids not exist.')
        traceback.print_exc()
    
    flag = 0
    for antigen_chain_id in antigen_chain_ids:
        if antigen_chain_id not in pdb_chain_id:
            logging.warning(f"antigen id: {antigen_chain_id} not exist")
            flag += 1
    
    if flag > 0:
        traceback.print_exc()
    # pdb.set_trace()
    try:
        feature = make_pdb_npz(struc, pdb_chain_id, heavy_chain_id, light_chain_id, antigen_chain_ids)
        # save_feature(feature, code, orig_heavy_chain_id, orig_light_chain_id, antigen_chain_ids, args.output_dir)
        logging.info(f'succeed: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id}')
        return feature


    except Exception as e:
        traceback.print_exc()
        logging.error(f'make structure: {mmcif_file} {orig_heavy_chain_id} {orig_light_chain_id} {str(e)}')






def pad_for_batch(items, batch_length, dtype):
    """Pad a list of items to batch_length using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All
            items are padded so that their length matches this number.
        dtype: A string ('seq', 'msk', 'crd') reperesenting the type of
            data included in items.

    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if dtype == 'seq':
        for seq in items:
            z = torch.ones(batch_length - seq.shape[0], dtype=seq.dtype) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype, device=msk.device)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == "crd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "crd_msk":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "ebd":
        for item in items:
            shape = [batch_length - item.shape[0]] + list(item.shape[1:])
            z = torch.zeros(shape, dtype=item.dtype, device = item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "pair":
        for item in items:
            c = F.pad(item, (0, 0, 0, batch_length - item.shape[-2], 0, batch_length - item.shape[-2]))
            batch.append(c)
    else:
        raise ValueError('Not implemented yet!')
    batch = torch.stack(batch, dim=0)
    return batch

def weights_from_file(filename):
    if filename:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in filter(lambda x: len(x)>0, map(lambda x: x.strip(), f)):
                items = line.split()
                yield float(items[0])

def embedding_get_labels(name, mat):
    if name == 'token':
        return [residue_constants.restypes_with_x[i if i < len(residue_constants.restypes_with_x) else -1]
                for i in range(mat.shape[0])]
    return None

def pdb_save(step, batch, headers, prefix='/tmp', is_training=False):
    
    for x, pid in enumerate(batch['name']):
        str_seq = batch['str_heavy_seq'][x] + batch['str_light_seq'][x]
        heavy_len = len(batch['str_heavy_seq'][x])
        N = len(str_seq)
        #aatype = batch['seq'][x,...].numpy()
        aatype = np.array([residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in str_seq])
        features = dict(aatype=aatype, residue_index=np.arange(N), heavy_len=heavy_len)

        if is_training:
            p = os.path.join(prefix, '{}_{}_{}.pdb'.format(pid, step, x))
        else:
            p = os.path.join(prefix, f'{pid}.pdb')

        with open(p, 'w') as f:
            coords = headers['folding']['final_atom_positions'].detach().cpu()  # (b l c d)
            _, _, num_atoms, _ = coords.shape
            coord_mask = np.asarray([residue_constants.restype_atom14_mask[restype][:num_atoms] for restype in aatype])

            result = dict(structure_module=dict(
                final_atom_mask = coord_mask,
                final_atom_positions = coords[x,:N].numpy()))
            prot = protein.from_prediction(features=features, result=result)
            f.write(protein.to_pdb(prot))

            #logging.debug('step: {}/{} length: {}/{} PDB save: {}'.format(step, x, masked_seq_len, len(str_seq), pid))

            #torsions = headers['folding']['traj'][-1]['torsions_sin_cos'][x,:len(str_seq)]
            #np.savez(os.path.join(prefix, f'{pid}_{step}_{x}.npz'), torsions = torsions.detach().cpu().numpy())

            if 'coord' in batch:
                if is_training:
                    p = os.path.join(prefix, '{}_{}_{}_gt.pdb'.format(pid, step, x))
                else:
                    p = os.path.join(prefix, f'{pid}_gt.pdb')

                with open(p, 'w') as f:
                    coord_mask = batch['coord_mask'].detach().cpu()
                    coords = batch['coord'].detach().cpu()
                    result = dict(structure_module=dict(
                        final_atom_mask = coord_mask[x,...].numpy(),
                        final_atom_positions = coords[x,...].numpy()))
                    prot = protein.from_prediction(features=features, result=result)
                    f.write(protein.to_pdb(prot))
                    #logging.debug('step: {}/{} length: {}/{} PDB save: {} (groundtruth)'.format(step, x, masked_seq_len, len(str_seq), pid))

def make_chain(aa_types, coords, chain_id, pLDDT, mask=None):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(i, aatype, coord, pLDDT):
        nonlocal serial_number
        
        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', i, ' '), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if atom_name == '':
                continue
            bfactor = pLDDT[j]
            atom = Atom(name=atom_name, 
                    coord=coord[j],
                    bfactor=bfactor, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue
    # pdb.set_trace()
    for i, (aa, coord) in enumerate(zip(aa_types, coords)):
        pLDDT_ = pLDDT[i]
        
        if mask is not None:
            if mask[i]:
                chain.add(make_residue(i+1, aa, coord, pLDDT_))
        else:
            chain.add(make_residue(i+1, aa, coord, pLDDT_))
    return chain

def save_pdb(str_heavy_seq, heavy_chain, str_light_seq, light_chain, coord, pdb_path, pLDDT, antigen_data):
    assert len(str_heavy_seq) + len(str_light_seq) == coord.shape[0]
    plddt_b_factors = np.repeat(
        pLDDT[..., None], residue_constants.atom_type_num, axis=-1
    )
    heavy_chain = make_chain(str_heavy_seq, coord[:len(str_heavy_seq)], heavy_chain, plddt_b_factors[:len(str_heavy_seq)])
    light_chain = make_chain(str_light_seq, coord[len(str_heavy_seq):], light_chain, plddt_b_factors[len(str_heavy_seq):])
    model = PDBModel(id=0)
    model.add(heavy_chain)
    model.add(light_chain)
    start = 0
    import pdb as pp
    # pp.set_trace()
    for i in range(len(antigen_data['antigen_chains'])):
        id = i + 2
        chain = antigen_data['antigen_chains'][i]
        chain_len = len(antigen_data['antigen_chain_ids'][antigen_data['antigen_chain_ids'] == id])
        plddt_b_factors = np.repeat(pLDDT[..., None], residue_constants.atom_type_num, axis=-1)
        plddt_b_factors = np.full((chain_len, residue_constants.atom_type_num), pLDDT[0])
        antigen_str_seq = antigen_data['antigen_str_seq'][start:start+chain_len]
        antigen_coords = antigen_data['antigen_coords'][start:start+chain_len]
        antigen_mask = antigen_data['antigen_coord_mask'][start:start+chain_len, residue_constants.atom_order['CA']]
        start += chain_len
        antigen_chain = make_chain(antigen_str_seq, antigen_coords, chain, plddt_b_factors, antigen_mask)
        model.add(antigen_chain)

    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


# def _mask_select_data(data, mask):
#     return {
#         k: _mask_select(v, mask)
#         for k, v in data.items()
#     }
