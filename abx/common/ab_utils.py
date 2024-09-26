import numpy as np

from collections import OrderedDict

from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue

from abx.utils import Kabsch

def get_antibody_regions(N, struc2seq, chain_id, schema='imgt'):
    assert chain_id in 'HL'
    assert schema in ['chothia', 'imgt']
    cdr_def_chothia = {
            'H': {
                'fr1' : (1,  25),
                'cdr1': (26, 32),
                'fr2' : (33, 51),
                'cdr2': (52, 56),
                'fr3' : (57, 94),
                'cdr3': (95, 102),
                'fr4' : (103,113),
            }, 
            'L': {    
                'fr1' : (1,  23),
                'cdr1': (24, 34),
                'fr2' : (35, 49),
                'cdr2': (50, 56),
                'fr3' : (57, 88),
                'cdr3': (89, 97),
                'fr4' : (98, 109),
                }
    }
    
    cdr_def_imgt = {
            'H': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            }, 
            'L': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            }, 
    }

    cdr_def = cdr_def_imgt if schema == 'imgt' else cdr_def_chothia 

    range_dict = cdr_def[chain_id]

    _schema = {'fr1':0,'cdr1':1,'fr2':2,'cdr2':3,'fr3':4,'cdr3':5,'fr4':6}

    def _get_region(i):
        r = None
        for k, v in range_dict.items():
            if i >= v[0] and i <= v[1]:
                r = k
                break
        if r is None:
            return -1
        return 7 * int(chain_id == 'L') + _schema[r]
    
    region_def = np.full((N,),-1)

    for (hetflag, resseq, icode), v in struc2seq.items():
        region_def[v] = _get_region(int(resseq))

    return region_def

def get_antibody_regions_seq(imgt_numbering, chain_id):
    cdr_def_imgt = {
            'H': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            }, 
            'L': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            }, 
    }

    cdr_def = cdr_def_imgt

    range_dict = cdr_def[chain_id]

    _schema = {'fr1':0,'cdr1':1,'fr2':2,'cdr2':3,'fr3':4,'cdr3':5,'fr4':6}

    def _get_region(i):
        r = None
        for k, v in range_dict.items():
            if i >= v[0] and i <= v[1]:
                r = k
                break
        if r is None:
            return -1
        return 7 * int(chain_id == 'L') + _schema[r]

    N = len(imgt_numbering)
    region_def = np.full((N,),-1)
    
    for i, (_, resseq, icode) in enumerate(imgt_numbering):
        region_def[i] = _get_region(resseq)

    return region_def

def calc_ab_metrics(gt_coord, pred_coord, cdr_def, gt_str_seq=None, pred_str_seq=None):
    # mask = coord_mask * (cdr_def != -1)
    # gt_coord, pred_coord, cdr_def = gt_coord[mask,:], pred_coord[mask, :], cdr_def[mask]
    gt_aligned, pred_aligned = Kabsch(
            np.transpose(gt_coord,[1,0]),
            np.transpose(pred_coord, [1, 0]))

    def _calc_rmsd(A, B):
        return np.sqrt(np.mean(np.sum(np.square(A-B), axis=0)))

    full_rmsd = _calc_rmsd(gt_aligned, pred_aligned)
    
    ret = OrderedDict()
    # ret.update({'full_len' : gt_aligned.shape[1]})
    # ret.update({'full_rmsd':full_rmsd})
    
    _schema = {'cdr1':1,'cdr2':3,'cdr3':5}
    cdr_idx = {v : 'heavy_' + k for k, v in _schema.items()}
    cdr_idx.update({v + 7 : 'light_' + k for k, v in _schema.items()})

    for k, v in cdr_idx.items():
        
        indices = (cdr_def == k)
        gt, pred = gt_aligned[:, indices], pred_aligned[:, indices]
        # gt_seq = gt_str_seq[indices]
        # pred_seq = pred_str_seq[indices]
        if gt_str_seq is not None:
            gt_s= ''.join([char for char, keep in zip(gt_str_seq, indices) if keep])
            pred_s = ''.join([char for char, keep in zip(pred_str_seq, indices) if keep])
            AAR = np.mean([a==b for a, b in zip(gt_s, pred_s)])
            ret.update({v + '_AAR':AAR})
            if k == 5:
                AAR_1 = np.mean([a==b for a, b in zip(gt_s[4:-2], pred_s[4:-2])])
                ret.update({v + '_Loop_AAR':AAR_1})

        rmsd = _calc_rmsd(gt, pred)
        # ret.update({v + '_len' : gt.shape[1]})
        ret.update({v + '_RMSD':rmsd})
        if k == 5:
            gt_, pred_ = gt[:,4:-2], pred[:,4:-2]
            # pdb.set_trace()
            rmsd_ = _calc_rmsd(gt_, pred_)
            ret.update({v + '_Loop_RMSD':rmsd_})
    return ret

def renum_chain_imgt(orig_chain, struc2seq, imgt_numbering):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in struc2seq:
            idx = struc2seq[residue.id]
            new_residue = Residue(id=imgt_numbering[idx], resname=residue.resname, segid=residue.segid)
            for atom in residue:
                atom.detach_parent()
                new_residue.add(atom)
            chain.add(new_residue)
    return chain
