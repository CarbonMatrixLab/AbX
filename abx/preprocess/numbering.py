import requests
import time
import numpy as np

from Bio.PDB.Chain import Chain as PDBChain
from anarci import anarci

def renumber_pdb_by_remote(old_pdb, renum_pdb):
    success = False
    time.sleep(1)
    for i in range(5):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f},
                )

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    if success:
        new_pdb_data = response.text
        with open(renum_pdb, "w") as f:
            f.write(new_pdb_data)
    else:
        print(
            "Failed to renumber PDB. This is likely due to a connection error or a timeout with the AbNum server."
        )


def get_ab_regions(domain_numbering, chain_id):
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

    N = len(domain_numbering)
    region_def = np.full((N,),-1)
    
    for i, (resseq, icode) in enumerate(domain_numbering):
        region_def[i] = _get_region(resseq)

    return region_def

def renumber_ab_seq(str_seq, allow, scheme='imgt'):
    results = anarci([('A',str_seq)], scheme=scheme, allow=allow,)
    # print(f"results: {results} {('A',str_seq)}")
    numbering, alignment_details, hit_tables = results
    
    if numbering[0] is None:
        return dict(domain_numbering=None, start=None, end=None)

    # only return the most significant one
    domain_numbering, start_index, end_index = numbering[0][0]
    end_index += 1
    
    domain_numbering = [x[0] for x in domain_numbering if x[1] != '-']
    
    assert end_index - start_index == len(domain_numbering)
    
    return dict(domain_numbering=domain_numbering,
            start=start_index,
            end=end_index) 

def extract_ab_seq(full_seq, scheme='imgt'):
    input_seqs = [('full', full_seq)]

    ret_seqs, numbers, alignments, hit_tables = anarci.run_anarci(
        input_seqs, assign_germline=False, scheme=scheme)#bit_score_threshold=10)

    hit_tables = hit_tables[0]
    head, hits = hit_tables[0], hit_tables[1:]
    h_seqs, l_seqs = [], []
    for hit in hits:
        x = dict(zip(head, hit))
        chain_id = x['id'].split('_')[-1]
        if chain_id in ['H']:
            h_seqs.append(full_seq[x['query_start']:x['query_end'] + 1])
        if chain_id in ['K', 'L']:
            l_seqs.append(full_seq[x['query_start']:x['query_end'] + 1])

    h_seq = h_seqs[0] if len(h_seqs) > 0 else ''
    l_seq = l_seqs[0] if len(l_seqs) > 0 else ''

    return h_seq, l_seq

def extract_variable_domain_struc(orig_chain, new_chain_id, scheme='imgt'):
    assert scheme in ['chothia', 'imgt']

    assert new_chain_id in ['H', 'L']

    # Chothia numbering for PDB
    pdb_num_begin = 1
    if scheme == 'chothia':
        pdb_num_end = 113 if new_chain_id == 'H' else 109
    elif scheme == 'imgt':
        pdb_num_end = 128

    
    def _res_filter(residue):
        hetflag, resseq, icode = residue.get_id()
        return hetflag == ' ' \
                and resseq >= pdb_num_begin \
                and resseq <= pdb_num_end
    
    def _add_residue(target_chain, residue):
        if _res_filter(residue):
            residue.detach_parent()
            target_chain.add(residue)
        return

    new_chain = PDBChain(new_chain_id)
    
    for residue in orig_chain:
        _add_residue(new_chain, residue)

    return new_chain
