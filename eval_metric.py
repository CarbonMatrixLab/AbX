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
import re
import csv
from abx.metric import eval_metric, make_coords, cdr_numbering, InterfaceEnergy

def parse_list(data_dir):
    input_fname_pattern = '\.pdb$'
    relax_fname_pattern = '\_relaxed.pdb$'
    visited = set()
    for parent, _, files in os.walk(data_dir):
        for fname in files:
            # print(f"fname: {fname}")
            fpath = os.path.join(parent, fname)
            if not re.search(input_fname_pattern, fname):
                continue
            if re.search(relax_fname_pattern, fname):
                continue
            if os.path.getsize(fpath) == 0:
                continue
            if fpath in visited:
                continue
            visited.add(fpath)

            yield fpath



def main(args):
    reference_data = {}
    results = []
    for ref_pdb in parse_list(os.path.join(args.data_dir, 'reference')):
        # if 'relaxed' not in ref_pdb:
        pdb_name = ((ref_pdb.split('/')[-1]).split('.pdb'))[0]
        code, heavy_chain, light_chain, antigen_chain = pdb_name.split('_')
        ref_ab_ca, ref_ab_str_seq, ref_heavy_str, ref_light_str = make_coords(ref_pdb)
        cdr_def = cdr_numbering(ref_heavy_str, ref_light_str)
        data = {
            'cdr_def': cdr_def,
            'coords': ref_ab_ca,
            'str_seq': ref_ab_str_seq
        }
        # if args.energy:
        #     ref_energy = InterfaceEnergy(ref_pdb)
        #     data.update(
        #         {'dG': ref_energy}
        #     )
        reference_data[f'{pdb_name}'] = data
    # print(f"ref: {reference_data}")
    func = functools.partial(eval_metric, args=args, reference_data=reference_data)
    with mp.Pool(args.cpus) as p:
        results = p.starmap(func, ((pdb_file,) for pdb_file in parse_list(args.data_dir)))

    # Average Results for each Metric
    df = pd.DataFrame(results)
    column_means = df.mean()
    filtered_means = column_means.filter(like='RMSD').append(column_means.filter(like='AAR'))
    print(f"---------------------")
    print(f"Average Results for each Metric")
    print(f"---------------------")
    print(filtered_means)

    csv_file_path = os.path.join(args.data_dir, 'results.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = results[0].keys() if results else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--data_dir', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-e', '--energy', type=bool, default=False)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()
    
    main(args)


