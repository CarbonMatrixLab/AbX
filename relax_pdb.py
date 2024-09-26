from abx.relax import Rosetta_relax
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


def parse_list(data_dir):
    input_fname_pattern = '\.pdb$'
    visited = set()
    for parent, _, files in os.walk(data_dir):
        for fname in files:
            # print(f"fname: {fname}")
            fpath = os.path.join(parent, fname)
            if not re.search(input_fname_pattern, fname):
                continue
            if os.path.getsize(fpath) == 0:
                continue
            if fpath in visited:
                continue
            visited.add(fpath)

            yield fpath



def main(args):
    # fpath = parse_list(args.data_dir)

    func = functools.partial(Rosetta_relax, args=args)
    # print(f"fpath: {fpath}")

    # for name in parse_list(args.data_dir):
    #     print(f"name: {name}")
    #     func(name)
    with mp.Pool(args.cpus) as p:
        p.starmap(func, ((pdb_file,) for pdb_file in parse_list(args.data_dir)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--data_dir', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    parser.add_argument('-g', '--generate_area', type=str, choices=['cdrs', 'H3'], default='cdrs')
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args.data_dir,'relax.log')
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


