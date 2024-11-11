import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import resource
import json
import copy
import time

from collections import OrderedDict
import ml_collections
import torch
import torch.multiprocessing as mp
from einops import rearrange
import numpy as np


from abx.data import dataset
from abx.data.utils import save_pdb
from abx.common import residue_constants
from abx.common.utils import index_to_str_seq

from abx.model.abx import ScoreNetwork, get_prev
from diffuser.full_diffuser import FullDiffuser
import pdb


def log_setup(args):
    # logging
    os.makedirs(os.path.join(args.output_dir, args.mode), exist_ok=True)
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                args.output_dir, args.mode,
                f'{os.path.splitext(os.path.basename(__file__))[0]}.log'))]

    def handler_apply(h, f, *arg):
        f(*arg)
        return h
    level = logging.DEBUG if args.verbose else logging.INFO
    handlers = list(map(lambda x: handler_apply(
        x, x.setLevel, level), handlers))
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'
    handlers = list(map(lambda x: handler_apply(
        x, x.setFormatter, logging.Formatter(fmt)), handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    log_queue = mp.Queue(-1)

    return log_queue, handlers


def worker_setup(rank, log_queue, args):  
    logger = logging.getLogger()
    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)

    if (args.gpu_list or args.map_location) and torch.cuda.is_available():
        world_size = len(args.gpu_list) if args.gpu_list else 1
        if len(args.gpu_list) > 1:
            logging.info('torch.distributed.init_process_group: rank=%d@%d, world_size=%d', rank, args.gpu_list[rank] if args.gpu_list else 0, world_size)
            torch.distributed.init_process_group(
                    backend='nccl',
                    #init_method=f'file://{args.ipc_file}',
                    rank=rank, world_size=world_size)

def worker_cleanup(args):  # pylint: disable=redefined-outer-name
    if (args.gpu_list or args.map_location) and torch.cuda.is_available():
        if len(args.gpu_list) > 1:
            torch.distributed.destroy_process_group()

def worker_device(rank, args):  # pylint: disable=redefined-outer-name
    if args.device == 'gpu':
        return args.gpu_list[rank]
    else:
        return torch.device('cpu')

def worker_load(rank, args):  # pylint: disable=redefined-outer-name
    def _feats_gen(feats, device):
        for fn, opts in feats:
            if 'device' in opts:
                opts['device'] = device
            yield fn, opts
    
    device = worker_device(rank, args)
    
    with open(args.model_config, 'r', encoding='utf-8') as f:
       config = json.loads(f.read())
       diff_feat = config['diffuser']
       config = ml_collections.ConfigDict(config)
    model_conf = config.model
    diff_conf = config.diffuser
    diff_conf.so3.use_cached_score = True
    diffuser = FullDiffuser.get(diff_conf)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    model = ScoreNetwork(model_conf=model_conf, diffuser=diffuser)
    model.load_state_dict(model_state_dict, strict=True)
    
    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.loads(f.read())
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
            if 'diffuse' in feat_name:
                feat_args.update(
                    {'diff_conf': diff_feat}
                )
                if 'optimize_steps' in feat_args:
                    optimize_steps = feat_args['optimize_steps']
                    del feat_args['optimize_steps']
    model = model.to(device=device)
    model.eval()
    if args.mode in ['design', 'trajectory']:
        return list(_feats_gen(feats, device)), model, diffuser, config
    else:
        return list(_feats_gen(feats, device)), model, diffuser, config, optimize_steps

def postprocess_one(name, str_heavy_seq, str_light_seq, coord, args, pLDDT, antigen_data, time=None):
    if time:
        pdb_file = f'{args.output_dir}/{name}@{time:.4f}.pdb'
    else:
        pdb_file = f'{args.output_dir}/{name}.pdb'
    heavy_chain = name.split('_')[1]
    light_chain = name.split('_')[2]

    save_pdb(str_heavy_seq, heavy_chain, str_light_seq, light_chain, coord, pdb_file, pLDDT, antigen_data)

def postprocess_trajectory(batch, traj, args):
    fields = ('name', 'str_heavy_seq', 'str_light_seq', 'antigen_origin_str_seq', 'antigen_origin_atom14_gt_positions', 'antigen_origin_atom14_gt_exists', 'antigen_origin_chain_ids')
    names, str_heavy_seqs, str_light_seqs, antigen_str_seq, antigen_coords, antigen_coords_mask, antigen_chain_ids = map(batch.get, fields)
    for data in traj:
        pLDDT = data['pLDDT']
        seq = data['seq']
        coords = data['atom14_results']
        time = data['time'] if len(traj) > 1 else None
        for i, (name, str_heavy_seq, str_light_seq, antigen_str_seq, antigen_coords, antigen_coords_mask, antigen_chain_ids) in enumerate(zip(names, str_heavy_seqs, str_light_seqs, antigen_str_seq, antigen_coords, antigen_coords_mask, antigen_chain_ids)):
            pLDDT_ = pLDDT[i]
            h_len = len(str_heavy_seq)
            l_len = len(str_light_seq)
            heavy_seq = seq[i, :h_len]
            light_seq = seq[i, h_len:h_len+l_len]
            antigen_chains = list(name.split('_'))[-1]
            antigen_chains = antigen_chains.split('|')
            antigen_data = {
                'antigen_str_seq': antigen_str_seq,
                'antigen_coords': antigen_coords,
                'antigen_coord_mask': antigen_coords_mask,
                'antigen_chain_ids': antigen_chain_ids,
                'antigen_chains': antigen_chains
            }
            # pdb.set_trace()
            str_heavy_seq_ = index_to_str_seq(heavy_seq)
            str_light_seq_ = index_to_str_seq(light_seq)
            postprocess_one(name, str_heavy_seq_, str_light_seq_, coords[i, :len(str_heavy_seq)+len(str_light_seq)], args, pLDDT_, antigen_data, time)
            


# SE3 Diffusion Inference
def _set_t_feats(feats, diffuser, t, t_placeholder):
    feats['t'] = t * t_placeholder
    rot_score_scaling, trans_score_scaling = diffuser.score_scaling(feats['t'])
    feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
    feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
    return feats

def _self_conditioning(batch, model, config):
    model_sc = model(batch)
    prev = get_prev(batch, model_sc, config)
    batch.update(prev)
    return batch


def sample_fn(data_init, config, diffuser, model, args, num_t=100, min_t=0.01, center=True, self_condition=True, noise_scale=1.0, eps=1e-8):
    """SE3 Diffusion Process Inference function.

    Args:
        data_init: Initial data values for sampling.
    """
    # Run reverse process.
    model_conf = config.model
    score_network_conf = model_conf.heads.diffusion_module

    batch = copy.deepcopy(data_init)
    device = batch['rigids_t'].device
    bb_mask = batch['atom14_gt_exists'][..., 0]
    diffuse_mask = (1 - batch['fixed_mask']) * bb_mask
    antibody_len = batch['anchor_flag'].shape[1]
    t_placeholder = torch.ones(batch['rigids_t'].shape[0], device=device, dtype=torch.float32)

    reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
    dt = 1/num_t
    dt = torch.tensor(dt, device=device)
    
    if args.mode == 'optimize':
        opt_step = batch['t'][0].cpu().numpy()
        if opt_step < 1.0:
            mask = reverse_steps <= opt_step + eps
            reverse_steps = reverse_steps[mask]

    with torch.no_grad():
        traj = []
        if score_network_conf.embed.embed_self_conditioning and self_condition and len(reverse_steps) > 0:
            batch = _set_t_feats(batch, diffuser, reverse_steps[0], t_placeholder)
            batch = _self_conditioning(batch, model, model_conf)

        for t in reverse_steps:
            start_time = time.time()
            if t > min_t:
                t_ = torch.tile(torch.tensor(t, device=device), (batch['rigids_t'].shape[0],))
                
                # t_ = torch.tensor(t, device=device)
                batch = _set_t_feats(batch, diffuser, t_, t_placeholder)

                # Calculate the score function
                model_out = model(batch)
                rot_score = model_out['heads']['folding']['rot_score']
                trans_score = model_out['heads']['folding']['trans_score']
                seq_logits = model_out['heads']['sequence_module']['logits']
                if score_network_conf.embed.embed_self_conditioning:
                    prev = get_prev(batch, model_out, model_conf)
                    batch.update(prev)
                
                # Reverse the diffusion process
                rigids_t, seq_t = diffuser.reverse(
                    rigid_t=batch['rigids_t'],
                    seq_t = batch['seq_t'],
                    rot_score = rot_score,
                    trans_score = trans_score,
                    logits_t = seq_logits,
                    diffuse_mask = diffuse_mask,
                    t=t_,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )
                
            else:
                model_out = model(batch)
                rigids_t = model_out['heads']['folding']['rigids']
                seq_t = model_out['heads']['sequence_module']['seq_0']
                
                
            batch['rigids_t'] = rigids_t
            batch['seq_t'] = seq_t

            pLDDT = model_out['heads']['predicted_lddt']['pLDDT']
            pLDDT_item = torch.sum(pLDDT * diffuse_mask, dim=1) / torch.sum(diffuse_mask, dim=1)
            pLDDT = torch.tile(pLDDT_item[:, None], (1,data_init['anchor_flag'].shape[1])).to('cpu').numpy()

            atom14_results = model_out['heads']['folding']['final_atom14_positions'][:,:antibody_len]
            seq = torch.clamp(seq_t[:,:antibody_len], min=0, max=19).long().to('cpu').numpy()

            data = {
                'seq': seq,
                'atom14_results': atom14_results,
                'pLDDT': pLDDT,
                'time': t
            }
            traj.append(data)
            end_time = time.time()
            # logging.info(f"t step: {t} time: {end_time - start_time}")

        if args.mode != 'trajectory':
            traj = [traj[-1]]

        postprocess_trajectory(batch, traj, args)



def design(rank, log_queue, args):
    worker_setup(rank, log_queue, args)

    if args.mode == 'optimize':
        feats, model, diffuser, config, optimize_steps = worker_load(rank, args)
    else:
        feats, model, diffuser, config = worker_load(rank, args)
    logging.info('feats: %s', feats)

    # name_idx = []
    # with open(args.name_idx) as f:
    #     name_idx = [x.strip() for x in f]
    inference_step = config.diffuser.inference_step
    num_samples = args.num_samples
    
    def inference_fn(args):
        for i, batch in enumerate(test_loader):
            try:
                logging.info('name: %s', ','.join(batch['name']))
                start_time = time.time()
                sample_fn(batch, config, diffuser, model, args, num_t=inference_step, min_t=0.01, center=True, self_condition=True, noise_scale=1.0)
                end_time = time.time()
                logging.info('time: %s', end_time - start_time)
                
            except:
                logging.error('fails in predicting', batch['name'])

    args.output_dir = os.path.join(args.output_dir, f'{args.mode}')
    output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'optimize':
        for step in optimize_steps:
            logging.info(f"--------------------")
            logging.info(f"Optimize Steps: {step}")
            for i in range(len(feats)):
                feat_name, feat_args = feats[i]
                if 'diffuse' in feat_name:
                    feat_args['diff_conf'].update(opt_step=step)
            test_loader = dataset.load_single(
                pdb_file=args.pdb_file,
                is_training=False,
                feats=feats,
                batch_size=args.batch_size)
            
            logging.info(f"Reference Batch")
            ref_dir = os.path.join(output_dir, f'reference')
            os.makedirs(ref_dir, exist_ok=True)

            for i, batch in enumerate(test_loader):
                antibody_len = batch['anchor_flag'].shape[1]
                # pdb.set_trace()
                ref_data = {
                    'atom14_results': batch['atom14_gt_positions'][:,:antibody_len],
                    'seq': batch['seq'][:,:antibody_len],
                    'pLDDT': np.full((args.batch_size, antibody_len), fill_value=100)
                }
                ref_data = [ref_data]
                args.output_dir = ref_dir
                postprocess_trajectory(batch, ref_data, args)

            opt_dir = os.path.join(output_dir, f'OPT-{step}')
            os.makedirs(opt_dir, exist_ok=True)
            for k in range(num_samples):
                logging.info(f"{k:04d}-th Sample Batch")
                args.output_dir = os.path.join(opt_dir, f'{k:04d}')
                os.makedirs(args.output_dir, exist_ok=True)
                inference_fn(args)

    else:
        test_loader = dataset.load_single(
            pdb_file=args.pdb_file,
            is_training=False,
            feats=feats,
            batch_size=args.batch_size)
        
        ref_dir = os.path.join(output_dir, f'reference')
        os.makedirs(ref_dir, exist_ok=True)
        logging.info(f"Reference Batch")
        for i, batch in enumerate(test_loader):
            antibody_len = batch['anchor_flag'].shape[1]
            ref_data = {
                'atom14_results': batch['atom14_gt_positions'][:,:antibody_len],
                'seq': batch['seq'][:,:antibody_len],
                'pLDDT': np.full((args.batch_size, antibody_len), fill_value=100)
            }
            ref_data = [ref_data]
            args.output_dir = ref_dir
            postprocess_trajectory(batch, ref_data, args)
        
        for k in range(num_samples):
            logging.info(f"{k:04d}-th Sample Batch")
            args.output_dir = os.path.join(output_dir, f'{k:04d}')
            os.makedirs(args.output_dir, exist_ok=True)
            inference_fn(args)

    worker_cleanup(args)

def main(args):
    
    mp.set_start_method('spawn', force=True)

    log_queue, handlers = log_setup(args)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    logging.info('-----------------')
    logging.info('Arguments: %s', args)
    logging.info('-----------------')

    if len(args.gpu_list) > 1:
        mp.spawn(design, args=(log_queue, args),
                nprocs=len(args.gpu_list) if args.gpu_list else 1,
                join=True)
    else:
        design(args.gpu_list[0], log_queue, args)

    listener.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu')

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_features', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)

    parser.add_argument('--pdb_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--mode', type=str, choices=['design', 'optimize', 'trajectory'], default='design')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100)

    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
