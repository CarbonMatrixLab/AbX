import numpy as np
from diffuser import so3_diffuser
from diffuser import r3_diffuser
from diffuser import discrete_diffuser
from abx.common import residue_constants
from scipy.spatial.transform import Rotation
import torch
# import logging
from abx.model.quat_affine import quat_to_rotvec, quat_multiply, invert_quat, rotvec_to_quat
diffuser_obj_dict = {}

def _extract_trans_rots(rigid):
    assert len(rigid.shape) == 3
    assert rigid.shape[-1] == 7
    quat = rigid[..., :4]
    rot = quat_to_rotvec(quat)
    tran = rigid[..., 4:]
    return tran, rot

def _assemble_rigid(rotvec, trans):
    assert len(rotvec.shape) == 3
    quat = rotvec_to_quat(rotvec)
    tensor = torch.zeros((*quat.shape[:-1], 7), device=trans.device, dtype=trans.dtype)
    tensor[..., :4] = quat
    tensor[..., 4:] = trans
    return tensor

class FullDiffuser:

    def __init__(self, diff_conf):
        self._diff_conf = diff_conf
        self._diffuser = diff_conf['diffuse'] 
        self._diffuse_rot = self._diffuser['diffuse_rot']
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._diff_conf['so3'])

        self._diffuse_trans = self._diffuser['diffuse_trans']
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._diff_conf['r3'])

        self._diffuse_seq = self._diffuser['diffuse_seq']
        self._seq_diffuser = discrete_diffuser.DiscreteDiffuser(self._diff_conf['seq'])


    @staticmethod
    def get(diff_conf):
        global diffuser_obj_dict

        name = 'diffuser'

        if name not in diffuser_obj_dict:
            diffuser_obj_dict[name] = FullDiffuser(diff_conf)

        return diffuser_obj_dict[name]
    
    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed
    
    def forward_marginal(
            self,
            rigids_0: torch.tensor,
            seq_0:torch.tensor,
            t: torch.tensor,
            diffuse_mask: np.ndarray = None,
        ):
        trans_0, rot_0 = _extract_trans_rots(rigids_0)
        device = t.device
        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                torch.zeros_like(rot_0, device=device),
                torch.ones_like(t, device=device)
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(
                rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                torch.zeros_like(trans_0, device=device),
                torch.ones_like(t, device=device)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0, t)
            trans_score_scaling = self._r3_diffuser.score_scaling(t) # Stable

        if not self._diffuse_seq:
            seq_t, q_t0, rate_t = (
                seq_0,
                torch.eye(residue_constants.restype_num,device=device).unsqueeze(0).expand(t.shape[0], -1, -1),
                torch.zeros((t.shape[0], residue_constants.restype_num, residue_constants.restype_num), device=device)
            )
        else:
            seq_t, q_t0, rate_t = self._seq_diffuser.forward_marginal(
                seq_0, t)

        if diffuse_mask is not None:
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])
            trans_score = self._apply_mask(
                trans_score,
                torch.zeros_like(trans_score, device=trans_score.device),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                torch.zeros_like(rot_score, device=rot_score.device),
                diffuse_mask[..., None])
            seq_t = self._apply_mask(
                seq_t, seq_0, diffuse_mask)

        rigids_t = _assemble_rigid(rot_t, trans_t)


        return {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
            'seq_t': seq_t,
            'q_t0': q_t0,
            'rate_t': rate_t,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, scale=scale)
    
    def calc_quat_score(self, quat_t, quat_0, t):
        n_samples = np.cumprod(quat_0.shape[:2])[-1]
        quat_0 = quat_0.reshape((n_samples,4))
        quat_0_inv = invert_quat(quat_0).reshape((*quat_t.shape[:2], 4))
        quat_t = quat_t.reshape((*quat_0_inv.shape[:2], 4))
        quats_0t = quat_multiply(quat_0_inv, quat_t)
        rotvec_0t = quat_to_rotvec(quats_0t)
        return self._so3_diffuser.score(rotvec_0t, t)

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0: torch.tensor,
            rigid_t: torch.tensor,
            t: torch.tensor):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = torch.zeros_like(rot_0,device=t.device)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = torch.zeros_like(tran_0,device=t.device)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: torch.tensor,
            seq_t: torch.tensor,
            rot_score: torch.tensor,
            trans_score: torch.tensor,
            logits_t: torch.tensor,
            t: torch.tensor,
            dt: torch.tensor,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale
                )
        if not self._diffuse_seq:
            trans_t_1 = trans_t
        else: 
            seq_t_1 = self._seq_diffuser.reverse(
                x_t=seq_t,
                logits_t=logits_t,
                t=t,
                dt=dt,
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])
            seq_t_1 = self._apply_mask(
                seq_t_1, seq_t, diffuse_mask)

        return _assemble_rigid(rot_t_1, trans_t_1), seq_t_1

    def sample_ref(
            self,
            n_samples: np.array,
            impute_rigids: torch.tensor=None,
            impute_seq: torch.tensor=None,
            diffuse_mask: torch.tensor=None
        ):
        if impute_rigids is not None:
            device = impute_rigids.device
            assert impute_rigids.shape[:2] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute_rigids)
            trans_impute = trans_impute.reshape((*n_samples, 3))
            rot_impute = rot_impute.reshape((*n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and (impute_rigids is None or impute_seq is None):
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute_rigids is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute_rigids is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_seq) and impute_seq is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples, device=device)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples, device=device
            )
        else:
            trans_ref = trans_impute

        if self._diffuse_seq:
            seq_ref = self._seq_diffuser.sample_ref(
                n_samples=n_samples, device=device)
        else:
            seq_ref = impute_seq


        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
            seq_ref = self._apply_mask(
                seq_ref, impute_seq, diffuse_mask)
        trans_ref = self._r3_diffuser._unscale(trans_ref)

        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        seq_t = seq_ref
        
        return {'rigids_t': rigids_t,
                'seq_t': seq_t,
                }
