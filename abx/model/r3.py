import torch
from einops import rearrange
import numpy as np
from abx.model.quat_affine import rot_to_quat

def rigids_op(rigids, op):
    return tuple(map(op, rigids))

def rigids_apply(rigids, points):
    rots, trans = rigids
    assert (points.shape[-1] == 3) and (points.ndim - trans.ndim  in [0, 1])

    if points.ndim == trans.ndim:
        return trans + torch.einsum('... l r d, ... l d -> ... l r', rots, points)
    else:
        return rearrange(trans, '... d -> ... () d') + torch.einsum('... l r d, ... l m d -> ... l m r', rots, points)

def rigids_mul_vecs(rigids, vecs):
    rots, trans = rigids
    assert vecs.ndim - trans.ndim  in [0, 1]

    if vecs.ndim == trans.ndim:
        return trans + torch.squeeze(torch.matmul(rots, vecs[..., None]), dim=-1) 
    else:
        return rearrange(trans, '... d -> ... () d') + torch.einsum('... r d, ... m d -> ... m r', rots, vecs)

def rots_mul_rots(rots_a, rots_b):
    assert rots_a.shape[-2:] == (3, 3) and rots_b.shape[-2:] == (3, 3)

    return torch.einsum('... r d, ... d m -> ... r m', rots_a, rots_b)

def rigids_mul_rots(rigids, rots_b):
    rots, trans = rigids
    return (rots_mul_rots(rots, rots_b), trans)

def rigids_mul_rigids(rigids_a, rigids_b):
    rots_a, trans_a = rigids_a
    rots_b, trans_b = rigids_b

    assert rots_a.ndim == rots_b.ndim and trans_a.ndim == trans_b.ndim

    rots = torch.einsum('... r d, ... d m -> ... r m', rots_a, rots_b)

    trans = torch.einsum('... r d, ...d -> ... r', rots_a, trans_b) + trans_a

    return (rots, trans)

def invert_rots(rots):
    return rearrange(rots, '... i j -> ... j i')

def rots_mul_vecs(rots, vecs):
    return torch.einsum('... r d, ... d -> ... r', rots, vecs)

def invert_rigids(rigids):
    rots, trans = rigids
    inv_rots = invert_rots(rots)
    inv_trans = -rots_mul_vecs(inv_rots, trans)
    
    return (inv_rots, inv_trans)

def rigids_from_tensor4x4(m):
    assert m.shape[-2:] == (4, 4)

    rots, trans = m[...,:3,:3], m[...,:3,3]

    return (rots, trans)

def rigids_to_tensor4x4(rigids):
    shape = rigids[0].shape[:-2] + (4, 4)
    tensor = torch.zeros(shape, dtype=rigids[0].dtype, device=rigids[0].device)
    tensor[...,:3,:3] = rigids[0]
    tensor[...,:3,3] = rigids[1]
    tensor[...,3,3] = 1.0
    return tensor

def vecs_robust_normalize(v, dim=-1, epsilon=1e-8):
  norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True) + epsilon)
  return v / norms

def vecs_cross_vecs(v1, v2):
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3
    
    return torch.stack([
        v1[...,1] * v2[...,2] - v1[...,2] * v2[...,1],
        v1[...,2] * v2[...,0] - v1[...,0] * v2[...,2],
        v1[...,0] * v2[...,1] - v1[...,1] * v2[...,0],
        ], dim=-1)

def rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane, epsilon=1e-6):
    # Shape (b, l, 3)
    assert point_on_neg_x_axis.shape[-1] == 3\
           and origin.shape[-1] == 3\
           and point_on_xy_plane.shape[-1] == 3

    e0_unnormalized = origin - point_on_neg_x_axis

    e1_unnormalized = point_on_xy_plane - origin

    e0 = vecs_robust_normalize(e0_unnormalized)
    
    c = torch.einsum('... c, ... c -> ... ', e1_unnormalized, e0)[..., None]
    e1 = e1_unnormalized - c * e0
    e1 = vecs_robust_normalize(e1)

    # e2 = torch.cross(e0, e1, dim=-1)
    e2 = vecs_cross_vecs(e0, e1)

    R = torch.stack((e0, e1, e2), dim=-1)
    return (R, origin)

def rigids_to_tensor7(rigids):
    rots = rigids[0]
    trans = rigids[1]
    quat = rot_to_quat(rots)
    tensor = torch.zeros((*quat.shape[:-1], 7), device=rots.device, dtype=rots.dtype)
    tensor[..., :4] = quat
    tensor[..., 4:] = trans
    return tensor


def matrix_to_rotvec(R):
    """Convert batch of 3x3 rotation matrices to rotation vectors"""
    assert R.shape[-2:] == (3, 3)
    batch_size = R.shape[0]
    device = R.device
    trace = torch.empty(batch_size,device=device)
    for i in range(batch_size):
        trace[i] = R[i].trace()
    theta = torch.acos((trace - 1) / 2)
    r = 1 / (2 * torch.sin(theta)[..., None]) * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    return theta.unsqueeze(-1) * r

def rotvec_to_matrix(rotvec):
    assert len(rotvec.shape) == 2
    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    r = rotvec / (theta + 1e-6)
    cost = torch.cos(theta)[...,None]
    sint = torch.sin(theta)[...,None]
    outer = torch.bmm(r.unsqueeze(-1), r.unsqueeze(-2))
    eye = torch.eye(r.shape[-1]).expand_as(outer).to(device=rotvec.device)
    matrix = cost * eye + (1 - cost) * outer + sint * skew_symmetric(r)
    
    return matrix

def skew_symmetric(v):
    zero = torch.zeros_like(v[..., 0]).to(device=v.device)
    return torch.stack((zero, -v[..., 2], v[..., 1],
                        v[..., 2], zero, -v[..., 0],
                        -v[..., 1], v[..., 0], zero), -1).reshape(v.shape[0], 3, 3) 

def compose_rotvec(r1,r2):
    assert len(r1.shape) == 2 and len(r2.shape) == 2
    
    R1 = rotvec_to_matrix(r1).float()
    R2 = rotvec_to_matrix(r2).float()
    cR = torch.einsum('...ij,...jk->...ik', R1,R2)

    return matrix_to_rotvec(cR)


