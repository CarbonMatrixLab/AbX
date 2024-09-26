import torch
from torch.nn import functional as F
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional

from einops import rearrange, parse_shape

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def l2_normalize(v, dim=-1, epsilon=1e-12):
    norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True) + epsilon)
    return v / norms

def squared_difference(x, y):
    return torch.square(x-y)

def batched_select(params, indices, dim=None, batch_dims=0):
    params_shape, indices_shape = list(params.shape), list(indices.shape)
    assert params_shape[:batch_dims] == indices_shape[:batch_dims]
   
    def _permute(dim, dim1, dim2):
        permute = []
        for i in range(dim):
            if i == dim1:
                permute.append(dim2)
            elif i == dim2:
                permute.append(dim1)
            else:
                permute.append(i)
        return permute

    if dim is not None and dim != batch_dims:
        params_permute = _permute(len(params_shape), dim1=batch_dims, dim2=dim)
        indices_permute = _permute(len(indices_shape), dim1=batch_dims, dim2=dim)
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        params_shape, indices_shape = list(params.shape), list(indices.shape)

    params, indices = torch.reshape(params, params_shape[:batch_dims+1] + [-1]), torch.reshape(indices, list(indices_shape[:batch_dims]) + [-1, 1])

    # indices = torch.tile(indices, params.shape[-1:])
    indices = indices.repeat([1] * (params.ndim - 1) + [params.shape[-1]])

    batch_params = torch.gather(params, batch_dims, indices.to(dtype=torch.int64))

    output_shape = params_shape[:batch_dims] + indices_shape[batch_dims:] + params_shape[batch_dims+1:]

    if dim is not None and dim != batch_dims:
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        
    return torch.reshape(batch_params, output_shape)

# def lddt(pred_points, true_points, points_mask, cutoff=15.):
#     """Computes the lddt score for a batch of coordinates.
#         https://academic.oup.com/bioinformatics/article/29/21/2722/195896
#         Inputs: 
#         * pred_coords: (b, l, d) array of predicted 3D points.
#         * true_points: (b, l, d) array of true 3D points.
#         * points_mask : (b, l) binary-valued array. 1 for points that exist in
#             the true points
#         * cutoff: maximum inclusion radius in reference struct.
#         Outputs:
#         * (b, l) lddt scores ranging between 0 and 1
#     """
#     assert len(pred_points.shape) == 3 and pred_points.shape[-1] == 3
#     assert len(true_points.shape) == 3 and true_points.shape[-1] == 3

#     eps = 1e-10

#     # Compute true and predicted distance matrices. 
#     pred_cdist = torch.sqrt(torch.sum(
#         torch.square(
#             rearrange(pred_points, 'b l c -> b l () c') -
#             rearrange(pred_points, 'b l c -> b () l c')),
#         dim=-1,
#         keepdims=False))
#     true_cdist = torch.sqrt(torch.sum(
#         torch.square(
#             rearrange(true_points, 'b l c -> b l () c') -
#             rearrange(true_points, 'b l c -> b () l c')),
#         dim=-1,
#         keepdims=False))
   
#     cdist_to_score = ((true_cdist < cutoff) *
#             (rearrange(points_mask, 'b i -> b i ()') *rearrange(points_mask, 'b j -> b () j')) *
#             (1.0 - torch.eye(true_cdist.shape[1], device=points_mask.device)))  # Exclude self-interaction

#     # Shift unscored distances to be far away
#     dist_l1 = torch.abs(true_cdist - pred_cdist)

#     # True lDDT uses a number of fixed bins.
#     # We ignore the physical plausibility correction to lDDT, though.
#     score = 0.25 * sum([dist_l1 < t for t in (0.5, 1.0, 2.0, 4.0)])

#     # Normalize over the appropriate axes.
#     return (torch.sum(cdist_to_score * score, dim=-1) + eps)/(torch.sum(cdist_to_score, dim=-1) + eps)


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def plddt(logits):
    """Compute per-residue pLDDT from logits
    """
    device = logits.device if hasattr(logits, 'device') else None
    # Shape (b, l, c)
    b, c = logits.shape[0], logits.shape[-1]
    width = 1.0 / c
    centers = torch.arange(start=0.5*width, end=1.0, step=width, device=device)
    probs = F.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * centers.view(*((1,) * len(probs.shape[:-1])), *centers.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100

