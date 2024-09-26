# utils for working with 3d-protein structures
import contextlib
from functools import wraps
from inspect import isfunction
import uuid

import numpy as np
import torch

# helpers
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def unique_id():
    """Generate a unique ID as specified in RFC 4122."""
    # See https://docs.python.org/3/library/uuid.html
    return str(uuid.uuid4())

def uniform_sample(n, min, max):
    rand_tensor = torch.FloatTensor(n).uniform_()
    sample_tensor = min + (max-min) * rand_tensor
    
    return sample_tensor


def torch_interp(x_new, x, y):
    """Batched 1D linear interpolation for PyTorch tensors."""
    # Sort x and y
    sorted_indices = x.argsort(dim=1)

    x = torch.gather(x, -1, sorted_indices)
    y = torch.gather(y, -1, sorted_indices)

    # Find the indices of the bins where x_new falls
    bin_indices = torch.sum((x.unsqueeze(2) < x_new.unsqueeze(1)), dim=1)
    
    # Clip bin_indices so that it's in [0, x.shape[1] - 2]
    bin_indices = torch.clamp(bin_indices, 0, x.shape[1] - 2)

    # Calculate weights
    x_low = torch.gather(x, -1, bin_indices)
    x_high = torch.gather(x, -1, bin_indices + 1)
    y_low = torch.gather(y, -1, bin_indices)
    y_high = torch.gather(y, -1, bin_indices + 1)

    w = (x_new - x_low) / (x_high - x_low + 1e-8)

    w[x_new > x[:, -1].unsqueeze(1)] = 1.0
    w[x_new < x[:, 0].unsqueeze(1)] = 0

    # Compute interpolated values
    y_new = y_low * (1 - w) + y_high * w

    return y_new








# decorators

def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        return fn(*args, **kwargs)
    return inner

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy

def expand_arg_dims(dim_len = 3):
    """ pack here for reuse. 
        turns input into (B x D x N)
    """
    def outer(fn):
        @wraps(fn)
        def inner(x, y, **kwargs):
            assert len(x.shape) == len(y.shape), "Shapes of A and B must match."
            remaining_len = dim_len - len(x.shape)
            x = expand_dims_to(x, length = remaining_len)
            y = expand_dims_to(y, length = remaining_len)
            return fn(x, y, **kwargs)
        return inner
    return outer

def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            backend = kwargs.pop('backend')
            passed_args = fn(*args, **kwargs)
            passed_args = list(passed_args)
            if isinstance(passed_args[-1], dict):
                passed_kwargs = passed_args.pop()
            else:
                passed_kwargs = {}
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

@contextlib.contextmanager
def torch_disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

# distance utils (distogram to dist mat + masking)

def center_distogram_torch(distogram, bins, min_t=1., center="mean", wide="std"):
    """ Returns the central estimate of a distogram. Median for now.
        Inputs:
        * distogram: (batch, N, N, B) where B is the number of buckets.
        * bins: (B,) containing the cutoffs for the different buckets
        * min_t: float. lower bound for distances.
        Outputs:
        * central: (batch, N, N)
        * dispersion: (batch, N, N)
        * weights: (batch, N, N)
    """
    shape, device = distogram.shape, distogram.device
    # threshold to weights and find mean value of each bin
    n_bins = ( bins - 0.5 * (bins[2] - bins[1]) ).to(device)
    n_bins[0]  = 1.5
    n_bins[-1] = 1.33*bins[-1] # above last threshold is ignored
    max_bin_allowed = torch.tensor(n_bins.shape[0]-1).to(device).long()
    # calculate measures of centrality and dispersion - 
    magnitudes = distogram.sum(dim=-1)
    if center == "median":
        cum_dist = torch.cumsum(distogram, dim=-1)
        medium   = 0.5 * cum_dist[..., -1:]
        central  = torch.searchsorted(cum_dist, medium).squeeze()
        central  = n_bins[ torch.min(central, max_bin_allowed) ]
    elif center == "mean":
        central  = (distogram * n_bins).sum(dim=-1) / magnitudes
    # create mask for last class - (IGNORE_INDEX)   
    mask = (central <= bins[-2].item()).float()
    # mask diagonal to 0 dist - don't do masked filling to avoid inplace errors
    diag_idxs = np.arange(shape[-2])
    central   = expand_dims_to(central, 3 - len(central.shape))
    central[:, diag_idxs, diag_idxs]  *= 0.
    # provide weights
    if wide == "var":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes
    elif wide == "std":
        dispersion = ((distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes).sqrt()
    else:
        dispersion = torch.zeros_like(central, device=device)
    # rescale to 0-1. lower std / var  --> weight=1. set potential nan's to 0
    weights = mask / (1+dispersion)
    weights[weights != weights] *= 0.
    weights[:, diag_idxs, diag_idxs] *= 0.
    return central, weights


# distance matrix to 3d coords: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279

def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distogram is (N x N) and symmetric
        Outs: 
        * best_3d_coords: (batch x 3 x N)
        * historic_stresses: (batch x steps)
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()
    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = [torch.tensor([np.inf]*batch, device=device)]

    # initialize by eigendecomposition: https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
    # follow : https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
    D = pre_dist_mat**2
    M =  0.5 * (D[:, :1, :] + D[:, :, :1] - D) 
    # do loop svd bc it's faster: (2-3x in CPU and 1-2x in GPU)
    # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
    svds = [torch.svd_lowrank(mi) for mi in M]
    u = torch.stack([svd[0] for svd in svds], dim=0)
    s = torch.stack([svd[1] for svd in svds], dim=0)
    v = torch.stack([svd[2] for svd in svds], dim=0)
    best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

    # only eigen - way faster but not weights
    if weights is None and eigen==True:
        return torch.transpose( best_3d_coords, -1, -2), torch.zeros_like(torch.stack(his, dim=0))
    elif eigen==True:
        if verbose:
            print("Can't use eigen flag if weights are active. Fallback to iterative")

    # continue the iterative way
    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        best_3d_coords = best_3d_coords.contiguous()
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[ dist_mat <= 0 ] += 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))

        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        his.append( stress / dis )

    return torch.transpose(best_3d_coords, -1,-2), torch.stack(his, dim=0)

def mds_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper. 
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress 
    """
    if weights is None:
        weights = np.ones_like(pre_dist_mat)

    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    his = [np.inf]
    # init random coords
    best_stress = np.inf * np.ones(batch)
    best_3d_coords = 2*np.random.rand(batch, 3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] - best_3d_coords[:, :, None, :], axis=-3)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum(axis=(-1, -2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio 
        B[:, np.arange(N), np.arange(N)] += ratio.sum(axis=-1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.matmul(best_3d_coords, B))
        dis = np.linalg.norm(coords, axis=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)

def get_dihedral_torch(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Can't use torch.dot bc it does not broadcast
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) ) 


def get_dihedral_numpy(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return np.arctan2( ( (np.linalg.norm(u2, axis=-1, keepdims=True) * u1) * np.cross(u2,u3, axis=-1)).sum(axis=-1),  
                       ( np.cross(u1,u2, axis=-1) * np.cross(u2, u3, axis=-1) ).sum(axis=-1) ) 

def calc_phis_torch(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = torch.transpose(pred_coords.detach(), -1 , -2).cpu()
    # ensure dims
    N_mask = expand_dims_to( N_mask, 2-len(N_mask.shape) )
    CA_mask = expand_dims_to( CA_mask, 2-len(CA_mask.shape) )
    if C_mask is not None: 
        C_mask = expand_dims_to( C_mask, 2-len(C_mask.shape) )
    else:
        C_mask = torch.logical_not(torch.logical_or(N_mask,CA_mask))
    # select points
    n_terms  = pred_coords_[:, N_mask[0].squeeze()]
    c_alphas = pred_coords_[:, CA_mask[0].squeeze()]
    c_terms  = pred_coords_[:, C_mask[0].squeeze()]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_torch(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return torch.stack([(x<0).float().mean() for x in phis], dim=0 ) 
    return phis


def calc_phis_numpy(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (N, ) boolean mask for N-term positions
        * CA_mask: (N, ) boolean mask for C-alpha positions
        * C_mask: (N, ) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
    """ 
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = np.transpose(pred_coords, (0, 2, 1))
    n_terms  = pred_coords_[:, N_mask.squeeze()]
    c_alphas = pred_coords_[:, CA_mask.squeeze()]
    # select c_term auto if not passed
    if C_mask is not None: 
        c_terms = pred_coords_[:, C_mask]
    else:
        c_terms  = pred_coords_[:, (np.ones_like(N_mask)-N_mask-CA_mask).squeeze().astype(bool) ]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_numpy(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop: 
        return np.array( [(x<0).mean() for x in phis] ) 
    return phis


# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/

def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    device = X.device
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C)
    
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_

def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_


# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html

def distmat_loss_torch(X=None, Y=None, X_mat=None, Y_mat=None, p=2, q=2,
                       custom=None, distmat_mask=None, clamp=None):
    """ Calculates a loss on the distance matrix - no need to align structs.
        Inputs: 
        * X: (N, d) tensor. the predicted structure. One of (X, X_mat) is needed.
        * X_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * Y: (N, d) tensor. the true structure. One of (Y, Y_mat) is needed.
        * Y_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * p: int. power for the distance calculation (2 for euclidean)
        * q: float. power for the scaling of the loss (2 for MSE, 1 for MAE, etc)
        * custom: func or None. custom loss over distance matrices. 
                  ex: lambda x,y: 1 - 1/ (1 + ((x-y))**2) (1 is very bad. 0 is good)
        * distmat_mask: (N, N) mask (boolean or weights for each ij pos). optional.
        * clamp: tuple of (min,max) values for clipping distance matrices. ex: (0,150)
    """
    assert (X is not None or X_mat is not None) and \
           (Y is not None or Y_mat is not None), "The true and predicted coords or dist mats must be provided"
    def to_mat(x):
        x = x.squeeze()
        if clamp is not None:
            x = torch.clip(x, *clamp)
        return torch.cdist(x, x, p=p)

    # calculate distance matrices
    if X_mat is None: 
        X = X.squeeze()
        if clamp is not None:
            X = torch.clip(X, *clamp)
        X_mat = torch.cdist(X, X, p=p)
    if Y_mat is None: 
        Y = Y.squeeze()
        if clamp is not None:
            Y = torch.clip(Y, *clamp)
        Y_mat = torch.cdist(Y, Y, p=p)
    if distmat_mask is None:
        distmat_mask = torch.ones_like(Y_mat).bool()

    # do custom expression if passed
    if custom is not None:
        return custom(X_mat.squeeze(), Y_mat.squeeze()).mean()
    # **2 ensures always positive. Later scale back to desired power
    else:
        loss = ( X_mat - Y_mat )**2 
        if q != 2:
            loss = loss**(q/2)
        return loss[distmat_mask].mean()

def rmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return torch.sqrt(torch.mean((X - Y)**2, axis=(-1, -2)))

def rmsd_numpy(X, Y):
    """ Assumes X,Y are both (B x D x N). See below for wrapper. """
    return np.sqrt(np.mean((X - Y)**2, axis=(-1, -2)))

def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes X,Y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    device = X.device
    if weights is None:
        weights = torch.ones(1, len(cutoffs))
    else:
        weights = torch.tensor([weights]).to(device)
    # set zeros and fill with values
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).float().mean(dim=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def gdt_numpy(X, Y, cutoffs, weights=None):
    """ Assumes X,Y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    if weights is None:
        weights = np.ones((1, len(cutoffs)))
    else:
        weights = np.array([weights])
    # set zeros and fill with values
    GDT = np.zeros((X.shape[0], len(cutoffs)))
    dist = np.sqrt(((X - Y)**2).sum(axis=1))
    # iterate over thresholds
    for i, cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).mean(axis=-1)
    # weighted mean
    return (GDT*weights).mean(-1)

def tmscore_torch(X, Y, L):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(21, L)
    d0 = 1.24 * (L - 15)**(1.0/3.0) - 1.8
    # get distance
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # formula (see wrapper for source): 
    return (1.0 / (1.0 + (dist/d0)**2)).mean(dim=-1)

def tmscore_numpy(X, Y, L):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(21, L)
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    # get distance
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # formula (see wrapper for source): 
    return (1 / (1 + (dist/d0)**2)).mean(axis=-1)


def mdscaling_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, 
                    eigen=False, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, eigen=eigen, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_torch(preds, N_mask, CA_mask, C_mask, prop=True)
    to_correct = torch.nonzero( (phi_ratios < 0.5)).view(-1)
    # fix mirrors by (-1)*Z if more (+) than (-) phi angles
    preds[to_correct, -1] = (-1)*preds[to_correct, -1]
    if verbose == 2:
        print("Corrected mirror idxs:", to_correct)
            
    return preds, stresses


def mdscaling_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel 
    preds, stresses = mds_numpy(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    # no need to caculate multiple mirrors - just correct Z axis
    phi_ratios = calc_phis_numpy(preds, N_mask, CA_mask, C_mask, prop=True)
    for i,pred in enumerate(preds):
        # fix mirrors by (-1)*Z if more (+) than (-) phi angles
        if phi_ratios < 0.5:
            preds[i, -1] = (-1)*preds[i, -1]
            if verbose == 2:
                print("Corrected mirror in struct no.", i)

    return preds, stresses


def lddt_ca_torch(true_coords, pred_coords, cloud_mask, r_0=15.):
    """ Computes the lddt score for each C_alpha.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * true_coords: (b, l, c, d) in sidechainnet format.
        * pred_coords: (b, l, c, d) in sidechainnet format.
        * cloud_mask : (b, l, c) adapted for scn format.
        * r_0: float. maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt for c_alpha scores (ranging between 0 and 1)
        See wrapper below.
    """
    device, dtype = true_coords.device, true_coords.type()
    thresholds = torch.tensor([0.5, 1, 2, 4], device=device).type(dtype)
    # adapt masks
    cloud_mask = cloud_mask.bool().cpu()
    c_alpha_mask  = torch.zeros(cloud_mask.shape[1:], device=device).bool() # doesn't have batch dim
    c_alpha_mask[..., 1] = True
    # container for c_alpha scores (between 0,1)
    wrapper = torch.zeros(true_coords.shape[:2], device=device).type(dtype)

    for bi, seq in enumerate(true_coords):
        # select atoms for study
        c_alphas = cloud_mask[bi]*c_alpha_mask # only pick c_alpha positions
        selected_pred = pred_coords[bi, c_alphas, :] 
        selected_target = true_coords[bi, c_alphas, :]
        # get number under distance
        dist_mat_pred = torch.cdist(selected_pred, selected_pred, p=2)
        dist_mat_target = torch.cdist(selected_target, selected_target, p=2) 
        under_r0_target = dist_mat_target < r_0
        compare_dists = torch.abs(dist_mat_pred - dist_mat_target)[under_r0_target]
        # measure diff below threshold
        score = torch.zeros_like(under_r0_target).float()
        max_score = torch.zeros_like(under_r0_target).float()
        max_score[under_r0_target] = 4.
        # measure under how many thresholds
        score[under_r0_target] = thresholds.shape[0] - \
                                 torch.bucketize( compare_dists, boundaries=thresholds ).float()
        # dont include diagonal
        l_mask = c_alphas.float().sum(dim=-1).bool()
        wrapper[bi, l_mask] = ( score.sum(dim=-1) - thresholds.shape[0] ) / \
                              ( max_score.sum(dim=-1) - thresholds.shape[0] )

    return wrapper


################
### WRAPPERS ###
################

@set_backend_kwarg
@invoke_torch_or_numpy(mdscaling_torch, mdscaling_numpy)
def MDScaling(pre_dist_mat, **kwargs):
    """ Gets distance matrix (-ces). Outputs 3d.  
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * N_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * CA_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
    pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
    return pre_dist_mat, kwargs

@expand_arg_dims(dim_len = 2)
@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
    """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs: 
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
    # run calcs - pick the 0th bc an additional dim was created
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(rmsd_torch, rmsd_numpy)
def RMSD(A, B):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs: 
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(gdt_torch, gdt_numpy)
def GDT(A, B, *, mode="TS", cutoffs=None, weights=None):
    """ Returns GDT(Global Distance Test) score as defined here (highre is better):
        Supports both TS and HA
        http://predictioncenter.org/casp12/doc/help.html
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
    # define cutoffs for each type of gdt and weights
    cutoffs = default(cutoffs, [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8])
    # calculate GDT
    return A, B, cutoffs, {'weights': weights}

@expand_arg_dims()
@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B, *, L):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding. 
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now. 
        Inputs: 
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    return A, B, dict(L=L)

def contact_precision_torch(pred, truth, ratios, ranges, mask=None, cutoff=8):
    # (..., l, l)
    assert truth.shape[-1] == truth.shape[-2]
    assert pred.shape == truth.shape

    seq_len = truth.shape[-1]
    mask1s = torch.ones_like(truth, dtype=torch.int8)
    if exists(mask):
        mask1s = mask1s * (mask[...,:,None] * mask[...,None,:])
    mask_ranges = map(
            lambda r: torch.triu(mask1s, default(r[0], 0)) - torch.triu(mask1s, default(r[1], seq_len)),
            ranges)

    pred_truth = torch.stack((pred, truth), dim=-1)
    for (i, j), mask in zip(ranges, mask_ranges):
        masked_pred_truth = pred_truth[mask.bool()]
        sorter_idx = (-masked_pred_truth[:, 0]).argsort()
        sorted_pred_truth = masked_pred_truth[sorter_idx]

        for ratio in ratios:
            num_tops = max(1, int(seq_len * ratio))
            assert 0 < num_tops <= seq_len
            top_labels = sorted_pred_truth[:num_tops, 1]
            num_corrects = ((0 < top_labels) & (top_labels < cutoff)).sum()
            yield (i, j), ratio, num_corrects / float(num_tops)

def contact_precision_numpy(pred, truth, ratios, ranges, mask=None, cutoff=8):
    # (l, l)
    assert truth.shape[-1] == truth.shape[-2]
    assert pred.shape == truth.shape

    seq_len = truth.shape[-1]
    mask1s = np.ones_like(truth, dtype=np.int8)
    if exists(mask):
        mask1s = mask1s * (mask[...:,None] * mask[...,None,:])
    mask_range = map(
            lambda r: np.triu(mask1s, default(r[0], 0)) - np.triu(mask1s, default(r[1], seq_len)),
            ranges)

    pred_truth = np.stack((pred, truth), dim=-1)

    for mask in mask_range:
        masked_pred_truth = pred_truth[mask.nonzero()]
        sorter_idx = (-masked_pred_truth[:, 0]).argsort()
        sorted_pred_truth = masked_pred_truth[sorter_idx]

        for ratio in ratios:
            num_tops = max(1, int(seq_len * ratio))
            assert 0 < num_tops <= seq_len
            top_labels = sorted_pred_truth[:num_tops, 1]
            num_corrects = ((0 < top_labels) & (top_labels < cutoff)).sum()
            yield num_corrects / float(num_tops)

@set_backend_kwarg
@invoke_torch_or_numpy(contact_precision_torch, contact_precision_numpy)
def contact_precision(pred, truth, ratios=None, ranges=None, **kwargs):
    if not exists(ratios):
        ratios = [1, .5, .2, .1]
    if not exists(ranges):
        ranges = [(6, 12), (12, 24), (24, None)]
    return pred, truth, ratios, ranges, kwargs
