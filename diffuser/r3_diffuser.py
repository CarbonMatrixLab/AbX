"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch
import pdb
import logging
logger = logging.getLogger(__name__)


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf['min_b']
        self.max_b = r3_conf['max_b']

    def _scale(self, x):
        return x * torch.tensor(self._r3_conf['coordinate_scaling'],device=x.device)

    def _unscale(self, x):
        return x / torch.tensor(self._r3_conf['coordinate_scaling'], device=x.device)

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return torch.tensor(self.min_b, device=t.device) + t*torch.tensor((self.max_b - self.min_b),device=t.device)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))[:, None, None]

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t)[:, None, None] * x

    def sample_ref(self, n_samples: np.array, device='cpu'):
        return torch.randn(size=(*n_samples,3),device=device)

    def marginal_b_t(self, t):
        return t*torch.tensor(self.min_b, device=t.device) + (1/2)*(t**2)*(torch.tensor(self.max_b-self.min_b, device=t.device))

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: torch.tensor, t: torch.tensor, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t, device=x_t_1.devcie)
        z_t_1 = torch.randn(size=x_t_1.shape, device=x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * torch.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: torch.tensor, t: torch.tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_0 = self._scale(x_0)

        log_mean_coeff = -0.5 * self.marginal_b_t(t)
        cast_shape = [log_mean_coeff.shape[0]] + [1] * (len(x_0.shape) - 1)
        log_mean_coeff = log_mean_coeff.view(*cast_shape)

        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        x_t = torch.normal(mean=mean, std=std).to(device=x_0.device)
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.tensor):
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: torch.tensor,
            score_t: torch.tensor,
            t: torch.tensor,
            dt: torch.tensor,
            mask: torch.tensor=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape, device=score_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * dt * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones(x_t.shape[:-1], device=x_t.device)
        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1, keepdims=True)
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, scale=False):
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)

        t = t[:, None, None]
        return -(x_t - torch.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
