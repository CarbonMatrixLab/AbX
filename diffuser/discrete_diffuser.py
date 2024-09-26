import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abx.common import residue_constants
import math
import pdb

class DiscreteDiffuser:
    def __init__(self, discrete_conf):
        self.discrete_conf = discrete_conf
        self.residue_num = residue_constants.restype_num 
        self.rate_const = discrete_conf.rate_const

        rate = self.rate_const * torch.ones((self.residue_num, self.residue_num))
        rate = rate - torch.diag(torch.diag(rate))
        row_sum = torch.diag(torch.sum(rate, dim=1))
        rate = rate - row_sum
        # rate = torch.zeros((self.residue_num, self.residue_num))
        # rate[:self.residue_num-1, :self.residue_num-1] = rate_    

        eigvals, eigvecs = torch.linalg.eigh(rate)

        self.rate_matrix = rate.float()
        self.eigvals = eigvals.float()
        self.eigvecs = eigvecs.float()

    @staticmethod
    def get(se3_conf):
        global diffuser_obj_dict

        name = 'diffuser'

        if name not in diffuser_obj_dict:
            diffuser_obj_dict[name] = DiscreteDiffuser(se3_conf)

        return diffuser_obj_dict[name]

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed
    
    def _integral_rate_scalar(self, t):
        return self.time_base * (self.time_exponential ** t) - self.time_base
    
    def _rate_scalar(self, t):
        return self.time_base * math.log(self.time_exponential) * (self.time_exponential ** t)

    def rate(self, t):
        batch_size = t.shape[0]
        
        return torch.tile(self.rate_matrix.view(1,self.residue_num,self.residue_num).to(t.device), (batch_size, 1, 1))
    
    def transition(self, t):
        batch_size = t.shape[0]
        eigvals = self.eigvals.to(device=t.device)
        eigvecs = self.eigvecs.to(device=t.device)
        t = t.float()
        transitions = eigvecs.reshape(1, self.residue_num, self.residue_num) @ \
            torch.diag_embed(torch.exp(eigvals.reshape(1, self.residue_num) * t.reshape(batch_size, 1))) @\
            eigvecs.T.reshape(1, self.residue_num, self.residue_num)

        if torch.min(transitions) < -1e-6:
            print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        transitions[transitions < 1e-8] = 0.0

        return transitions

    def sample_ref(self, n_samples: np.array, device='cpu'):
        return torch.randint(low=0, high=self.residue_num, size=(n_samples[0], n_samples[1]),device=device)

    def forward_marginal(self, x_0: torch.tensor, t: torch.tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_tilde: [..., n] positions at time t in Angstroms.
        """
        batch_size = t.shape[0]
        length = x_0.shape[1]
        qt0 = self.transition(t).to(device=t.device) # (B, S, S)
        rate = self.rate(t).to(device=t.device) # (B, S, S)
        x_0  = torch.clamp(x_0, min=0, max=self.residue_num-1)
        
        # --------------- Sampling x_t, x_tilde --------------------
        qt0_rows_reg = qt0[
            torch.arange(batch_size, device=t.device).repeat_interleave(length),
            x_0.long().flatten(),
            :
        ] # (B*D, S)
        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(batch_size, length)
        rate_vals_square = rate[
            torch.arange(batch_size, device=t.device).repeat_interleave(length),
            x_t.long().flatten(),
            :
        ] # (B*D, S)
        rate_vals_square[
            torch.arange(batch_size*length, device=t.device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.reshape(batch_size, length, self.residue_num)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).reshape(batch_size, length)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
            torch.arange(batch_size, device=t.device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(batch_size, device=t.device),
            square_dims
        ] = square_newval_samples
        # x_tilde = x_tilde.long()
        # x_tilde (B, D)
        return x_tilde, qt0, rate


    def reverse(
            self,
            *,
            x_t: torch.tensor,
            logits_t: torch.tensor,
            t: torch.tensor,
            dt: torch.tensor,
            eps_ratio: float=1e-9,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t:  current positions at time t in angstroms.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        n_samples = x_t.shape[:2]
        x_t  = torch.clamp(x_t, min=0, max=self.residue_num-1)

        p0t = F.softmax(logits_t, dim=2) # (N, D, S)
        qt0 = self.transition(t * torch.ones((n_samples[0],), device=t.device)) # (N, S, S)
        rate = self.rate(t * torch.ones((n_samples[0],), device=t.device)) # (N, S, S)
        qt0_denom = qt0[
            torch.arange(n_samples[0], device=t.device).repeat_interleave(n_samples[1] * self.residue_num),
            torch.arange(self.residue_num, device=t.device).repeat(n_samples[0]*n_samples[1]),
            x_t.long().flatten().repeat_interleave(self.residue_num)
        ].view(*n_samples,self.residue_num) + torch.tensor(eps_ratio, device=t.device) # (N, D, S)

        # First S is x0 second S is x tilde

        qt0_numer = qt0 # (N, S, S)

        forward_rates = rate[
            torch.arange(n_samples[0], device=t.device).repeat_interleave(n_samples[1]*self.residue_num),
            torch.arange(self.residue_num, device=t.device).repeat(n_samples[0]*n_samples[1]),
            x_t.long().flatten().repeat_interleave(self.residue_num)
        ].view(*n_samples, self.residue_num).to(device=t.device)

        inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)
        reverse_rates = forward_rates * inner_sum # (N, D, S)

        reverse_rates[
            torch.arange(n_samples[0], device=t.device).repeat_interleave(n_samples[1]),
            torch.arange(n_samples[1], device=t.device).repeat(n_samples[0]),
            x_t.long().flatten()
        ] = 0.0

        diffs = torch.arange(self.residue_num, device=t.device).view(1,1,self.residue_num) - x_t.view(*n_samples,1)
        poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * dt)
        jump_nums = poisson_dist.sample()
        adj_diffs = jump_nums * diffs
        overall_jump = torch.sum(adj_diffs, dim=2)
        xp = x_t + overall_jump
        x_new = torch.clamp(xp, min=0, max=self.residue_num-1).to(dtype=torch.int32)
        # x_new = xp

        return x_new





