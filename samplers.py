"""
The set of all diffusion samplers.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch
from tqdm import tqdm

from training.noising_kernels import NoisingKernel

from contextlib import contextmanager
import torch

@contextmanager
def freeze_model(net: torch.nn.Module):
    # Store the original requires_grad state for all parameters
    original_states = {param: param.requires_grad for param in net.parameters()}
    
    try:
        # Set requires_grad to False to freeze the model
        for param in net.parameters():
            param.requires_grad = False
        yield net  # Yield control back to the caller with the model frozen
    finally:
        # Restore original requires_grad states after the block of code finishes
        for param, requires_grad in original_states.items():
            param.requires_grad = requires_grad


class BaseSampler(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError("The method __call__ must be implemented.")
    

class EDMSampler(BaseSampler):
    
    def __init__(
        self,
        verbose: bool = False,
    ):
        self.verbose = verbose 
    
    def __call__(
        self,
        net, 
        coords, 
        latents, 
        conditioning=None,
        num_steps=18, 
        sigma_min=0.002, 
        sigma_max=80, 
        rho=7,
    ):
        with freeze_model(net):
            # Adjust noise levels based on what's supported by the network.
            
            sigma_min = max(sigma_min, getattr(net, 'sigma_min', 0))
            sigma_max = min(sigma_max, getattr(net, 'sigma_max', float('inf')))

            # Time step discretization.
            step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

            # Main sampling loop.
            x_next = latents.to(torch.float32)# * t_steps[0] TODO: check why this is added here!
            if self.verbose:
                iterable_range = tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), desc="Sampling")
            else:
                iterable_range = enumerate(zip(t_steps[:-1], t_steps[1:]))
            for i, (t_cur, t_next) in iterable_range: # 0, ..., N-1
                x_cur = x_next

                x_hat = x_cur
                t_hat = t_cur

                # Euler step.
                denoised = net(coords=coords, samples=x_hat, sigma=t_hat.repeat(x_hat.shape[0]), conditioning=conditioning).to(torch.float32)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    denoised = net(coords=coords, samples=x_next, sigma=t_hat.repeat(x_hat.shape[0]), conditioning=conditioning).to(torch.float32)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


class RepaintImputer(ABC):
    """
    A sampler equippied with imputation that uses the Tweedie estimates.

    This is based on: https://arxiv.org/pdf/2411.00359
    """
    def __init__(
        self,
        noising_kernel: NoisingKernel,
        verbose: bool = False,
        l: float = 0.1,
        r: float = 1.,
        conditional_opt_steps: int = 1,
        conditional_lr: float = 0.1,
    ):
        self.noising_kernel = noising_kernel
        self.verbose = verbose

        # the kernel parameters
        self.l = l
        self.r = r
        self.conditional_lr = conditional_lr
        self.conditional_opt_steps = conditional_opt_steps

    def __call__(
        self,
        net, # the model
        batch_size, # the batch size
        coords, # [dim, N1]
        coord_ref, # [dim, N2]
        values_ref, # [batch_size, N2]
        conditioning=None,
        num_steps=18, 
        sigma_min=0.002, 
        sigma_max=80, 
        rho=7.,
    ):
        with freeze_model(net):
            
            # Adjust noise levels based on what's supported by the network.
            sigma_min = max(sigma_min, getattr(net, 'sigma_min', 0))
            sigma_max = min(sigma_max, getattr(net, 'sigma_max', float('inf')))

            # Time step discretization.
            step_indices = torch.arange(num_steps, dtype=torch.float32, device=coords.device)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

            # Main sampling loop.
            all_coords = torch.cat([coords, coord_ref], dim=1) # [dim, N1 + N2]




            # print(torch.max(torch.abs(torch.linalg.eigvalsh(conditional_cov))), torch.min(torch.abs(torch.linalg.eigvalsh(conditional_cov))))
            # print("<<>>")
            # print(x0_hat)

            x_next = self.noising_kernel.sample(all_coords.repeat(batch_size, 1, 1)).float()* t_steps[0] # [N1 + N2]
            if self.verbose:
                iterable_range = tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), desc="Sampling with imputation")
            else:
                iterable_range = enumerate(zip(t_steps[:-1], t_steps[1:]))
            for i, (t_cur, t_next) in iterable_range: # 0, ..., N-1
                x_cur = x_next

                x_hat = x_cur
                t_hat = t_cur

                denoised = net(coords=all_coords, samples=x_hat, sigma=t_hat.repeat(batch_size), conditioning=conditioning).to(torch.float32)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    denoised = net(coords=all_coords, samples=x_next, sigma=t_hat.repeat(batch_size), conditioning=conditioning).to(torch.float32)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                
                # Apply conditional update
                if i < num_steps - 1:
                    # Precompute the conditional covariance for the GP regression
                    C = t_hat ** 2 * torch.exp(-torch.cdist(all_coords.permute(1, 0), all_coords.permute(1, 0), p=2) / (2 * self.l**2))
                    C_nr = C[:, coords.shape[1]:]
                    C_r = C[coords.shape[1]:, coords.shape[1]:]
                    C_r_inv = torch.linalg.pinv(C_r)
                    conditional_cov = C - C_nr @ C_r_inv @ C_nr.T
                    x0_hat = torch.einsum("ij,bj->bi", C_nr @ C_r_inv, values_ref)
                    conditional_cov_inv = torch.linalg.pinv(conditional_cov)

                    for ii in range(self.conditional_opt_steps):
                        
                        if self.verbose:
                            iterable_range.set_description(f"Optimizing conditional estimate [{ii+1}/{self.conditional_opt_steps}]")
                        x_next.requires_grad = True

                        log_density = -0.5 * torch.einsum("bi,ij,bj->b", (x_next - x0_hat, conditional_cov_inv, x_next - x0_hat))
                        # print(log_density)
                        # print(x0_hat, x_next)
                        # print("**")
                        # check if log_density is nan
                        assert not torch.isnan(log_density).any(), "Log density contains NaNs!"

                        log_density.sum().backward()
                        x_next_grad = x_next.grad.clone()
                        x_next_grad = x_next_grad
                        x_next.requires_grad = False
                        # print(torch.max(x_next_grad), torch.min(x_next_grad))
                        x_next = x_next + self.conditional_lr * x_next_grad

        return x_next  

class TweedieImputer(ABC):
    """
    A sampler equippied with imputation that uses the Tweedie estimates.

    This is based on: https://arxiv.org/pdf/2411.00359
    """
    def __init__(
        self,
        noising_kernel: NoisingKernel,
        verbose: bool = False,
        lambd: float = 1.0,
        l: float = 0.05,
        r: float = 0.1,
    ):
        self.noising_kernel = noising_kernel
        self.verbose = verbose
        self.lambd = lambd

        # the kernel parameters
        self.l = l
        self.r = r

    def __call__(
        self,
        net, # the model
        batch_size, # the batch size
        coords, # [dim, N1]
        coord_ref, # [dim, N2]
        values_ref, # [N2]
        conditioning=None,
        num_steps=18, 
        sigma_min=0.002, 
        sigma_max=80, 
        rho=7.,
    ):
        with freeze_model(net):
            
            # Adjust noise levels based on what's supported by the network.
            sigma_min = max(sigma_min, getattr(net, 'sigma_min', 0))
            sigma_max = min(sigma_max, getattr(net, 'sigma_max', float('inf')))

            # Time step discretization.
            step_indices = torch.arange(num_steps, dtype=torch.float32, device=coords.device)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

            # Main sampling loop.
            all_coords = torch.cat([coords, coord_ref], dim=1) # [dim, N1 + N2]


            C = torch.exp(-(torch.cdist(all_coords.permute(1, 0), all_coords.permute(1, 0), p=2))/ (2 * self.l**2))
            C_nr = C[:, coords.shape[1]:]
            C_r = C[coords.shape[1]:, coords.shape[1]:]
            C_inv = torch.linalg.inv(C + self.r**2 * torch.eye(C.shape[0]).to(C.device))
            conditional_chunk = C_r - C_nr.T @ C_inv @ C_nr
            conditional_mu_mat = C_nr.T @ C_inv
            conditional_cov_inv = torch.linalg.inv(conditional_chunk)


            x_next = self.noising_kernel.sample(all_coords.repeat(batch_size, 1, 1)).float()* t_steps[0] # [N1 + N2]
            if self.verbose:
                iterable_range = tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), desc="Sampling with imputation")
            else:
                iterable_range = enumerate(zip(t_steps[:-1], t_steps[1:]))
            for i, (t_cur, t_next) in iterable_range: # 0, ..., N-1
                
                x_hat = x_next
                t_hat = t_cur

                
                # Apply conditional optimization
                # -----------------------------
                x_hat.requires_grad = True
                x0_hat = x_hat
                
                # TODO: check if the following is correct
                for t_cur1, t_next1 in zip(t_steps[i:-1], t_steps[i+1:]):
                    
                    denoised = net(coords=all_coords, samples=x0_hat, sigma=t_cur1.repeat(batch_size), conditioning=conditioning).to(torch.float32)
                    d_cur = (x0_hat - denoised) / t_cur1
                    x0_hat = x0_hat + (t_next1 - t_cur1) * d_cur
        
                obs_mu = torch.einsum("ij,bj->bi", conditional_mu_mat, x0_hat)
                log_density = -0.5 * torch.einsum("bi,ij,bj->b", (values_ref - obs_mu, conditional_cov_inv, values_ref - obs_mu))
                
                # check if log_density is nan
                assert not torch.isnan(log_density).any(), "Log density contains NaNs!"

                log_density.sum().backward()
                x_hat_grad = x_hat.grad.clone()
                x_hat_grad = x_hat_grad
                x_hat.requires_grad = False
                # guide the generation using this gradient
                x_hat = x_hat + self.lambd * x_hat_grad / coord_ref.shape[1] # adjusting the gradient based on the number of reference points
                
                # apply the actual update
                # -----------------------------
                denoised = net(coords=all_coords, samples=x_hat, sigma=t_hat.repeat(batch_size), conditioning=conditioning).to(torch.float32)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction
                # -----------------------------
                if i < num_steps - 1:
                    denoised = net(coords=all_coords, samples=x_next, sigma=t_hat.repeat(batch_size), conditioning=conditioning).to(torch.float32)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                
        return x_next
