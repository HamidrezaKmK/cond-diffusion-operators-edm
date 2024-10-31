"""
The set of all diffusion samplers.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch
from tqdm import tqdm

from training.noising_kernels import NoisingKernel

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
            # print(coords.shape, x_hat.shape, t_hat.repeat(x_hat.shape[0]).shape)
            # print("---")
            denoised = net(coords=coords, samples=x_hat, sigma=t_hat.repeat(x_hat.shape[0]), conditioning=conditioning).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(coords=coords, samples=x_next, sigma=t_hat.repeat(x_hat.shape[0]), conditioning=conditioning).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


class ImputationEDMSampler(ABC):
    """
    A sampler that also performs imputation.
    """
    def __init__(
        self,
        noising_kernel: NoisingKernel,
        verbose: bool = False,
    ):
        self.noising_kernel = noising_kernel
        self.verbose = verbose
    
    def __call__(
        self,
        net, # the model
        coords, # [dim, N1]
        coord_ref, # [dim, N2]
        values_ref, # [N2]
        conditioning=None,
        num_steps=18, 
        sigma_min=0.002, 
        sigma_max=80, 
        rho=7.,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, getattr(net, 'sigma_min', 0))
        sigma_max = min(sigma_max, getattr(net, 'sigma_max', float('inf')))

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=coords.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        all_coords = torch.cat([coords, coord_ref], dim=1) # [dim, N1 + N2]
        x_next = self.noising_kernel.sample(all_coords.unsqueeze(0)).float().squeeze() * t_steps[0] # [N1 + N2]
        if self.verbose:
            iterable_range = tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), desc="Sampling with imputation")
        else:
            iterable_range = enumerate(zip(t_steps[:-1], t_steps[1:]))
        for i, (t_cur, t_next) in iterable_range: # 0, ..., N-1
            x_cur = x_next

            x_hat = x_cur
            t_hat = t_cur

            # Euler step.
            noised_out_reference = t_cur * self.noising_kernel.sample(coord_ref.unsqueeze(0)).float().squeeze() + values_ref
            x_hat = torch.cat([x_hat[:coords.shape[1]], noised_out_reference])
            denoised = net(coords=all_coords, samples=x_hat.unsqueeze(0), sigma=t_hat.unsqueeze(0), conditioning=conditioning).to(torch.float32).squeeze()
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                x_next = torch.cat([x_next[:coords.shape[1]], noised_out_reference])
                denoised = net(coords=all_coords, samples=x_next.unsqueeze(0), sigma=t_hat.unsqueeze(0), conditioning=conditioning).to(torch.float32).squeeze()
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next#[:coords.shape[1]]