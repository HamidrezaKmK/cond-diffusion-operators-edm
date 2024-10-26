"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from abc import ABC, abstractmethod

from training.augment_pipe import AugmentPipe
from training.noising_kernels import NoisingKernel

class ScoreMatchingLoss(ABC):
    """Abstract base class for loss functions."""
    @abstractmethod
    def __call__(
        self, 
        net: torch.nn.Module, 
        coords: torch.Tensor, 
        samples: torch.Tensor, 
        conditioning: torch.Tensor | None = None, 
        augment_pipe: AugmentPipe | None = None,
    ):
        raise NotImplementedError   
    
#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss(ScoreMatchingLoss):
    def __init__(
        self, 
        noising_kernel: NoisingKernel,
        P_mean: float = -1.2, 
        P_std: float = 1.2, 
        sigma_data: float = 0.5,
    ):
        self.noising_kernel = noising_kernel
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self, 
        net: torch.nn.Module, 
        coords: torch.Tensor, 
        samples: torch.Tensor, 
        conditioning: torch.Tensor | None = None, 
        augment_pipe: AugmentPipe | None = None,
    ):
        """
        Compute the denoising score matching loss.

        Args:
            net: The denoising diffusion model.
            coords: The coordinates of the samples.
            samples: The value of the samples.
            conditioning: 
                Conditioning samples; this is for example used for imputation and should either be 
                non-existent or have the same size and shape as the original samples.
            augment_pipe:
                The augmentation pipeline used to augment the samples and conditioning.
        Returns:
            The loss value.
        """
        # get the noise scheduler and the loss weight   
        batch_size = samples.shape[0]  
        rnd_normal = torch.randn(batch_size).to(dtype=samples.dtype).to(device=samples.device) # (related to timestep)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        loss_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # We want to augment pipe both input and conditioning at once.
        
        x_dim = samples.shape[1]
        if conditioning:
            _samples = torch.cat((samples, conditioning), dim=1)
            # repeat coordinates along the second axis
            _coords = coords.repeat(1, 2)
            _augmented_samples = augment_pipe(_coords, _samples)
            samples_augmented = _augmented_samples[:, :x_dim]
            conditioning_augmented = _augmented_samples[:, x_dim:]
        else:
            conditioning_augmented = None
            samples_augmented = augment_pipe(coords, samples)
        
        # get correlated Gaussian noise from the Gaussian random field and scale using the scheduler
        # multuply 
        if len(coords.shape) == 3:
            n = sigma[:, None] * self.noising_kernel.sample(coords)
        else:
            n = sigma[:, None] * self.noising_kernel.sample(coords.unsqueeze(0).repeat(batch_size, 1, 1))
        denoised_samples = net(
            coords=coords,
            samples=samples_augmented + n,
            sigma=sigma,
            conditioning=conditioning,
            conditioning_augmented=conditioning_augmented,
        )
        return loss_weight[:, None] * (denoised_samples - samples_augmented) ** 2
    
