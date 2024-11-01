"""
This is a low frequency 1D dataset that is used to test the diffusion-based neural operator.
"""

from typing import Literal

import torch

from torch.utils.data import Dataset
import torch

class PinkNoise1D(Dataset):
    """
    Generate 1D pink noise using frequency-domain filtering.
    
    Args:
        total_count (int): Number of samples that can be generated from the dataset
    
    Returns:
        Tensor: A 1D tensor containing the pink noise signal.
    """
    def __init__(
        self, 
        total_count: int,
        num_samples: int,
        low_pass_threshold: float | None = 0.1,
        threshold_type: Literal["relative", "absolute"] = "relative",
        make_irregular: bool = False,
        fixed_irregularity_seed: int | None = None,
        bernoulli_p: float = 0.8,
    ):
        self.total_count = total_count
        self.num_samples = num_samples
        self.low_pass_threshold = low_pass_threshold
        self.make_irregular = make_irregular
        self.fixed_irregularity_seed = fixed_irregularity_seed
        self.bernoulli_p = bernoulli_p
        self.threshold_type = threshold_type
    
    def __len__(self):
        return self.total_count
    
    def __getitem__(self, idx):
        
        if idx >= self.total_count:
            raise IndexError("Index out of bounds")
        
        # generate white noise and rescale the power spectrum
        white_noise = torch.randn(self.num_samples)
        freqs = torch.fft.rfft(white_noise)
        freqs_idx = torch.arange(1, freqs.shape[-1] + 1, device=freqs.device)
        rescale_freqs = freqs / torch.sqrt(freqs_idx.float())
        if self.low_pass_threshold is not None:
            if self.threshold_type == "relative":
                rescale_freqs[freqs_idx > self.low_pass_threshold * self.num_samples] = 0
            elif self.threshold_type == "absolute":
                rescale_freqs[freqs_idx > self.low_pass_threshold] = 0
            else:
                raise ValueError("Invalid threshold type")
        # Perform inverse FFT to convert the signal back to the time domain
        pink_noise = torch.fft.irfft(rescale_freqs, n=self.num_samples)
        pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std()
        
        coords = torch.linspace(0, 1, self.num_samples).unsqueeze(0) # [dim x n]

        if self.make_irregular:
            if self.fixed_irregularity_seed is not None:
                rng = torch.Generator().manual_seed(self.fixed_irregularity_seed)
            else:
                rng = torch.Generator().manual_seed(idx)
            mask = torch.bernoulli(self.bernoulli_p * torch.ones(self.num_samples), generator=rng).bool()
            coords = coords[:, mask]
            pink_noise = pink_noise[mask]
        
        return coords, pink_noise 
