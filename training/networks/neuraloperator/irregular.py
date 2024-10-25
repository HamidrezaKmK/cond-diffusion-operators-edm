import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
from torch import nn
from neuralop.layers.spectral_convolution import SpectralConv
import dnnlib
from neuralop.models import GINO

class GINOScoreNet(torch.nn.Module):
    """
    A score network used for the diffusion denoising operator which has the backbone of GINO.
    """
    def __init__(
        self,
        num_coords,
        dropout:int=0,
    ):
        super().__init__()
        self.num_coords = num_coords
        self._gino = GINO(
            in_channels=num_coords,
            out_channels=num_coords,
            fno_channel_mlp_dropout=dropout,
        )
    
    def forward(self, x, class_labels=None):
        return self._gino(x, class_labels)
    
@persistence.persistent_class
class EDMPreconditioner(torch.nn.Module):
    def __init__(self,
        score_net: nn.Module,                   # The score network to use.
        rank                = 1,                # Rank of the EDM preconditioner.
        fmult               = 1,                # Factor to multiply the EDM preconditioner by.
        dropout             = 0,                # Dropout probability.
        num_blocks          = 3,                # Number of blocks in the model.
        use_fp16            = False,            # Execute the underlying model at FP16 precision?
        sigma_min           = 0,                # Minimum supported noise level.
        sigma_max           = float('inf'),     # Maximum supported noise level.
        sigma_data          = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.model = score_net
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.num_blocks = num_blocks

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        pass

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
