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
        self.dummy = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        coords: torch.Tensor,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
    ):
        return samples * self.dummy
    
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

    def forward(
        self, 
        coords: torch.Tensor,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
    ):
        # TODO: fix this
        return self.model(
            coords=coords,
            samples=samples,
            sigma=sigma,
            conditioning=conditioning,
            conditioning_augmented=conditioning_augmented,
        )
    
        # x = x.to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        # # TODO
        # class_labels = None if self.label_dim == 0 else \
        #     torch.zeros([1, self.label_dim], device=x.device) if class_labels is None \
        #     else class_labels.to(torch.float32) #.reshape(-1, self.label_dim)
        
        # dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        # c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        # c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        # c_noise = sigma.log() / 4

        # F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        # assert F_x.dtype == dtype
        # D_x = c_skip * x + c_out * F_x.to(torch.float32)
        # return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
