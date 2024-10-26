import torch
from torch import nn
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
