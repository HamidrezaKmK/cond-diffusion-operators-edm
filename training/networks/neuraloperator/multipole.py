import torch
from torch import nn
from neuralop.layers.gno_block import GNOBlock

class MultiPoleScoreNet(torch.nn.Module):
    """
    A score network based on multipole graph kernel networks which is designed to mimic UNets.
    """
    def __init__(
        self,
        num_coords,
        radius: float = 0.035, 
    ):
        super().__init__()
        self.num_coords = num_coords
        self.block0 = GNOBlock(
            in_channels=1, out_channels=1, coord_dim=num_coords, radius=radius
        )
    
    def forward(
        self, 
        coords: torch.Tensor,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
    ):

        ret = self.block0(
            y=coords.permute(1, 0),
            x=coords.permute(1, 0),
            f_y=samples.unsqueeze(-1),
        )

        return ret.squeeze(-1)
