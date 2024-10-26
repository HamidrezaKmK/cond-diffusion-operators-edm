from typing import List

import torch
from torch import nn
from neuralop.layers.gno_block import GNOBlock
from torch.nn import functional as F

class MultiPoleScoreNet(torch.nn.Module):
    """
    A score network based on multipole graph kernel networks which is designed to mimic UNets in that each
    next layer is a subsample of the previous layer but with more long-range connections when going deeper into
    the U structure. When going up, the connections are more local and skip connections are used to help with
    the vanishing gradient problem.
    """
    def __init__(
        self,
        coord_dim: int,
        num_channels: int,
        latent_channels: int = 128,
        radius: float = 0.035,
        radii_mult: List[float] = [0.035],
        iter_count: int = 3,
    ):
        """
        coord_dim: int
            The number of coordinates, for example, a 3D space would have 3 coordinates,
            (time, latitude, longitude, wind vertical component, wind horizontal component) 
            would have 5 coordinates, etc.
        num_channels: int
            The number of channels in the input/output space.
        latent_channels: int
            The input intensities will be lifted to this number of channels.
        radius: float
            The radius used for the base (1st layer) GNOBlock.
        radii_mult: List[float]
            The radii tend to increase as the depth increases in a v-cycle. This list is the cumulative multiplier
            for the radii. For example, if the radii_mult is [1, 2, 3], and radius is 0.1 then the actual
            radii would be [0.1, 0.2, 0.6]. The length of this would also determine the depth of the network.
        iter_count: int
            The number of times to run the feedback loop 
            (the capital T in the original multipole paper: https://arxiv.org/pdf/2006.09535)
        """
        super().__init__()
        self.radii = [radius]
        for mult in radii_mult:
            self.radii.append(self.radii[-1] * mult)
        
        self.coord_dim = coord_dim
        self.iter_count = iter_count
        self.num_levels = len(self.radii) - 1 # levels are indexed from 0 to num_levels
        self.num_channels = num_channels

        self.fc_lifting = nn.Linear(num_channels, latent_channels) # encoder
        self.fc_projection = nn.Linear(latent_channels, num_channels) # decoder

        use_open3d_neighbor_search = True if coord_dim == 3 else False
        for l in range(self.num_levels + 1):
            # Skip Kernels
            self.register_module(
                f"GK_{l}_{l}",
                GNOBlock(
                    in_channels=latent_channels,
                    out_channels=latent_channels,
                    coord_dim=coord_dim,
                    radius=self.radii[l],
                    use_open3d_neighbor_search=use_open3d_neighbor_search
                )
            )
            # Direct Affine Operators
            self.register_module(
                f"W_{l}",
                nn.Linear(latent_channels, latent_channels)
            )
            if l < self.num_levels:
                # Downsampling Kernels
                self.register_module(
                    f"GK_{l}_{l+1}",
                    GNOBlock(
                        in_channels=latent_channels,
                        out_channels=latent_channels,
                        coord_dim=coord_dim,
                        radius=max(self.radii[l], self.radii[l + 1]),
                        use_open3d_neighbor_search=use_open3d_neighbor_search,
                    )
                )
                # Upsampling Kernels
                self.register_module(
                    f"GK_{l+1}_{l}",
                    GNOBlock(
                        in_channels=latent_channels,
                        out_channels=latent_channels,
                        coord_dim=coord_dim,
                        radius=max(self.radii[l], self.radii[l + 1]), # TODO: not sure about the radius here
                        use_open3d_neighbor_search=use_open3d_neighbor_search,
                    )
                )
    
    def forward(
        self, 
        coords: torch.Tensor,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
    ):
        if len(coords.shape) != 2:
            raise ValueError("coords must be of shape (dim, num_samples)")
        
        # subsample the coordinates across the depth for the v-cycle, each level will have half as many coordinates
        coord_across_depths = [coords.permute(1, 0)] # (num_samples, dim)
        for l in range(1, self.num_levels + 1):
            coord_across_depths.append(coords[:, ::(1 << l)].permute(1, 0)) # TODO: is this a GPU access nightmare?

        # uplift everything to a latent space that will be projected down at the end
        uplifted = self.fc_lifting(samples) # (batch_size, num_samples, latent_channels)

        # coords_across_depths[d] will indicate the coordinates at depth d
        current_v_up = [uplifted] + [None] * self.num_levels
        current_v_down = [uplifted] + [None] * self.num_levels
        for _ in range(self.iter_count):

            # (1) the downward pass going from 0 to num_levels (fine-grained to course-grained)
            for l in range(1, self.num_levels + 1):
                # downsample
                v_downsampled = self.get_submodule(f"GK_{l-1}_{l}")(
                    y=coord_across_depths[l-1],
                    x=coord_across_depths[l],
                    f_y=current_v_down[l-1],
                )
                # direct sum if last iteration is available
                if current_v_up[l] is not None:
                    v_downsampled = v_downsampled + current_v_up[l]
                current_v_down[l] = F.relu(v_downsampled)
            
            # (2) the upward pass going from num_levels to 0 (course-grained to fine-grained)
            current_v_up[self.num_levels] = F.relu(
                self.get_submodule(f"W_{self.num_levels}")(current_v_down[self.num_levels]) + \
                self.get_submodule(f"GK_{self.num_levels}_{self.num_levels}")(
                    x=coord_across_depths[self.num_levels],
                    y=coord_across_depths[self.num_levels],
                    f_y=current_v_down[self.num_levels]),
            )
            for l in range(self.num_levels - 1, -1, -1):
                # upsample
                v_upsampled = self.get_submodule(f"GK_{l+1}_{l}")(
                    y=coord_across_depths[l+1],
                    x=coord_across_depths[l],
                    f_y=current_v_up[l+1],
                )
                v_upsampled = v_upsampled + (
                    self.get_submodule(f"W_{l}")(current_v_down[l]) + \
                    self.get_submodule(f"GK_{l}_{l}")(
                        x=coord_across_depths[l],
                        y=coord_across_depths[l],
                        f_y=current_v_down[l],
                    )
                )
                current_v_up[l] = F.relu(v_upsampled)
        
        return self.fc_projection(current_v_up[0])
