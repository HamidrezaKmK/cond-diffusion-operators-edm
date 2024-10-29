from typing import List, Literal

import torch
from torch import nn
from neuralop.layers.gno_block import GNOBlock
from torch.nn import functional as F

from training.networks.utils import PositionalEmbedding, FourierEmbedding

class MPGKScoreOperator(torch.nn.Module):
    """
    A score operator based on multipole graph kernel networks which is designed to mimic UNets in that each
    next layer is a subsample of the previous layer but with more long-range connections when going deeper into
    the U structure. When going up, the connections are more local and skip connections are used to help with
    the vanishing gradient problem.
    """
    def __init__(
        self,
        coord_dim: int,
        latent_channels: int = 128,
        emb_channels: int = 128,
        radius: float = 0.035,
        radii_mult: List[float] = [0.035],
        iter_count: int = 3,
        embedding_type: Literal['positional', 'fourier'] = 'positional',
        group_norm: bool = True,
        num_groups: int = 32,
    ):
        """
        coord_dim: int
            The number of coordinates, for example, a 3D space would have 3 coordinates,
            (time, latitude, longitude, wind vertical component, wind horizontal component) 
            would have 5 coordinates, etc.
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
        embedding_type: Literal['positional', 'fourier']
            This is the positional encoding used for the score operator scheduling input.
        group_norm: bool
            Whether to use group normalization in the GNOBlocks.
        num_groups: int
            The number of groups used in group normalization.
        """
        super().__init__()
        self.radii = [radius]
        for mult in radii_mult:
            self.radii.append(self.radii[-1] * mult)
        
        if embedding_type == 'positional':
            self.schedule_embedding = PositionalEmbedding(num_channels=emb_channels, endpoint=True)
        elif embedding_type == 'fourier':
            self.schedule_embedding = FourierEmbedding(num_channels=emb_channels)
        else:
            raise ValueError("embedding_type must be either 'positional' or 'fourier'")
        
        self.coord_dim = coord_dim
        self.iter_count = iter_count
        self.num_levels = len(self.radii) - 1 # levels are indexed from 0 to num_levels
        self.latent_channels = latent_channels

        self.fc_lifting = nn.Linear(1, latent_channels) # encoder
        self.fc_projection = nn.Linear(latent_channels, 1) # decoder

        use_open3d_neighbor_search = True if coord_dim == 3 else False
        for l in range(self.num_levels + 1):
            # Embedding Network for the Scheduling Variable
            self.register_module(
                f"schedule_embedding_net_{l}",
                nn.Sequential(
                    nn.Linear(emb_channels, latent_channels * 2),
                    nn.SiLU(),
                    nn.Linear(latent_channels * 2, latent_channels * 2),
                    nn.SiLU(),
                )
            )
            # Group Normalization for the GNOBlocks
            self.register_module(
                f"norm_{l}",
                nn.GroupNorm(num_groups=num_groups, num_channels=latent_channels) if group_norm else nn.Identity()
            )
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
        coords: torch.Tensor, # [dim, num_samples]
        samples: torch.Tensor, # [batch_size, num_samples]
        sigma: torch.Tensor, # [batch_size]
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
    ):
        if len(coords.shape) != 2:
            raise ValueError("coords must be of shape (dim, num_samples)")
      
        samples = samples.unsqueeze(2) # [batch_size, num_samples, 1]   
        emb: torch.Tensor = self.schedule_embedding(sigma)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos

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
                current_v_down[l] = F.silu(v_downsampled)
            
            # (2) the upward pass going from num_levels to 0 (course-grained to fine-grained)
            params = self.get_submodule(f"schedule_embedding_net_{self.num_levels}")(emb) # [batch_size, latent_channels * 2]
            shift, scale = torch.chunk(params, 2, dim=1)
            shift = shift.unsqueeze(1).repeat(1, current_v_down[self.num_levels].shape[1], 1)
            scale = scale.unsqueeze(1).repeat(1, current_v_down[self.num_levels].shape[1], 1)
            normed_val = self.get_submodule(f"norm_{self.num_levels}")(current_v_down[self.num_levels].reshape(-1, self.latent_channels)).reshape_as(current_v_down[self.num_levels])
            current_v_up[self.num_levels] = F.silu(
                self.get_submodule(f"W_{self.num_levels}")(current_v_down[self.num_levels]) + \
                self.get_submodule(f"GK_{self.num_levels}_{self.num_levels}")(
                    x=coord_across_depths[self.num_levels],
                    y=coord_across_depths[self.num_levels],
                    f_y=F.silu(torch.addcmul(shift, normed_val, scale + 1)), # add the effect of the scheduling variable
                ),
            )
            for l in range(self.num_levels - 1, -1, -1):
                # upsample
                v_upsampled = self.get_submodule(f"GK_{l+1}_{l}")(
                    y=coord_across_depths[l+1],
                    x=coord_across_depths[l],
                    f_y=current_v_up[l+1],
                )
                params = self.get_submodule(f"schedule_embedding_net_{l}")(emb) # [batch_size, latent_channels * 2]
                shift, scale = torch.chunk(params, 2, dim=1)
                shift = shift.unsqueeze(1).repeat(1, current_v_down[l].shape[1], 1)
                scale = scale.unsqueeze(1).repeat(1, current_v_down[l].shape[1], 1)
                normed_val = self.get_submodule(f"norm_{l}")(current_v_down[l].reshape(-1, self.latent_channels)).reshape_as(current_v_down[l])
                v_upsampled = v_upsampled + (
                    self.get_submodule(f"W_{l}")(current_v_down[l]) + \
                    self.get_submodule(f"GK_{l}_{l}")(
                        x=coord_across_depths[l],
                        y=coord_across_depths[l],
                        f_y=F.silu(torch.addcmul(shift, normed_val, scale + 1)),
                    )
                )
                current_v_up[l] = F.silu(v_upsampled)
        
        return self.fc_projection(current_v_up[0]).squeeze(2) # [batch_size, num_samples]
