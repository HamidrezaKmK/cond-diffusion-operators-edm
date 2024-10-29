import torch
from torch_utils import persistence
from torch import nn


@persistence.persistent_class
class EDMPreconditioner(torch.nn.Module):
    def __init__(self,
        score_net: nn.Module,                   # The score operator to use.
        sigma_min           = 0,                # Minimum supported noise level.
        sigma_max           = float('inf'),     # Maximum supported noise level.
        sigma_data          = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.model = score_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(
        self, 
        coords: torch.Tensor,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        conditioning_augmented: torch.Tensor | None = None,
        **model_kwargs,
    ):
        # TODO: make sure this is correct! 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x: torch.Tensor = self.model(
            coords=coords,
            samples=(c_in[:, None] * samples).float(),
            sigma=c_noise,
            conditioning=conditioning,
            conditioning_augmented=conditioning_augmented,
            **model_kwargs,
        )
        D_x = c_skip[:, None] * samples + c_out[:, None] * F_x.float()
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

# @persistence.persistent_class
# class VEPrecond(torch.nn.Module):
#     def __init__(
#         self,
#         score_net: nn.Module,           # The score operator to use.     
#         label_dim       = 0,            # Number of class labels, 0 = unconditional.
#         use_fp16        = False,        # Execute the underlying model at FP16 precision?
#         sigma_min       = 0.0002,         # Minimum supported noise level.
#         sigma_max       = 80,          # Maximum supported noise level.
#         model_type      = 'SongUNet',   # Class name of the underlying model.
#         **model_kwargs,                 # Keyword arguments for the underlying model.
#     ):
#         super().__init__()
#         self.label_dim = label_dim
#         self.use_fp16 = use_fp16
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

#     def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
#         x = x.to(torch.float32)
#         sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
#         class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
#         dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

#         c_skip = 1
#         c_out = sigma
#         c_in = 1
#         c_noise = (0.5 * sigma).log()

#         F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
#         assert F_x.dtype == dtype
#         D_x = c_skip * x + c_out * F_x.to(torch.float32)
#         return D_x

#     def round_sigma(self, sigma):
#         return torch.as_tensor(sigma)
