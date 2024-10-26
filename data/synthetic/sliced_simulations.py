import os
import inspect

from typing import Callable, Tuple
from pathlib import Path
import functools

import dotenv
import torch
import numpy as np
from noise import pnoise1, pnoise2, pnoise3

from torch.utils.data import Dataset

def perlin_mask_sampler(
    shape: Tuple[int],
    scale: float = 10.0,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    cutoff: float = -0.5, # after generating the perlin noise, mask according to a cutoff value
    seed: int = 0,
):
    if not (1 <= len(shape) <= 3): 
        raise ValueError("Only 1D, 2D, and 3D masks are supported.")
    elif len(shape) == 1:
        noise_sampler = functools.partial(pnoise1, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed)
    elif len(shape) == 2:
        noise_sampler = functools.partial(pnoise2, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed)
    elif len(shape) == 3:
        noise_sampler = functools.partial(pnoise3, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed)

    mesh = np.meshgrid(*tuple([np.linspace(0, s / scale, s) for s in shape]), indexing='ij')
    noise = np.vectorize(noise_sampler)(*mesh)

    return noise > cutoff # mask according to the cutoff value

class SpatioTemporalSlice(Dataset):
    
    """Temporal and spatial slices of a spatio-temporal simulation.

    Parameters
    ----------
    entire_simulation: np.ndarray
        A simulation of shape [n_sumulations, n_timesteps, *spatial_shape]
        that is stored in a numpy array. The first coordinate is one of the independent simulations,
        the second coordinate is the time dimension, and the rest of the coordinates are the spatial dimensions.
    spatial_window: Tuple | int
        The size of the spatial window rectangle to sample from the data. The dimension of spatial_window
        should match the spatial dimensions of the simulation. If an integer is provided, the spatial window
        is an interval of that size, if it is 2D then it means that it is a rectangle with edges spatial_window[0] and spatial_window[1]
    temporal_window: int
        The size of the temporal window to sample from the data. It is an interval of that size.
    mask_sampler: Callable | None
        This should be a callable function that takes in a tuple `shape` and a `seed` and returns a binary numpy array
        with the same shape as `shape`. The mask_sampler is used to mask out values of the simulation to artificially
        make it irregular
    total_count: int
        The length of the dataset, i.e. the number of spatio-temporal slices to sample from the simulation
    dtype: torch.dtype, optional
        The data type of the output tensors, by default torch.float32
    """
    def __init__(
        self,
        entire_simulation: np.ndarray | str,
        spatial_window: Tuple | int,
        temporal_window: int,
        mask_sampler: Callable | None,
        total_count: int,
        dtype: torch.dtype = torch.float32,
    ):
        # load the simulation
        if isinstance(entire_simulation, str):
            dotenv.load_dotenv(override=True)
            entire_simulation = Path(os.getenv("DATA_PATH", "./outputs/data")) / entire_simulation
            # check if the name ends with .npy and the file exists
            if not entire_simulation.exists():
                raise FileNotFoundError(f"File {entire_simulation} not found.")
            if not entire_simulation.name.endswith(".npy"):
                raise ValueError("Simulation file should be a numpy file.")
            entire_simulation = np.load(entire_simulation)
        self.entire_simulation = entire_simulation
        
        # store the parameters
        self.spatial_window = (spatial_window,) if isinstance(spatial_window, int) else spatial_window
        self.temporal_window = temporal_window
        self.mask_sampler = mask_sampler
        self.total_count = total_count
        self.dtype = dtype 

        # load the simulations and check if the windows are valid, self.entire_simulation should have the shape [S, T, X1, X2, ...]   
        if len(self.entire_simulation.shape) != len(self.spatial_window) + 2:
            raise ValueError("Spatial window dimensions don't match simulation dimensions.")
        for i, window_size in enumerate(self.spatial_window):
            if window_size > self.entire_simulation.shape[i + 2]:
                raise ValueError(f"Spatial window size {window_size} is larger than simulation size {self.entire_simulation.shape[i + 2]}.")
        if temporal_window > self.entire_simulation.shape[1]:
            raise ValueError(f"Temporal window size {temporal_window} is larger than simulation size {self.entire_simulation.shape[1]}.")
        

        # inspect the mask_sampler to ensure that it takes in a shape and seed argument
        if mask_sampler is not None:
            if not callable(mask_sampler):
                raise ValueError("mask_sampler must be a callable function.")
            # inspect parameters of the mask_sampler
            mask_sampler_shape = inspect.signature(mask_sampler)
            if "shape" not in mask_sampler_shape.parameters:
                raise ValueError("mask_sampler must have a shape argument.")
            if "seed" not in mask_sampler_shape.parameters:
                raise ValueError("mask_sampler must have a seed argument.")
    
    def __len__(self):
        return self.total_count
    
    def __getitem__(self, index):
        """
        Returns a random spatio-temporal slice from the simulation data.

        The resulting slice is a set of tensors with the same flat length, organized as follows:

        - a tensor of timesteps
        - a tuple tensor of spatial coordinates (one tensor per dimension)
        - a tensor of values
        """
        if index >= self.total_count:
            raise IndexError(f"Index {index} out of bounds for dataset of size {self.total_count}.")
        
        # get a numpy generator
        rng = np.random.default_rng(seed=index)
        
        selected_simulation = self.entire_simulation[rng.integers(0, self.entire_simulation.shape[0])]
        
        # get the spatial and temporal random slices
        temporal_index = rng.integers(0, selected_simulation.shape[0] - self.temporal_window)
        spatial_corner = []
        for i, window_size in enumerate(self.spatial_window):
            spatial_corner.append(rng.integers(0, selected_simulation.shape[i + 1] - window_size))
        spatial_corner = tuple(spatial_corner)
        simulation_slice = selected_simulation[
            (
                slice(temporal_index,temporal_index + self.temporal_window), 
                *[slice(corner, corner + size) for corner, size in zip(spatial_corner, self.spatial_window)]
            )
        ]

        # get the mask
        if self.mask_sampler is None:
            mask = np.ones((self.temporal_window, *self.spatial_window))
        else:
            mask = self.mask_sampler((self.temporal_window, *self.spatial_window), seed=rng.integers(0, 128))
        all_coords = np.where(mask == 1)
        simulated_values = torch.from_numpy(simulation_slice[tuple(all_coords)]).to(dtype=self.dtype)
        all_coords = torch.stack([torch.from_numpy(coords).to(dtype=self.dtype) for coords in all_coords])
        
        return all_coords, simulated_values
