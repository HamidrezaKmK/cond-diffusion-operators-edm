import numpy as np

from torch.utils.data import Dataset
import torch
from pyproj import CRS, Transformer

class RegionalDataset(Dataset):
    def __init__(
        self,
        base_dset,
        square_size: float, # in km
        total_count: int,
    ):
        self.total_count = total_count
        self.base_dset = base_dset

        self.half_distance = square_size / 2

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        rng = torch.Generator().manual_seed(idx)
        # take a random element in base_dset
        chosen_idx = torch.randint(0, len(self.base_dset), (1,), generator=rng).item()
        latitudes, longitudes, times, attr = self.base_dset[chosen_idx]

        latitudes = latitudes.flatten()
        longitudes = longitudes.flatten()
        times = times.flatten()
        attr = attr.flatten()

        # randomly select a center of the region
        center_selected = torch.randint(0, len(latitudes), (1,), generator=rng).item()

        center_lat = latitudes[center_selected]
        center_lon = longitudes[center_selected]
        
        crs_wgs84 = CRS.from_epsg(4326)  # EPSG code for WGS84
        proj_string = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +units=km +no_defs +ellps=WGS84"
        crs_local = CRS.from_proj4(proj_string)
        # Transformer from WGS84 to local coordinate system
        transformer_to_local = Transformer.from_crs(crs_wgs84, crs_local, always_xy=True)

        x, y = transformer_to_local.transform(longitudes.cpu().numpy(), latitudes.cpu().numpy())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # remove points that are too far away
        msk = (x >= -self.half_distance) & (x <= self.half_distance) & (y >= -self.half_distance) & (y <= self.half_distance)

        return x[msk], y[msk], times[msk], attr[msk]
        

        

