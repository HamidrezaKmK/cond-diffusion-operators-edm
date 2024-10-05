import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import shutil

import numpy as np
from tqdm import tqdm
import dotenv
import torch
import xarray as xr
from torch.utils.data import Dataset
import eumdac
from datetime import datetime
import zipfile
from zipfile import BadZipFile

class EumstatDataset(Dataset, ABC):

    def __init__(
        self,
        geometry: List[Tuple[float, float]],
        start_time: datetime | str,
        end_time: datetime | str,
        collection_id: str,
        path: str = os.path.join("outputs", "data", "eumstat"),
        query_elements: dict | None = None,
    ):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        dotenv.load_dotenv(override=True)

        # Insert your personal key and secret
        consumer_key = os.getenv('EUMESTAT_CONSUMER_KEY')
        consumer_secret = os.getenv('EUMESTAT_CONSUMER_SECRET')
        
        if not consumer_key or not consumer_secret:
            raise ValueError("Please set the EUMESTAT_CONSUMER_KEY and EUMESTAT_CONSUMER_SECRET environment variables.")
        
        credentials = (consumer_key, consumer_secret)
        token = eumdac.AccessToken(credentials)
        self.datastore = eumdac.DataStore(token)

        selected_collection = self.datastore.get_collection(collection_id)
        self.start_time = start_time if isinstance(start_time, datetime) else datetime.fromisoformat(start_time)
        self.end_time = end_time if isinstance(end_time, datetime) else datetime.fromisoformat(end_time)
        products = selected_collection.search(
            geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in geometry])),
            dtstart=self.start_time, 
            dtend=self.end_time,
            **(query_elements or {}),
        )
        self.all_product_names = []
        pbar = tqdm(products, desc="downloading ...")
        for product in pbar:
            self.all_product_names.append(str(product))
            with product.open() as fsrc:
                zipfile_name = os.path.join(self.path, fsrc.name)
                extraction_path = os.path.join(self.path, str(product))
                try:
                    if not os.path.exists(zipfile_name):
                        pbar.set_description(f"Downloading {zipfile_name}")
                        with open(os.path.join(self.path, fsrc.name), mode='wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
                    else:
                        pbar.set_description(f"File {zipfile_name} already exists")
                    os.makedirs(extraction_path, exist_ok=True)
                    with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
                        zip_ref.extractall(extraction_path)
                except KeyboardInterrupt as e:
                    os.remove(zipfile_name)
                    shutil.rmtree(extraction_path)
                    raise e
                except BadZipFile as e:
                    os.remove(zipfile_name)
                    shutil.rmtree(extraction_path)
                    # take a step back
                    pbar.update(-1)
                    pbar.set_description(f"BadZipFile {zipfile_name}: retrying ...")

class ASCATDataset(EumstatDataset):
    
    COLLECTION_ID = "EO:EUM:DAT:METOP:OSI-104"

    def __init__(
        self,
        geometry: List[Tuple[float, float]],
        start_time: datetime | str,
        end_time: datetime | str,
        attribute: str,
        path: str = os.path.join("outputs", "data", "eumstat"),
        query_elements: dict | None = None,
    ):
        super().__init__(geometry, start_time, end_time, ASCATDataset.COLLECTION_ID, path, query_elements)

        # store all nc arrays
        self.nc_arrays = []
        pbar = tqdm(self.all_product_names, desc="loading ...")
        for product_name in pbar:
            nc_file_name = os.path.join(self.path, product_name, f"{product_name}.nc")
            self.nc_arrays.append(xr.open_dataset(nc_file_name, mask_and_scale=False))
        
        self.attribute = attribute
    
    def __len__(self):
        return len(self.nc_arrays)

    def __getitem__(self, idx):
        ds = self.nc_arrays[idx]
        
        # >> latitudes
        latitudes_scale_factor = ds.coords['lat'].attrs['scale_factor']
        latitudes_offset = ds.coords['lat'].attrs['add_offset']
        latitudes = latitudes_scale_factor * torch.tensor(ds.coords['lat'].values, dtype=torch.float32) + latitudes_offset

        # >> longitudes
        longitudes_scale_factor = ds.coords['lon'].attrs['scale_factor']
        longitudes_offset = ds.coords['lon'].attrs['add_offset']
        longitudes = longitudes_scale_factor * torch.tensor(ds.coords['lon'].values, dtype=torch.float32) + longitudes_offset

        # >> attriute
        attribute_scale_factor = ds[self.attribute].attrs['scale_factor']
        attribute_offset = ds[self.attribute].attrs['add_offset']
        attribute = attribute_scale_factor * torch.tensor(ds[self.attribute].values, dtype=torch.float32) + attribute_offset

        # >> times
        times = self.nc_arrays[idx].time.values
        times = (ds['time'].values - np.datetime64(self.start_time)) / np.timedelta64(1, 's') / 3600.
        times = torch.tensor(times, dtype=torch.float32)

        return latitudes, longitudes, times, attribute


class HIRSDataset(EumstatDataset):
    
    COLLECTION_ID = "EO:EUM:DAT:0647"

    def __init__(
        self,
        geometry: List[Tuple[float, float]],
        start_time: datetime | str,
        end_time: datetime | str,
        channel: int,
        path: str = os.path.join("outputs", "data", "eumstat"),
        query_elements: dict | None = None,
    ):
        super().__init__(geometry, start_time, end_time, HIRSDataset.COLLECTION_ID, path, query_elements)

        # store all nc arrays
        self.nc_arrays = []
        pbar = tqdm(self.all_product_names, desc="loading ...")
        for product_name in pbar:
            nc_file_name = os.path.join(self.path, product_name, f"{product_name}.nc")
            self.nc_arrays.append(xr.open_dataset(nc_file_name, engine='h5netcdf'))
        self.channel = channel
        
    def __len__(self):
        return len(self.nc_arrays)
    
    def __getitem__(self, idx):
        ds = self.nc_arrays[idx]
        
        # >> latitudes
        latitudes = torch.tensor(ds.coords['latitude'].values, dtype=torch.float32)

        # >> longitudes
        longitudes = torch.tensor(ds.coords['longitude'].values, dtype=torch.float32)

        # >> temperatures
        temperatures =  torch.tensor(ds['btemps'].values, dtype=torch.float32)

        # >> times
        times = self.nc_arrays[idx].time.values
        times = (ds['time'].values - np.datetime64(self.start_time)) / np.timedelta64(1, 's') / 3600.
        times = torch.tensor(times, dtype=torch.float32)

        num, x, channels = temperatures.shape

        times = times.repeat(1, x)

        latitudes = latitudes.flatten()
        longitudes = longitudes.flatten()
        times = times.flatten()
        temperatures = temperatures.reshape(-1, channels)
        return latitudes, longitudes, times, temperatures[:, self.channel]

