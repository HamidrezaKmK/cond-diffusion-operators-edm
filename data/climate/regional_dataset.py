from typing import List, Tuple
from abc import abstractmethod, ABC
from datetime import datetime, timedelta
import bisect

import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
from pyproj import CRS, Transformer, Proj

from .eumetsat import EumstatDataset

class RegionalDataset(Dataset, ABC):
    """
    This is a (square) regional dataset that samples a region that can be further used for analysis and training.
    Different variations of this type of dataset exists. For example, one variation is from the Era5 re-analysis dataset
    and another is from the Eumetsat dataset. 
    """

    def __init__(
        self,
        NS_length: float, # in km
        EW_length: float, # in km
        total_count: int,
        timedelta_size: timedelta,
    ):
        """
        Args:
            square_size (float): The size of the square in km.
            total_count (int): The total number of samples to generate.
        """
        self.total_count = total_count
        self.NS_length = NS_length
        self.EW_length = EW_length
        self.timedelta_size = timedelta_size
    
    def __len__(self) -> int:
        return self.total_count

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in the derived class.")    

class EumstatRegionalDataset(RegionalDataset):
    def __init__(
        self,
        NS_length: float, # in km
        EW_length: float, # in km
        total_count: int,
        base_dset: EumstatDataset,
    ):
        """
        This variation of the regional dataset uses an EumstateDataset as the base dataset.
        """
        super().__init__(NS_length, EW_length, total_count)
        self.base_dset = base_dset

    def __getitem__(self, idx):
        """
        Take a random region by sampling a random coordinate center and then taking all the points that are within the square.
        The coordinates are then transformed into a local coordinate system with the center as the origin.
        """
        rng = torch.Generator().manual_seed(idx)
        
        chosen_idx = torch.randint(0, len(self.base_dset), (1,), generator=rng).item()
        latitudes, longitudes, times, attr = self.base_dset[chosen_idx]

        assert len(latitudes) == len(longitudes) == len(times) == len(attr), "All the arrays should have the same length."
        assert latitudes.numel() == longitudes.numel() == times.numel() == attr.numel(), "All the arrays should have the same number of elements."

        # randomly select a center of the region
        center_selected = torch.randint(0, len(latitudes), (1,), generator=rng).item()

        center_lat = latitudes[center_selected]
        center_lon = longitudes[center_selected]
        time_center = times[center_selected]

                
        crs_wgs84 = CRS.from_epsg(4326)  # EPSG code for WGS84
        proj_string = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +units=km +no_defs +ellps=WGS84"
        crs_local = CRS.from_proj4(proj_string)
        # Transformer from WGS84 to local coordinate system
        transformer_to_local = Transformer.from_crs(crs_wgs84, crs_local, always_xy=True)

        x, y = transformer_to_local.transform(longitudes.cpu().numpy(), latitudes.cpu().numpy())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # remove points that are too far away
        msk = (x >= -self.EW_length / 2) & (x <= self.EW_length / 2) & (y >= -self.NS_length / 2) & (y <= self.NS_length / 2)
        
        # remove the points that are outside the time range
        msk = msk & (times >= time_center - self.timedelta_size) & (times <= time_center + self.timedelta_size)

        return x[msk], y[msk], times[msk], attr[msk]

class ERA5RegionalDataset(RegionalDataset):
    @staticmethod
    def all_attributes():
        """The list of all attributes that are stored in this subset of ERA5"""
        return [
            '100m_u_component_of_wind',
            '100m_v_component_of_wind',
            '10m_u_component_of_neutral_wind',
            '10m_u_component_of_wind',
            '10m_v_component_of_neutral_wind',
            '10m_v_component_of_wind',
            '10m_wind_gust_since_previous_post_processing',
            '2m_dewpoint_temperature',
            '2m_temperature',
            'air_density_over_the_oceans',
            'angle_of_sub_gridscale_orography',
            'anisotropy_of_sub_gridscale_orography',
            'benjamin_feir_index',
            'boundary_layer_dissipation',
            'boundary_layer_height',
            'charnock',
            'clear_sky_direct_solar_radiation_at_surface',
            'cloud_base_height',
            'coefficient_of_drag_with_waves',
            'convective_available_potential_energy',
            'convective_inhibition',
            'convective_precipitation',
            'convective_rain_rate',
            'convective_snowfall',
            'convective_snowfall_rate_water_equivalent',
            'downward_uv_radiation_at_the_surface',
            'duct_base_height',
            'eastward_gravity_wave_surface_stress',
            'eastward_turbulent_surface_stress',
            'evaporation',
            'forecast_albedo',
            'forecast_logarithm_of_surface_roughness_for_heat',
            'forecast_surface_roughness',
            # 'fraction_of_cloud_cover',
            'free_convective_velocity_over_the_oceans',
            'friction_velocity',
            # 'geopotential',
            'geopotential_at_surface',
            'gravity_wave_dissipation',
            'high_cloud_cover',
            'high_vegetation_cover',
            'ice_temperature_layer_1',
            'ice_temperature_layer_2',
            'ice_temperature_layer_3',
            'ice_temperature_layer_4',
            'instantaneous_10m_wind_gust',
            'instantaneous_eastward_turbulent_surface_stress',
            'instantaneous_large_scale_surface_precipitation_fraction',
            'instantaneous_moisture_flux',
            'instantaneous_northward_turbulent_surface_stress',
            'instantaneous_surface_sensible_heat_flux',
            'k_index',
            'lake_bottom_temperature',
            'lake_cover',
            'lake_depth',
            'lake_ice_depth',
            'lake_ice_temperature',
            'lake_mix_layer_depth',
            'lake_mix_layer_temperature',
            'lake_shape_factor',
            'lake_total_layer_temperature',
            'land_sea_mask',
            'large_scale_precipitation',
            'large_scale_precipitation_fraction',
            'large_scale_rain_rate',
            'large_scale_snowfall',
            'large_scale_snowfall_rate_water_equivalent',
            'leaf_area_index_high_vegetation',
            'leaf_area_index_low_vegetation',
            'low_cloud_cover',
            'low_vegetation_cover',
            'maximum_2m_temperature_since_previous_post_processing',
            'maximum_individual_wave_height',
            'maximum_total_precipitation_rate_since_previous_post_processing',
            'mean_boundary_layer_dissipation',
            'mean_convective_precipitation_rate',
            'mean_convective_snowfall_rate',
            'mean_direction_of_total_swell',
            'mean_direction_of_wind_waves',
            'mean_eastward_gravity_wave_surface_stress',
            'mean_eastward_turbulent_surface_stress',
            'mean_evaporation_rate',
            'mean_gravity_wave_dissipation',
            'mean_large_scale_precipitation_fraction',
            'mean_large_scale_precipitation_rate',
            'mean_large_scale_snowfall_rate',
            'mean_northward_gravity_wave_surface_stress',
            'mean_northward_turbulent_surface_stress',
            'mean_period_of_total_swell',
            'mean_period_of_wind_waves',
            'mean_potential_evaporation_rate',
            'mean_runoff_rate',
            'mean_sea_level_pressure',
            'mean_snow_evaporation_rate',
            'mean_snowfall_rate',
            'mean_snowmelt_rate',
            'mean_square_slope_of_waves',
            'mean_sub_surface_runoff_rate',
            'mean_surface_direct_short_wave_radiation_flux',
            'mean_surface_direct_short_wave_radiation_flux_clear_sky',
            'mean_surface_downward_long_wave_radiation_flux',
            'mean_surface_downward_long_wave_radiation_flux_clear_sky',
            'mean_surface_downward_short_wave_radiation_flux',
            'mean_surface_downward_short_wave_radiation_flux_clear_sky',
            'mean_surface_downward_uv_radiation_flux',
            'mean_surface_latent_heat_flux',
            'mean_surface_net_long_wave_radiation_flux',
            'mean_surface_net_long_wave_radiation_flux_clear_sky',
            'mean_surface_net_short_wave_radiation_flux',
            'mean_surface_net_short_wave_radiation_flux_clear_sky',
            'mean_surface_runoff_rate',
            'mean_surface_sensible_heat_flux',
            'mean_top_downward_short_wave_radiation_flux',
            'mean_top_net_long_wave_radiation_flux',
            'mean_top_net_long_wave_radiation_flux_clear_sky',
            'mean_top_net_short_wave_radiation_flux',
            'mean_top_net_short_wave_radiation_flux_clear_sky',
            'mean_total_precipitation_rate',
            'mean_vertical_gradient_of_refractivity_inside_trapping_layer',
            'mean_vertically_integrated_moisture_divergence',
            'mean_wave_direction',
            'mean_wave_direction_of_first_swell_partition',
            'mean_wave_direction_of_second_swell_partition',
            'mean_wave_direction_of_third_swell_partition',
            'mean_wave_period',
            'mean_wave_period_based_on_first_moment',
            'mean_wave_period_based_on_first_moment_for_swell',
            'mean_wave_period_based_on_first_moment_for_wind_waves',
            'mean_wave_period_based_on_second_moment_for_swell',
            'mean_wave_period_based_on_second_moment_for_wind_waves',
            'mean_wave_period_of_first_swell_partition',
            'mean_wave_period_of_second_swell_partition',
            'mean_wave_period_of_third_swell_partition',
            'mean_zero_crossing_wave_period',
            'medium_cloud_cover',
            'minimum_2m_temperature_since_previous_post_processing',
            'minimum_total_precipitation_rate_since_previous_post_processing',
            'minimum_vertical_gradient_of_refractivity_inside_trapping_layer',
            'model_bathymetry',
            'near_ir_albedo_for_diffuse_radiation',
            'near_ir_albedo_for_direct_radiation',
            'normalized_energy_flux_into_ocean',
            'normalized_energy_flux_into_waves',
            'normalized_stress_into_ocean',
            'northward_gravity_wave_surface_stress',
            'northward_turbulent_surface_stress',
            'ocean_surface_stress_equivalent_10m_neutral_wind_direction',
            'ocean_surface_stress_equivalent_10m_neutral_wind_speed',
            # 'ozone_mass_mixing_ratio',
            'peak_wave_period',
            'period_corresponding_to_maximum_individual_wave_height',
            'potential_evaporation',
            # 'potential_vorticity',
            'precipitation_type',
            'runoff',
            'sea_ice_cover',
            'sea_surface_temperature',
            'significant_height_of_combined_wind_waves_and_swell',
            'significant_height_of_total_swell',
            'significant_height_of_wind_waves',
            'significant_wave_height_of_first_swell_partition',
            'significant_wave_height_of_second_swell_partition',
            'significant_wave_height_of_third_swell_partition',
            'skin_reservoir_content',
            'skin_temperature',
            'slope_of_sub_gridscale_orography',
            'snow_albedo',
            'snow_density',
            'snow_depth',
            'snow_evaporation',
            'snowfall',
            'snowmelt',
            'soil_temperature_level_1',
            'soil_temperature_level_2',
            'soil_temperature_level_3',
            'soil_temperature_level_4',
            'soil_type',
            # 'specific_cloud_ice_water_content',
            # 'specific_cloud_liquid_water_content',
            # 'specific_humidity',
            'standard_deviation_of_filtered_subgrid_orography',
            'standard_deviation_of_orography',
            'sub_surface_runoff',
            'surface_latent_heat_flux',
            'surface_net_solar_radiation',
            'surface_net_solar_radiation_clear_sky',
            'surface_net_thermal_radiation',
            'surface_net_thermal_radiation_clear_sky',
            'surface_pressure',
            'surface_runoff',
            'surface_sensible_heat_flux',
            'surface_solar_radiation_downward_clear_sky',
            'surface_solar_radiation_downwards',
            'surface_thermal_radiation_downward_clear_sky',
            'surface_thermal_radiation_downwards',
            # 'temperature',
            'temperature_of_snow_layer',
            'toa_incident_solar_radiation',
            'top_net_solar_radiation',
            'top_net_solar_radiation_clear_sky',
            'top_net_thermal_radiation',
            'top_net_thermal_radiation_clear_sky',
            'total_cloud_cover',
            'total_column_cloud_ice_water',
            'total_column_cloud_liquid_water',
            'total_column_ozone',
            'total_column_rain_water',
            'total_column_snow_water',
            'total_column_supercooled_liquid_water',
            'total_column_water',
            'total_column_water_vapour',
            'total_precipitation',
            'total_sky_direct_solar_radiation_at_surface',
            'total_totals_index',
            'trapping_layer_base_height',
            'trapping_layer_top_height',
            'type_of_high_vegetation',
            'type_of_low_vegetation',
            # 'u_component_of_wind',
            'u_component_stokes_drift',
            'uv_visible_albedo_for_diffuse_radiation',
            'uv_visible_albedo_for_direct_radiation',
            # 'v_component_of_wind',
            'v_component_stokes_drift',
            'vertical_integral_of_divergence_of_cloud_frozen_water_flux',
            'vertical_integral_of_divergence_of_cloud_liquid_water_flux',
            'vertical_integral_of_divergence_of_geopotential_flux',
            'vertical_integral_of_divergence_of_kinetic_energy_flux',
            'vertical_integral_of_divergence_of_mass_flux',
            'vertical_integral_of_divergence_of_moisture_flux',
            'vertical_integral_of_divergence_of_ozone_flux',
            'vertical_integral_of_divergence_of_thermal_energy_flux',
            'vertical_integral_of_divergence_of_total_energy_flux',
            'vertical_integral_of_eastward_cloud_frozen_water_flux',
            'vertical_integral_of_eastward_cloud_liquid_water_flux',
            'vertical_integral_of_eastward_geopotential_flux',
            'vertical_integral_of_eastward_heat_flux',
            'vertical_integral_of_eastward_kinetic_energy_flux',
            'vertical_integral_of_eastward_mass_flux',
            'vertical_integral_of_eastward_ozone_flux',
            'vertical_integral_of_eastward_total_energy_flux',
            'vertical_integral_of_eastward_water_vapour_flux',
            'vertical_integral_of_energy_conversion',
            'vertical_integral_of_kinetic_energy',
            'vertical_integral_of_mass_of_atmosphere',
            'vertical_integral_of_mass_tendency',
            'vertical_integral_of_northward_cloud_frozen_water_flux',
            'vertical_integral_of_northward_cloud_liquid_water_flux',
            'vertical_integral_of_northward_geopotential_flux',
            'vertical_integral_of_northward_heat_flux',
            'vertical_integral_of_northward_kinetic_energy_flux',
            'vertical_integral_of_northward_mass_flux',
            'vertical_integral_of_northward_ozone_flux',
            'vertical_integral_of_northward_total_energy_flux',
            'vertical_integral_of_northward_water_vapour_flux',
            'vertical_integral_of_potential_and_internal_energy',
            'vertical_integral_of_potential_internal_and_latent_energy',
            'vertical_integral_of_temperature',
            'vertical_integral_of_thermal_energy',
            'vertical_integral_of_total_energy',
            # 'vertical_velocity',
            'vertically_integrated_moisture_divergence',
            'volumetric_soil_water_layer_1',
            'volumetric_soil_water_layer_2',
            'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4',
            'wave_spectral_directional_width',
            'wave_spectral_directional_width_for_swell',
            'wave_spectral_directional_width_for_wind_waves',
            'wave_spectral_kurtosis',
            'wave_spectral_peakedness',
            'wave_spectral_skewness',
            'zero_degree_level',
        ]

    def __init__(
        self,
        NS_length: float, # in km
        EW_length: float, # in km
        total_count: int,
        attribute: str,
        timedelta_size: timedelta,
    ):
        """
        Use the ERA5 dataset, in particular, the one available at:
        https://github.com/google-research/arco-era5?tab=readme-ov-file
        which is a super-set of what NeuralGCM and GraphCast have been trained on.
        The dataset is available on Google Cloud Storage
        """
        super().__init__(NS_length, EW_length, total_count, timedelta_size)
        
        assert attribute in ERA5RegionalDataset.all_attributes(), f"Attribute {attribute} is not in the list of all attributes."

        ds = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
            chunks=None,
            storage_options=dict(token='anon'),
        )
        self.all_observations = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))[attribute]

    def __getitem__(self, idx):

        # set the random generator
        rng = torch.Generator().manual_seed(idx)

        time_index = torch.randint(0, len(self.all_observations.coords['time']), (1,), generator=rng).item()
        all_times = [datetime.fromtimestamp(t.item() / 1e9) for t in self.all_observations.coords['time'].values]
        time_center = all_times[time_index]
        time_start = time_center - self.timedelta_size / 2
        time_end = time_center + self.timedelta_size / 2
        time_left_index = bisect.bisect(all_times, time_start)
        time_right_index = bisect.bisect(all_times, time_end)

        center_lat_index = torch.randint(0, len(self.all_observations.coords['latitude']), (1,), generator=rng).item()
        center_lon_index = torch.randint(0, len(self.all_observations.coords['longitude']), (1,), generator=rng).item()
        
        center_lat = self.all_observations.coords['latitude'].values[center_lat_index]
        center_lon = self.all_observations.coords['longitude'].values[center_lon_index]
        
        # ---------------------
        # Define a custom projection centered on the given latitude and longitude using the azimuthal equidistant projection
        projection = Proj(proj="aeqd", lat_0=center_lat, lon_0=center_lon, datum="WGS84")

        # Use the transformer to convert from geographic (lat, lon) to planar (x, y)
        transformer = Transformer.from_proj(proj_from="epsg:4326", proj_to=projection)

        _, x_all_for_comparison = transformer.transform(self.all_observations.coords['latitude'].values, center_lon * np.ones(len(self.all_observations.coords['latitude'].values)))
        _, y_all_for_comparison = transformer.transform(center_lat * np.ones(len(self.all_observations.coords['longitude'].values)), self.all_observations.coords['longitude'].values)
        
        x_all_for_comparison *= -1
        y_all_for_comparison[center_lon_index:] *= -1


        x_left = bisect.bisect(x_all_for_comparison, x_all_for_comparison[center_lat_index] - self.EW_length * 10)
        x_right = bisect.bisect(x_all_for_comparison, x_all_for_comparison[center_lat_index] + self.EW_length * 10)
        y_left = bisect.bisect(y_all_for_comparison, y_all_for_comparison[center_lon_index] - self.NS_length * 10)
        y_right = bisect.bisect(y_all_for_comparison, y_all_for_comparison[center_lon_index] + self.NS_length * 10)

        filtered = self.all_observations.isel(time=slice(time_left_index, time_right_index), latitude=slice(x_left, x_right), longitude=slice(y_left, y_right))
        stacked = filtered.stack(sample=('time', 'latitude', 'longitude'))

        lats = stacked['latitude'].values
        lons = stacked['longitude'].values
        times = stacked['time'].values
        attrs = stacked.values
        
        x, y = transformer.transform(lats, lons)
        times = times.astype('timedelta64[s]').astype(float) / 3600.

        msk = (x >= -self.EW_length) & (x <= self.EW_length) & (y >= -self.NS_length / 2) & (y <= self.NS_length / 2)

        return torch.from_numpy(x).float()[msk], torch.from_numpy(y).float()[msk], torch.from_numpy(times).float()[msk], torch.from_numpy(attrs).float()[msk], center_lat, center_lon, time_center
