import pandas as pd
import numpy as np

from scipy.spatial import cKDTree

import os

import netCDF4 as nc
import xarray as xr

import re
from glob import glob

import time

import sys

if len(sys.argv) != 2:
    print("Usage: python your_script_name.py <n_adj_points>")
    sys.exit(1)

# Convertir el argumento a un entero
n_adj_points = int(sys.argv[1])

file_path = '../01_data/01_biological_data'
file_name = 'metadata.tsv'
sat_data_path = '../01_data/00_satellite_data'

output_dir = '../01_data/02_satellite_data_processed'
os.makedirs(output_dir, exist_ok=True)

file = os.path.join(file_path, file_name)
md = pd.read_csv(file, sep='\t', index_col=0)

md_srf = md[md.Layer == 'SRF'].copy()
md_srf['Event.date.YYYYMM'] = md_srf['Event.date'].str[:7].str.replace('-', '')
md_srf['Event.date.YYYYMM01'] = md_srf['Event.date'].str[:7].str.replace('-', '')+'01'

satellite_features = [
    'CHL.chlor_a', 'FLH.nflh', 'FLH.ipar', 'IOP.adg_unc_443', 'IOP.adg_443',
    'IOP.aph_unc_443', 'IOP.aph_44', 'IOP.bbp_s', 'IOP.adg_s', 'bbp_unc_443', 
    'IOP.bbp_443', 'IOP.a_412', 'IOP.a_443', 'IOP.a_469', 'IOP.a_488', 'IOP.a_531', 
    'IOP.a_547', 'IOP.a_555', 'IOP.a_645', 'IOP.a_667', 'IOP.a_678', 'IOP.bb_412', 
    'IOP.bb_443', 'IOP.bb_469', 'IOP.bb_488', 'IOP.bb_531', 'IOP.bb_547', 'IOP.bb_555', 
    'IOP.bb_645', 'IOP.bb_667', 'IOP.bb_678', 'KD.Kd_490', 'NSST.sst', 'PAR.par', 
    'PIC.pic', 'POC.poc', 'RRS.aot_869', 'RRS.angstrom', 'RRS.Rrs_412', 'RRS.Rrs_443', 
    'RRS.Rrs_469', 'RRS.Rrs_488', 'RRS.Rrs_531', 'RRS.Rrs_547', 'RRS.Rrs_555', 
    'RRS.Rrs_645', 'RRS.Rrs_667', 'RRS.Rrs_678', 'SST.sst'
]

satellite_data_terra = pd.DataFrame(index=md_srf.index, columns=satellite_features)
satellite_data_aqua = pd.DataFrame(index=md_srf.index, columns=satellite_features)

def find_satellite_file(directory, pattern):
    regex = re.compile(pattern)
    for file in os.listdir(directory):
        if regex.match(file):
            return os.path.join(directory, file)
    return None

def select_n_nearest_valid(ds, feature, latitude, longitude, n):
    latitudes = ds['lat'].values
    longitudes = ds['lon'].values
    
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing='ij')
    distances = np.sqrt((lat_grid - latitude)**2 + (lon_grid - longitude)**2)
    
    sorted_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
    
    valid_points = []
    for i in range(len(sorted_indices[0])):
        lat_idx = sorted_indices[0][i]
        lon_idx = sorted_indices[1][i]
        data_point = ds[feature].isel(lat=lat_idx, lon=lon_idx)
        if not np.isnan(data_point.values):
            valid_points.append(data_point.values)
        if len(valid_points) >= n:
            break
    
    if len(valid_points) > 0:
        return np.mean(valid_points)
    else:
        return np.nan

start_time = time.time()

for index, row in md_srf.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']
    date = row['Event.date.YYYYMM']
    
    for feature in satellite_features:
        resolution = '9km'
        
        pattern_terra = rf"TERRA_MODIS\.{date}01_{date}\d{{2}}\.L3m\.MO\.{feature}\.{resolution}\.nc"
        file_path_terra = find_satellite_file(sat_data_path, pattern_terra)
        pattern_aqua = rf"AQUA_MODIS\.{date}01_{date}\d{{2}}\.L3m\.MO\.{feature}\.{resolution}\.nc"
        file_path_aqua = find_satellite_file(sat_data_path, pattern_aqua)
        
        if file_path_terra and file_path_aqua:
            ds_terra = xr.open_dataset(file_path_terra)
            ds_aqua = xr.open_dataset(file_path_aqua)
            try:
                variable_name = feature.split('.')[1]
                
                data_point_terra = select_n_nearest_valid(ds_terra, variable_name, latitude, longitude, n_adj_points)
                satellite_data_terra.at[index, feature] = data_point_terra

                data_point_aqua = select_n_nearest_valid(ds_aqua, variable_name, latitude, longitude, n_adj_points)
                satellite_data_aqua.at[index, feature] = data_point_aqua

            except KeyError:
                print(f"Feature {feature} not found in dataset")
            finally:
                ds_terra.close()
                ds_aqua.close()
        else:
            if not file_path_terra:
                print(f"No TERRA file found for pattern: {pattern_terra}")
            if not file_path_aqua:
                print(f"No AQUA file found for pattern: {pattern_aqua}")

satellite_data_avg = (satellite_data_terra.astype(float) + satellite_data_aqua.astype(float)) / 2

end_time = time.time()
execution_time = end_time - start_time

print("satellite nan values")
print(satellite_data_avg.isna().sum())

print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

output_path = os.path.join(output_dir, f'matrix_tara_world_adj_grids_{n_adj_points}.tsv')
satellite_data_avg.to_csv(output_path, sep='\t')