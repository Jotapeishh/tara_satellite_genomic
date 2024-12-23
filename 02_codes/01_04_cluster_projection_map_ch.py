import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import re
import matplotlib.pyplot as plt

input_dir = '../03_results/out_genomic_clusters'
filename = 'kmeans_results_ch.tsv'

env_data_dir = '../01_data/01_biological_data'
env_filename = 'metadata_chile.tsv'

output_dir = '../03_results/out_genomic_clusters/map_projections_ch'
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(input_dir,filename)
md_path = os.path.join(env_data_dir,env_filename)

clusters = pd.read_csv(file_path, sep='\t', index_col=0)
for col in clusters.columns:
    clusters[col] = clusters[col].astype('Int64')
md = pd.read_csv(md_path, sep='\t', index_col=0)
col_selection = ['lat_cast','lon_cast','Temperature [ºC]',
       'Salinity [PSU]', 'Oxygen [%]',
       'Fluorescence [mg/m3]', 'Orthophosphate [uM]', 'Silicic-acid [uM]',
       'Nitrite [uM]', 'Nitrates [uM]', 'NP ratio']
merged_data = md.join(clusters)

def plot_clusters_on_map(merged_data, cluster_column):
    filtered_data = merged_data[~merged_data[cluster_column].isna()] # filter out rows where cluster_column is NaN
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()  # flatten the array of axes for easy iteration
    
    plot_titles = [
        f'Clusters Projection: {cluster_column}',
        'Temperature [ºC]',
       'Salinity [PSU]', 'Oxygen [%]',
       'Fluorescence [mg/m3]', 'Orthophosphate [uM]', 'Silicic-acid [uM]',
       'Nitrite [uM]', 'Nitrates [uM]', 'NP ratio'
    ]
    data_columns = [
        cluster_column,
        'Temperature [ºC]',
       'Salinity [PSU]', 'Oxygen [%]',
       'Fluorescence [mg/m3]', 'Orthophosphate [uM]', 'Silicic-acid [uM]',
       'Nitrite [uM]', 'Nitrates [uM]', 'NP ratio'
    ]
    
    unique_clusters = filtered_data[cluster_column].unique()
    num_clusters = len(unique_clusters)
    marker_styles = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', 'H', '*', 'x', '+', 'D']
    if num_clusters > len(marker_styles):
        marker_styles = (marker_styles * ((num_clusters // len(marker_styles)) + 1))[:num_clusters]
    cluster_marker_map = dict(zip(unique_clusters, marker_styles))
    
    env_vars = ['Temperature [ºC]',
       'Salinity [PSU]', 'Oxygen [%]',
       'Fluorescence [mg/m3]', 'Orthophosphate [uM]', 'Silicic-acid [uM]',
       'Nitrite [uM]', 'Nitrates [uM]', 'NP ratio']
    norms = {}
    
    for data_column in env_vars:
        vmin = filtered_data[data_column].min()
        vmax = filtered_data[data_column].max()
        norms[data_column] = Normalize(vmin=vmin, vmax=vmax)
    
    for idx, ax in enumerate(axes):
        ax.set_extent([-80, -67, -55,-17])

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.BORDERS)
        
        ax.set_title(plot_titles[idx])
        
        data_column = data_columns[idx]
        plot_data = filtered_data[~filtered_data[data_column].isna()]
        
        if idx == 0:
            for cluster_id in unique_clusters:
                cluster_points = plot_data[plot_data[cluster_column] == cluster_id]
                ax.scatter(
                    cluster_points['lon_cast'],
                    cluster_points['lat_cast'],
                    label=f'Cluster {cluster_id}',
                    s=35,
                    marker=cluster_marker_map[cluster_id],
                    transform=ccrs.PlateCarree()
                )
            ax.legend(loc='upper left')
        else:
            norm = norms[data_column]
            for cluster_id in unique_clusters:
                cluster_points = plot_data[plot_data[cluster_column] == cluster_id]
                sc = ax.scatter(
                    cluster_points['lon_cast'],
                    cluster_points['lat_cast'],
                    c=cluster_points[data_column],
                    s=35,
                    cmap='viridis',
                    marker=cluster_marker_map[cluster_id],
                    edgecolors='black',
                    norm=norm,
                    transform=ccrs.PlateCarree()
                )
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.5)
            cbar.set_label(data_column)
            handles = []
            for cluster_id in unique_clusters:
                marker = cluster_marker_map[cluster_id]
                handle = plt.Line2D([], [], color='black', marker=marker, linestyle='', markersize=8, label=f'Cluster {cluster_id}')
                handles.append(handle)
            ax.legend(handles=handles, loc='upper left')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'clusters_{cluster_column}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

for column in clusters.columns:
    plot_clusters_on_map(merged_data, column)