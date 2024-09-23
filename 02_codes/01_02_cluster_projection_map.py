import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

input_dir = '../03_results/out_genomic_clusters'
filename = 'kmeans_results.tsv'

env_data_dir = '../01_data/01_biological_data'
env_filename = 'metadata.tsv'

output_dir = '../03_results/out_genomic_clusters/map_projections'
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(input_dir,filename)
md_path = os.path.join(env_data_dir,env_filename)

clusters = pd.read_csv(file_path, sep='\t', index_col=0)
md = pd.read_csv(md_path, sep='\t', index_col=0)

merged_data = clusters.join(md[['Latitude', 'Longitude']])

def plot_clusters_on_map(merged_data, cluster_column):
    filtered_data = merged_data[~merged_data[cluster_column].isna()] # Filter nan values
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_global()
    
    unique_clusters = filtered_data[cluster_column].unique()
    for cluster_id in unique_clusters:
        cluster_data = filtered_data[filtered_data[cluster_column] == cluster_id]
        plt.scatter(cluster_data['Longitude'], 
                    cluster_data['Latitude'], 
                    label=f'Cluster {cluster_id}',
                    s=25, 
                    transform=ccrs.PlateCarree())
    
    plt.title(f'Clusters Projection: {cluster_column}')
    plt.legend(loc='upper left')
    #plt.show()

    output_path = os.path.join(output_dir, f'clusters_{cluster_column}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

for column in clusters.columns:
    plot_clusters_on_map(merged_data, column)