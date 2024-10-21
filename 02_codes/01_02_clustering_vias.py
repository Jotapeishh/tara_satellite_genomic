import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

input_dir = '../01_data/01_biological_data'
output_dir = '../03_results/out_genomic_clusters'
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)
matrix_files = sorted([f for f in files if f.startswith('Matrix_world_GEN_M1-vias') and f.endswith('.tsv')])

all_metrics_results = []
clustering_results_dict = {}

def perform_kmeans_clustering(matrix, matrix_type_subsample, n_clusters_list):
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
        kmeans.fit(matrix)
        
        cluster_labels = kmeans.labels_
        
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(matrix, cluster_labels)
        davies_bouldin = davies_bouldin_score(matrix, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(matrix, cluster_labels)
        
        all_metrics_results.append({
            'matrix': f"{matrix_type_subsample}",
            'n_clusters': n_clusters,
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        })
        
        col_name = f"{matrix_type_subsample}_kmeans_{n_clusters}"
        results = pd.DataFrame({col_name: cluster_labels}, index=matrix.index)
        
        if col_name not in clustering_results_dict:
            clustering_results_dict[col_name] = results
        else:
            clustering_results_dict[col_name] = pd.concat([clustering_results_dict[col_name], results], axis=1)

n_clusters_list = [3, 4, 5, 6, 7, 8]
for matrix_file in matrix_files:
    file_path = os.path.join(input_dir, matrix_file)
    matrix = pd.read_csv(file_path, sep='\t', index_col=0)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    matrix_type_subsample = "_".join(base_filename.split('_')[3:])
    
    perform_kmeans_clustering(matrix, matrix_type_subsample, n_clusters_list)

combined_clustering_results = pd.concat(clustering_results_dict.values(), axis=1)

existing_results_file = os.path.join(output_dir, 'kmeans_results.tsv')

if os.path.exists(existing_results_file):
    existing_results = pd.read_csv(existing_results_file, sep='\t', index_col=0)
    combined_clustering_results = pd.concat([existing_results, combined_clustering_results], axis=1)

combined_clustering_results.to_csv(existing_results_file, sep='\t', index=True)

metrics_df = pd.DataFrame(all_metrics_results)

metrics_output_filename = os.path.join(output_dir, 'kmeans_metrics.tsv')

if os.path.exists(metrics_output_filename):
    existing_metrics = pd.read_csv(metrics_output_filename, sep='\t')
    metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)

metrics_df.to_csv(metrics_output_filename, sep='\t', index=False)
