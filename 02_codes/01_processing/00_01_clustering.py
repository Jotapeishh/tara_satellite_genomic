import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.

    Parameters:
    data (pandas.DataFrame): A DataFrame with samples as rows and components as columns.

    Returns:
    pandas.DataFrame: A clr-normalized DataFrame.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")

    # Add small amount to cells with a value of 0
    if (data <= 0).any().any():
        data = data.replace(0, eps)

    # Calculate the geometric mean of each row
    gm = np.exp(data.apply(np.log).mean(axis=1))

    # Perform clr transformation
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)

    return clr_data

input_dir = '../../01_data/01_biological_data'
output_dir = '../../03_out_results/out_genomic_clusters'
os.makedirs(output_dir, exist_ok=True)

# Read matrices of interest and sort them alphabetically
files = os.listdir(input_dir)
matrix_files = sorted([f for f in files if f.startswith('Matrix_world_GEN_') and f.endswith('.tsv')])

all_metrics_results = []
clustering_results_dict = {}

def perform_kmeans_clustering(matrix, matrix_type_subsample, n_clusters_list, clr=False):
    suffix = 'clr_' if clr else ''
    # Perform K-Means for different 'n'
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
        kmeans.fit(matrix)
        
        cluster_labels = kmeans.labels_
        
        # Calculate evaluation metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(matrix, cluster_labels)
        davies_bouldin = davies_bouldin_score(matrix, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(matrix, cluster_labels)
        
        all_metrics_results.append({
            'matrix': f"{suffix}{matrix_type_subsample}",
            'n_clusters': n_clusters,
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        })
        
        col_name = f"{suffix}{matrix_type_subsample}_kmeans_{n_clusters}" # Create a DataFrame for the cluster labels with appropriate column names
        results = pd.DataFrame({col_name: cluster_labels}, index=matrix.index)
        
        if col_name not in clustering_results_dict:
            clustering_results_dict[col_name] = results
        else:
            clustering_results_dict[col_name] = pd.concat([clustering_results_dict[col_name], results], axis=1)

# Perform K-Means for different n-clusters for each matrix
n_clusters_list = [3, 4, 5, 6, 7, 8]
for matrix_file in matrix_files:
    file_path = os.path.join(input_dir, matrix_file)
    matrix = pd.read_csv(file_path, sep='\t', index_col=0)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    matrix_type_subsample = "_".join(base_filename.split('_')[3:])
    
    perform_kmeans_clustering(matrix, matrix_type_subsample, n_clusters_list, clr=False)
    # CLR normalized matrix clustering
    clr_matrix = clr_(matrix)
    perform_kmeans_clustering(clr_matrix, matrix_type_subsample, n_clusters_list, clr=True)

combined_clustering_results = pd.concat(clustering_results_dict.values(), axis=1)
#combined_clustering_results = combined_clustering_results.sort_index(axis=1)

# Results of the kmeans
output_filename = 'kmeans_results.tsv'
combined_clustering_results.to_csv(os.path.join(output_dir, output_filename), sep='\t', index=True)

# Results of the metrics of the kmeans clustering
metrics_df = pd.DataFrame(all_metrics_results)
metrics_output_filename = 'kmeans_metrics.tsv'
metrics_df.to_csv(os.path.join(output_dir, metrics_output_filename), sep='\t', index=False)

# Plot metrics
unique_matrices = metrics_df['matrix'].unique()
for matrix_type_subsample in unique_matrices:
    matrix_metrics_df = metrics_df[metrics_df['matrix'] == matrix_type_subsample]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia', color='tab:blue')
    ax1.plot(matrix_metrics_df['n_clusters'], matrix_metrics_df['inertia'], color='tab:blue', label='Inertia')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score', color='tab:orange')
    ax2.plot(matrix_metrics_df['n_clusters'], matrix_metrics_df['silhouette_score'], color='tab:orange', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.axhline(y=0.25, color='tab:orange', linestyle='--', linewidth=1, label='Silhouette Score Threshold (0.25)')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Davies-Bouldin Score', color='tab:green')
    ax3.plot(matrix_metrics_df['n_clusters'], matrix_metrics_df['davies_bouldin_score'], color='tab:green', label='Davies-Bouldin Score')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.axhline(y=1.50, color='tab:green', linestyle='--', linewidth=1, label='Davies-Bouldin Score Threshold (1.50)')

    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel('Calinski-Harabasz Score', color='tab:red')
    ax4.plot(matrix_metrics_df['n_clusters'], matrix_metrics_df['calinski_harabasz_score'], color='tab:red', label='Calinski-Harabasz Score')
    ax4.tick_params(axis='y', labelcolor='tab:red')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    fig.tight_layout()
    plt.title(f'Evaluation Metrics for {matrix_type_subsample}')

    # Save the plot
    plot_filename = f'kmeans_metrics_{matrix_type_subsample}.pdf'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.close()
