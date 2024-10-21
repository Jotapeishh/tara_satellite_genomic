import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import xgboost as xgb

from glob import glob
import os

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder

input_sat_dir = '../01_data/02_satellite_data_processed'

desired_files = [
    # 'matrix_tara_world_adj_grids_01.tsv',
    # 'matrix_tara_world_adj_grids_09.tsv', 
    'matrix_tara_world_adj_grids_25.tsv',  # In this case of study we will only use 25 adjacent grids
    # 'matrix_tara_world_adj_grids_49.tsv'
]

predictor_files = sorted([f for f in glob(os.path.join(input_sat_dir, 'matrix_tara_world_adj_grids_*.tsv')) 
                          if os.path.basename(f) in desired_files])


input_kmeans_dir = '../03_results/out_genomic_clusters'
target_vars_filename = 'kmeans_results.tsv'
target_vars_path = os.path.join(input_kmeans_dir, target_vars_filename)

target_vars = pd.read_csv(target_vars_path, sep='\t', index_col=0)
target_vars = target_vars.map(lambda x: f"C{x}")

# Identify only the new columns related to the M1-vias data
desired_clusters = {'5', '6', '7', '8'}  # Only consider this number of clusters
columns_to_use = [col for col in target_vars.columns if 'M1-vias' in col and col.split('_')[-1] in desired_clusters]

results_df = pd.DataFrame(index=[os.path.basename(file) for file in predictor_files], columns=columns_to_use)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return (accuracy, f1)

n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

le = LabelEncoder()

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro')
}

for file in predictor_files:
    file_name = os.path.basename(file)
    df = pd.read_csv(file, sep='\t', index_col=0)

    aligned_predictor = df.loc[df.index.intersection(target_vars.index)]  # Align with satellite data

    for target_column in columns_to_use:
        X = aligned_predictor
        y = target_vars.loc[aligned_predictor.index, target_column]

        non_nan_indices = y.dropna().index
        X = X.loc[non_nan_indices]
        y = y.loc[non_nan_indices]

        y_encoded = le.fit_transform(y)

        unique, counts = np.unique(y_encoded, return_counts=True)
        min_samples = n_splits

        X_resampled = X.copy()
        y_resampled = y_encoded.copy()

        for cls, count in zip(unique, counts):
            if count < min_samples:
                diff = min_samples - count
                cls_indices = np.where(y_encoded == cls)[0]
                indices_to_duplicate = np.random.choice(cls_indices, diff, replace=True)
                X_resampled = np.concatenate([X_resampled, X.iloc[indices_to_duplicate]], axis=0)
                y_resampled = np.concatenate([y_resampled, y_encoded[indices_to_duplicate]], axis=0)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        cv_results = cross_validate(model, X_resampled, y_resampled, cv=rskf, scoring=scoring, return_train_score=False)

        avg_accuracy = np.mean(cv_results['test_accuracy'])
        avg_f1_macro = np.mean(cv_results['test_f1_macro'])

        results_df.at[file_name, target_column] = (avg_accuracy, avg_f1_macro)
            
print(results_df)

# Path to the existing predictions file
predictions_file = '../03_results/out_predictions/predictions_kmeans.tsv'

if os.path.exists(predictions_file):
    existing_results_df = pd.read_csv(predictions_file, sep='\t', index_col=0)
    results_df = pd.concat([existing_results_df, results_df], axis=1)

# Save the updated results
results_df.to_csv(predictions_file, sep='\t')
