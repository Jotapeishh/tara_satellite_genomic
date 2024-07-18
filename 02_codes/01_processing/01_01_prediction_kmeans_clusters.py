import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import xgboost as xgb

from glob import glob
import os

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

input_sat_dir = '../../01_data/02_satellite_data_processed'
predictor_files = sorted(glob(os.path.join(input_sat_dir, 'matrix_tara_world_adj_grids_*.tsv')))


input_kmeans_dir = '../../03_results/out_genomic_clusters'
target_vars_filename = 'kmeans_results.tsv'
target_vars_path = os.path.join(input_kmeans_dir, target_vars_filename)

target_vars = pd.read_csv(target_vars_path, sep='\t', index_col=0)
target_vars = target_vars.map(lambda x: f"C{x}")
#target_vars.head()

results_df = pd.DataFrame(index=[os.path.basename(file) for file in predictor_files], columns=target_vars.columns)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred, average='macro')
    #precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    #roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
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
    # Nombre del archivo como identificador
    file_name = os.path.basename(file)
    
    # Cargar el predictor
    df = pd.read_csv(file, sep='\t', index_col=0)
    
    # Alinear los predictores con las muestras de target_vars
    aligned_predictor = df.loc[df.index.intersection(target_vars.index)]
    
    for target_column in target_vars.columns:
        X = aligned_predictor
        y = target_vars.loc[aligned_predictor.index, target_column]

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

        #cv_results = cross_validate(model, X, y_encoded, cv=rskf, scoring=scoring, return_train_score=False)
        cv_results = cross_validate(model, X_resampled, y_resampled, cv=rskf, scoring=scoring, return_train_score=False)

        avg_accuracy = np.mean(cv_results['test_accuracy'])
        avg_f1_macro = np.mean(cv_results['test_f1_macro'])

        results_df.at[file_name, target_column] = (avg_accuracy, avg_f1_macro)
            
print(results_df)

results_df.to_csv('../../03_results/out_predictions/predictions_kmeans.tsv', sep='\t')