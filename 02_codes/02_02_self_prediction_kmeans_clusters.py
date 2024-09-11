import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np

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

# Directories and filenames
input_kmeans_dir = '../03_results/out_genomic_clusters'
target_vars_filename = 'kmeans_results.tsv'
target_vars_path = os.path.join(input_kmeans_dir, target_vars_filename)
biological_data_dir = '../01_data/01_biological_data/'
results_filename = '../03_results/out_predictions/self_predictions_kmeans.tsv'

# Load the kmeans target variables
target_vars = pd.read_csv(target_vars_path, sep='\t', index_col=0)

# Define valid matrix types and minimum cluster number m
valid_matrix_types = ['M1', 'guidi', 'salazar', 'stress']

# Function to filter valid columns based on matrix type and m >= 5
def filter_columns(column_name):
    parts = column_name.split('_')
    matrix_type = parts[0]
    cluster_number = int(parts[-1].replace('kmeans', ''))
    return matrix_type in valid_matrix_types and cluster_number >= 5

# Apply the filter to get the relevant target variables
filtered_target_vars = target_vars.loc[:, target_vars.columns.map(filter_columns)]

# XGBoost Classifier setup
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Cross-validation setup
n_splits = 3
n_repeats = 1
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

# Initialize label encoder
le = LabelEncoder()

# Initialize results DataFrame
results_df = pd.DataFrame()

# Function to load and align predictors with target variables
def load_and_align_predictor(matrix_type, subsample):
    predictor_file = f"Matrix_world_GEN_{matrix_type}_{subsample}.tsv"
    predictor_path = os.path.join(biological_data_dir, predictor_file)
    
    # Load predictor matrix
    if os.path.exists(predictor_path):
        predictor_matrix = pd.read_csv(predictor_path, sep='\t', index_col=0)
        return clr_(predictor_matrix)
    else:
        print(f"Warning: Predictor file {predictor_file} not found!")
        return None

# Function to evaluate model using cross-validation
def evaluate_model(X, y, n_splits=5):
    # Perform cross-validation
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro')
    }
    
    # Cross-validate
    cv_results = cross_validate(xgb_classifier, X, y, cv=rskf, scoring=scoring, return_train_score=False)
    
    # Get average metrics
    avg_accuracy = np.mean(cv_results['test_accuracy'])
    avg_f1_macro = np.mean(cv_results['test_f1_macro'])
    
    return avg_accuracy, avg_f1_macro

# Iterate over each filtered target variable
for col in filtered_target_vars.columns:
    # Extract matrix_type and subsample from the column name
    parts = col.split('_')
    matrix_type, subsample = parts[0], parts[1]
    
    # Load the corresponding predictor matrix
    predictor_matrix = load_and_align_predictor(matrix_type, subsample)
    
    if predictor_matrix is not None:
        # Drop NaN values from the target variable
        target_col = filtered_target_vars[col].dropna()
        
        # Align samples by finding the common samples between predictors and target variables
        common_samples = target_col.index.intersection(predictor_matrix.index)
        aligned_target = target_col.loc[common_samples]
        aligned_predictors = predictor_matrix.loc[common_samples]
        
        if not aligned_target.empty and not aligned_predictors.empty:
            # Encode the target variable
            y_encoded = le.fit_transform(aligned_target)
            
            # Handle class imbalance and ensure each class has at least 'n_splits' samples
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            X_resampled = aligned_predictors.copy()
            y_resampled = y_encoded.copy()

            for cls, count in zip(unique_classes, class_counts):
                if count < n_splits:
                    diff = n_splits - count
                    cls_indices = np.where(y_encoded == cls)[0]
                    indices_to_duplicate = np.random.choice(cls_indices, diff, replace=True)
                    X_resampled = pd.concat([X_resampled, aligned_predictors.iloc[indices_to_duplicate]], axis=0)
                    y_resampled = np.concatenate([y_resampled, y_encoded[indices_to_duplicate]])

            # Perform cross-validation and calculate metrics
            avg_accuracy, avg_f1_macro = evaluate_model(X_resampled, y_resampled)

            # Store results
            results_df.at[col, 'accuracy'] = avg_accuracy
            results_df.at[col, 'f1_macro'] = avg_f1_macro

# Save the results to a file
results_df.to_csv(results_filename, sep='\t')

print(f"Results saved to {results_filename}")
