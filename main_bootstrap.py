from pathlib import Path
from collections import Counter, defaultdict # Added defaultdict
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import time # Added for timing


from mapie.classification import MapieClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold,ShuffleSplit, train_test_split, StratifiedShuffleSplit, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample # Added for bootstrapping
from sklearn.calibration import CalibratedClassifierCV
from venn_abers import VennAbersCalibrator, VennAbers


def drop_correlate_feature(cor_matrix, df, threshold = 0.8):
    """Drops highly correlated features."""
    column_corr_sums = {}
    for col in cor_matrix.columns:
        column_corr_sums[col] = (cor_matrix[col]>threshold).sum() - 1  # Exclude self-correlation

    # Sort columns by their sum of correlations in descending order
    sorted_columns = sorted(column_corr_sums.items(), key=lambda x: x[1], reverse=True)

    # Iterate over the sorted columns and drop features based on the threshold
    features_to_drop = set() # Use set for faster lookups
    cols_to_check = list(cor_matrix.columns)

    for col1, _ in sorted_columns:
        if col1 in features_to_drop:
            continue
        for col2, _ in sorted_columns:
            if col1 == col2 or col2 in features_to_drop:
                continue
            # Check correlation only if it exists in the matrix (it should)
            if col1 in cor_matrix and col2 in cor_matrix[col1]:
                 if cor_matrix.loc[col1, col2] >= threshold:
                    # Keep the feature with lower sum of correlations if counts differ,
                    # otherwise keep the one encountered first (col2 in this loop structure)
                    if column_corr_sums[col1] > column_corr_sums[col2]:
                         features_to_drop.add(col1)
                         break # Move to the next col1
                    elif column_corr_sums[col1] < column_corr_sums[col2]:
                         features_to_drop.add(col2)
                    else:
                         # If correlation counts are equal, drop the one later in the sorted list (col1)
                         features_to_drop.add(col1)
                         break # Move to the next col1

    return df.drop(columns=list(features_to_drop)), list(features_to_drop)

def get_eso_data(
        eso_path, eso_feature_path, data_type, grade_cutpoint = 1):

    eso_df = pd.read_csv(eso_path)
    eso_df = eso_df.dropna()
    ####################### Get clinical features
    if 'eso' in str(eso_path):
        clinical_features = eso_df[['sex', 'age', 'surgery' ]]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(eso_df[['smoke' ]])
        smoke_onehot = enc.transform(eso_df[['smoke' ]]).toarray()
        smoke_onehot = pd.DataFrame(smoke_onehot, columns=enc.get_feature_names_out())
        clinical_features = pd.concat([clinical_features, smoke_onehot], axis=1)
    
    ##############################################

    eso_df['type']=0
    eso_features_df = pd.read_csv(eso_feature_path)

    eso_features_df = eso_features_df.drop(columns = [col for col in eso_features_df if 
                                                        ("gldm" in col) or
                                                        ("eud_dose" in col) or 
                                                        ("ntcp_dose" in col) or
                                                        ("shape" in col)])
    eso_features_df = eso_features_df.reset_index(drop = True)

    data = pd.DataFrame()
    for data_type in data_types:
            data = pd.concat([data, eso_features_df.drop(columns = [col for col in eso_features_df.columns if (data_type not in col)])], axis = 1)
    y = eso_df[['grade']]

    y.grade[y.grade<grade_cutpoint] = 0
    y.grade[y.grade>=grade_cutpoint] = 1
    return data, y.values.ravel()


def feature_processing(X, y, return_pandas = False, correlation_threshold = 0.8):

    X_numeric = deepcopy(X)

    max_d = X_numeric.max()
    min_d = X_numeric.min()

    X_norm_numeric = (X_numeric - min_d) / (max_d - min_d)
    X_norm_numeric = X_norm_numeric.dropna(axis='columns') 

    cor_matrix = X_norm_numeric.corr(method='spearman').abs()
    selected_numeric_data, _ = drop_correlate_feature(cor_matrix, X_norm_numeric, threshold = correlation_threshold)


    selected_data = selected_numeric_data
    selected_columns = selected_data.columns
    X_final = deepcopy(selected_data.values)
    y_copy = deepcopy(y)

    kept_numeric_cols = selected_numeric_data.columns
    return X_final, y_copy, max_d[kept_numeric_cols], min_d[kept_numeric_cols], selected_columns


def get_pvalue_conformal(uncertainty_score_cal, uncertainty_score_test, esp = 1e-6):
    uncertainty_score_cal = -np.log(uncertainty_score_cal + esp)
    uncertainty_score_test = -np.log(uncertainty_score_test + esp)
    results = uncertainty_score_cal >= uncertainty_score_test

    p_value_1 = (results.sum(axis=0)+1) / (results.shape[0]+1) 
    p_value_0 = 1 - p_value_1 
    p_values = np.vstack([p_value_0, p_value_1]).T 

    return p_values


def train_loo(X, y, all_models = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'penalty': ['l2'], 'C': [1,0.1,10, 100, 0.01], 'solver': ['lbfgs'],
                'class_weight': ['balanced'], 'max_iter': [500]
            }),
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': [3, 5, 10, 50, 100], 'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5], 'class_weight': ['balanced'],
            })
    },
    save_suffix = "", save_folder = 'training_results_exper',
    correlation_threshold = 0.8,
    n_bootstraps = 10 
    ):

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame for feature processing.")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("Input y must be a pandas Series or numpy array.")
    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=X.index) # Ensure y has same index as X for splitting


    leave_one_out = LeaveOneOut()
    n_splits = leave_one_out.get_n_splits(X)
    prob_results = defaultdict(lambda: defaultdict(list))
    uncertainty_results = defaultdict(lambda: defaultdict(list))
    data_results = defaultdict(dict)

    fold_counter = 0
    start_time = time.time()

    for train_index, test_index in leave_one_out.split(X, y):
        fold_counter += 1
        loo_test_subject_index = test_index[0] # The actual index in the original DataFrame
        print(f"\n--- (Test Subject Index: {loo_test_subject_index}) ---")

        # --- Data Splitting and Preprocessing for the Fold ---
        X_all_fold = deepcopy(X)
        y_all_fold = deepcopy(y)
        X_train_orig, X_test_orig = X_all_fold.iloc[train_index], X_all_fold.iloc[test_index]
        y_train_orig, y_test = y_all_fold.iloc[train_index], y_all_fold.iloc[test_index] # y_test is final

        # Store fold-specific data (once per fold)
        data_results[loo_test_subject_index]['X_test'] = X_test_orig
        data_results[loo_test_subject_index]['y_test'] = y_test.values # Store y_test numpy array

        ##################### Features selction
        _, _, max_d, min_d, selected_columns = feature_processing(
        X_train_orig, y_train_orig, correlation_threshold=correlation_threshold
        )

        X_test_processed = X_test_orig[selected_columns]
        X_test_processed = (X_test_processed-min_d)/(max_d-min_d)
        X_test_np = X_test_processed.values 
        ######################################


        # === BOOTSTRAP LOOP within the LOO fold ===
        print(f"Starting {n_bootstraps} bootstrap iterations...")
        for i_boot in range(n_bootstraps):
            print(f"  Bootstrap iteration {i_boot+1}/{n_bootstraps}")

            # Create bootstrap sample from the processed training data for this fold
            # Use resample on the processed DataFrame and Series to keep alignment
            X_train_boot, y_train_boot = resample(
                X_train_orig, y_train_orig,
                replace=True, # Sample with replacement
                n_samples=len(y_train_orig), # Keep original size
                random_state=(i_boot*941225)%18912561 # For reproducibility
            )
            # X_train_boot, y_train_boot = X_train_orig, y_train_orig

            X_train_boot = X_train_boot[selected_columns]
            X_train_boot = (X_train_boot-min_d)/(max_d-min_d)

            # Convert to numpy for sklearn functions
            X_train_boot_np = X_train_boot.values
            y_train_boot_np = y_train_boot.values
            
            # #############! remove this to train bootstrap
            # X_train_boot_np = deepcopy(X_train_orig[selected_columns])
            # X_train_boot_np = (X_train_boot_np-min_d)/(max_d-min_d)
            # X_train_boot_np = X_train_boot_np.values 
            # y_train_boot_np = deepcopy(y_train_orig.values)
            # ####################!

            X_cal = X_train_boot_np
            y_cal = y_train_boot_np

            # --- Train models on the current bootstrap sample ---
            for model_name, (_model_class, param_grid) in all_models.items():
                #! check XGBoost
                if model_name == 'XGBoost':
                    weight = (y_train_boot_np == 0).sum() / (y_train_boot_np == 1).sum()
                    param_grid['scale_pos_weight'] = [weight]

                # Perform GridSearchCV on the bootstrap sample
                grid_search = GridSearchCV(estimator=_model_class(), param_grid=param_grid, cv=3, scoring = 'balanced_accuracy', n_jobs=-1) # Use multiple cores if available
                grid_search.fit(X_train_boot_np, y_train_boot_np)

                # Train the best model on the *entire* bootstrap sample
                model = _model_class(**grid_search.best_params_)
                clf = model.fit(X_train_boot_np, y_train_boot_np) # Train on bootstrap

                # --- Predict ---
                pred = clf.predict_proba(X_test_np) # Shape (1, n_classes)
                pred_cal_boot = clf.predict_proba(X_cal) # Predictions on the data used for training clf

                prob_results[model_name][loo_test_subject_index].append(pred[0])
                uncertainty_results[model_name][loo_test_subject_index].append(pred[0])

                ############################ Sigmoid Calibration
                calibrated_model_sig = CalibratedClassifierCV(deepcopy(clf), method='sigmoid', cv='prefit')
                # Fit calibrator using the bootstrap data
                calibrated_model_sig.fit(X_cal, y_cal)
                pred_calibrated_sig = calibrated_model_sig.predict_proba(X_test_np)

                prob_results[f"{model_name}_sigmoid"][loo_test_subject_index].append(pred_calibrated_sig[0])
                uncertainty_results[f"{model_name}_sigmoid"][loo_test_subject_index].append(pred_calibrated_sig[0])

                ############################ Isotonic Calibration 
                calibrated_model_iso = CalibratedClassifierCV(deepcopy(clf), method='isotonic', cv='prefit')
                    # Fit calibrator using the bootstrap data
                calibrated_model_iso.fit(X_cal, y_cal)
                pred_calibrated_iso = calibrated_model_iso.predict_proba(X_test_np)

                prob_results[f"{model_name}_isotonic"][loo_test_subject_index].append(pred_calibrated_iso[0])
                uncertainty_results[f"{model_name}_isotonic"][loo_test_subject_index].append(pred_calibrated_iso[0])
                ############################ Venn-Abers Calibration
                va = VennAbersCalibrator()
                # Use predictions on bootstrap train set (pred_cal_boot) as calibration scores
                pred_calibrated_va, p0p1 = va.predict_proba(
                    p_cal=pred_cal_boot, y_cal=y_cal,
                    p_test=pred, # Use raw prediction on test set
                    p0_p1_output=True
                )
                # Store the base classifier, VA is applied during prediction

                prob_results[f"{model_name}_venn_abers"][loo_test_subject_index].append(pred_calibrated_va[0]) # p0p1 interval [p0, p1]
                # Use the interval width or p1 as uncertainty measure
                uncertainty_results[f"{model_name}_venn_abers"][loo_test_subject_index].append(p0p1[0]) # Store [p0, p1] as uncertainty
                ############################# Conformal Prediction (using simple 
                cal_scores_cp = pred_cal_boot[:, 1] # P(class 1) for calibration set (bootstrap train)
                test_scores_cp = pred[:, 1]       # P(class 1) for test set

                # Calculate p-values for class 0 and class 1
                pvalues = get_pvalue_conformal(cal_scores_cp, test_scores_cp) # Shape (1, 2)

                prob_results[f"{model_name}_conformal"][loo_test_subject_index].append(pred[0])
                # Store p-values [p0, p1] as uncertainty measure
                uncertainty_results[f"{model_name}_conformal"][loo_test_subject_index].append(pvalues[0])

    print("\nSaving results...")
    output_folder = Path(save_folder)
    output_folder.mkdir(parents=True, exist_ok=True) 


    prob_results_final = {k: dict(v) for k, v in prob_results.items()}
    uncertainty_results_final = {k: dict(v) for k, v in uncertainty_results.items()}
    data_results_final = dict(data_results)

    with open(output_folder / f'loo_boot_prob_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(prob_results_final, f)

    with open(output_folder / f'loo_boot_data_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(data_results_final, f)

    with open(output_folder / f'loo_boot_uncertainty_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(uncertainty_results_final, f)


    print(f"Results saved to {output_folder}")

    return prob_results_final, uncertainty_results_final # Optionally return results


if __name__ == '__main__':
    np.random.seed(941225)

    correlation_threshold = 0.8
    correlated_all = True 
    save_folder = 'training_results_paper_boot' # Changed folder name
    save_suffix="_eso_noclinical_boot" 
    data_types = ['dosimetric', 'dosiomic', 'radiomic'] 
    n_bootstraps_main = 100 
    eso_path = Path(r"data.csv")
    eso_feature_path = Path(r"final_features_eso_15mm.csv")


    print("Loading data...")
    X, y = get_eso_data(
        eso_path, eso_feature_path,data_types,
        grade_cutpoint=1
    )

    print("\nData dimensions before LOO:")
    print(f'X shape: {X.shape}')
    print(f'y counts: {Counter(y)}')

    all_models = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'penalty': ['l2', 'l1'], 
                'C': [1,0.1, 0.01, 10, 100],
                'solver': ['liblinear'], 
                'class_weight': ['balanced'], 
                'max_iter': [2000]
            }),
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': [5, 20, 50],
                'min_samples_leaf': [2, 4],
                'min_samples_split': [2, 4],
                'class_weight': ['balanced'], 
            }),
        'SVM': (
            SVC,
            {
                'C': [1, 0.1, 10, 0.01],
                'kernel': ['rbf'],
                'gamma': ['scale'],
                'class_weight': ['balanced'],
                'probability': [True],
            }),
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': [5, 20, 50],
                'learning_rate': [0.1, 0.01],
                'subsample': [0.5, 1],
                'scale_pos_weight': [1], 
            }),

    }

    print(f"\nStarting LOO training with {n_bootstraps_main} bootstraps per fold...")
    train_loo(
        X, y,
        all_models = all_models,
        save_suffix=save_suffix,
        save_folder=save_folder,
        correlation_threshold=correlation_threshold, # Pass the threshold
        n_bootstraps=n_bootstraps_main # Pass the number of bootstraps
    )

    print("\n--- Script Finished ---")