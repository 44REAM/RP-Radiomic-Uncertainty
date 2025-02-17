from pathlib import Path
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy




from mapie.classification import MapieClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold,ShuffleSplit, train_test_split, StratifiedShuffleSplit, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.calibration import CalibratedClassifierCV
from venn_abers import VennAbersCalibrator, VennAbers


def drop_correlate_feature(cor_matrix, df, threshold = 0.8):
    column_corr_sums = {}
    for col in cor_matrix.columns:
        column_corr_sums[col] = (cor_matrix[col]>threshold).sum() - 1  # Exclude self-correlation

    # Sort columns by their sum of correlations in descending order
    sorted_columns = sorted(column_corr_sums.items(), key=lambda x: x[1], reverse=True)

    # Iterate over the sorted columns and drop features based on the threshold
    features_to_drop = []

    for col1, _ in sorted_columns:
        for col2, _ in sorted_columns:
            if col1 != col2 and cor_matrix.loc[col1, col2] >= threshold:
                if (col1 not in features_to_drop) and col2 not in features_to_drop:
                    features_to_drop.append(col1)

    return df.drop(columns=features_to_drop)


def get_eso_data(
        eso_path, eso_feature_path, data_types = ['dosiomic', 'dosimetric', 'radiomic'], 
        correlation_threshold = 0.8, correlated_all = True, grade_cutpoint = 1,
        clinical = True, return_pandas = False):

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
    print(eso_features_df.shape)

    eso_features_df = eso_features_df.drop(columns = [col for col in eso_features_df if 
                                                        ("gldm" in col) or
                                                        ("eud_dose" in col) or 
                                                        ("ntcp_dose" in col) or
                                                        ("shape" in col)])
    eso_features_df = eso_features_df.reset_index(drop = True)

    y = eso_df[['grade']]

    y.grade[y.grade<grade_cutpoint] = 0
    y.grade[y.grade>=grade_cutpoint] = 1

    scaler = StandardScaler()
    # X_norm = scaler.fit_transform(eso_features_df)
    # X_norm = pd.DataFrame(X_norm, columns=eso_features_df.columns)
    X_norm = (eso_features_df-eso_features_df.min())/(eso_features_df.max()-eso_features_df.min())
    X_norm = X_norm.dropna(axis='columns')

    datas = {}

    if correlated_all:
        data = pd.DataFrame()
        for data_type in data_types:
            data = pd.concat([data, X_norm.drop(columns = [col for col in X_norm if (data_type not in col)])], axis = 1)

        cor_matrix = data.corr(method='spearman').abs()
        selected_data = drop_correlate_feature(cor_matrix, data, threshold = correlation_threshold)

        if return_pandas:
            X = deepcopy(selected_data)
            if clinical:
                X = pd.concat([selected_data, clinical_features], axis=1)
            y_copy = deepcopy(y)
            return X, y_copy
        
        else:
            X = deepcopy(selected_data.values)
            if clinical:
                X = np.concatenate((X, clinical_features.values), axis=1)
            y_copy = deepcopy(y.values.ravel())
            return X, y_copy

    else:
        for data_t in data_type:
            data = X_norm.drop(columns = [col for col in X_norm if (data_t  not in col)])
            cor_matrix = data.corr(method='spearman').abs()
            dosiomic_drop = drop_correlate_feature(cor_matrix, data, threshold = correlation_threshold)
            X = deepcopy(dosiomic_drop.values)
            y_copy = deepcopy(y.values.ravel())
            datas[data_t] = {'X':X, 'y':y_copy}
        if clinical:
            datas['clinical'] = {'X':clinical_features.values, 'y':y.values.ravel()}
    return datas

def train_loo(X, y, all_models = {
        'LogisticRegression': (
            LogisticRegression, 
            {
                'penalty': ['l2'],
                'C': [1,0.1,10, 100, 0.01],
                'solver': ['lbfgs'],
                # 'l1_ratio': [0, 0.5, 1],
                'class_weight': ['balanced'],
                'max_iter': [500]
            }),
        'RandomForest': (
            RandomForestClassifier, 
            {
                'n_estimators': [3, 5, 10, 50, 100],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced'],
            }
            )
    }, save_suffix = "", save_folder = 'training_results_exper'):

    leave_one_out = LeaveOneOut()
    leave_one_out.get_n_splits(X)
    prob_results = {}
    uncertainty_results = {}
    model_results = {}
    data_results = {}


    for train_index, test_index in leave_one_out.split(X):
        print(f"Training fold {test_index[0]}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  
        # Corrected for correct y_test usage

        for model_name, (_model_class, param_grid) in all_models.items():
            model_results[test_index[0]] = {}
            # Perform GridSearchCV with Stratified K-Fold
            grid_search = GridSearchCV(estimator=_model_class(), param_grid=param_grid, cv=3, scoring = 'balanced_accuracy')
            grid_search.fit(X_train, y_train)

            model = _model_class(**grid_search.best_params_)
            clf = model.fit(X_train, np.ravel(y_train))
            model_results[test_index[0]][model_name] = clf
            pred = clf.predict_proba(X_test)

            if model_name not in prob_results.keys():
                prob_results[model_name] = []
            prob_results[model_name].append(pred[0])

            if model_name not in uncertainty_results.keys():
                uncertainty_results[model_name] = []
            uncertainty_results[model_name].append(pred[0])

            ########################### Calibrate sigmoid
            calibrated_model = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_train, y_train)  # Calibration fit
            model_results[test_index[0]][f"{model_name}_sigmoid"] = calibrated_model
            pred_calibrated = calibrated_model.predict_proba(X_test)

            if f"{model_name}_sigmoid" not in prob_results.keys():
                prob_results[f"{model_name}_sigmoid"] = []
            prob_results[f"{model_name}_sigmoid"].append(pred_calibrated[0])

            if f"{model_name}_sigmoid" not in uncertainty_results.keys():
                uncertainty_results[f"{model_name}_sigmoid"] = []
            uncertainty_results[f"{model_name}_sigmoid"].append(pred_calibrated[0])

            ########################### Calibrate isotonic
            calibrated_model = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            calibrated_model.fit(X_train, y_train)
            model_results[test_index[0]][f"{model_name}_isotonic"] = calibrated_model
            pred_calibrated = calibrated_model.predict_proba(X_test)

            if f"{model_name}_isotonic" not in prob_results.keys():
                prob_results[f"{model_name}_isotonic"] = []
            prob_results[f"{model_name}_isotonic"].append(pred_calibrated[0])

            if f"{model_name}_isotonic" not in uncertainty_results.keys():
                uncertainty_results[f"{model_name}_isotonic"] = []
            uncertainty_results[f"{model_name}_isotonic"].append(pred_calibrated[0])

            ########################### Calibrate VENN ABERS and p0p1
            model_results[test_index[0]][f"{model_name}_venn_abers"] = clf
            va = VennAbersCalibrator()
            p_cal = clf.predict_proba(X_train)
            p_test = clf.predict_proba(X_test)
            pred_calibrated, p0p1 = va.predict_proba(p_cal=p_cal, y_cal=y_train, p_test=p_test, p0_p1_output = True)

            if f"{model_name}_venn_abers" not in prob_results.keys():
                prob_results[f"{model_name}_venn_abers"] = []
            prob_results[f"{model_name}_venn_abers"].append(pred_calibrated[0])

            if f"{model_name}_venn_abers" not in uncertainty_results.keys():
                uncertainty_results[f"{model_name}_venn_abers"] = []
            uncertainty_results[f"{model_name}_venn_abers"].append(pred_calibrated[0])

            # ########################### Calibrate p0p1


            # prob_results[f"{model_name}_venn_abers_p0p1"] = []
            # model_results[test_index[0]][f"{model_name}_venn_abers_p0p1"] = clf
            # prob_results[f"{model_name}_venn_abers_p0p1"].append(p0p1[0])

            ############################ Conformal Prediction

            model_results[test_index[0]][f"{model_name}_conformal"] = clf
            p_cal = clf.predict_proba(X_train)

            # mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="lac")
            # mapie_score.fit(X_train, y_train)
            pvalues = get_pvalue_conformal(p_cal[:, 1], pred[:, 1])

            if f"{model_name}_conformal" not in prob_results.keys():
                prob_results[f"{model_name}_conformal"] = []
            prob_results[f"{model_name}_conformal"].append(pred[0])

            if f"{model_name}_conformal" not in uncertainty_results.keys():
                uncertainty_results[f"{model_name}_conformal"] = []
            uncertainty_results[f"{model_name}_conformal"].append(pvalues[0])

            


        data_results[test_index[0]] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    
    with open(f'{save_folder}/loo_trained_models{save_suffix}.pkl', 'wb') as f:
        pickle.dump(model_results, f)
    
    with open(f'{save_folder}/loo_prob_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(prob_results, f)

    with open(f'{save_folder}/loo_data_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(data_results, f)

    with open(f'{save_folder}/loo_uncertainty_results{save_suffix}.pkl', 'wb') as f:
        pickle.dump(uncertainty_results, f)

def get_pvalue_conformal(uncertainty_score_train, uncertainty_score_test, esp = 1e-6):
    uncertainty_score_train = -np.log(uncertainty_score_train + esp)
    uncertainty_score_test = -np.log(uncertainty_score_test + esp)
    # this will got boolean array of shape (n_train, n_test )
    results = uncertainty_score_train >= uncertainty_score_test

    # plus one for test set also in conformal set
    results_1 = (results.sum(axis=0)+1) / (results.shape[0]+1)

    results_0 = 1-results_1
    results = np.vstack([results_0, results_1]).T
    return results

if __name__ == '__main__':
    np.random.seed(941225)
    # X_dosiomic = data['dosiomic']['X']
    # X_radiomic = data['radiomic']['X']
    # X = np.concatenate((X_dosiomic, X_radiomic), axis=1)
    # y = data['dosiomic']['y']

    correlation_threshold = 0.8
    correlated_all = True
    save_folder = 'training_results_paper'

    eso_path = Path(r"F:\Research backup\radio\Data\esophagus\eso_numpy_15mm\data.csv")
    eso_feature_path  = Path(r"F:\Research backup\radio\Data\final_features_eso_15mm.csv")
    data_types = ['dosimetric']
    X, y = get_eso_data(
        eso_path, eso_feature_path, 
        data_types = data_types, 
        correlation_threshold=correlation_threshold, 
        correlated_all=correlated_all, 
        grade_cutpoint=1, clinical = False)
    
    counter = Counter(y)
    scale_pos_weight = counter[0]/counter[1]
    print(counter)
    all_models = {
        'LogisticRegression': (
            LogisticRegression, 
            {
                'penalty': ['l2', 'l1'],
                'C': [1,0.1,10, 100, 0.01],
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
                'probability': [True]
            }),
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': [5, 20, 50],
                'learning_rate': [0.1, 0.01],
                'subsample': [0.5, 1],
                'scale_pos_weight': [scale_pos_weight],
            }),
        
    }
    train_loo(X, y, all_models = all_models, save_suffix="_eso_dosimetric",save_folder = save_folder)

    # eso_path = Path(r"F:\Research backup\radio\Data\lung\lung_numpy\data.csv")
    # eso_feature_path  = Path(r"F:\Research backup\radio\Data\final_features_lung_15mm.csv")
    # X, y = get_eso_data(eso_path, eso_feature_path, correlation_threshold=correlation_threshold, correlated_all=correlated_all, grade_cutpoint=2, clinical = False)
    # train_loo(X, y, all_models = all_models, save_suffix="_lung")

