import pickle
from itertools import cycle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mapie.metrics as mapie_metrics


from sklearn import metrics
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

calibration_name_convert = {
    'sigmoid': 'PS',
    'isotonic': 'IR',
    'venn_abers': 'VAs',
    'conformal': 'CP',
    'conformal_cal': 'CP-PS'
}
metric_name_convert = {
    'accuracy_': 'ACC',
    'f1': 'F1 Score',
    'precision': 'Precision',
    'coverage': 'Coverage',
    'balanced_accuracy_': 'BACC',
    'auc_': 'AUROC',
    'prauc_': 'AUPRC',
}

model_name_convert = {
    'LogisticRegression': 'LR',
    'SVM': 'SVM',
    'XGBoost': 'XGB',
    'RandomForest': 'RF',
}


def calculate_metrics(pred, pred_prob, y_true, suffix = ""):
    metrics_dict = {
        f'accuracy_{suffix}': np.nan,
        f'precision_{suffix}': np.nan,
        f'recall_{suffix}': np.nan,
        f'f1_{suffix}': np.nan,
        f'balanced_accuracy_{suffix}': np.nan,
        f'specificity_{suffix}': np.nan,
        f'auc_{suffix}': np.nan,
        f'prauc_{suffix}': np.nan
    }
    if len(y_true) == 0:
        return metrics_dict



    metrics_dict[f'accuracy_{suffix}'] = metrics.accuracy_score(y_true, pred)
    metrics_dict[f'precision_{suffix}'] = metrics.precision_score(y_true, pred)
    metrics_dict[f'recall_{suffix}'] = metrics.recall_score(y_true, pred)
    metrics_dict[f'f1_{suffix}'] = metrics.f1_score(y_true, pred)
    metrics_dict[f'balanced_accuracy_{suffix}'] = metrics.balanced_accuracy_score(y_true, pred)
    metrics_dict[f'specificity_{suffix}'] = metrics.recall_score(y_true, pred, pos_label=0)
    try:
        metrics_dict[f'auc_{suffix}'] = metrics.roc_auc_score(y_true, pred_prob)
    except ValueError as e:
        print(e)
        metrics_dict[f'auc_{suffix}'] = np.nan
    
    try:
        precision, recall, threshold = metrics.precision_recall_curve(y_true, pred_prob)
        metrics_dict[f'prauc_{suffix}'] = metrics.auc(recall, precision)
    except ValueError:
        metrics_dict[f'prauc_{suffix}'] = np.nan
    
    

    return metrics_dict

def cal_uncertainty_score(uncertainty, pred_prob):

    return 1 - uncertainty[np.array(range(len(uncertainty))).astype(int), pred_prob.argmax(axis=1).astype(int)]

def get_coverage(y_test, proba, uncertainty, pred = None):
    all_metrics = []
    coverages = []
    
    if pred is None:
        pred = proba.argmax(axis=1)

    uncertainty_score = cal_uncertainty_score(uncertainty, proba)

    coverage = 1

    all_metrics.append(calculate_metrics(pred, proba[:,1], y_test))
    coverages.append(coverage)

    for idx in uncertainty_score.argsort()[::-1]:
        proba_remove = uncertainty_score[idx]
        index_remove = np.where(uncertainty_score > proba_remove)[0]

        proba_test = np.delete(proba[:,1], index_remove, axis=0)
        pred_test = np.delete(pred, index_remove, axis=0)
        y_test_test = np.delete(y_test, index_remove, axis=0)
        coverage = len(proba_test)/len(uncertainty)

        if coverage not in coverages:
            coverages.append(coverage)
            all_metrics.append(calculate_metrics(pred_test, proba_test, y_test_test))

    keys = all_metrics[0].keys()
    new_data = {key: [] for key in keys}

    # Iterate through the list of dictionaries and append values to the corresponding lists
    for metric in all_metrics:
        for key, value in metric.items():
            new_data[key].append(value)
    
    return np.array(coverages), new_data



def test_coverage(axs,
        y_true, prob_results, uncertainty_results, metric_name, 
        model_names= ['LogisticRegression','SVM','XGBoost', 'RandomForest'], 
        coverages_threshold = 0.1, suffix = "", save_folder = 'results_exper', use_orignal_pred = False, row = 0):

    lines = ["-","--","-.",":"]
    for i, model_name in enumerate(model_names):

        linecycler = cycle(lines)
        for key, value in uncertainty_results.items():

            if 'conformal' in key.lower():
                use_orignal_pred = True
            if model_name.lower() not in key.lower():
                continue

            if use_orignal_pred:
                pred = np.array(prob_results[key]).argmax(axis=1)
                # pred = np.array(value).argmax(axis=1)
                coverages, metrics = get_coverage(y_true, np.array(prob_results[key]), np.array(value), pred)
            else:
                coverages, metrics = get_coverage(y_true, np.array(prob_results[key]), np.array(value))

            axs[row][i].plot(coverages[coverages>=coverages_threshold], np.array(metrics[metric_name])[coverages>=coverages_threshold],next(linecycler), label=key)

        if row == 0:
            axs[row][i].set_title(model_name_convert[model_name], color='black', fontsize=10)
        if row == 1:
            axs[row][i].set_xlabel('Coverage', color='black', fontsize=10)
        if i == 0:
            axs[row][i].set_ylabel(metric_name_convert[metric_name], color='black', fontsize=10)
        axs[row][i].tick_params(axis='both', colors='black', labelsize=10)
        axs[row][i].set_ylim([0.5, 1])


def test_threshold(y_true, prob_results, uncertainty_results, 
        model_names= ['logistic','svm','xgboost', 'forest'], 
        uncertainty_thresholds = 0.8):
    model_metrics = {}
    for i, model_name in enumerate(model_names):
        model_metrics[model_name] = {}
        for key, value in uncertainty_results.items():
 
            if model_name.lower() not in key.lower():
                continue
            prob = np.array(prob_results[key])
            uncertainty = np.array(value)
            certainty = 1-cal_uncertainty_score(uncertainty, prob)


            pred = prob.argmax(axis=1)
            model_metrics[model_name][key] = calculate_metrics(pred[certainty>= uncertainty_thresholds], prob[certainty>= uncertainty_thresholds][:,1], y_true[certainty>= uncertainty_thresholds])
            model_metrics[model_name][key].update({'coverage': len(pred[certainty>= uncertainty_thresholds])/len(pred)})
    return model_metrics

def test_all_threshold(y_true, prob_results, uncertainty_results,
                selected_columns = ['coverage', 'balanced_accuracy_'],
                selected_thresholds = [0.5, 0.8, 0.9]):
    
    df_all_threshold = None
    for prob_threshold in selected_thresholds:
        model_metrics = test_threshold(y_true, prob_results, uncertainty_results, uncertainty_thresholds = prob_threshold)
        df_specific_threshold = None
        for key, value in model_metrics.items():
            if df_specific_threshold is None:
                df_specific_threshold = pd.DataFrame(value).T[selected_columns]
            else:
                df_specific_threshold = pd.concat([df_specific_threshold, pd.DataFrame(value).T[selected_columns]])
        if df_all_threshold is None:
            df_all_threshold = df_specific_threshold
        else:
            df_all_threshold = pd.concat([df_all_threshold, df_specific_threshold], axis=1)
    
    model_names = []
    calibration_methods = []
    for name in df_specific_threshold.index:
        model_name = name.split('_')[0]
        if len(name.split('_'))>1:
            calibration_method = '_'.join(name.split('_')[1:])
            calibration_method = calibration_name_convert[calibration_method]
        else:
            calibration_method = "UC"
        model_names.append(model_name_convert[model_name])
        calibration_methods.append(calibration_method)

    df_all_threshold['Model'] = model_names
    df_all_threshold['Uncertainty Method'] = calibration_methods

    df_all_threshold.set_index(['Model', 'Uncertainty Method'], inplace=True)

    columns_names = [(f"Cutpoint {thres}", metric_name_convert[metric]) for thres in selected_thresholds for metric in selected_columns]

    df_all_threshold.columns = pd.MultiIndex.from_tuples(columns_names)
    return df_all_threshold

if __name__ == '__main__':
    suffix = '_eso_noclinical'
    train_folder = 'training_results_paper'
    save_folder = 'results_paper'

    with open(f'{train_folder}/loo_trained_models{suffix}.pkl', 'rb') as f:
        models = pickle.load(f)

    # Load the data
    with open(f'{train_folder}/loo_data_results{suffix}.pkl', 'rb') as f:
        datasets = pickle.load(f)

    with open(f'{train_folder}/loo_prob_results{suffix}.pkl', 'rb') as f:
        prob_results = pickle.load(f)

    with open(f'{train_folder}/loo_uncertainty_results{suffix}.pkl', 'rb') as f:
        uncertainty_results = pickle.load(f)

    y_true = []
    x_test = []
    y_train = []
    for key, value in datasets.items():
        y_true.append(value['y_test'])
        x_test.append(value['X_test'])
        y_train.append(value['y_train'])
    y_true  = np.array(y_true)
    y_true = y_true.reshape(-1)

    y_train = np.array(y_train)
    print(y_train.shape)

    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, x_test.shape[-1])
    print(y_true.shape, x_test.shape)

    coverages_threshold = 0.1

    # # ##############################
    results = test_all_threshold(y_true, prob_results, uncertainty_results, selected_columns=['coverage', 'accuracy_', 'auc_', 'prauc_'], selected_thresholds=[0.0, 0.8, 0.9])
    results.to_csv(f'{save_folder}/coverage_threshold{suffix}.csv')
    # # #############################
    test_metrics = ['auc_', 'prauc_']
    fig, axes = plt.subplots(len(test_metrics), 4, figsize=(12,5))
    for i, test_metric in enumerate(test_metrics):
        test_coverage(axes,
            y_true, prob_results, uncertainty_results,
            test_metric, ['LogisticRegression','SVM','XGBoost', 'RandomForest'], 
            coverages_threshold = coverages_threshold, 
            suffix=suffix+test_metric, save_folder=save_folder, row = i)
        
    handles, labels = axes.flat[0].get_legend_handles_labels()
    calibration_methods = []
    for name in labels:
        model_name = name.split('_')[0]
        if len(name.split('_'))>1:
            calibration_method = '_'.join(name.split('_')[1:])
            calibration_method = calibration_name_convert[calibration_method]
        else:
            calibration_method = "UC"
        calibration_methods.append(calibration_method)
    fig.legend(handles, calibration_methods)
    fig.tight_layout()

    plt.savefig(f'{save_folder}/coverage{suffix}.png', dpi = 800)
    # #############################
