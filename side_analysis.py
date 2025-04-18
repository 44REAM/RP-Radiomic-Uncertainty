import pickle
import numpy as np
import pickle
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
import sklearn.calibration as calibration
import mapie.metrics as mapie_metrics
import sklearn.metrics as metrics
from sklearn.calibration import CalibrationDisplay
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_bootstrap import reformat_results

def cal_metric(y_true, prob_results, model_names=['LogisticRegression', 'SVM', 'XGBoost', 'RandomForest'], metric='ece'):
    all_result = {}

    for i, model_name in enumerate(model_names):
        all_result[model_name] = {}
        for key, value in prob_results.items():

            if model_name.lower() not in key.lower():
                continue
            prob = np.array(value)

            pred = prob.argmax(axis=1)
            class_0 = np.zeros_like(y_true)
            class_0[pred == 0] = 1

            # this should be adaptive calibration error
            if metric == 'ece':
                ece_0 = mapie_metrics.expected_calibration_error(
                    class_0, prob[:, 0], num_bins=10, split_strategy='quantile')
                ece_1 = mapie_metrics.expected_calibration_error(
                    y_true, prob[:, 1], num_bins=10, split_strategy='quantile')
                results = (ece_0 + ece_1)/2
            elif metric == 'acc':
                results = metrics.accuracy_score(y_true, pred)
            elif metric == 'auroc':
                results = metrics.roc_auc_score(y_true, prob[:, 1])
            elif metric == 'auprc':
                precision, recall, threshold = metrics.precision_recall_curve(
                    y_true, prob[:, 1])
                results = metrics.auc(recall, precision)

            all_result[model_name][key] = results
    return all_result
def combine_metric(results):
    new_results = defaultdict(lambda: defaultdict(list))

    for boot in results:
        for model in boot:
            for method in boot[model]:
                    new_results[model][method].append(boot[model][method])
    return new_results

def get_mean_pvalue_compare_touc(results):
    mean = defaultdict(lambda: defaultdict(list))
    pvalue = defaultdict(lambda: defaultdict(list))
    for model in results:
        uc = results[model][model]
        for method in results[model]:
            if method == model:
                continue
            mean[model][method] = np.array(results[model][method]) - np.array(uc)
            se = np.std(mean[model][method], ddof=1) / np.sqrt(len(mean[model][method]))

            mean[model][method] = mean[model][method].mean()

            t = mean[model][method] / se
            pvalue[model][method] = 2 * stats.t.cdf(-abs(t), df=len(uc)-1)

    return mean, pvalue

def get_mean_pvalue_compare_to_custom(main_ersults, subtract_results):
    mean_diff = defaultdict(lambda: defaultdict(list))
    pavalue_diff = defaultdict(lambda: defaultdict(list))
    for model in subtract_results:
        for method in subtract_results[model]:
            if method != model:
                continue
            mean_diff[model][method] = np.array(main_ersults[model][method]) - np.array(subtract_results[model][model])
            se = np.std(mean_diff[model][method], ddof=1) / np.sqrt(len(mean_diff[model][method]))
            mean_diff[model][method] = mean_diff[model][method].mean()
            t = mean_diff[model][method] / se
            pavalue_diff[model][method] = 2 * stats.t.cdf(-abs(t), df=len(subtract_results[model][model])-1)
    return mean_diff, pavalue_diff
def plot_compare_to_uc(mean_dict, p_value_dict, save_folder):
    
    # --- Data Processing ---
    models = list(mean_dict.keys())
    methods_to_plot = ['sigmoid', 'isotonic', 'venn_abers'] # Exclude 'conformal'

    means_by_method = {method: [] for method in methods_to_plot}
    pvals_by_method = {method: [] for method in methods_to_plot}

    for model in models:
        for method in methods_to_plot:
            # Construct the full key used in the dictionaries
            mean_key = f"{model}_{method}"
            pval_key = f"{model}_{method}"

            # Append mean if key exists
            if mean_key in mean_dict[model]:
                means_by_method[method].append(mean_dict[model][mean_key])
            else:
                means_by_method[method].append(np.nan) # Or handle as error

            # Append p-value if key exists
            if pval_key in p_value_dict[model]:
                pvals_by_method[method].append(p_value_dict[model][pval_key])
            else:
                pvals_by_method[method].append(np.nan) # Or handle as error


    # --- Plotting Setup ---
    n_models = len(models)
    n_methods = len(methods_to_plot)
    x = np.arange(n_models)  # the label locations
    width = 0.2  # the width of the bars (adjust as needed)
    multiplier = 0

    fig, ax = plt.subplots(figsize=(7, 3)) # Adjust figure size if needed

    # --- Plot Bars and Significance Markers ---
    bar_containers = {} # To store bar container objects

    for i, method in enumerate(methods_to_plot):
        offset = width * i - width * (n_methods -1) / 2 # Calculate offset for centering group
        means = means_by_method[method]
        pvals = pvals_by_method[method]

        rects = ax.bar(x + offset, means, width, label=method.replace('_', ' ').title())
        bar_containers[method] = rects # Store container

        for j, rect in enumerate(rects):
            pval = pvals[j]
            # Check if pval is not nan and is significant
            if not math.isnan(pval) and pval < 0.05:
                height = rect.get_height()
                marker_y = height / 2  # Center the asterisk in the bar

                ax.text(rect.get_x() + rect.get_width() / 2., marker_y, '*',
                        ha='center', va='center', color='black', fontsize=14, fontweight='bold')

    # --- Add labels, title, and customisations ---
    ax.set_ylabel('Mean ACE difference')
    ax.set_xlabel('')
    ax.set_title('Calibration methods comparison')
    model_names_change = {
        'LogisticRegression': 'LR',
        'SVM': 'SVM',
        'XGBoost': 'XGB',
        'RandomForest': 'RF',
    }
    xtick = [model_names_change[model] for model in models]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick)
    handles, labels = ax.get_legend_handles_labels()
    model_names_change = {
        'Sigmoid': 'PS',
        'Isotonic': 'IR',
        'Venn Abers': 'VAs',
    }
    labels = [model_names_change[label] for label in labels]
    legend = ax.legend(handles, labels, loc='lower center', ncol=n_methods)
    legend.get_frame().set_edgecolor('black')  # Black border for the legend frame


    ax.axhline(0, color='grey', linewidth=0.8) # Add y=0 line
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

    fig.tight_layout() # Adjust layout to prevent labels overlapping
    
    plt.savefig(f'{save_folder}/compare_to_uc.png', dpi=800, bbox_inches='tight') # Save the figure

if __name__ == "__main__":
    ############## Load the data dose radiomic
    suffix = '_eso_noclinical_boot'
    training_folder = 'training_results_paper_boot'
    save_folder = 'results_paper_boot'

    with open(f'{training_folder}/loo_boot_data_results{suffix}.pkl', 'rb') as f:
        datasets = pickle.load(f)
    with open(f'{training_folder}/loo_boot_prob_results{suffix}.pkl', 'rb') as f:
        prob_results = pickle.load(f)

    prob_results = reformat_results(prob_results)
    y_true = []

    for key, value in datasets.items():
        y_true.append(value['y_test'])
    y_true  = np.array(y_true)
    y_true = y_true.reshape(-1)

    dose_radiomic_ece = []
    dose_radiomic_auroc = []
    dose_radiomic_auprc = []

    for key in prob_results.keys():

        dose_radiomic_ece.append(cal_metric(y_true, prob_results[key], metric = 'ece'))
        dose_radiomic_auroc.append(cal_metric(y_true, prob_results[key], metric = 'auroc'))
        dose_radiomic_auprc.append(cal_metric(y_true, prob_results[key], metric = 'auprc'))

    dose_radiomic_ece = combine_metric(dose_radiomic_ece)
    dose_radiomic_auroc = combine_metric(dose_radiomic_auroc)
    dose_radiomic_auprc = combine_metric(dose_radiomic_auprc)

    #####################################

    ###############################################
    suffix = '_eso_dose_boot'

    # Load the data
    with open(f'{training_folder}/loo_boot_data_results{suffix}.pkl', 'rb') as f:
        datasets = pickle.load(f)

    with open(f'{training_folder}/loo_boot_prob_results{suffix}.pkl', 'rb') as f:
        prob_results = pickle.load(f)
    prob_results = reformat_results(prob_results)
    y_true = []
    for key, value in datasets.items():
        y_true.append(value['y_test'])
    y_true  = np.array(y_true)
    y_true = y_true.reshape(-1)

    dose_ece = []
    dose_auroc = []
    dose_auprc = []

    for key in prob_results.keys():

        dose_ece.append(cal_metric(y_true, prob_results[key], metric = 'ece'))
        dose_auroc.append(cal_metric(y_true, prob_results[key], metric = 'auroc'))
        dose_auprc.append(cal_metric(y_true, prob_results[key], metric = 'auprc'))

    dose_ece = combine_metric(dose_ece)
    dose_auroc = combine_metric(dose_auroc)
    dose_auprc = combine_metric(dose_auprc)
    ###############################################
    suffix = '_eso_dosimetric_boot'

    with open(f'{training_folder}/loo_boot_data_results{suffix}.pkl', 'rb') as f:
        datasets = pickle.load(f)
    with open(f'{training_folder}/loo_boot_prob_results{suffix}.pkl', 'rb') as f:
        prob_results = pickle.load(f)
    prob_results = reformat_results(prob_results)
    y_true = []
    for key, value in datasets.items():
        y_true.append(value['y_test'])
    y_true  = np.array(y_true)
    y_true = y_true.reshape(-1)

    dosimetric_ece = []
    dosimetric_auroc = []
    dosimetric_auprc = []

    for key in prob_results.keys():

        dosimetric_ece.append(cal_metric(y_true, prob_results[key], metric = 'ece'))
        dosimetric_auroc.append(cal_metric(y_true, prob_results[key], metric = 'auroc'))
        dosimetric_auprc.append(cal_metric(y_true, prob_results[key], metric = 'auprc'))

    dosimetric_ece = combine_metric(dosimetric_ece)
    dosimetric_auroc = combine_metric(dosimetric_auroc)
    dosimetric_auprc = combine_metric(dosimetric_auprc)

    #################################################

    mean, pvalue = get_mean_pvalue_compare_touc(dose_radiomic_ece)
    plot_compare_to_uc(mean, pvalue, save_folder)
    ####################################################
    
    mean_radiomic_to_dose_ece, pvalue_radiomic_to_dose_ece = get_mean_pvalue_compare_to_custom(dose_radiomic_ece, dose_ece)
    mean_radiomic_to_dose_auroc, pvalue_radiomic_to_dose_auroc = get_mean_pvalue_compare_to_custom(dose_radiomic_auroc, dose_auroc)
    mean_radiomic_to_dose_auprc, pvalue_radiomic_to_dose_auprc = get_mean_pvalue_compare_to_custom(dose_radiomic_auprc, dose_auprc)

    mean_radiomic_to_dosimetric_ece, pvalue_radiomic_to_dosimetric_ece = get_mean_pvalue_compare_to_custom(dose_radiomic_ece, dosimetric_ece)
    mean_radiomic_to_dosimetric_auroc, pvalue_radiomic_to_dosimetric_auroc = get_mean_pvalue_compare_to_custom(dose_radiomic_auroc, dosimetric_auroc)
    mean_radiomic_to_dosimetric_auprc, pvalue_radiomic_to_dosimetric_auprc = get_mean_pvalue_compare_to_custom(dose_radiomic_auprc, dosimetric_auprc)

    mean_dose_to_dosimetric_ece, pvalue_dose_to_dosimetric_ece = get_mean_pvalue_compare_to_custom(dose_ece, dosimetric_ece)
    mean_dose_to_dosimetric_auroc, pvalue_dose_to_dosimetric_auroc = get_mean_pvalue_compare_to_custom(dose_auroc, dosimetric_auroc)
    mean_dose_to_dosimetric_auprc, pvalue_dose_to_dosimetric_auprc = get_mean_pvalue_compare_to_custom(dose_auprc, dosimetric_auprc)

    
    # Function to extract data from the nested dictionary structure
    def extract_data(mean_dict, pvalue_dict, comparison, metric):
        data = []
        for model, mean_val in mean_dict.items():
            # Access the actual mean value
            mean = list(mean_val.values())[0]
            # Access the actual p-value
            pvalue = list(pvalue_dict[model].values())[0]
            data.append({'Comparison': comparison, 'Metric': metric, 'Model': model, 'Mean': mean, 'PValue': pvalue})
        return data

    # Extract data for all comparisons and metrics
    all_data = []
    all_data.extend(extract_data(mean_radiomic_to_dose_ece, pvalue_radiomic_to_dose_ece, 'radiomic_to_dose', 'ECE'))
    all_data.extend(extract_data(mean_radiomic_to_dose_auroc, pvalue_radiomic_to_dose_auroc, 'radiomic_to_dose', 'AUROC'))
    all_data.extend(extract_data(mean_radiomic_to_dose_auprc, pvalue_radiomic_to_dose_auprc, 'radiomic_to_dose', 'AUPRC'))
    all_data.extend(extract_data(mean_radiomic_to_dosimetric_ece, pvalue_radiomic_to_dosimetric_ece, 'radiomic_to_dosimetric', 'ECE'))
    all_data.extend(extract_data(mean_radiomic_to_dosimetric_auroc, pvalue_radiomic_to_dosimetric_auroc, 'radiomic_to_dosimetric', 'AUROC'))
    all_data.extend(extract_data(mean_radiomic_to_dosimetric_auprc, pvalue_radiomic_to_dosimetric_auprc, 'radiomic_to_dosimetric', 'AUPRC'))

    # Create a pandas DataFrame
    df = pd.DataFrame(all_data)

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Define colors for each model
    model_colors = {'LogisticRegression': 'skyblue', 'SVM': 'lightcoral', 'XGBoost': 'lightgreen', 'RandomForest': 'gold'}
    models = list(model_colors.keys())

    # Function to add significance stars inside the bars
    def add_significance_stars(ax, data_subset):
        """
        Add significance stars to bars based on p-values in data.
        Places stars inside the bars rather than above them.
        """
        # Get the unique x-categories and hue values
        metrics = data_subset['Metric'].unique()
        
        # For each model-metric combination
        for metric in metrics:
            metric_df = data_subset[data_subset['Metric'] == metric]
            
            for model in models:
                model_row = metric_df[metric_df['Model'] == model]
                
                if not model_row.empty:
                    p_value = model_row['PValue'].iloc[0]
                    mean_value = model_row['Mean'].iloc[0]
                    
                    # Find the matching bar for this model and metric
                    for i, bar in enumerate(ax.patches):
                        # Check if this bar represents our model and metric
                        # Height is the most reliable way to match
                        if abs(bar.get_height() - mean_value) < 0.0001:  # Small tolerance for float comparison
                            x = bar.get_x() + bar.get_width() / 2
                            # Position the star at the middle of the bar
                            y = bar.get_height() / 2  
                            
                            # Add significance star if p-value is less than 0.05
                            if p_value < 0.05:
                                ax.text(x, y, '*', ha='center', va='center', 
                                        color='black', fontsize=15, fontweight='bold')

    # Plot 1: radiomic_to_dose ECE
    data_subset = df[(df['Comparison'] == 'radiomic_to_dose') & (df['Metric'] == 'ECE')]
    sns.barplot(x='Metric', y='Mean', hue='Model', data=data_subset, ax=axes[0], 
                palette=model_colors, order=['ECE'])
    axes[0].set_title('Compare all features to dosimetric + dosiomic')
    axes[0].set_ylabel('Mean ACE difference')
    axes[0].set_xlabel('')
    axes[0].set_xticklabels([])
    axes[0].set_ylim(-0.04, 0.005)

    axes[0].grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    add_significance_stars(axes[0], data_subset)

    # Plot 2: radiomic_to_dosimetric ECE
    data_subset = df[(df['Comparison'] == 'radiomic_to_dosimetric') & (df['Metric'] == 'ECE')]
    sns.barplot(x='Metric', y='Mean', hue='Model', data=data_subset, ax=axes[1], 
                palette=model_colors, order=['ECE'])
    axes[1].set_title('Compare all features to dosimetric')
    axes[1].set_ylabel('Mean ACE difference')
    axes[1].set_xlabel('')
    axes[1].set_xticklabels([])
    axes[1].set_ylim(-0.04, 0.005)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    add_significance_stars(axes[1], data_subset)

    # Plot 3: radiomic_to_dose AUROC and AUPRC
    data_subset = df[(df['Comparison'] == 'radiomic_to_dose') & (df['Metric'].isin(['AUROC', 'AUPRC']))]
    sns.barplot(x='Metric', y='Mean', hue='Model', data=data_subset, ax=axes[2], 
                palette=model_colors, order=['AUROC', 'AUPRC'])
    axes[2].set_title('Compare all features to dosimetric + dosiomic')
    axes[2].set_ylabel('Mean performance difference')
    axes[2].set_xlabel('')
    axes[2].set_ylim(0, 0.065)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    add_significance_stars(axes[2], data_subset)

    # Plot 4: radiomic_to_dosimetric AUROC and AUPRC
    data_subset = df[(df['Comparison'] == 'radiomic_to_dosimetric') & (df['Metric'].isin(['AUROC', 'AUPRC']))]
    sns.barplot(x='Metric', y='Mean', hue='Model', data=data_subset, ax=axes[3], 
                palette=model_colors, order=['AUROC', 'AUPRC'])
    axes[3].set_title('Compare all features to dosimetric')
    axes[3].set_ylabel('Mean performance difference')
    axes[3].set_xlabel('')
    axes[3].set_ylim(0, 0.065)
    axes[3].grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    add_significance_stars(axes[3], data_subset)

    # Remove redundant legends from individual subplots
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Create a single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    model_names_change = {
        'LogisticRegression': 'LR',
        'SVM': 'SVM',
        'XGBoost': 'XGB',
        'RandomForest': 'RF'
    }
    labels = [model_names_change[label] for label in labels]
    legend = fig.legend(handles, labels, loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, 0))
    legend.get_frame().set_edgecolor('black')  # Black border for the legend frame

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.08, 1, 0])
    plot_filename = f'{save_folder}/compare_dose_radiomic_dosimetric.png'
    plt.savefig(plot_filename, dpi=800, bbox_inches='tight') # Use bbox_inches='tight' for better fit