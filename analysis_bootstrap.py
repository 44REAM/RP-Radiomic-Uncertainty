import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import sklearn.metrics as metrics
import mapie.metrics as mapie_metrics
from scipy import stats
# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Constants for display names
CALIBRATION_NAMES = {
    'sigmoid': 'PS',
    'isotonic': 'IR',
    'venn_abers': 'VAs',
    'conformal': 'CP',
    'conformal_cal': 'CP-PS'
}

METRIC_NAMES = {
    'accuracy_': 'ACC',
    'f1': 'F1 Score',
    'precision': 'Precision',
    'coverage': 'Coverage',
    'balanced_accuracy_': 'BACC',
    'auc_': 'AUROC',
    'prauc_': 'AUPRC',
}

MODEL_NAMES = {
    'LogisticRegression': 'LR',
    'SVM': 'SVM',
    'XGBoost': 'XGB',
    'RandomForest': 'RF',
}

# Core calculation functions
def calculate_metrics(pred, pred_prob, y_true, suffix=""):
    """Calculate various classification metrics."""
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
        precision, recall, _ = metrics.precision_recall_curve(y_true, pred_prob)
        metrics_dict[f'prauc_{suffix}'] = metrics.auc(recall, precision)
    except ValueError:
        metrics_dict[f'prauc_{suffix}'] = np.nan
    
    return metrics_dict

def calculate_threshold_metrics(y_true, prob_results, uncertainty_results, 
                               model_names=None, 
                               uncertainty_threshold=0.8):
    """Calculate metrics at a specific uncertainty threshold."""
    if model_names is None:
        model_names = ['LogisticRegression', 'SVM', 'XGBoost', 'RandomForest']
        
    model_metrics = {}
    for model_name in model_names:
        model_metrics[model_name] = {}
        for key, value in uncertainty_results.items():
            if model_name.lower() not in key.lower():
                continue
                
            prob = np.array(prob_results[key])
            uncertainty = np.array(value)
            certainty = 1 - cal_uncertainty_score(uncertainty, prob)

            pred = prob.argmax(axis=1)
            filtered_indices = certainty >= uncertainty_threshold
            
            model_metrics[model_name][key] = calculate_metrics(
                pred[filtered_indices], 
                prob[filtered_indices][:,1], 
                y_true[filtered_indices]
            )
            model_metrics[model_name][key].update({
                'coverage': len(pred[filtered_indices])/len(pred)
            })
            
    return model_metrics

def calculate_all_thresholds(y_true, prob_results, uncertainty_results,
                           selected_columns=None, 
                           selected_thresholds=None):
    """Calculate metrics for multiple uncertainty thresholds."""
    if selected_columns is None:
        selected_columns = ['coverage', 'balanced_accuracy_']
    if selected_thresholds is None:
        selected_thresholds = [0.5, 0.8, 0.9]
    
    df_all_threshold = None
    
    for prob_threshold in selected_thresholds:
        model_metrics = calculate_threshold_metrics(
            y_true, prob_results, uncertainty_results, 
            uncertainty_threshold=prob_threshold
        )
        
        df_specific_threshold = None
        for key, value in model_metrics.items():
            if df_specific_threshold is None:
                df_specific_threshold = pd.DataFrame(value).T[selected_columns]
            else:
                df_specific_threshold = pd.concat(
                    [df_specific_threshold, pd.DataFrame(value).T[selected_columns]]
                )
                
        if df_all_threshold is None:
            df_all_threshold = df_specific_threshold
        else:
            df_all_threshold = pd.concat([df_all_threshold, df_specific_threshold], axis=1)
    
    # Format the dataframe
    model_names = []
    calibration_methods = []
    
    for name in df_specific_threshold.index:
        model_name = name.split('_')[0]
        if len(name.split('_')) > 1:
            calibration_method = '_'.join(name.split('_')[1:])
            calibration_method = CALIBRATION_NAMES[calibration_method]
        else: 
            calibration_method = "UC"
            
        model_names.append(MODEL_NAMES[model_name])
        calibration_methods.append(calibration_method)

    df_all_threshold['Model'] = model_names
    df_all_threshold['Uncertainty Method'] = calibration_methods
    df_all_threshold.set_index(['Model', 'Uncertainty Method'], inplace=True)

    columns_names = [
        (f"Cutpoint {thres}", METRIC_NAMES.get(metric, metric)) 
        for thres in selected_thresholds 
        for metric in selected_columns
    ]

    df_all_threshold.columns = pd.MultiIndex.from_tuples(columns_names)
    return df_all_threshold

def cal_uncertainty_score(uncertainty, pred_prob):
    """Calculate uncertainty score based on predicted probabilities."""
    return 1 - uncertainty[np.arange(len(uncertainty)).astype(int), 
                         pred_prob.argmax(axis=1).astype(int)]

def get_coverage_metrics(y_test, proba, uncertainty, pred=None):
    """Calculate metrics across different coverage levels."""
    all_metrics = []
    coverages = []
    
    if pred is None:
        pred = proba.argmax(axis=1)

    uncertainty_score = cal_uncertainty_score(uncertainty, proba)

    # Add metrics for full coverage
    coverage = 1
    all_metrics.append(calculate_metrics(pred, proba[:,1], y_test))
    coverages.append(coverage)

    # Calculate metrics for each coverage level
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

    # Reformat results
    keys = all_metrics[0].keys()
    new_data = {key: [] for key in keys}

    for metric in all_metrics:
        for key, value in metric.items():
            new_data[key].append(value)
    
    return np.array(coverages), new_data


def get_all_coverage_metrics(y_true, prob_results, uncertainty_results):

    all_results = {}
    y_true = np.array(y_true) # Ensure y_true is a numpy array

    # Iterate through each model and calibration method combination
    for key in prob_results.keys():
        # Ensure the corresponding uncertainty results exist
        if key not in uncertainty_results:
            raise ValueError(f"Warning: Uncertainty results not found for key '{key}'. Skipping.")

        # Retrieve probabilities and uncertainty values
        proba = np.array(prob_results[key])
        uncertainty = np.array(uncertainty_results[key])

        # Basic check for consistent lengths
        if len(y_true) != len(proba) or len(y_true) != len(uncertainty):
             raise ValueError(f"Warning: Data length mismatch for key '{key}'. "
                   f"y_true: {len(y_true)}, proba: {len(proba)}, uncertainty: {len(uncertainty)}. Skipping.")

        if len(y_true) == 0:
            raise ValueError(f"Warning: Empty y_true array for key '{key}'. Skipping.")

        # Determine if original predictions should be used (for conformal methods)
        # This logic mirrors the plotting function's handling
        use_original_pred = 'conformal' in key.lower()

        pred_arg = None
        if use_original_pred:
            # Calculate predictions based on probabilities if needed for get_coverage_metrics
            # This is important if get_coverage_metrics expects 'pred' for conformal
            if proba.ndim == 2 and proba.shape[1] > 1:
                 pred_arg = proba.argmax(axis=1)
            else:
                 # Handle cases where proba might not be 2D (e.g., already class labels)
                 # Adjust this logic if necessary based on conformal output format
                 raise ValueError(f"Warning: Unexpected probability shape for conformal key '{key}'. Shape: {proba.shape}")

        # Calculate coverage metrics for the current combination
        try:
            coverages, metrics_data = get_coverage_metrics(
                y_test=y_true,
                proba=proba,
                uncertainty=uncertainty,
                pred=pred_arg # Pass pred only if needed (use_original_pred is True)
            )

            # Store the results
            all_results[key] = {
                'coverages': coverages,
                'metrics': metrics_data
            }
        except Exception as e:
            print(f"Error calculating coverage metrics for key '{key}': {e}")
            # Store empty or error state if desired
            all_results[key] = {
                'coverages': np.array([]),
                'metrics': {m: [] for m in calculate_metrics([], [], []).keys()} # Get metric keys structure
            }


    return all_results

def plot_coverage_metrics(axs, all_coverage_data, metric_name,
                          model_names=None, coverages_threshold=0.1,
                          row=0, font_size=10):

    if model_names is None:
        model_names = ['LogisticRegression', 'SVM', 'XGBoost', 'RandomForest']

    lines = ["-", "--", "-.", ":"] # Linestyles for different calibration methods

    # Iterate through each MODEL subplot column
    for i, model_name in enumerate(model_names):
        linecycler = cycle(lines) # Reset linestyle cycle for each model

        # Find all keys in all_coverage_data that start with the current model_name
        # Sort keys to ensure consistent plotting order of calibration methods
        model_specific_keys = sorted([
            key for key in all_coverage_data
            if key.lower().startswith(model_name.lower() + '_') or key.lower() == model_name.lower()
        ])

        if not model_specific_keys:
             print(f"Note: No data found for model '{model_name}' in all_coverage_data.")

        # Plot data for each calibration method associated with the current model
        for key in model_specific_keys:

            data_for_key = all_coverage_data[key]
            coverages = np.array(data_for_key.get('coverages', np.array([])))

            metrics_dict = data_for_key.get('metrics', {})

            # Check if the requested metric exists and data is valid
            if metric_name not in metrics_dict:
                raise ValueError(f"Warning: Metric '{metric_name}' not found for key '{key}'. Skipping plot for this line.")

            metric_values = np.array(metrics_dict[metric_name])

            if len(coverages) == 0 or len(metric_values) == 0 or len(coverages) != len(metric_values):
                raise ValueError(f"Warning: Invalid or mismatched data length for key '{key}', metric '{metric_name}'. Skipping plot.")

            # Filter data based on coverage threshold
            valid_indices = coverages >= coverages_threshold

            if not np.any(valid_indices):
                raise ValueError(f"Note: No data points above coverage threshold {coverages_threshold} for key '{key}'.")

            # --- Plotting ---
            # NO call to get_coverage_metrics here!
            axs[row][i].plot(
                coverages[valid_indices],
                metric_values[valid_indices],
                next(linecycler),
                label=key # Use the full key for unique legend identification later
            )

        # --- Set plot labels and styling (Remains the same) ---
        display_model_name = MODEL_NAMES.get(model_name, model_name)
        if row == 0:
            axs[row][i].set_title(display_model_name, color='black', fontsize=font_size)

        # Determine if this is the last row being plotted (dynamic check)
        num_rows_in_figure = axs.shape[0]
        if row == num_rows_in_figure - 1:
            axs[row][i].set_xlabel('Coverage', color='black', fontsize=font_size)

        if i == 0:
            display_metric_name = METRIC_NAMES.get(metric_name, metric_name)
            axs[row][i].set_ylabel(display_metric_name, color='black', fontsize=font_size)

        axs[row][i].tick_params(axis='both', colors='black', labelsize=font_size)
        # TODO: Consider making ylim dynamic based on metric, or pass as argument
        axs[row][i].set_ylim([0.5, 1]) # Keeping original limits for now
        axs[row][i].grid(True, linestyle='--', alpha=0.6) # Add grid for better readability



def create_coverage_plot(all_coverage_data, # MODIFIED: Takes pre-calculated data
                         metrics_to_plot=None,
                         save_folder='results_paper',
                         suffix="", font_size = 11):

    if metrics_to_plot is None:
        metrics_to_plot = ['auc_', 'prauc_']

    if not all_coverage_data:
        print("Error: all_coverage_data is empty. Cannot create plot.")
        return

    # Determine the models present in the data to set the number of columns
    model_base_names_present = sorted(list(set(key.split('_')[0] for key in all_coverage_data.keys())))

    # Use a standard order if models are present, otherwise use detected order
    default_model_order = ['LogisticRegression', 'SVM', 'XGBoost', 'RandomForest']
    model_names_to_plot = [m for m in default_model_order if m in model_base_names_present]
    # If standard models aren't found, use the ones that are actually present
    if not model_names_to_plot:
        model_names_to_plot = model_base_names_present

    num_models = len(model_names_to_plot)
    if num_models == 0:
        print("Error: No models identified in all_coverage_data keys. Cannot create plot.")
        return

    num_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(num_metrics, num_models,
                             figsize=(3 * num_models, 2.7 * num_metrics), # Slightly taller plots
                             squeeze=False) # Ensure axes is always 2D array

    # --- Plot each specified metric in a separate row ---
    for i, metric in enumerate(metrics_to_plot):
        plot_coverage_metrics( # Call the REVISED plot function
            axs=axes,
            all_coverage_data=all_coverage_data, # MODIFIED: Pass pre-calculated data
            metric_name=metric,
            model_names=model_names_to_plot, # Pass the models we determined to plot
            coverages_threshold=0.1,
            row=i,
            font_size=font_size
        )

    # --- Add legend (Logic relies on labels set in plot_coverage_metrics) ---
    handles, labels = [], []
    # Try to get handles/labels from the first subplot that has lines
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break # Stop once we find handles

    if not handles:
        print("Warning: No lines were plotted (or found). Skipping legend generation.")
    else:
        # Generate unique calibration method names and corresponding handles from the full labels (keys)
        unique_calib_methods = {} # Dict to store {display_name: handle}

        for handle, label in zip(handles, labels):
            parts = label.split('_')
            if len(parts) > 1:
                calibration_method_key = '_'.join(parts[1:])
                # Use display name from CALIBRATION_NAMES, fall back to key if not found
                calibration_method_name = CALIBRATION_NAMES.get(calibration_method_key, calibration_method_key)
            else:
                # Assume it's the base model (Uncalibrated)
                calibration_method_name = "UC" # Uncalibrated

            # Store the first handle found for each unique calibration display name
            if calibration_method_name not in unique_calib_methods:
                 unique_calib_methods[calibration_method_name] = handle

        sorted_calib_names = unique_calib_methods.keys()
        handles_unique = [unique_calib_methods[name] for name in sorted_calib_names]
        labels_unique = sorted_calib_names


        # Position legend at the bottom, adjusting space based on number of items
        num_legend_items = len(labels_unique)

        fig.subplots_adjust(bottom=0.2)

        legend = fig.legend(
            handles_unique, labels_unique,
            loc="lower center",         # Position: bottom center of the figure
            bbox_to_anchor=(0.5, 0),    # Anchor point: middle-bottom of the figure
            # bbox_transform=fig.transFigure, # Relative to figure coordinates
            frameon=True,               # Draw frame around legend
            fancybox=False,             # Use simple box
            shadow=False,               # No shadow
            ncol=num_legend_items,      # Arrange items horizontally
            fontsize=font_size      # Slightly smaller font for legend
        )
        legend.get_frame().set_edgecolor('black') # Black border for the frame


    # Adjust layout tightly within the specified rectangle (leaving space for legend)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # --- Save the figure ---
    os.makedirs(save_folder, exist_ok=True) # Ensure save folder exists
    plot_filename = f'{save_folder}/coverage{suffix}.png'
    try:
        plt.savefig(plot_filename, dpi=800, bbox_inches='tight') # Use bbox_inches='tight' for better fit
        print(f"Coverage plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot to {plot_filename}: {e}")
    plt.close(fig) # Close the figure to free memory

def reformat_results(results):
    transposed_dict = {}

    for model_name, inner_dict in results.items():
        for key, value in inner_dict.items():
            num_bootstrap = len(value)
            for i in range(num_bootstrap):
                if i not in transposed_dict:
                    transposed_dict[i] = {}
                if model_name not in transposed_dict[i]:

                    transposed_dict[i][model_name] = []

                transposed_dict[i][model_name].append(value[i])

    return transposed_dict

def get_process_calculate_threshold(df):
    new_df = pd.DataFrame(columns = df.columns, index = df.index)
    for index in df.index:
        if 'UC' in index[1]:
            new_df.loc[index] = df.loc[index]
            continue

        results = df.loc[index] - df.loc[index[0], 'UC'].replace(np.nan, 0)
        for i, value in enumerate(results):
            results.iloc[i] = value
        new_df.loc[index] = results
    new_df = new_df.replace(np.nan, 0)
    return new_df

def get_process_calculate_threshold2(df):
    new_df = pd.DataFrame(columns = df.columns, index = df.index)
    for index in df.index:
        new_df.loc[index] = df.loc[index]
    new_df = new_df.replace(np.nan, 0)
    return new_df

def combine_dataframes_mean_sd(data_dict):
    num_dataframes = len(data_dict)
    if num_dataframes == 0:
        return None, None

    first_df = list(data_dict.values())[0]
    average_data = {}
    std_dev_data = {}

    for index in first_df.index:
        average_data[index] = {}
        std_dev_data[index] = {}
        for col in first_df.columns:
            values = [data_dict[i].loc[index, col] for i in range(num_dataframes)]

            average_data[index][col] = np.mean(values)
            std_dev_data[index][col] = np.std(values, ddof=1)

    average_df = pd.DataFrame.from_dict(average_data, orient='index', columns=first_df.columns)
    std_dev_df = pd.DataFrame.from_dict(std_dev_data, orient='index', columns=first_df.columns)
    average_df.columns = first_df.columns
    std_dev_df.columns = first_df.columns

    return average_df, std_dev_df

def get_mean_sd_calculate_threshold(data):

    numeric_dfs = {}
    for key, df in data.items():

        numeric_dfs[key] = df.select_dtypes(include=[np.number])

        numeric_dfs[key] = numeric_dfs[key].replace(np.nan, 0)
    combined = pd.concat(numeric_dfs, axis=0)

    avg = combined.groupby(level=1).mean()
    sd = combined.groupby(level=1).std()

    print("Elementwise average of numeric columns across dataframes:")
    print(avg)
    print("\nElementwise standard deviation of numeric columns across dataframes:")
    print(sd)

    first_df = data[next(iter(data))]
    result_avg = first_df.copy()
    for col in avg.columns:
        result_avg[col] = avg[col]

    first_df_std = first_df.copy()
    for col in sd.columns:
        first_df_std[col] = sd[col]
    return result_avg, first_df_std

def combine_dataframes(avg, sd):
    combined = avg.copy()
    for col in avg.columns:
        for item in avg.index:
            if isinstance(avg.loc[item, col], str):
                avg_item = avg.loc[item, col]
            else:
                prefix = ""
                if avg.loc[item, col]>0 and item[1]!='UC':
                    prefix = "+"
                avg_item = f"{prefix}{round(avg.loc[item, col],2)}"
                
            if isinstance(sd.loc[item, col], str):
                sd_item = sd.loc[item, col]
            else:

                sd_item = f"{round(sd.loc[item, col],2)}"

            significant = ""
            if sd.loc[item, col] == 0 and item[1]!='UC':
                significant = "†"
            else:
                if not isinstance(sd.loc[item, col], str) and not isinstance(avg.loc[item, col], str) and item[1]!='UC':
                    t = avg.loc[item, col] / sd.loc[item, col]

                    p = 2 * stats.t.cdf(-abs(t), df=99) # 100 bootstrap samples - 1
                    if p <0.05:
                        significant = "†"

            combined.loc[item, col] = f"{avg_item}±{sd_item}{significant}"
    return combined


def interpolate_metrics(data_dict, new_coverage_range=np.linspace(0.1, 1, num=91)):
    output_data = {}
    for outer_key, model_data in data_dict.items():
        if outer_key not in output_data:
            output_data[outer_key] = {}
        for model_name, data in model_data.items():
            if model_name not in output_data[outer_key]:
                output_data[outer_key][model_name] = {}
            coverages = np.array(data['coverages'])[::-1]
            auc = np.array(data['metrics']['auc_'])[::-1]
            prauc = np.array(data['metrics']['prauc_'])[::-1]
            interpolated_auc = np.interp(new_coverage_range, coverages, auc, left = np.nan)
            interpolated_prauc = np.interp(new_coverage_range, coverages, prauc, left = np.nan)

            output_data[outer_key][model_name]['coverages'] = new_coverage_range
            output_data[outer_key][model_name]['metrics'] = {
                'auc_': interpolated_auc,
                'prauc_': interpolated_prauc
            }

    return output_data

def avg_coverage_data(all_coverage_data):
    avg_all_coverage_data = {}
    for key in all_coverage_data:
        for model_name in all_coverage_data[key]:
            if model_name not in avg_all_coverage_data:
                avg_all_coverage_data[model_name] = {}
            if 'coverages' not in avg_all_coverage_data[model_name]:
                avg_all_coverage_data[model_name]['coverages'] = []
            if 'metrics' not in avg_all_coverage_data[model_name]:
                avg_all_coverage_data[model_name]['metrics'] = {}
            avg_all_coverage_data[model_name]['coverages'].append(all_coverage_data[key][model_name]['coverages'])
            
            if 'auc_' not in avg_all_coverage_data[model_name]['metrics']:
                avg_all_coverage_data[model_name]['metrics']['auc_'] = []
            if 'prauc_' not in avg_all_coverage_data[model_name]['metrics']:
                avg_all_coverage_data[model_name]['metrics']['prauc_'] = []
            avg_all_coverage_data[model_name]['metrics']['auc_'].append(all_coverage_data[key][model_name]['metrics']['auc_'])
            avg_all_coverage_data[model_name]['metrics']['prauc_'].append(all_coverage_data[key][model_name]['metrics']['prauc_'])

    for model_name in all_coverage_data[key]:
        avg_all_coverage_data[model_name]['coverages'] = np.array(avg_all_coverage_data[model_name]['coverages']).mean(axis = 0)
        avg_all_coverage_data[model_name]['metrics']['auc_'] = np.nanmedian(np.array(avg_all_coverage_data[model_name]['metrics']['auc_']),axis = 0)
        avg_all_coverage_data[model_name]['metrics']['prauc_'] = np.nanmedian(np.array(avg_all_coverage_data[model_name]['metrics']['prauc_']), axis = 0)
    return avg_all_coverage_data

if __name__ == '__main__':
    """Run the full analysis pipeline."""
    # Create save directory if it doesn't exist
    suffix='_eso_noclinical_boot'
    train_folder = 'training_results_paper_boot'
    save_folder = 'results_paper_boot'
    os.makedirs(save_folder, exist_ok=True)

    with open(f'{train_folder}/loo_boot_data_results{suffix}.pkl', 'rb') as f:
        datasets = pickle.load(f)

    with open(f'{train_folder}/loo_boot_prob_results{suffix}.pkl', 'rb') as f:
        prob_results = pickle.load(f)

    with open(f'{train_folder}/loo_boot_uncertainty_results{suffix}.pkl', 'rb') as f:
        uncertainty_results = pickle.load(f)

    # Prepare the data
    y_true = []
    for key, value in datasets.items():
        y_true.append(value['y_test'])
    y_true = np.array(y_true).reshape(-1)
    del datasets
    prob_results = reformat_results(prob_results)
    uncertainty_results = reformat_results(uncertainty_results)

    ################## Table #####################
    all_threshold_data = {}
    all_threshold_data_non_process= {}
    for bootstrap_key, prob_value in prob_results.items():

        a  = calculate_all_thresholds(
            y_true, prob_results[bootstrap_key], 
            uncertainty_results[bootstrap_key], 
            selected_thresholds = [0.0, 0.8, 0.9], selected_columns = ['coverage','auc_', 'prauc_'])
        all_threshold_data[bootstrap_key] = get_process_calculate_threshold(a)
        all_threshold_data_non_process[bootstrap_key] = get_process_calculate_threshold2(a)
    avg_non_process, sd_non_process = combine_dataframes_mean_sd(all_threshold_data_non_process)
    sd_non_process = sd_non_process/np.sqrt(100)
    avg_non_process.to_csv(f'{save_folder}/coverage_table_mean{suffix}.csv', encoding='utf-8-sig')
    sd_non_process.to_csv(f'{save_folder}/coverage_table_sd{suffix}.csv', encoding='utf-8-sig')

    avg, sd = combine_dataframes_mean_sd(all_threshold_data)
    sd = sd/np.sqrt(100)
    combine = combine_dataframes(avg, sd)
    combine.to_csv(f'{save_folder}/coverage_table{suffix}.csv', encoding='utf-8-sig')
    ################## Table #####################

    ################# Figure #####################
    all_coverage_data = {}
    for proba_key, prob_value in prob_results.items():
        prob_results[proba_key]

        uncertainty_results[proba_key]
        all_coverage_data[proba_key] = get_all_coverage_metrics(y_true, prob_results[proba_key], uncertainty_results[proba_key])

    all_coverage_data = interpolate_metrics(all_coverage_data)
    all_coverage_data = avg_coverage_data(all_coverage_data)
    create_coverage_plot( # Call the REVISED function
        all_coverage_data=all_coverage_data, # Pass the results here
        metrics_to_plot=['auc_', 'prauc_'],   # Specify metrics to plot
        save_folder=save_folder,
        suffix=suffix
    )
    ################# Figure #####################