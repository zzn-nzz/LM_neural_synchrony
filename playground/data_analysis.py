import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from utils import *
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

R2_DATA_PATH = "../affine_transformation"
DATA_MODE = "combined_metrics"

def get_eval_file_names(model_1, model_2, episode_id, seed=None):
    """
    Returns:
        List of evaluation file names for (model_1, model_2) with different seeds (0, 1, 2).
    """
    dialog_base = f"../sotopia_results/dialogs/{model_1}_{model_2}"
    eval_base = f"{dialog_base}/eval_oss"
    
    eval_files = []
    if seed is None:
        for seed in [0, 1, 2, 3, 4]:
            eval_file = f"{eval_base}/{episode_id}_temp0.7_seed{seed}.json"
            eval_files.append(eval_file)
    else:
        eval_file = f"{eval_base}/{episode_id}_temp0.7_seed{seed}.json"
        eval_files.append(eval_file)
    
    return eval_files

def get_eval_file_name(model_1, model_2, episode_id, seed=None):
    """
    Returns:
        Evaluation file name for (model_1, model_2) with seed 0 (for backward compatibility).
    """
    dialog_base = f"../sotopia_results/dialogs/{model_1}_{model_2}"
    eval_base = f"{dialog_base}/eval_oss"
    if seed is None:
        eval_file = f"{eval_base}/{episode_id}_temp0.7_seed0.json"
    else:
        eval_file = f"{eval_base}/{episode_id}_temp0.7_seed{seed}.json"
    return eval_file

def get_scores_multi_seed(model_1, model_2, episode_id, seed=None):
    """
    Returns:
        scores (dict): Combined scores from all available seeds
        d1_sum_avg (float): Average d1_sum across available seeds  
        d1_goal_avg (float): Average d1_goal across available seeds
        d2_sum_avg (float): Average d2_sum across available seeds
        d2_goal_avg (float): Average d2_goal across available seeds
    """
    if seed is None:
        eval_files = get_eval_file_names(model_1, model_2, episode_id)
    else:
        eval_files = get_eval_file_names(model_1, model_2, episode_id, seed)
    
    valid_scores = []
    d1_sums = []
    d1_goals = []
    d2_sums = []
    d2_goals = []
    
    files_exist = [os.path.exists(eval_file) for eval_file in eval_files]
    
    # If no files exist at all, raise error immediately
    if not any(files_exist):
        if seed is not None:
            return get_scores_multi_seed(model_1, model_2, episode_id)
        else:
            raise FileNotFoundError(f"All three seed files (seeds 0, 1, 2) are missing for {model_1}_{model_2}, episode {episode_id}")
    
    for eval_file in eval_files:
        if os.path.exists(eval_file):
            try:
                scores, d1_sum, d1_goal, d2_sum, d2_goal = get_scores(eval_file)
                valid_scores.append(scores)
                d1_sums.append(d1_sum)
                d1_goals.append(d1_goal)
                d2_sums.append(d2_sum)
                d2_goals.append(d2_goal)
            except (json.JSONDecodeError, KeyError, FileNotFoundError, TypeError, IndexError):
                continue
    
    if not valid_scores:
        if seed is not None:
            return get_scores_multi_seed(model_1, model_2, episode_id)
        else:
            raise FileNotFoundError(f"All seed files for {model_1}_{model_2}, episode {episode_id} are corrupted or invalid")
    
    d1_sum_avg = np.mean(d1_sums)
    d1_goal_avg = np.mean(d1_goals)
    d2_sum_avg = np.mean(d2_sums)
    d2_goal_avg = np.mean(d2_goals)
    
    combined_scores = valid_scores[0]
    
    return combined_scores, d1_sum_avg, d1_goal_avg, d2_sum_avg, d2_goal_avg

def get_performance_separate(model_1, model_2):
    """
    Returns: 
        goal score, overall score for model_1, goal score, overall score for model_2
    """
    
    model1_goal_sum = 0
    model1_overall_sum = 0
    model2_goal_sum = 0
    model2_overall_sum = 0
    
    total_episodes = 450
    valid_episodes = 0
    overall = []
    
    for i in range(total_episodes):
        try:
            scores, d1_sum, d1_goal, d2_sum, d2_goal = get_scores_multi_seed(model_1, model_2, i)
            
            model1_goal_sum += d1_goal
            model1_overall_sum += d1_sum / 7
            
            model2_goal_sum += d2_goal
            model2_overall_sum += d2_sum / 7

            overall.append((d1_sum + d2_sum) / 14)
            valid_episodes += 1
        except FileNotFoundError:
            continue
    
    if valid_episodes == 0:
        raise ValueError(f"No valid evaluation files found for any episode of {model_1}_{model_2}")
    
    return model1_goal_sum / valid_episodes, model1_overall_sum / valid_episodes, model2_goal_sum / valid_episodes, model2_overall_sum / valid_episodes

def get_performance_together(model_1, model_2):
    """
    Returns:
        The mean goal, overall score of (model_1, model_2)
    """
    g1, o1, g2, o2 = get_performance_separate(model_1, model_2)
    print(model_1, model_2, g1, o1, g2, o2)
    return (g1 + g2) / 2, (o1 + o2) / 2

def get_performance_averaged(model_1, model_2):
    """
    Returns:
        The mean goal, overall score of (model_1, model_2) and (model_2, model_1)
        (They're different because the utterance order matters)
    """
    g1, o1 = get_performance_together(model_1, model_2)
    g2, o2 = get_performance_together(model_2, model_1)
    return (g1 + g2) / 2, (o1 + o2) / 2

def get_scores_by_mode(model_1, model_2):
    """
    Returns:
        A dictionary whose 1st key is mode name, 2nd key is performance metric.
    """
    modes, episodes_id = get_all_modes() 
    metrics_by_mode = {}

    for mode in modes:
        metrics_by_mode[mode] = {}
        metrics_by_mode[mode][(model_1, model_2)] = {}

        g1, g2, o1, o2 = 0, 0, 0, 0
        valid_episodes = 0
        
        for i in episodes_id[mode]:
            try:
                scores, d1_sum, d1_goal, d2_sum, d2_goal = get_scores_multi_seed(model_1, model_2, i)
                g1 += d1_goal
                g2 += d2_goal
                o1 += d1_sum / 7
                o2 += d2_sum / 7
                valid_episodes += 1
            except FileNotFoundError:
                continue
        
        if valid_episodes > 0:
            metrics_by_mode[mode][(model_1, model_2)]['goal'] = (g1 + g2) / 2 / valid_episodes
            metrics_by_mode[mode][(model_1, model_2)]['goal_1'] = g1 / valid_episodes
            metrics_by_mode[mode][(model_1, model_2)]['goal_2'] = g2 / valid_episodes
            metrics_by_mode[mode][(model_1, model_2)]['overall'] = (o1 + o2) / 2 / valid_episodes
            metrics_by_mode[mode][(model_1, model_2)]['overall_1'] = o1 / valid_episodes
            metrics_by_mode[mode][(model_1, model_2)]['overall_2'] = o2 / valid_episodes
        else:
            metrics_by_mode[mode][(model_1, model_2)]['goal'] = 0
            metrics_by_mode[mode][(model_1, model_2)]['goal_1'] = 0
            metrics_by_mode[mode][(model_1, model_2)]['goal_2'] = 0
            metrics_by_mode[mode][(model_1, model_2)]['overall'] = 0
            metrics_by_mode[mode][(model_1, model_2)]['overall_1'] = 0
            metrics_by_mode[mode][(model_1, model_2)]['overall_2'] = 0
    
    return metrics_by_mode

def get_r2_results_single(models_identifier, layer_A, layer_B, data_dir=None, selected_keys=None, setting=None, data_mode="combined_metrics", anotherread=False):
    if data_mode is None:
        data_mode = DATA_MODE
    if anotherread:
        file_path = f"{data_dir}/{data_mode}_{models_identifier}anotherread[0, 1, 2, 3, 4]_layerA{layer_A}_layerB{layer_B}_allreps.json"
    else:
        file_path = f"{data_dir}/{data_mode}_{models_identifier}[0, 1, 2, 3, 4]_layerA{layer_A}_layerB{layer_B}_allreps.json"
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    processed_results = {}
    for model_name, model_data in results.items():
        if selected_keys and model_name not in selected_keys:
            continue
            
        processed_results[model_name] = {
            'r2_mean': model_data['r2_mean'],
            'r2_std': model_data['r2_std'],
            'r2_var': [std**2 for std in model_data['r2_std']],
            'r2_all_seeds': model_data['r2_all_seeds'],
            'train_r2_mean': model_data.get('train_r2_mean', [0]),
            'train_r2_std': model_data.get('train_r2_std', [0]),
            'train_r2_var': [std**2 for std in model_data.get('train_r2_std', [0])],
        }
        
    return processed_results

def get_layer_range(model):
    if "3B" in model:
        return (0, 27)
    else:
        return (0, 31)


def plot_heatmap(results, metric, model_name, save_dir, vmin, vmax):
    
    n, m = results.shape

    results = results.T
    flipped_results = np.flipud(results)
    plt.figure(figsize=(10, 10))
    
    mask_negative = flipped_results < 0
    
    ax = sns.heatmap(flipped_results, cmap='Greens', vmin=vmin, vmax=vmax, 
                    mask=mask_negative)
    
    gray_color = (0.5, 0.5, 0.5, 0.4)
    ax2 = sns.heatmap(flipped_results, cmap=mcolors.ListedColormap([gray_color]), 
                     vmin=-1, vmax=0, mask=~mask_negative, cbar=False)

    cbar = ax.collections[0].colorbar 
    cbar.ax.tick_params(labelsize=24) 
    
    ax.set_xlabel('Model A Layer', fontsize=24)
    ax.set_ylabel('Model B Layer', fontsize=24)

    ax.set_xticks(np.arange(0, n, 5) + 0.5, labels=np.arange(0, n, 5), fontsize=24)
    ax.set_yticks(np.arange(m - 1, -1, -5) + 0.5, labels=np.arange(0, m, 5), fontsize=24)

    ax.set_yticklabels(np.arange(0, m, 5), rotation=0, fontsize=24)
        
    for j in range(flipped_results.shape[1]):
        max_idx = np.argmax(flipped_results[:, j], axis=0)
        ax.add_patch(plt.Rectangle((j, max_idx), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/new_{model_name}_{metric}_heatmap.pdf'
    print(f"saved to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_r2_data(model_1, model_2, setting, layer_range_A=(0, 31), layer_range_B = (0, 31), shuffled=False):
    # Just returns r2 data (32, 32) or (28, 32), etc.
    
    if shuffled:
        suf = "_shuffled"
    else:
        suf = ""
    data_dir_template = f"{R2_DATA_PATH}{suf}/{setting}" + "/layerA{}/data"
    if "A" in setting:
        layer_range_A = get_layer_range(model_1)
        layer_range_B = get_layer_range(model_2)
    else:
        layer_range_A = get_layer_range(model_2)
        layer_range_B = get_layer_range(model_1)
        
    layers_A = list(range(layer_range_A[0], layer_range_A[1] + 1))
    layers_B = list(range(layer_range_B[0], layer_range_B[1] + 1))
    model_data = {}
    
    first_layer = layers_A[0]
    first_layer_data_dir = data_dir_template.format(first_layer)
    models_identifier = get_short_names_and_identifiers([model_1 + "_" + model_2])[0][0]
    
    results = get_r2_results_single(models_identifier, first_layer, first_layer, first_layer_data_dir, setting=setting)
    
    for model_name in results.keys():
        model_data[model_name] = (np.zeros((len(layers_A), len(layers_B))), np.zeros((len(layers_A), len(layers_B)))) # mean, var
    
    for layer_A in layers_A:
        layer_data_dir = data_dir_template.format(layer_A)
        for layer_B in layers_B:
            try:
                results = get_r2_results_single(
                    models_identifier, 
                    layer_A, 
                    layer_B, 
                    layer_data_dir,
                )
                for model_name, model_results in results.items():
                    if model_name in model_data:
                        model_data[model_name][0][layer_A][layer_B] = model_results['r2_mean'][0] if model_results['r2_mean'][0] >= 0 else 0
                        model_data[model_name][1][layer_A][layer_B] = model_results['r2_var'][0]
            except Exception as e:
                print(f"Warning: Could not load data for layer {layer_A}, {layer_B}: {str(e)}")

    return model_data[model_name][0]

def get_r2_scores_setting(model_1, model_2, setting, layer_range_A=(0, 31), layer_range_B = (0, 31), shuffled=False, data_mode="combined_metrics", anotherread=False, get_se=False):
    """
    Returns:
        The R2-related metric of (model_1, model_2) under setting.
        Layer_range indicates 
    """
    if shuffled:
        suf = "_shuffled"
    else:
        suf = ""
    data_dir_template = f"{R2_DATA_PATH}{suf}/{setting}" + "/layerA{}/data"
    if "A" in setting:
        layer_range_A = get_layer_range(model_1)
        layer_range_B = get_layer_range(model_2)
    else:
        layer_range_A = get_layer_range(model_2)
        layer_range_B = get_layer_range(model_1)
        
    layers_A = list(range(layer_range_A[0], layer_range_A[1] + 1)) # Layers of model_1
    layers_B = list(range(layer_range_B[0], layer_range_B[1] + 1)) # Layers of model_2
    model_data = {}
    
    first_layer = layers_A[0]
    first_layer_data_dir = data_dir_template.format(first_layer)
    models_identifier = get_short_names_and_identifiers([model_1 + "_" + model_2])[0][0]
    
    try:
        results = get_r2_results_single(models_identifier, first_layer, first_layer, first_layer_data_dir, data_mode=data_mode, anotherread=anotherread)

    except Exception as e:
        print(f"Warning: Could not load data for layer {first_layer}, {first_layer}, {first_layer_data_dir}, {models_identifier}, {data_mode}: {str(e)}")
        quit()

    for model_name in results.keys():
        model_data[model_name] = (np.zeros((len(layers_A), len(layers_B))), np.zeros((len(layers_A), len(layers_B)))) # mean, var
    print("Loading data for ", models_identifier, setting, data_mode, anotherread)
    for layer_A in layers_A:
        layer_data_dir = data_dir_template.format(layer_A)
        for layer_B in layers_B:
            try:
                results = get_r2_results_single(
                    models_identifier, 
                    layer_A, 
                    layer_B, 
                    layer_data_dir,
                    data_mode=data_mode,
                    anotherread=anotherread
                )
                for model_name, model_results in results.items():
                    if model_name in model_data:
                        model_data[model_name][0][layer_A][layer_B] = model_results['r2_mean'][0]
                        model_data[model_name][1][layer_A][layer_B] = model_results['r2_var'][0]
            except Exception as e:
                print(f"Warning: Could not load data for layer {layer_A}, {layer_B}, {layer_data_dir}, {models_identifier}, {data_mode}: {str(e)}")
                quit()

    for model_name in model_data.keys():
        print('Setting', setting, 'Data Mode', data_mode, 'Anotherread', anotherread)
        if get_se:
            return compute_avg_and_se_metric_row(model_data[model_name][0], model_data[model_name][1], "R^2", model_name)
        else:
            return compute_avg_metric_row(model_data[model_name][0], "R^2", model_name)

def compute_avg_and_se_metric_row(results, variances, metric, model_name):
    row_max_values = []
    row_variances = []
    max_coords = []
    
    for i in range(results.shape[0]):
        row_max = np.max(results[i])
        if row_max >= 0:
            row_variances.append(variances[i, np.argmax(results[i])])
        else:
            row_variances.append(0)
        if row_max < 0:
            row_max = 0
        row_max_values.append(row_max)
        max_col = np.argmax(results[i])
        max_coords.append((i, max_col))
    
    avg_val = np.mean(row_max_values)
    se_val = np.sqrt(np.sum(row_variances)) / len(row_variances)
    
    return avg_val, se_val

def compute_avg_metric_row(results, metric, model_name):
    
    row_max_values = []
    max_coords = []
    
    for i in range(results.shape[0]):
        row_max = np.max(results[i])
        if row_max < 0:
            row_max = 0
        row_max_values.append(row_max)
        
        max_col = np.argmax(results[i])
        max_coords.append((i, max_col))
    
    avg_val = np.mean(row_max_values)
    global_max_value = np.max(results)
    global_max_coords = np.unravel_index(np.argmax(results), results.shape)
    
    print(f"{model_name} {metric}: {avg_val}")
    print(f"{model_name} global max value: {global_max_value} at coordinate {global_max_coords}")
    
    return avg_val

def get_layers(model):
    if "3B" in model and "Llama" in model:
        return (0, 27)
    else:
        return (0, 31)


def plot_all_pairs_combined_mode(combs, mode="anotherread"):

    os.makedirs(f"{R2_DATA_PATH}/figs/{mode}", exist_ok=True)

    anotherread = False
    if mode == "anotherread":
        anotherread = True
    
    if "step" in mode:
        data_mode = mode
    else:
        data_mode = "combined_metrics"

    all_normal = []
    all_mode = []
    pair_labels = []
    
    for model_1, model_2 in combs:
        r2_normal = []
        r2_mode = [] 
        
        for s in ["A_forward", "B_forward"]:
            r2_normal.append(
                get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), shuffled=False)
            )
            r2_mode.append(
                get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), shuffled=False, anotherread=anotherread, data_mode=data_mode)
            )
            
            r2_normal.append(
                get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), shuffled=False)
            )
            r2_mode.append(
                get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), shuffled=False, anotherread=anotherread, data_mode=data_mode)
            )
        
        mean_normal = np.mean(r2_normal)
        mean_mode = np.mean(r2_mode)
        
        all_normal.append(mean_normal)
        all_mode.append(mean_mode)
        
        model_1_short = model_1.split('_None')[0].split('-')[0]
        model_2_short = model_2.split('_None')[0].split('-')[0]
        pair_labels.append(f"{model_1_short}-{model_2_short}")

    _plot_grouped_bars(all_normal, all_mode, pair_labels, f"{R2_DATA_PATH}/figs/{mode}/grouped_bars.pdf")


def _plot_grouped_bars(normal_values, shuffled_values, pair_labels, output_path):
    """Create grouped bar chart for all model pairs"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_pairs = len(normal_values)
    print(pair_labels)
    print(n_pairs, "Pairs")
    x = np.arange(n_pairs)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_values, width, label='Experimental', 
                   color='#2E86C1', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, shuffled_values, width, label='Control', 
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    if n_pairs <= 10:
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Model Pairs', fontsize=14, fontweight='bold')
    ax.set_ylabel('SyncR²', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_by_model_family(combs, separate=False):

    models_id = []
    goals = []
    overalls = []
    r2s = []
    family_labels = [] 

    mistral_family_combs = []
    llama_family_combs = []
    cross_family_combs = []
    
    for c in combs:
        mis, lla = 0, 0
        if "Mistral" in c[0]:
            mis += 1
        elif "Llama" in c[0]:
            lla += 1
        if "Mistral" in c[1]:
            mis += 1
        elif "Llama" in c[1]:
            lla += 1
        if mis == 2:
            mistral_family_combs.append(c)
        elif lla == 2:
            llama_family_combs.append(c)
        else:
            cross_family_combs.append(c)
    
    if separate:
        reversed_comb = []
        for c in combs:
            reversed_comb.append((c[1], c[0]))
        new_combs = combs + reversed_comb
    else:
        new_combs = combs

    for model_1, model_2 in new_combs:
        if separate:
            g, o = get_performance_together(model_1, model_2)
        else:
            g, o = get_performance_averaged(model_1, model_2)
        goals.append(g)
        overalls.append(o)

        r2 = []
        for s in ["A_forward", "B_forward"]:
            r2.append(get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2)))
            if not separate:
                r2.append(get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1)))
        r2s.append(np.mean(r2))
        models_id.append(get_short_names_and_identifiers([model_1 + "_" + model_2])[0][0])
        
        current_comb = (model_1, model_2)
        if current_comb in mistral_family_combs or (separate and (model_2, model_1) in mistral_family_combs):
            family_labels.append('Mistral')
        elif current_comb in llama_family_combs or (separate and (model_2, model_1) in llama_family_combs):
            family_labels.append('Llama')
        else:
            family_labels.append('Cross')
    
    os.makedirs(f"{R2_DATA_PATH}/figs", exist_ok=True)
    suffix = "_separate" if separate else ""
    
    create_family_correlation_scatter(overalls, r2s, "Overall", "SyncR²", models_id, family_labels,
                                    save_path=f"{R2_DATA_PATH}/figs/overall_r2_by_family{suffix}.pdf")

def compute_avg(s1, s2):
    return (s1 + s2) / 2

def get_instruction_performance(model_1, model_2):
    inst_performances = {
        "Mistral-7B-Instruct-v0.1": 0.4487,
        "Mistral-7B-Instruct-v0.2": 0.5496,
        "Mistral-7B-Instruct-v0.3": 0.5465,
        "Llama-2-7B-Chat": 0.3986,
        "Llama-3-8B-Instruct": 0.7408,
        "Llama-3.2-3B-Instruct": 0.7393
    }
    return compute_avg(inst_performances[model_1.replace('_None_0', '')], inst_performances[model_2.replace('_None_0', '')])

def get_musr_performance(model_1, model_2):
    musr_performances = {
        "Mistral-7B-Instruct-v0.1": 0.0613,
        "Mistral-7B-Instruct-v0.2": 0.0761,
        "Mistral-7B-Instruct-v0.3": 0.0430,
        "Llama-2-7B-Chat": 0.0328,
        "Llama-3-8B-Instruct": 0.0160,
        "Llama-3.2-3B-Instruct": 0.0808
    }
    return compute_avg(musr_performances[model_1.replace('_None_0', '')], musr_performances[model_2.replace('_None_0', '')])



def compute_partial_correlation(combs, metric):
    
    scores = []
    r2s = []
    social_performances = []
    for model_1, model_2 in combs:
        if metric == "IFEval":
            scores.append(get_instruction_performance(model_1, model_2))
        elif metric == "musr":
            scores.append(get_musr_performance(model_1, model_2))

        social_performances.append(get_performance_together(model_1, model_2)[1])
        r2 = []
        r2.append(get_r2_scores_setting(model_1, model_2, "A_forward", get_layers(model_1), get_layers(model_2)))
        r2.append(get_r2_scores_setting(model_2, model_1, "A_forward", get_layers(model_2), get_layers(model_1)))
        r2s.append(np.mean(r2))
    
    data = pd.DataFrame({
        'social_performance': social_performances,
        'r2_scores': r2s,
        'control_metric': scores
    })
    
    import pingouin as pg
    result = pg.partial_corr(data=data, x='social_performance', y='r2_scores', 
                            covar='control_metric', method='pearson')
    partial_corr = result['r'].values[0]
    p_value = result['p-val'].values[0]

    print(f"Partial correlation between social performance and R^2, controlling for {metric}:")
    print(f"  r = {partial_corr:.3f}, p = {p_value:.3f}")
    
    return partial_corr, p_value, len(combs)

def compute_partial_correlation_by_family(combs, metric):
    """
    Compute partial correlation between social performance and R^2 scores,
    controlling for the specified metric, separately for each model family.
    """
    mistral_family_combs = []
    llama_family_combs = []
    cross_family_combs = []
    
    for c in combs:
        mis, lla = 0, 0
        if "Mistral" in c[0]:
            mis += 1
        elif "Llama" in c[0]:
            lla += 1
        if "Mistral" in c[1]:
            mis += 1
        elif "Llama" in c[1]:
            lla += 1
        if mis == 2:
            mistral_family_combs.append(c)
        elif lla == 2:
            llama_family_combs.append(c)
        else:
            cross_family_combs.append(c)
    
    families = {
        'Mistral': mistral_family_combs,
        'Llama': llama_family_combs,
        'Cross-family': cross_family_combs
    }
    
    results = {}
    
    for family_name, family_combs in families.items():
        if len(family_combs) < 4:
            results[family_name] = {
                'partial_corr': None,
                'p_value': None,
                'n': len(family_combs),
                'note': 'Insufficient data points'
            }
            continue
            
        # Collect data for this family
        scores = []
        r2s = []
        social_performances = []
        
        for model_1, model_2 in family_combs:
            if metric == "IFEval":
                scores.append(get_instruction_performance(model_1, model_2))
            elif metric == "musr":
                scores.append(get_musr_performance(model_1, model_2))

            social_performances.append(get_performance_together(model_1, model_2)[1])
            r2 = []
            r2.append(get_r2_scores_setting(model_1, model_2, "A_forward", get_layers(model_1), get_layers(model_2)))
            r2.append(get_r2_scores_setting(model_2, model_1, "A_forward", get_layers(model_2), get_layers(model_1)))
            r2s.append(np.mean(r2))
        
        data = pd.DataFrame({
            'social_performance': social_performances,
            'r2_scores': r2s,
            'control_metric': scores
        })
        
        r_XY, _ = pearsonr(social_performances, r2s)
        r_XZ, _ = pearsonr(social_performances, scores)
        r_YZ, _ = pearsonr(r2s, scores)
        print(f"{family_name}: r_social_r2 = {r_XY:.3f}, r_social_score = {r_XZ:.3f}, r_r2_score = {r_YZ:.3f}")

        import pingouin as pg
        result = pg.partial_corr(data=data, x='social_performance', y='r2_scores', 
                                covar='control_metric', method='pearson')
        partial_corr = result['r'].values[0]
        p_value = result['p-val'].values[0]

        results[family_name] = {
            'partial_corr': partial_corr,
            'p_value': p_value,
            'n': len(family_combs),
            'note': 'Success'
        }
    
    print(f"\nPartial correlations by model family (controlling for {metric}):")
    print("-" * 70)
    for family_name, result in results.items():
        if result['partial_corr'] is not None:
            print(f"{family_name:12} (n={result['n']:2}): r = {result['partial_corr']:6.3f}, "
                  f"p = {result['p_value']:6.3f}")
        else:
            print(f"{family_name:12} (n={result['n']:2}): {result['note']}")
    
    return results

def transform_labels(labels):
    trans_dict = {
        "Lla2": "L2",
        "Lla3.2-3B": "L3.2",
        "Lla3": "L3",
        "Misv0.1": "M1",
        "Misv0.2": "M2",
        "Misv0.3": "M3"
    }
    for i, l in enumerate(labels):
        for k, v in trans_dict.items():
            if k in l:
                l = l.replace(k, v)
        labels[i] = l
    return labels


def create_family_correlation_scatter(x_data, y_data, x_label, y_label, point_labels, family_labels, figsize=(12, 8), save_path=None):

    point_labels = transform_labels(point_labels)

    df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'label': point_labels,
        'family': family_labels
    })
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#020035',
        'axes.facecolor': '#fafafa'
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.grid(True, which='major', color="#a9a8a8", linewidth=0.8, alpha=0.7)
    ax.grid(True, which='minor', color="#9b9a9a", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_color('#020035')
        spine.set_linewidth(2)
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=12))
    
    family_colors = {
        'Mistral': '#E69F00',  
        'Llama':   '#56B4E9',
        'Cross':   '#009E73'
    }
    
    family_markers = {
        'Mistral': 'o',
        'Llama': 's',
        'Cross': '^'
    }
    
    families = df['family'].unique()
    
    for family in families:
        family_data = df[df['family'] == family]
        if len(family_data) == 0:
            continue
            
        color = family_colors.get(family, '#7f8c8d')
        marker = family_markers.get(family, 'o')
        
        ax.scatter(family_data['x'], family_data['y'], 
                  s=250, c=color, marker=marker, alpha=0.88, linewidths=2.5, 
                  label=f'{family} Family')
        
        if len(family_data) >= 2:
            z = np.polyfit(family_data['x'], family_data['y'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(family_data['x'].min(), family_data['x'].max(), 100)
            y_line = p(x_line)
            
            ax.plot(x_line, y_line, color=color, linewidth=6, alpha=0.30, linestyle='--')
            
            corr_family, p_val_family = pearsonr(family_data['x'], family_data['y'])
            print(f'{family} Family - Correlation: {corr_family:.3f}, p-value: {p_val_family:.3f}')
    
    for i, row in df.iterrows():
        ax.annotate(row['label'], 
                   (row['x'], row['y']), 
                   xytext=(7, 7), 
                   textcoords='offset points', 
                   fontsize=10,
                   fontweight='normal',
                   color='#020035')
    
    ax.tick_params(axis='both', colors='#020035', labelsize=20)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, edgecolor='#020035', fontsize=15)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)


def plot_different_conditions(comb, conditions, condition_name):

    model_1, model_2 = comb
    
    condition_scores = {}
    for cond in conditions:
        r2 = []
        for s in ["A_forward", "B_forward"]:
            r2.append(get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), data_mode=cond))
            r2.append(get_r2_scores_setting(model_2, model_1, s, get_layers(model_1), get_layers(model_2), data_mode=cond))
        condition_scores[cond] = np.mean(r2)
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(conditions, [condition_scores[cond] for cond in conditions], 
                   color='#4472C4', alpha=0.8, width=0.6)
    
    plt.xlabel(condition_name, fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    filename = f"{R2_DATA_PATH}/figs/different_conditions/{model_1}_{model_2}_{condition_name}_conditions.pdf"
    os.makedirs(f'{R2_DATA_PATH}/figs/different_conditions', exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    
def plot_step_line_chart(combs, filename="step_comparison", get_se=True):
    
    steps = [1, 3, 5]
    step_modes = ['combined_metrics', 'step_3', 'step_5']
    
    all_data = {}
    all_se = {}
    pair_labels = []
    
    for model_1, model_2 in combs:

        pair_label = get_short_names_and_identifiers([model_1 + "_" + model_2])[0][0]
        pair_labels.append(pair_label)
        
        step_scores = []
        step_se = []
        for mode in step_modes:
            r2 = []
            se = []
            for s in ["A_forward", "B_forward"]:
                results1 = get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), data_mode=mode, get_se=True)
                results2 = get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), data_mode=mode, get_se=True)
                r2.append(results1[0])
                r2.append(results2[0])
                se.append(results1[1])
                se.append(results2[1])
            step_scores.append(np.mean(r2))
            step_se.append(np.mean(se))
        
        all_data[pair_label] = step_scores
        all_se[pair_label] = step_se
    
    pair_labels = transform_labels(pair_labels)
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#020035',
        'axes.facecolor': '#fafafa'
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.grid(True, which='major', color='#d0d0d0', linewidth=0.8, alpha=0.7)
    ax.grid(True, which='minor', color='#e8e8e8', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_color('#020035')
        spine.set_linewidth(2)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(combs)))
    
    for i, pair_label in enumerate(pair_labels):
        scores = all_data[get_short_names_and_identifiers([combs[i][0] + "_" + combs[i][1]])[0][0]]
        se = all_se[get_short_names_and_identifiers([combs[i][0] + "_" + combs[i][1]])[0][0]]
        
        ax.plot(steps, scores, 
               marker='o', linewidth=2.5, markersize=8,
               color=colors[i], alpha=0.95,
               label=pair_label)
        
        ax.fill_between(steps, 
                       np.array(scores) - np.array(se), 
                       np.array(scores) + np.array(se),
                       color=colors[i], alpha=0.15)
    
    ax.set_xlabel('Step', fontsize=14, fontweight='bold', color='#020035')
    ax.set_ylabel('SyncR²', fontsize=14, fontweight='bold', color='#020035')
    ax.set_xticks(steps)
    ax.set_xticklabels(steps)
    
    ax.tick_params(axis='both', colors='#020035', labelsize=12)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, shadow=True,
             framealpha=0.9, edgecolor='#020035', fontsize=10)
    
    plt.tight_layout()

    save_path = f"{R2_DATA_PATH}/figs/{filename}.pdf"
    os.makedirs(f"{R2_DATA_PATH}/figs", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Step line chart saved to {save_path}")
    
    print("\nStep comparison summary:")
    print("-" * 50)
    for i, pair_label in enumerate(pair_labels):
        original_label = get_short_names_and_identifiers([combs[i][0] + "_" + combs[i][1]])[0][0]
        scores = all_data[original_label]
        print(f"{pair_label:15}: Step1={scores[0]:.3f}, Step3={scores[1]:.3f}, Step5={scores[2]:.3f}")

    
def plot_step_line_chart_by_family(combs, filename="step_comparison_by_family", get_se=True):
    steps = [1, 3, 5]
    step_modes = ['combined_metrics', 'step_3', 'step_5']
    
    mistral_family_combs = []
    llama_family_combs = []
    cross_family_combs = []
    
    for c in combs:
        mis, lla = 0, 0
        if "Mistral" in c[0]:
            mis += 1
        elif "Llama" in c[0]:
            lla += 1
        if "Mistral" in c[1]:
            mis += 1
        elif "Llama" in c[1]:
            lla += 1
        if mis == 2:
            mistral_family_combs.append(c)
        elif lla == 2:
            llama_family_combs.append(c)
        else:
            cross_family_combs.append(c)
    
    family_groups = [
        (mistral_family_combs, 'Mistral', '#E69F00'),
        (llama_family_combs, 'Llama', '#56B4E9'), 
        (cross_family_combs, 'Cross', '#009E73')
    ]
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#020035',
        'axes.facecolor': '#fafafa'
    })
    
    os.makedirs(f"{R2_DATA_PATH}/figs", exist_ok=True)
    
    for family_combs, family_name, base_color in family_groups:
        if not family_combs: 
            continue
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.grid(True, which='major', color='#d0d0d0', linewidth=0.8, alpha=0.7)
        ax.grid(True, which='minor', color='#e8e8e8', linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_color('#020035')
            spine.set_linewidth(2)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(family_combs)))
        colors = [(r*0.8, g*0.8, b*0.8, a) for r, g, b, a in colors]
        
        family_data = {}
        family_se = {}
        family_labels = []
        
        for model_1, model_2 in family_combs:
            pair_label = get_short_names_and_identifiers([model_1 + "_" + model_2])[0][0]
            family_labels.append(pair_label)
            
            step_scores = []
            step_se = []
            for mode in step_modes:
                r2 = []
                se = []
                for s in ["A_forward", "B_forward"]:
                    results1 = get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), data_mode=mode, get_se=True)
                    results2 = get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), data_mode=mode, get_se=True)
                    r2.append(results1[0])
                    r2.append(results2[0])
                    se.append(results1[1])
                    se.append(results2[1])
                step_scores.append(np.mean(r2))
                step_se.append(np.mean(se))
            
            family_data[pair_label] = step_scores
            family_se[pair_label] = step_se
            
        family_labels = transform_labels(family_labels)
        
        for i, pair_label in enumerate(family_labels):
            original_label = get_short_names_and_identifiers([family_combs[i][0] + "_" + family_combs[i][1]])[0][0]
            scores = family_data[original_label]
            se = family_se[original_label]
            
            ax.plot(steps, scores, 
                   marker='o', linewidth=4, markersize=12,
                   color=colors[i], alpha=0.9,
                   label=pair_label)
            
            ax.fill_between(steps, 
                           np.array(scores) - np.array(se), 
                           np.array(scores) + np.array(se),
                           color=colors[i], alpha=0.15)
        ax.set_xticks(steps)
        ax.set_xticklabels([1, 2, 3])
        ymin, ymax = ax.get_ylim()
        
        ymin = np.floor(ymin * 10) / 10
        ymax = np.ceil(ymax * 10) / 10

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + 0.1, 0.1))
        
        ax.tick_params(axis='both', colors='#020035', labelsize=24)
        
        ax.legend(
            loc='best',
            frameon=True, fancybox=True, shadow=True,
            framealpha=0.8, edgecolor='#020035', fontsize=16
        )

        
        plt.tight_layout()
        
        save_path = f"{R2_DATA_PATH}/figs/{filename}_{family_name.lower()}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{family_name} family step line chart saved to {save_path}")
    
    
def plot_all_pairs_combined_mode_by_family(combs, mode="anotherread"):
    """
    Plot average SyncR² values grouped by model family.
    Each family gets two bars for comparison: normal vs mode.
    
    Args:
        combs: List of tuples containing model pairs
        mode: Mode for comparison (anotherread)
    """
    os.makedirs(f"{R2_DATA_PATH}/figs/{mode}", exist_ok=True)

    anotherread = False
    if mode == "anotherread":
        anotherread = True
    
    if "step" in mode:
        data_mode = mode
    else:
        data_mode = "combined_metrics"

    mistral_family_combs = []
    llama_family_combs = []
    cross_family_combs = []
    
    for c in combs:
        mis, lla = 0, 0
        if "Mistral" in c[0]:
            mis += 1
        elif "Llama" in c[0]:
            lla += 1
        if "Mistral" in c[1]:
            mis += 1
        elif "Llama" in c[1]:
            lla += 1
        if mis == 2:
            mistral_family_combs.append(c)
        elif lla == 2:
            llama_family_combs.append(c)
        else:
            cross_family_combs.append(c)
    
    families = {
        'Mistral': mistral_family_combs,
        'Cross-family': cross_family_combs,
        'Llama': llama_family_combs,
    }
    
    family_names = []
    family_normal_means = []
    family_mode_means = []
    family_normal_ste = []
    family_mode_ste = []
    
    for family_name, family_combs in families.items():
        if len(family_combs) == 0:
            continue
            
        family_normal = []
        family_mode = []
        
        for model_1, model_2 in family_combs:
            r2_normal = []
            r2_mode = []
            
            for s in ["A_forward", "B_forward"]:
                r2_normal.append(
                    get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), shuffled=False)
                )
                r2_mode.append(
                    get_r2_scores_setting(model_1, model_2, s, get_layers(model_1), get_layers(model_2), shuffled=False, anotherread=anotherread, data_mode=data_mode)
                )
                
                r2_normal.append(
                    get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), shuffled=False)
                )
                r2_mode.append(
                    get_r2_scores_setting(model_2, model_1, s, get_layers(model_2), get_layers(model_1), shuffled=False, anotherread=anotherread, data_mode=data_mode)
                )
            
            family_normal.append(np.mean(r2_normal))
            family_mode.append(np.mean(r2_mode))
        
        family_names.append(family_name)
        family_normal_means.append(np.mean(family_normal))
        family_mode_means.append(np.mean(family_mode))
        print(family_normal)
        print(np.std(family_normal))
        print(np.sqrt(len(family_normal)))
        family_normal_ste.append(np.std(family_normal) / np.sqrt(len(family_normal) * 3))
        family_mode_ste.append(np.std(family_mode) / np.sqrt(len(family_mode) * 3))
        
        print(f"{family_name} Family - Normal: {np.mean(family_normal):.4f}, {mode}: {np.mean(family_mode):.4f} (n={len(family_combs)} pairs)")

    print(family_mode_ste)
    print(family_normal_ste)
    
    _plot_family_grouped_bars(family_normal_means, family_mode_means, family_normal_ste, family_mode_ste, family_names, 
                             f"{R2_DATA_PATH}/figs/{mode}/family_comparison.pdf", mode)


def _plot_family_grouped_bars(normal_values, mode_values, normal_stes, mode_stes, family_names, output_path, mode):

    df = pd.DataFrame({
        "Family": np.tile(family_names, 2),
        "Value": np.concatenate([normal_values, mode_values]),
        "STE": np.concatenate([normal_stes, mode_stes]),
        "Condition": (["Experimental"] * len(family_names)) + \
                     ([mode] * len(family_names))
    })

    mode_label = mode.capitalize()
    if mode == "anotherread":
        mode_label = "Another Read"

    df["Condition"] = df["Condition"].replace({mode: mode_label})

    palette = sns.color_palette("Set2", 2)

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(
        data=df,
        x="Family",
        y="Value",
        hue="Condition",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        ci=None,
    )

    n_families = len(family_names)
    x_positions = np.arange(n_families)
    width = 0.4
    
    for i, family in enumerate(family_names):
        
        ax.errorbar(x_positions[i] - width/2, normal_values[i], yerr=normal_stes[i], 
                   fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
        
        ax.errorbar(x_positions[i] + width/2, mode_values[i], yerr=mode_stes[i], 
                   fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)

    ax.tick_params(axis="y", labelsize=24)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine()

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="", fontsize=20, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved family comparison plot to {output_path}")

if __name__ == "__main__":
    combs = []
    model_list = [
        "Mistral-7B-Instruct-v0.1_None_0",
        "Mistral-7B-Instruct-v0.2_None_0",
        "Mistral-7B-Instruct-v0.3_None_0",
        "Llama-2-7B-Chat_None_0",
        "Llama-3-8B-Instruct_None_0",
        "Llama-3.2-3B-Instruct_None_0",
    ]
    for m1 in model_list:
        for m2 in model_list:
            if ((m1, m2) not in combs) and ((m2, m1) not in combs):
                combs.append((m1, m2))

    plot_by_model_family(combs)