import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import random
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
from utils import set_seed, get_short_names_and_identifiers, get_all_modes
import time

def colorize(text, color):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "end": "\033[0m",
    }
    return colors[color] + text + colors["end"]

def print_colored(text, color):
    print(colorize(text, color))

def save_projection_matrix(G, save_path):
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save({
        'G_state_dict': G.state_dict(),
        'G_input_dim': G.linear.weight.shape[1],
        'G_output_dim': G.linear.weight.shape[0]
    }, save_path)

def load_projection_matrix(load_path):

    checkpoint = torch.load(load_path)
    G = LinearProjection(checkpoint['G_input_dim'], checkpoint['G_output_dim'])
    G.load_state_dict(checkpoint['G_state_dict'])
    return G.to('cuda')

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_affine_transformation(train_X, train_Y, test_X, test_Y,
                               device='cuda', verbose=True, seed=0, alpha=0.1, save_compute=False):
    """
    Returns:
        r2_score: R² coefficient for test set (higher is better)
        nmse: Normalized Mean Square Error for test set (lower is better)
        mse: Mean Square Error for test set (original loss metric)
        train_r2_score: R² coefficient for training set
        train_nmse: NMSE for training set
        train_mse: MSE for training set
        G: Analytical projection matrix from X to Y
    """
    start_time = time.time()
    set_seed(seed)
    
    train_X = train_X.clone().detach().float().to(device)
    train_Y = train_Y.clone().detach().float().to(device)
    test_X = test_X.clone().detach().float().to(device)
    test_Y = test_Y.clone().detach().float().to(device)
    
    if verbose:
        print("Computing analytical solution using least squares...")
    
    n_samples = train_X.shape[0]
    X_b = torch.cat([torch.ones(n_samples, 1, device=device), train_X], dim=1)

    p = X_b.shape[1]
    I = torch.eye(p, device=device)
    I[0, 0] = 0 # this is for removing bias term from regularization
    print(f'Until Preparation: {time.time() - start_time}.')
    # W = (X_b.T @ X_b + alpha * I)^-1 @ X_b.T @ y
    A = X_b.T @ X_b + alpha * I
    b_vec = X_b.T @ train_Y
    print(f'Until Calculation: {time.time() - start_time}.')
    weights = torch.linalg.solve(A, b_vec, )
    print(f'Until linalg solve: {time.time() - start_time}.')

    bias = weights[0]
    W = weights[1:].T
        
    G = LinearProjection(train_X.shape[1], train_Y.shape[1]).to(device)
    with torch.no_grad():
        G.linear.weight.copy_(W.to(dtype=torch.float32, device='cuda'))
        G.linear.bias.copy_(bias.to(dtype=torch.float32, device='cuda'))

    print(f"Regression solved in {time.time() - start_time:.4f} seconds")
    # Step 2: Evaluation on test set
    start_time = time.time()

    if not save_compute:
        train_dataset = TensorDataset(train_X.cpu(), train_Y.cpu())
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    test_dataset = TensorDataset(test_X.cpu(), test_Y.cpu())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    G.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred_y = G(batch_x)
            all_y_true.append(batch_y.cpu())
            all_y_pred.append(pred_y.cpu())
    
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    if not save_compute:
        mse = mean_squared_error(all_y_true, all_y_pred)
        y_true_var = np.var(all_y_true, axis=0)
        avg_var = np.mean(y_true_var)
        nmse = mse / avg_var
    val_r2 = r2_score(all_y_true, all_y_pred)
    
    if not save_compute:
        all_train_y_true = []
        all_train_y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred_y = G(batch_x)
                all_train_y_true.append(batch_y.cpu())
                all_train_y_pred.append(pred_y.cpu())
        
        all_train_y_true = torch.cat(all_train_y_true, dim=0).numpy()
        all_train_y_pred = torch.cat(all_train_y_pred, dim=0).numpy()
        
        train_mse = mean_squared_error(all_train_y_true, all_train_y_pred)
        train_y_true_var = np.var(all_train_y_true, axis=0)
        train_avg_var = np.mean(train_y_true_var)
        train_nmse = train_mse / train_avg_var
        train_r2 = r2_score(all_train_y_true, all_train_y_pred)

    print(f"Training and Evaluating takes {time.time() - start_time} seconds.")

    if save_compute:
        return val_r2, None, None, None, None, None, G
    
    return val_r2, nmse, mse, train_r2, train_nmse, train_mse, G

def get_preloaded_features(modes, episodes_id, model, token=None, setting=None, data_mode=None, seed_list=None, another_read=False):
    # The seed_list is for gathering episodes with different seeds, t=0.7

    allAs = []
    allBs = []
    seed_boundaries = []  # Store boundary information for each episode

    token = None

    for mode in tqdm(modes):
        for episode in episodes_id[mode]:
            if another_read:
                orig_model = model.replace('_anotherread', '')
                sA, _, boundaries = get_all_states_interactive_with_boundaries(model, mode, episode, token=token, setting=setting, seed_list=seed_list, data_mode=data_mode)
                _, sB, boundaries = get_all_states_interactive_with_boundaries(orig_model, mode, episode, token=token, setting=setting, seed_list=seed_list, data_mode=data_mode)
            else:
                sA, sB, boundaries = get_all_states_interactive_with_boundaries(model, mode, episode, token=token, setting=setting, seed_list=seed_list, data_mode=data_mode)
            
            allAs.append(sA) # sA[episode_id] = layers, dim
            allBs.append(sB)
            seed_boundaries.append(boundaries)  # boundaries[episode_id] = [end_idx_seed0, end_idx_seed1, ...]
    
    return allAs, allBs, seed_boundaries

def get_features_from_preloaded(preloaded_features_A, preloaded_features_B, layer_A=10, layer_B=10):
    return preloaded_features_A[:, layer_A, :], preloaded_features_B[:, layer_B, :]

def get_all_states_interactive_with_boundaries(model_name, mode, episode, token=None, setting=None, seed_list=None, suff="", data_mode="combined_metrics"):

    state_path = f'sotopia_results{suff}/{model_name}/{mode}/episode_{episode}'
    states_A = []
    states_B = []
    seed_boundaries = []

    step_size = 1
    if data_mode == "step_3":
        step_size = 3
    elif data_mode == "step_5":
        step_size = 5
    
    def get_agent(x):
        if x % 2 == 0:
            return 1
        else:
            return 2
        
    if seed_list:
        current_total = 0
        for s in seed_list:
            state_path_with_seed = f"{state_path}/t0.7_seed{s}"
            if setting == None or setting[0] == "A":
                turn = 0
            elif setting[0] == "B":
                turn = 1
            
            # Count turns for this specific seed
            seed_turn_count = 0
            while True:
                a_path = f'{state_path_with_seed}/agent_{get_agent(turn)}_turn_{turn}/states.npy'
                b_path = f'{state_path_with_seed}/agent_{get_agent(turn + step_size)}_turn_{turn + step_size}/states.npy'
                if os.path.exists(a_path) and os.path.exists(b_path):
                    try:
                        A = np.load(a_path)
                        B = np.load(b_path)
                    except Exception as e:
                        print(f"Error loading {a_path} or {b_path}: {e}")
                        break
                    if token == -1:
                        states_A.append(A[:, -1:, :])
                        states_B.append(B[:, -1:, :])
                    else:
                        states_A.append(A)
                        states_B.append(B)
                    seed_turn_count += 1
                    turn += 2  # Move to next pair
                else:
                    break  # Stop when a file doesn't exist
            
            current_total += seed_turn_count
            seed_boundaries.append(current_total)
            
    if not states_A:  # If no valid turns were found
        return np.array([]), np.array([]), []
    
    states_A = np.array(states_A)
    states_B = np.array(states_B)
    
    if states_A.ndim < 4:
        print(states_A.shape, mode, episode)
    if states_A.ndim >= 4:
        states_A = states_A.squeeze(axis=2)
        states_B = states_B.squeeze(axis=2)
    
    if setting == None or setting[1] == "forward":
        return states_A, states_B, seed_boundaries
    elif setting[1] == "backward":
        return states_B, states_A, seed_boundaries

def sampling_with_full_episodes(As, Bs, sample_num, shuffle=False, seed_boundaries=None):
    
    start_time = time.time()
    train_sample_num = int(0.8 * sample_num)
    test_sample_num = sample_num - train_sample_num
    
    num_episodes = len(As)
    all_episode_indices = np.random.permutation(num_episodes)
    
    # Pre-calculate episode sizes for efficient memory allocation
    episode_sizes = np.array([As[i].shape[0] for i in range(num_episodes)])
    
    # Find episodes for test set
    test_set_idx = []
    test_samples_collected = 0
    test_episode_idx = 0
    
    while test_samples_collected < test_sample_num and test_episode_idx < num_episodes:
        curr_episode = all_episode_indices[test_episode_idx]
        test_set_idx.append(curr_episode)
        test_samples_collected += episode_sizes[curr_episode]
        test_episode_idx += 1
    
    # Find episodes for train set
    train_episodes = []
    train_samples_collected = 0
    train_episode_idx = test_episode_idx
    
    while train_samples_collected < train_sample_num and train_episode_idx < num_episodes:
        curr_episode = all_episode_indices[train_episode_idx]
        train_episodes.append(curr_episode)
        train_samples_collected += episode_sizes[curr_episode]
        train_episode_idx += 1
    
    if shuffle:
        print('Using shuffled features with per-seed shuffling')
    print("Until Finding Episodes: ", time.time() - start_time)
    # Pre-allocate arrays with exact sizes
    test_total_samples = sum(episode_sizes[i] for i in test_set_idx)
    train_total_samples = sum(episode_sizes[i] for i in train_episodes)
    
    # Get feature dimensions from first episode
    feature_dima1 = As[0].shape[1]
    feature_dima2 = As[0].shape[2]
    feature_dimb1 = Bs[0].shape[1]
    feature_dimb2 = Bs[0].shape[2]
    
    test_A_samples = np.empty((test_total_samples, feature_dima1, feature_dima2), dtype=As[0].dtype)
    test_B_samples = np.empty((test_total_samples, feature_dimb1, feature_dimb2), dtype=Bs[0].dtype)
    train_A_samples = np.empty((train_total_samples, feature_dima1, feature_dima2), dtype=As[0].dtype)
    train_B_samples = np.empty((train_total_samples, feature_dimb1, feature_dimb2), dtype=Bs[0].dtype)
    
    # Fill test arrays
    test_idx = 0
    for episode_idx in test_set_idx:
        episode_A = As[episode_idx]
        episode_B = Bs[episode_idx]
        num_turns = min(episode_A.shape[0], episode_B.shape[0])
        if num_turns == 0:
            continue
        test_A_samples[test_idx:test_idx + num_turns] = episode_A[:num_turns]
        test_B_samples[test_idx:test_idx + num_turns] = episode_B[:num_turns]
        test_idx += num_turns
    print(f"Until Filling test data: {time.time() - start_time} seconds.")
    # Fill train arrays with optional shuffling
    train_idx = 0
    for episode_idx in train_episodes:
        episode_A = As[episode_idx]
        episode_B = Bs[episode_idx]
        if episode_A.shape[0] == 0:
            continue
        # print(episode_A.shape)
        num_turns = min(episode_A.shape[0], episode_B.shape[0])
        if num_turns == 0:
            continue
        
        if shuffle and seed_boundaries:
            # Optimized per-seed shuffling
            boundaries = seed_boundaries[episode_idx]
            episode_B = episode_B.copy()  # Only copy when needed
            
            start_idx = 0
            for end_idx in boundaries:
                segment_len = end_idx - start_idx
                if segment_len > 1:  # Only shuffle if more than 1 element
                    shuffled_indices = np.random.permutation(segment_len)
                    episode_B[start_idx:end_idx] = episode_B[start_idx:end_idx][shuffled_indices]
                start_idx = end_idx
            
        elif shuffle:
            if num_turns > 1:  # Only shuffle if more than 1 element
                shuffled_indices = np.random.permutation(num_turns)
                episode_B = episode_B[shuffled_indices]
        
        train_A_samples[train_idx:train_idx + num_turns] = episode_A[:num_turns]
        train_B_samples[train_idx:train_idx + num_turns] = episode_B[:num_turns]
        train_idx += num_turns
    print(f"Until Filling train data: {time.time() - start_time} seconds.")
    print(f"Sampling data: {time.time() - start_time} seconds.")
    return train_A_samples, train_B_samples, test_A_samples, test_B_samples, test_set_idx, train_episodes


def linear_all_reps_layer_by_layer(models, seeds=(0, 1, 2), setting=None, seed_list=None, precomputed_sample_num=None, shuffle=False, A_range=32, B_range=32,
                                   data_seed_list=None, data_mode="combined_metrics", save_weights=False, specific_layer_pair=None, another_read=False):
    if data_seed_list is None:
        data_seed_list = seed_list
    
    model_short_names, models_identifier = get_short_names_and_identifiers(models)
    print(model_short_names, models_identifier)
    
    modes, episodes_id = get_all_modes()

    all_model_data = {}

    for i, model_name in tqdm(enumerate(models), total=len(models)):
        short_name = model_short_names[i]
        
        allAs, allBs, seed_boundaries = get_preloaded_features(modes, episodes_id, model_name, setting=setting, data_mode=data_mode, seed_list=data_seed_list, another_read=another_read)

        model_data = {
            "allAs": allAs,
            "allBs": allBs,
            "seed_boundaries": seed_boundaries
        }
        all_model_data[short_name] = model_data

    if seed_list:
        models_identifier += str(seed_list)

    for layer_A in range(A_range):

        suffix = ""
        if shuffle is True:
            suffix = "_shuffled"
        data_dir = f"affine_transformation{suffix}/{setting[0]}_{setting[1]}/layerA{layer_A}/data"
        weights_dir = f"affine_transformation{suffix}/{setting[0]}_{setting[1]}/layerA{layer_A}/weights"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)

        for layer_B in range(B_range):
            if specific_layer_pair is not None:
                if not (layer_A == specific_layer_pair[0] and layer_B == specific_layer_pair[1]):
                    continue

            results_file = f"{data_dir}/{data_mode}_{models_identifier}_layerA{layer_A}_layerB{layer_B}_allreps.json"
            # Check if results have content
            if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not save_weights:
                print(f"Results already exist at {results_file}, skipping computation")
                continue
            
            print(f"Layer A: {layer_A}, Layer B: {layer_B}")
            results = {}
            
            for _, model_name in tqdm(enumerate(models), desc="Processing models with equal samples", total=len(models)):
                short_name = model_short_names[_]
                
                all_r2_results = []
                all_nmse_results = []
                all_mse_results = []
                all_train_r2_results = []
                all_train_nmse_results = []
                all_train_mse_results = []
                all_test_set_idx = []
                all_train_set_idx = []
                allAs, allBs, seed_boundaries = all_model_data[short_name]["allAs"], all_model_data[short_name]["allBs"], all_model_data[short_name]["seed_boundaries"]

                for seed in seeds:

                    set_seed(seed)
                    train_X, train_Y, test_X, test_Y, test_set_idx, train_set_idx = sampling_with_full_episodes(allAs, allBs, precomputed_sample_num, shuffle=shuffle, seed_boundaries=seed_boundaries)
                    train_A, train_B, test_A, test_B = train_X[:, layer_A, :], train_Y[:, layer_B, :], test_X[:, layer_A, :], test_Y[:, layer_B, :]

                    print(f"\nProcessing seed {seed}")
                    set_seed(seed)
                    
                    r2_values = []
                    nmse_values = []
                    mse_values = []
                    train_r2_values = []
                    train_nmse_values = []
                    train_mse_values = []
                    
                    train_A_tensor = torch.tensor(train_A).to('cuda')
                    train_B_tensor = torch.tensor(train_B).to('cuda')
                    test_A_tensor = torch.tensor(test_A).to('cuda')
                    test_B_tensor = torch.tensor(test_B).to('cuda')
                    
                    r2_score, nmse, mse, train_r2_score, train_nmse, train_mse, G = train_affine_transformation(
                        train_X=train_A_tensor,
                        train_Y=train_B_tensor,
                        test_X=test_A_tensor,
                        test_Y=test_B_tensor,
                        device='cuda',
                        verbose=False,
                        seed=seed,
                    )
                    r2_values.append(r2_score)
                    nmse_values.append(nmse)
                    mse_values.append(mse)
                    train_r2_values.append(train_r2_score)
                    train_nmse_values.append(train_nmse)
                    train_mse_values.append(train_mse)
                    print(f"Validation - R² score: {r2_score:.4f}")
                    
                    if save_weights:
                        weight_file = f"{weights_dir}/{data_mode}_{short_name}_layerA{layer_A}_layerB{layer_B}_seed{seed}.pt"
                        save_projection_matrix(G, weight_file)
                        print(f"Saved model weights to {weight_file}")
                        
                        # Save train and test episode IDs
                        episode_ids_file = f"{weights_dir}/{data_mode}_{short_name}_layerA{layer_A}_layerB{layer_B}_seed{seed}_episodes.json"
                        episode_ids = {
                            'train_episodes': [int(ep) for ep in train_set_idx],
                            'test_episodes': [int(ep) for ep in test_set_idx]
                        }
                        with open(episode_ids_file, 'w') as f:
                            json.dump(episode_ids, f, indent=2)
                        print(f"Saved episode IDs to {episode_ids_file}")
                        print(f"  Train episodes ({len(train_set_idx)}): {train_set_idx[:10]}{'...' if len(train_set_idx) > 10 else ''}")
                        print(f"  Test episodes ({len(test_set_idx)}): {test_set_idx[:10]}{'...' if len(test_set_idx) > 10 else ''}")

                    all_test_set_idx.append(test_set_idx)
                    all_train_set_idx.append(train_set_idx)
                    all_r2_results.append(r2_values)
                    all_nmse_results.append(nmse_values)
                    all_mse_results.append(mse_values)
                    all_train_r2_results.append(train_r2_values)
                    all_train_nmse_results.append(train_nmse_values)
                    all_train_mse_results.append(train_mse_values)
                
                all_r2_results = np.array(all_r2_results)
                all_nmse_results = np.array(all_nmse_results)
                all_mse_results = np.array(all_mse_results)
                all_train_r2_results = np.array(all_train_r2_results)
                all_train_nmse_results = np.array(all_train_nmse_results)
                all_train_mse_results = np.array(all_train_mse_results)
                
                r2_mean = np.mean(all_r2_results, axis=0)
                r2_std = np.std(all_r2_results, axis=0)
                mse_mean = np.mean(all_mse_results, axis=0)
                mse_std = np.std(all_mse_results, axis=0)
                
                train_r2_mean = np.mean(all_train_r2_results, axis=0)
                train_r2_std = np.std(all_train_r2_results, axis=0)
                train_mse_mean = np.mean(all_train_mse_results, axis=0)
                train_mse_std = np.std(all_train_mse_results, axis=0)
                
                # Store results
                results[short_name] = {
                    'r2_mean': r2_mean.tolist(),
                    'r2_std': r2_std.tolist(),
                    'r2_all_seeds': all_r2_results.tolist(),
                    'mse_mean': mse_mean.tolist(),
                    'mse_std': mse_std.tolist(),
                    'mse_all_seeds': all_mse_results.tolist(),
                    'train_r2_mean': train_r2_mean.tolist(),
                    'train_r2_std': train_r2_std.tolist(),
                    'train_r2_all_seeds': all_train_r2_results.tolist(),
                    'train_mse_mean': train_mse_mean.tolist(),
                    'train_mse_std': train_mse_std.tolist(),
                    'train_mse_all_seeds': all_train_mse_results.tolist(),
                    'test_set_idx': [[int(ep) for ep in episodes] for episodes in all_test_set_idx],
                    'train_set_idx': [[int(ep) for ep in episodes] for episodes in all_train_set_idx]
                }
            
            with open(results_file, 'w') as f:
                json.dump(convert_numpy_types(results), f, indent=2)
            
            print(f"Results data saved to {results_file}")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    seeds = (0, 1, 2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0")
    parser.add_argument('--setting', type=str, default="A_forward")
    parser.add_argument('--data_mode', type=str, default="combined_metrics")
    parser.add_argument('--another_read', action="store_true", default=False)
    parser.add_argument('--A_range', type=int, default=32)
    parser.add_argument('--B_range', type=int, default=32)
    parser.add_argument('--save_weights', action="store_true")
    parser.add_argument('--layer_A', type=int, default=None, help="Specific layer A to run (if not set, runs all layers)")
    parser.add_argument('--layer_B', type=int, default=None, help="Specific layer B to run (if not set, runs all layers)")
    args = parser.parse_args()
    print(args)
    if args.another_read is True:
        args.model += "_anotherread"
    
    setting = args.setting
    if '3.2-3B' in args.model.split("None_0")[0]:
        if setting[0] == "A":
            args.A_range = 28
        else:
            args.B_range = 28
    if '3.2-3B' in args.model.split("None_0")[1]:
        if setting[0] == "A":
            args.B_range = 28
        else:
            args.A_range = 28
    models = [args.model]
    setting = (args.setting.split('_')[0], args.setting.split('_')[1])
    seed_list = [0, 1, 2, 3, 4]
    
    # Set specific_layer_pair if both layer_A and layer_B are provided
    specific_layer_pair = None
    if args.layer_A is not None and args.layer_B is not None:
        specific_layer_pair = (args.layer_A, args.layer_B)
        print(f"Running specific layer pair: Layer A={args.layer_A}, Layer B={args.layer_B}")
    elif args.layer_A is not None or args.layer_B is not None:
        print("Warning: Both --layer_A and --layer_B must be specified to run a specific layer pair. Running all layers instead.")
    
    linear_all_reps_layer_by_layer(models, seeds, setting, seed_list=seed_list, precomputed_sample_num=6500, shuffle=False, A_range=args.A_range, B_range=args.B_range, data_seed_list=None, data_mode=args.data_mode, save_weights=args.save_weights, specific_layer_pair=specific_layer_pair, another_read=args.another_read)