import os
import torch
import numpy as np
import random
import json

def enh_print(x, color="green"):
    if color == "green":
        print(f"\033[92m{x}\033[0m")
    elif color == "red":
        print(f"\033[91m{x}\033[0m")
    elif color == "yellow":
        print(f"\033[93m{x}\033[0m")
    else:
        print(x)

def create_folder_if_not_there(path): 
    if not os.path.exists(path):
      os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_short_names_and_identifiers(models):

    def shorten(x):
        first = False
        if x[0] == "M":
            first = True
        a_ver = "Mis" + x.split('Mistral-7B-Instruct-')[1].split('Llama')[0].split('_')[0]
        a_mode = ''.join(x.split('Mistral-7B-Instruct-')[1].split('Llama')[0].split('interv')[0].split('_')[1:])
        b_ver = "Lla" + x.split('Llama-')[1].split('-')[0]
        b_mode = ''.join(x.split('Llama-')[1].split('Mistral')[0].split('interv')[0].split('_')[1:])
        if b_mode == "None0":
            b_mode = ""
        if a_mode == "None0":
            a_mode = ""
        if "Llama-3.2-3B-Instruct" in x:
            b_ver += "-3B"
        if first:
            return a_ver + a_mode + b_ver + b_mode
        else:
            return b_ver + b_mode + a_ver + a_mode

    short_names = []
    models_identifier = ""
    for m in models:
        if "Mistral" in m and "Llama" in m:
            model_short_name = shorten(m)
            if "diff_backbones" not in models_identifier:
                models_identifier += "diff_backbones"
        elif "Qwen" not in m:
            if "Mistral" in m:
                a_ver = "Mis" + m.split('Mistral-7B-Instruct-')[1].split('_')[0]
                a_mode = ''.join(m.split('Mistral-7B-Instruct-')[1].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                b_ver = "Mis" + m.split('Mistral-7B-Instruct-')[2].split('_')[0]
                b_mode = ''.join(m.split('Mistral-7B-Instruct-')[2].split('interv')[0].split('_')[1:])
                if b_mode == "None0":
                    b_mode = ""
                model_short_name = a_ver + a_mode + b_ver + b_mode
            if "Llama" in m:
                a_ver = "Lla" + m.split('Llama-')[1].split('-')[0]
                a_mode = ''.join(m.split('Llama-')[1].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                b_ver = "Lla" + m.split('Llama-')[2].split('-')[0]
                b_mode = ''.join(m.split('Llama-')[2].split('interv')[0].split('_')[1:])
                if b_mode == "None0":
                    b_mode = ""
                model_short_name = a_ver + a_mode + b_ver + b_mode
        else:
            if "Mistral" in m:
                first = False
                if m[0] == "Q":
                    first = True
                a_ver = "Mis" + m.split('Mistral-7B-Instruct-')[1].split('Qwen')[0].split('_')[0]
                a_mode = ''.join(m.split('Mistral-7B-Instruct-')[1].split('Qwen')[0].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                model_short_name = "Q3" + a_ver + a_mode if first else a_ver + a_mode + "Q3"
            if "Llama" in m:
                first = False
                if m[0] == "Q":
                    first = True
                a_ver = "Lla" + m.split('Llama-')[1].split('-')[0]
                a_mode = ''.join(m.split('Llama-')[1].split('Qwen')[0].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                model_short_name = "Q3" + a_ver + a_mode if first else a_ver + a_mode + "Q3"
        model_short_name = model_short_name.replace('None0', '')
        short_names.append(model_short_name)

    if len(models) == 1:
        models_identifier = short_names[0]
    # print(models, short_names)
    return short_names, models_identifier


def get_all_modes():
    modes = []
    for subdir in os.listdir('../sotopia_results/Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0'):
        if '.npy' not in subdir:
            modes.append(subdir)

    episodes_id = {}
    for mode in modes:
        try:
            spath = f'../sotopia_results/Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0/{mode}'
            all_items = os.listdir(spath)
        except FileNotFoundError:
            spath = f'sotopia_results/Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0/{mode}'
            all_items = os.listdir(spath)
        subfolders_id = [int(item.split('_')[-1]) for item in all_items if os.path.isdir(os.path.join(spath, item))]
        subfolders_id = sorted(subfolders_id)
        episodes_id[mode] = subfolders_id
        # print(mode, len(episodes_id[mode]))
    return modes, episodes_id

def get_scores(json_file):
    scores = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = data['result']
        d1 = []
        d1_sum = 0
        d1_goal = data["agent_1_evaluation"]["goal"][1]
        for k, v in data["agent_1_evaluation"].items():
            d1.append((k, v[1]))
            d1_sum += v[1]
        
        d2 = []
        d2_sum = 0
        d2_goal = data["agent_2_evaluation"]["goal"][1]
        for k, v in data["agent_2_evaluation"].items():
            d2.append((k, v[1]))
            d2_sum += v[1]
        
        scores["agent_1"] = d1
        scores["agent_2"] = d2

    return scores, d1_sum, d1_goal, d2_sum, d2_goal


def get_all_scores(json_file):
    scores = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = data['result']
        
        d1 = []
        for k, v in data["agent_1_evaluation"].items():
            d1.append((k, v[1]))
        
        d2 = []
        for k, v in data["agent_2_evaluation"].items():
            d2.append((k, v[1]))
        
        scores["agent_1"] = d1
        scores["agent_2"] = d2
    return scores