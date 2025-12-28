import torch
import csv
from sot_env import SotopiaEnv
from utils import *
from tqdm import tqdm
from copy import copy
import argparse

parser = argparse.ArgumentParser(description='Run Sotopia environment with specified affected types array')
parser.add_argument('--model_1', type=str, default="Mistral-7B-Instruct-v0.3")
parser.add_argument('--model_2', type=str, default="Mistral-7B-Instruct-v0.3")
parser.add_argument('--affected_type', type=str, default="None")
parser.add_argument('--value', type=float, default=0)
args = parser.parse_args()

max_new_tokens = 300
max_turns = 16
episodes_num = 450
temp = 0.7
seeds = [0, 1, 2, 3, 4]

if args.value == 0.0:
    args.value = 0

model_1_name = f"{args.model_1}_{args.affected_type}_{args.value}"
model_2_name = f"{args.model_2}_{args.affected_type}_{args.value}"

save_dir_base = f"sotopia_results/{model_1_name}_{model_2_name}"
dialog_base = f"sotopia_results/dialogs/{model_1_name}_{model_2_name}"
create_folder_if_not_there(dialog_base)

for seed in seeds:
    print(save_dir_base, seed)
    set_seed(seed)
    verbose = False
    turns = []

    env = SotopiaEnv(model_1_name=model_1_name, 
                        model_2_name=model_2_name, 
                        max_turns=max_turns, 
                        saving_dir_base=save_dir_base, 
                        save_states=True, 
                        save_states_only_dialog=False,
                        save_states_dialog_imperson=False,
                        save_states_without_goal=False,
                        probing_goal=False,
                        probing_self_goal=False,
                        max_new_tokens=max_new_tokens,
                        verbose=verbose,
                        temperature=temp,
                        seed=seed)
    
    for episode_id in tqdm(range(episodes_num)):
        set_seed(seed)
        if os.path.exists(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv'):
            if os.path.getsize(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv') > 0:
                print(f"Episode {episode_id} already exists, skipping.")
                continue
        with open(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Intro', 'Dialog', 'Intended Dialog'])

        done = False
        env.reset(env_id=episode_id)
        
        cur_turn = 0
        while not done:
            done = env.step()
            cur_turn = copy(env.cur_turn)

        turns.append(cur_turn)

        intro, dialog, intended_dialog = env.complete_intro, env.cur_dialog, env.intended_dialog
        with open(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([intro, dialog, intended_dialog])

        create_folder_if_not_there(f"{dialog_base}/prompt_records")
        with open(f'{dialog_base}/prompt_records/{episode_id}_temp0.7_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['save_states',
                            ])
            for i in range(cur_turn):
                writer.writerow([
                    env.prompts[i],
                ])

    print('turns of each episode:', turns)
    del env
                

