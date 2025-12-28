import torch
import csv
import json
import numpy as np
import os
from sot_env import *
from utils import *
from tqdm import tqdm
import argparse

def get_agent(x):
    if x % 2 == 0:
        return 1
    else:
        return 2

def load_prompts_from_csv(csv_file_path):
    
    prompts = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    prompts.append(row[0])  # First column contains the prompt
    except FileNotFoundError:
        print(f"Prompt file not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error loading prompts from {csv_file_path}: {e}")
        return None
    
    return prompts

def transform_prompt(prompt, now_profiles, actor, another_read):
    original_prompt = prompt
    print("Original Prompt:")
    print(prompt)
    try:
        if another_read:
            
            agent_name = prompt.split("Imagine you are ")[1].split(',')[0]
            before_conversation = prompt.split('Turn')[0].strip()
            conversation = ('Turn' + 'Turn'.join(prompt.split('Turn')[1:])).strip()

            agent1_name = now_profiles["agent1_profile"]["first_name"] + " " + now_profiles["agent1_profile"]["last_name"]
            agent2_name = now_profiles["agent2_profile"]["first_name"] + " " + now_profiles["agent2_profile"]["last_name"]
            current_agent_name = agent1_name
            
            first_agent_name = prompt.split("'s background")[0].split('\n')[-1]
            before_agent_background = prompt.split(f"{first_agent_name}'s background:")[0]
            agent_background_and_after = prompt.split(f"{first_agent_name}'s background:")[1:]

            inst = prompt.split("\nYou can find")[0]
            inst_transformed = f"Imagine you are {current_agent_name}. You are reading an interaction between other two people."
            prompt = prompt.replace(inst, inst_transformed)

            prompt = prompt.replace(f"Note that {agent_name}'s goal is only visible to you.\n", "")
            prompt = prompt.replace(f"You can find {agent_name}'s goal (or background) in the 'Here is the context of the interaction' field.\n", "")
            prompt = prompt.replace('If your goal as the agent has been achieved, you should leave the conversation by saying "LEAVE". If your goal has not been achieved, you should say something to interact with the other. Keep your words concise.\n', "")
            orig_ending = prompt.split("\n")[-1]
            split_newline_list = [s.strip() for s in prompt.split("\n")[:-1]]
            if '' in split_newline_list:
                split_newline_list.remove('')
            prompt = "\n".join(split_newline_list)
            print('Transformed prompt:', prompt)

    except Exception as e:
        print(f"Error transforming prompt: {e}")
        return original_prompt

    return prompt


def save_states_only(env, model_1_name, model_2_name, episode_id, seed, temp=0.7, 
                    max_new_tokens=1, record_dialog_base=None, saving_dir_base=None, save_prompt_base=None,
                    token_position=-1, random_profile=False, now_profiles=None, another_read=False):
    """
    Save internal states without generating conversations by loading existing prompts.
    Changes will be made to existing prompts. New prompts will be stored in save_prompt_base.
    New states will be stored in saving_dir_base.
    
    Args:
        model_1_name: Name of model 1
        model_2_name: Name of model 2  
        episode_id: Episode ID to process
        seed: Random seed used
        temp: Temperature parameter
        max_new_tokens: Max tokens for generation
        record_dialog_base: Base directory where prompt records are stored
        saving_dir_base: Base directory where states will be saved
        save_prompt_base: Base directory where prompts will be saved
        token_position: Token position for state extraction
        random_profile: Whether random profile was used
        now_profiles: Now profiles
    """
    
    prompt_file = f'{record_dialog_base}/prompt_records/{episode_id}_temp{temp}_seed{seed}.csv'
    prompts = load_prompts_from_csv(prompt_file)
    
    for i in range(len(prompts)):
        if i % 2 == 0:
            actor = 1
        else:
            actor = 2
        prompts[i] = transform_prompt(prompts[i], now_profiles, actor, another_read)
        
    # Save Transformed prompts to save_prompt_base
    create_folder_if_not_there(save_prompt_base)
    with open(f'{save_prompt_base}/episode_{episode_id}_temp{temp}_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['save_states'])
        for i in range(len(prompts)):
            writer.writerow([prompts[i]])

    if prompts is None:
        print(f"Failed to load prompts for episode {episode_id}, seed {seed}")
        return False
    
    print(f"Loaded {len(prompts)} prompts for episode {episode_id}")
    
    for turn, prompt in enumerate(prompts):
        print(f"Processing turn {turn}/{len(prompts)-1}")
        
        if turn % 2 == 0:
            current_model = env.model_1
            actor = 1
        else:
            current_model = env.model_2
            actor = 2
        
        try:
            with torch.no_grad():
                
                intended_result, result, states, attn_states = current_model.generate_response_and_get_state(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    layer="All", 
                    token=token_position
                )
                
                saving_dir = f"{saving_dir_base}/{env.source_name}"
                
                data_dir = f"{saving_dir}/episode_{episode_id}/agent_{actor}_turn_{turn}"
                if temp != 0:
                    data_dir = f"{saving_dir}/episode_{episode_id}/t{temp}_seed{seed}/agent_{actor}_turn_{turn}"
                
                create_folder_if_not_there(data_dir)
                
                save_path = f'{data_dir}/states.npy'
                attn_save_path = f'{data_dir}/attn.npy'
                
                print(f"Saving states to: {save_path}")
                np.save(save_path, states)
                np.save(attn_save_path, attn_states)
                
        except Exception as e:
            print(f"Error processing turn {turn}: {e}")
            continue
    
    # Cleanup
    del env
    print(f"Completed state saving for episode {episode_id}")
    return True

def main():
    """
    Main function to run state-only saving for specified episodes.
    """
    parser = argparse.ArgumentParser(description='Save internal states without generating conversations')
    parser.add_argument('--model_1', type=str, default="Mistral-7B-Instruct-v0.3")
    parser.add_argument('--model_2', type=str, default="Mistral-7B-Instruct-v0.3") 
    parser.add_argument('--affected_type', type=str, default="None")
    parser.add_argument('--value', type=float, default=0)
    parser.add_argument('--random_profile', action="store_true")
    parser.add_argument('--another_read', action="store_true")
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='Seeds to process')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature used in original generation')
    
    args = parser.parse_args()
    args.random_profile = True
    all_random_profiles = []
    args.seeds = [0, 1, 2]
    # Setup same naming convention as original
    if args.value == 0.0:
        args.value = 0
    
    if args.another_read:
        random_suff = "_anotherread"
    else:
        random_suff = ""
    
    model_1_name = f"{args.model_1}_{args.affected_type}_{args.value}"
    model_2_name = f"{args.model_2}_{args.affected_type}_{args.value}"
    save_dir_base = f"sotopia_results/{model_1_name}_{model_2_name}{random_suff}"
    save_prompt_base = f"sotopia_results/dialogs/{model_1_name}_{model_2_name}{random_suff}/prompt_records"

    record_dialog_base = f"sotopia_results/dialogs/{model_1_name}_{model_2_name}"
    
    print(f"Using dialog base: {record_dialog_base}")
    print(f"Saving states to: {save_dir_base}")
    print(f"Saving prompts to: {save_prompt_base}")
    
    success_count = 0
    total_count = 0

    env = None
    for seed in args.seeds:
        del env
        env = SotopiaEnv(
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            max_turns=16,
            saving_dir_base=save_dir_base,
            save_states=True, 
            save_states_only_dialog=False,
            save_states_dialog_imperson=False,
            save_states_without_goal=False,
            probing_goal=False,
            probing_self_goal=False,
            max_new_tokens=1,
            verbose=False,
            temperature=args.temp,
            seed=seed
        )
        set_seed(seed)
        all_random_profiles = []
        for i in range(450):
            all_random_profiles.append(random_sample_agents_from_pool_with_descriptions_standalone())

        for episode_id in tqdm(range(450)):
            total_count += 1
            
            # Check if prompt records exist
            prompt_file = f'{record_dialog_base}/prompt_records/{episode_id}_temp{args.temp}_seed{seed}.csv'
            if not os.path.exists(prompt_file):
                print(f"Prompt file not found: {prompt_file}, skipping...")
                continue

            # Reset environment to load the same episode scenario
            env.reset(env_id=episode_id)
            
            # Calculate how many states were saved from prompt_file
            original_turns = len(load_prompts_from_csv(prompt_file))
            # Check if states already exist to avoid recomputation
            state_check_path = f"{save_dir_base}/{env.source_name}/episode_{episode_id}/t{args.temp}_seed{seed}/agent_{get_agent(original_turns - 1)}_turn_{original_turns - 1}/states.npy"
            if os.path.exists(state_check_path) and os.path.getsize(state_check_path) > 0:
                print(f"States already exist for episode {episode_id}, seed {seed}, skipping...")
                continue

            success = save_states_only(
                env=env,
                model_1_name=model_1_name,
                model_2_name=model_2_name, 
                episode_id=episode_id,
                seed=seed,
                temp=args.temp,
                max_new_tokens=1,
                record_dialog_base=record_dialog_base,
                saving_dir_base=save_dir_base,
                save_prompt_base=save_prompt_base,
                token_position=-1,
                random_profile=args.random_profile,
                now_profiles=all_random_profiles[episode_id],
                another_read=args.another_read,
            )
            
            if success:
                success_count += 1
    
    print(f"Completed: {success_count}/{total_count} episodes processed successfully")

if __name__ == "__main__":
    main()