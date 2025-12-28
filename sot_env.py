import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LM_hf import *
from typing import Literal
from sotopia_utils.utils import  *
import random
from openai import OpenAI
import os
import numpy as np
import re
import json
import pickle
import time
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from utils import *

current_file_path = os.path.abspath(__file__)
directory = os.path.dirname(current_file_path)


def getLM(model_name_all, model_paths, temperature, seed):
    if "_" not in model_name_all:
        return LM_nnsight(model_paths[model_name], model_name, temperature=temperature, seed=seed)
    
    parsed_name = model_name_all.split('_')
    model_name = parsed_name[0]
    
    return LM_nnsight(model_paths[model_name], model_name, temperature=temperature, seed=seed)

class SotopiaEnv():
    def __init__(self, 
                model_1_name: str,
                model_2_name: str,
                token_position = -1,
                env_model: str = "gpt-4",
                max_turns: int = 20,
                temperature: float = 0.7,
                test_baseline: bool = False,
                test_gpt: bool = False,
                test_mode: str = "None",
                max_new_tokens: int = 50,
                save_states: bool = False,
                save_states_only_dialog: bool = False,
                save_states_dialog_imperson: bool = False,
                save_states_without_goal: bool = False,
                probing_goal: bool = False,
                probing_self_goal: bool = False,
                saving_dir: str = None,
                saving_dir_base: str = None,
                mask_history: bool=False,
                verbose: bool=False,
                seed: int=0,
    ):
        with open('model_paths.json', 'r') as f:
            model_paths = json.load(f)
        
        if saving_dir is not None:
            create_folder_if_not_there(saving_dir)

        if ('None' in model_1_name or 'Replace' not in model_1_name) and model_1_name == model_2_name:
            self.model_1 = getLM(model_1_name, model_paths, temperature, seed)
            self.model_2 = self.model_1
            print('Loaded only one language model for both agents')
        else:
            self.model_1 = getLM(model_1_name, model_paths, temperature, seed)
            self.model_2 = getLM(model_2_name, model_paths, temperature, seed)
        
        if 'Self' in model_1_name and 'Self' in model_2_name:
            self.mask_other_utterances = True
        else:
            self.mask_other_utterances = False

        self.actor_role = 'agent1'
        self.env_model = env_model
        self.cur_turn = 0
        self.max_turns = max_turns
        self.temperature = temperature
        self.agent1_intro = ''
        self.agent2_intro = ''
        self.complete_intro = ''
        self.cur_dialog = []
        self.intended_dialog = []
        self.cur_rewards = []
        self.actions = []
        self.cur_states = []
        self.next_states = []
        self.terminations = []
        self.p1_scores = None
        self.p2_scores = None
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.agent1_name = ''
        self.agent2_name = ''
        self.agent1_goal = ''
        self.agent2_goal = ''
        self.additional_instruction = "If your goal as the agent has been achieved, you should leave the conversation by saying \"LEAVE\". If your goal has not been achieved, you should say something to interact with the other. Keep your words concise.\n"
        self.test_mode = test_mode
        self.max_new_tokens = max_new_tokens
        self.competition_episodes = []
        self.cooperation_episodes = []
        self.episode = 0
        self.save_states = save_states
        self.save_states_only_dialog = save_states_only_dialog
        self.save_states_dialog_imperson = save_states_dialog_imperson
        self.save_states_without_goal = save_states_without_goal
        self.probing_goal = probing_goal
        self.probing_self_goal = probing_self_goal
        self.load_scenarios()
        self.prompts = []
        self.prompts_only_dialogs = []
        self.prompts_dialog_imperson = []
        self.prompts_without_goal = []
        self.prompts_probing_goal = []
        self.prompts_probing_self_goal = []
        self.token_position = token_position
        self.mask_history = mask_history
        self.saving_dir_base = saving_dir_base
        self.seed = seed
        self.source_name = None

    def load_scenarios(self):
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/env_agent_combos.json'), 'r') as f:
            self.env_agent_combos = json.load(f)

        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/envs.json'), 'r') as f:
            self.envs = json.load(f)

        cooperation_episodes = []
        competition_episodes = []
        for k, v in self.envs.items():
            if v["source"] == "mutual_friends":
                cooperation_episodes.append(k)
            if v["source"] == "craigslist_bargains":
                competition_episodes.append(k)

        cooperation_episodes_number = []
        competition_episodes_number = []
        for i, comb in enumerate(self.env_agent_combos):
            if comb["env_id"] in cooperation_episodes:
                cooperation_episodes_number.append(i)
            elif comb["env_id"] in competition_episodes:
                competition_episodes_number.append(i)
        
        self.cooperation_episodes = cooperation_episodes_number
        self.competition_episodes = competition_episodes_number    
        # print(len(cooperation_episodes), len(competition_episodes))

        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/agents.json'), 'r') as f:
            self.agent_profiles = json.load(f)

    def get_current_prompt(self, only_dialog_imperson = False, without_goal = False, probing_goal = False, probing_self_goal = False):
        # accumulate the current dialog into a prompt
        prompt = "\n".join(self.cur_dialog)
        prompt = self.agent1_intro + prompt if self.cur_turn % 2 == 0 else self.agent2_intro + prompt
        return prompt
    
    def _clean_tags(self, text):
        # use regex to remove the tags in the text
        return re.sub(r'<.*?>', '', text)
    
    def _mask_intro(self, intro, role: int):
        # Mask the opponent's goal and secret in the intro
        opponent_name = self.agent2_name if role == 0 else self.agent1_name
        # mask secrets
        if f"<p viewer='agent_{1-role}'>" in intro:
            # replace the text contained in the tag
            first_name, last_name = opponent_name.split()
            intro = re.sub(fr"<p viewer='agent_{1-role}'>.*?<\/p>", "", intro)

        intro = re.sub(fr" <p viewer=\"environment\">.*?<\/p>", "", intro)

        # mask goal
        
        intro = re.sub(fr"{opponent_name}'s goal: .*?(\n|$)", f"{opponent_name}'s goal: Unknown\n", intro)
        intro = re.sub(
            fr"{opponent_name}'s goal: .*?(\n<extra_info>.*?\n</extra_info>|\n|$)",
            f"{opponent_name}'s goal: Unknown\n",
            intro,
            flags=re.DOTALL
        )
        return intro
    
    def _judge_terminate(self) -> bool:
        # 1. exceeds the max_turns

        if self.cur_turn >= self.max_turns:
            return True
        elif any([_ in self.intended_dialog[-1][:10].lower() or _ in self.intended_dialog[-1][-10:].lower() for _ in ["left", "leave"]]):
            return True
        else:
            return False
        
    def save_conversation_history(self, seed):
        # the seed is how the action is randomized in the actor not how the engine works
        # convert to numpy arrays
        self.cur_rewards = np.array(self.cur_rewards)
        self.cur_states = np.array(self.cur_states)
        self.actions = np.array(self.actions)
        self.next_states = np.array(self.next_states)
        self.terminations = np.array(self.terminations)

        tbd = dict(
            complete_intro=self.complete_intro, 
            dialog=self.cur_dialog, 
            rewards=self.cur_rewards,
            observations=self.cur_states,
            actions=self.actions,
            next_observations=self.next_states,
            terminals=self.terminations,
            agent1_scores=self.p1_scores,
            agent2_scores=self.p2_scores,
        )
        with open(os.path.join(self.saving_dir, f"env_id={self.env_id}_role={self.actor_role}_seed={seed}.pkl"), 'wb') as f:
            pickle.dump(tbd, f)

    def reset(self, env_id = None, actor_role = "agent1"):
        # Reset the environment and return the initial state
        self.cur_dialog = []
        self.intended_dialog = []
        self.cur_rewards = []
        self.cur_states = []
        self.actions = []
        self.next_states = []
        self.terminations = []
        self.prompts = []

        # Sample a new scenario
        self.actor_role = actor_role

        if env_id is None:
            env_id = random.randint(0, 449)
        self.episode = env_id
        env_agent_combo_storage = self.env_agent_combos[env_id]
        
        self.env_id = env_id
        env_id = env_agent_combo_storage['env_id']
        
        env_profile = self.envs[env_id]

        source_name =  env_profile["source"]
        self.source_name = source_name
        self.saving_dir = f"{self.saving_dir_base}/{source_name}"

        agent_ids = env_agent_combo_storage['agent_ids']
        agent_profiles = [self.agent_profiles[id] for id in agent_ids]

        self.agent1_name = agent_profiles[0]['first_name'] + " " + agent_profiles[0]['last_name']
        self.agent2_name = agent_profiles[1]['first_name'] + " " + agent_profiles[1]['last_name']

        self.agent1_goal = f"{env_profile['agent_goals'][0]}"
        self.agent2_goal = f"{env_profile['agent_goals'][1]}"
        background = AgentBackground(
            scenario=env_profile['scenario'],
            p1_background=get_bio(
                env_profile['relationship'],
                agent_profiles[0],
                agent_id=0,
            ),
            p2_background=get_bio(
                env_profile['relationship'],
                agent_profiles[1],
                agent_id=1,
            ),
            p1_goal=f"{env_profile['agent_goals'][0]}",
            p2_goal=f"{env_profile['agent_goals'][1]}",
            p1_name=self.agent1_name,
            p2_name=self.agent2_name,
        )
        # rewrite test.py; save prompts
        # form the prompt
        agent1 = self.agent1_name
        agent2 = self.agent2_name
        self.agent1_intro = self._mask_intro(background.to_natural_language(agent1), 0) + self.additional_instruction
        self.agent2_intro = self._mask_intro(background.to_natural_language(agent2), 1) + self.additional_instruction
        
        self.complete_intro = self._clean_tags(background.to_complete_intro())
        self.agent1_intro = self._clean_tags(self.agent1_intro)
        self.agent2_intro = self._clean_tags(self.agent2_intro)

        self.agent1_goal = self._clean_tags(self.agent1_goal)
        self.agent2_goal = self._clean_tags(self.agent2_goal)

        self.cur_turn = 0
        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent1_name} said:")
        self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent1_name} said:")

        if self.actor_role == 'agent2':
            self._act(action=None, actor=False)
            self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:")
            self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:")

        return None
    
    def get_final_reward(self):
        template = """{history}{schema}"""
        history = self.complete_intro + "\n".join(self.cur_dialog)
        with open(os.path.join(directory, 'sotopia_utils/output_schema.txt'), 'r') as f:
            schema = f.read()

        prompt = template.format(history=history, schema=schema)

        # call self.env_model to get the reward
        client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )
        
        response = client.chat.completions.create(
            model=self.env_model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        output_text = response.choices[0].message.content

        # parse output_text to get the reward
        output_parser = output_parser=PydanticOutputParser[EnvResponse](
            pydantic_object=EnvResponse
        )
        try:
            parsed_result = output_parser.parse(output_text)
    
        except Exception as e:
            
            reformat_parsed_result = format_bad_output(
                output_text, format_instructions=output_parser.get_format_instructions()
            )
            parsed_result = output_parser.parse(reformat_parsed_result)

        d = parsed_result.agent_1_evaluation.dict() if self.actor_role == 'agent1' else parsed_result.agent_2_evaluation.dict()
        overall_score = 0
        for dimension in d:
            overall_score += d[dimension][1]

        overall_score /= len(d)
        return overall_score, parsed_result.agent_1_evaluation.dict(), parsed_result.agent_2_evaluation.dict()
    
    def get_prompt_for_fake_dialog(self, agent=1, type=None):
        if agent == 1:
            intro = self.agent1_intro
        else:
            intro = self.agent2_intro
        res = '\n'.join(intro.split('\n')[:-1]).replace(self.additional_instruction[:-1], '')
        if type == 0: # Relevant
            res += "\nGenerate 16 sentences in the format of 1. 'Sentence_1', 2. 'Sentence_2' ..., 16. 'Sentence_16', simulating possible words the agent might say in this interaction. Keep in mind the agent's goal. Generation:"
        elif type == 1: # Irrelevant
            res += "\nGenerate 16 sentences in the format of 1. 'Sentence_1', 2. 'Sentence_2' ..., 16. 'Sentence_16'. The sentences should be the least likely things the agent would say in this interaction. Generation:"
        return res
        # print(self.agent1_intro)
        # print(self.agent1_goal)

    def _act(self, actor: int, save_states: bool = False, turn: int = 0):
        with torch.no_grad():
            prompt = self.get_current_prompt()
            self.prompts.append(prompt)
            if actor == 1:
                now_model = self.model_1
            else:
                now_model = self.model_2
                
            if save_states is True:
                data_dir = f"{self.saving_dir}/episode_{self.episode}/agent_{actor}_turn_{turn}"
                if self.temperature != 0:
                    data_dir = f"{self.saving_dir}/episode_{self.episode}/t{self.temperature}_seed{self.seed}/agent_{actor}_turn_{turn}"
                # print(self.temperature, data_dir)
                create_folder_if_not_there(data_dir)
                save_path = f'{data_dir}/states.npy'
                attn_save_path = f'{data_dir}/attn.npy'
                intended_result, result, states, attn_states = now_model.generate_response_and_get_state(prompt=prompt, 
                                                                           max_new_tokens = self.max_new_tokens,
                                                                           layer="All", token=self.token_position)
                # print(states.shape)
                np.save(save_path, states)
                np.save(attn_save_path, attn_states)
            else:
                result = now_model.generate_response(prompt, max_new_tokens=self.max_new_tokens)

        result = result.split('\n')[0]
        intended_result = intended_result.split('\n')[0]

        if ': ' in result:
            result = ''.join(result.split(': ')[1:])
        if ': ' in intended_result:
            intended_result = ''.join(intended_result.split(': ')[1:])

        self.cur_turn += 1
        self.cur_dialog[-1] += result
        self.intended_dialog[-1] += intended_result
        

    def step(self):
        # self.cur_states.append(cur_state.detach().cpu().numpy())
        # self.actions.append(action.detach().cpu().numpy())
        # take an action
        self._act(actor=1, save_states=self.save_states, turn=self.cur_turn)

        # judge whether terminate or not
        done = self._judge_terminate()

        if done:
            return done 
        
        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")
        self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")

        self._act(actor=2, save_states=self.save_states, turn=self.cur_turn)

        done = self._judge_terminate()
        
        if done:
            return done

        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")
        self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")

        self.cur_rewards.append(0)
        self.terminations.append(False)

        # return the new state, reward, done, info
        return done

def random_sample_agents_from_pool_with_descriptions_standalone(agents_file_path=None,
                                                               relationship_type=None, seed=None):
    """
    Standalone function to randomly sample agent pairs from pool with descriptions.
    """
    
    if relationship_type is None:
        relationship_type = RelationshipType.stranger
    
    if agents_file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_file_path = os.path.join(current_dir, 'sotopia_utils/sotopia_data/agents.json')
    
    with open(agents_file_path, 'r') as f:
        agent_profiles = json.load(f)
    
    all_agent_ids = list(agent_profiles.keys())
    pairs = []
    
    # Clean tags function
    def clean_tags(text):
        return re.sub(r'<.*?>', '', text)
    
    sampled_agents = random.sample(all_agent_ids, 2)
    
    agent1_profile = agent_profiles[sampled_agents[0]].copy()
    agent1_profile['agent_id'] = sampled_agents[0]
    agent2_profile = agent_profiles[sampled_agents[1]].copy()
    agent2_profile['agent_id'] = sampled_agents[1]
    
    # Generate bios
    agent1_bio = get_bio(relationship_type, agent1_profile, agent_id=0)
    agent2_bio = get_bio(relationship_type, agent2_profile, agent_id=1)
    
    agent1_clean_bio = clean_tags(agent1_bio)
    agent2_clean_bio = clean_tags(agent2_bio)
    
    pair_info = {
        'agent1_profile': agent1_profile,
        'agent2_profile': agent2_profile,
        'agent1_bio': agent1_bio,
        'agent2_bio': agent2_bio,
        'agent1_clean_bio': agent1_clean_bio,
        'agent2_clean_bio': agent2_clean_bio,
        'relationship_type': relationship_type
    }
    
    return pair_info

if __name__ == "__main__":
    env = SotopiaEnv('1', '1')