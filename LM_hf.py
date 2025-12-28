from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
import torch
from transformers.generation.utils import GenerationConfig
from utils import *
import random

class BaseLM:
    def __init__(self, model_path, model_name, device="cuda", temperature=0., seed=0, **kwargs):
        self.device = device
        self.model_name = model_name
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None and self.model_name != "gpt2":
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_model.resize_token_embeddings(len(tokenizer))
        
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
        else:
            base_model.generation_config.do_sample = True
            base_model.generation_config.temperature = temperature
        
        base_model.generation_config.seed = seed
        if isinstance(base_model.generation_config.eos_token_id, list):
            base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id[0]
        else:
            base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
        
        base_model.to(self.device)
        base_model.eval()
        self.temperature=temperature
        self.seed=seed
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
    
    def get_inst(self, base_prompt):
        if self.model_name == "Mistral-7B-Instruct-v0.1":
            return f'[INST]{base_prompt}[/INST]'
        elif self.model_name == "Mistral-7B-Instruct-v0.2":
            return f'[INST]{base_prompt}[/INST]'
        elif self.model_name == "Mistral-7B-Instruct-v0.3":
            return f'[INST]{base_prompt}[/INST]'
        elif "Llama" in self.model_name and "Instruct" in self.model_name:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
{base_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"""
        elif "Llama" in self.model_name and "Chat" in self.model_name:
            return f'[INST]{base_prompt}[/INST]'
        elif self.model_name == "gemma-2-9b-it":
            return f"""<start_of_turn>user
{base_prompt}<end_of_turn>
<start_of_turn>model"""
        elif self.model_name == "gpt2":
            return base_prompt
        elif self.model_name == "Qwen2.5-3B-Instruct":
            return f'<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n'

    def get_response_format(self, resp):
        if self.model_name == "Mistral-7B-Instruct-v0.1":
            return resp.split('[/INST]')[-1]
        elif self.model_name == "Mistral-7B-Instruct-v0.2":
            return resp.split('[/INST]')[-1]
        elif self.model_name == "Mistral-7B-Instruct-v0.3":
            return resp.split('[/INST]')[-1]
        elif "Llama" in self.model_name and "Chat" in self.model_name:
            return resp.split('[/INST]')[-1]
        elif "Llama" in self.model_name:
            return resp.split('<|end_header_id|>')[-1].replace('<|eot_id|>', '')
        elif self.model_name == "gemma-2-9b-it":
            return resp.split('<start_of_turn>model')[-1]
        elif self.model_name == "gpt2":
            return resp
        elif self.model_name == "Qwen2.5-3B-Instruct":
            return resp.split('<|im_start|>assistant\n')[-1]
            
    def get_result_from_output(self, value):
        result = self.model.tokenizer.batch_decode(value)[0]
        result = self.get_response_format(result)
        # enh_print(result, 'red')
        result = result.replace('</s>', '')
        result = result.replace('<|im_end|>', '')
        result = result.strip()
        return result


class LM_nnsight(BaseLM):
    def __init__(self, model_path, model_name, device="cuda", temperature=0., seed=0., affected_level=0, affected_type="None"):
        super().__init__(model_path, model_name, device, temperature, seed)

    def generate_response(self, prompt, max_new_tokens=2000):
        prompt = self.get_inst(prompt)
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
            with generator.invoke(prompt) as invoker:
                output = self.model.generator.output.save()
        # print(output)
        return self.get_result_from_output(output.value)

    def generate_response_and_get_state(self, prompt, max_new_tokens, layer, token=-1, save_states_fake_agents=None, verbose=False, save_attn=False):
        reps = []
        attn_reps = []
        n_heads = self.model.model.config.num_attention_heads
        
        prompt = self.get_inst(prompt)
        prompt_tokens = self.model.tokenizer.encode(prompt)
        
        loi = [] # layer of interest
        if layer == "All":
            for l in self.model.model.layers:
                loi.append(l)
        else:
            loi.append(self.model.model.layers[layer])

        with self.model.generate(max_new_tokens=max_new_tokens) as tracer:
            with tracer.invoke(prompt) as invoker:
                for i, l in enumerate(loi):
                    reps.append([l.output[0][:, -1:, :].save()]) # at last token of prompt: producing the first token
                    if save_attn:
                        attn_reps.append([l.self_attn.output[0][:, -1:, :].save()])
                    if token == -1:
                        continue
                    for _ in range(max_new_tokens):
                        l.next()
                        l.self_attn.next()
                        reps[i].append(l.output[0].save())
                        if save_attn:
                            attn_reps[i].append(l.self_attn.output[0].save())

                original = self.model.generator.output.save()
        
        reps_numpy = []
        att_reps_numpy = []
        output_tokens_num = len(original.value[0].tolist()) - len(prompt_tokens)
        if save_attn:
            for rep, att_rep in zip(reps, attn_reps):
                hidden_states = torch.cat(rep[:output_tokens_num], dim=1).cpu().numpy()
                reps_numpy.append(hidden_states[0])
                
                atts = torch.cat(att_rep[:output_tokens_num], dim=1).cpu().numpy()
                reshaped_atts = atts.reshape(atts.shape[0], output_tokens_num, n_heads, -1)
                att_reps_numpy.append(reshaped_atts[0])
        else:
            for rep in reps:
                hidden_states = torch.cat(rep[:output_tokens_num], dim=1).cpu().numpy()
                reps_numpy.append(hidden_states[0])
            
        reps_numpy = np.array(reps_numpy)
        att_reps_numpy = np.array(att_reps_numpy)

        ret = self.get_result_from_output(original.value)

        return ret, ret, reps_numpy, att_reps_numpy

    def __call__(self, prompt, max_new_tokens=2000):
        ans = self.generate_response(prompt, max_new_tokens)
        return ans