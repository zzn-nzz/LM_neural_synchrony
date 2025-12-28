import csv
from openai import OpenAI
import os
import json
from tqdm import tqdm
from utils import *
import time

api_num = 0
prompt_cache = {}
CACHE_FILE = 'cache/oss_prompt_cache_base_models.json'
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)


class InterruptProtection:
    def __init__(self):
        self.old_handler = None
    
    def __enter__(self):
        import signal
        self.old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import signal
        if self.old_handler is not None:
            signal.signal(signal.SIGINT, self.old_handler)

def load_cache():
    global prompt_cache
    
    with InterruptProtection():
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    prompt_cache = json.load(f)
                with open('cache/oss_prompt_cache.json', 'r', encoding='utf-8') as f:
                    base_cache = json.load(f)
                prompt_cache.update(base_cache)
                print(f"Loaded cache with {len(prompt_cache)} entries")
            except json.JSONDecodeError:
                print(f"Error loading cache file, starting with empty cache")
                prompt_cache = {}
            except Exception as e:
                print(f"Unexpected error loading cache: {e}")
                prompt_cache = {}
        else:
            prompt_cache = {}
            print("No cache file found, starting with empty cache")

def save_cache():
    import tempfile
    
    with InterruptProtection():
        temp_filename = None
        try:
            cache_dir = os.path.dirname(CACHE_FILE) or '.'
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding='utf-8', 
                dir=cache_dir,
                prefix='cache_temp_',
                suffix='.json',
                delete=False
            ) as temp_file:
                temp_filename = temp_file.name
                json.dump(prompt_cache, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_filename, CACHE_FILE)
            print(f"Cache saved with {len(prompt_cache)} entries")
            
        except Exception as e:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except:
                    pass
            print(f"Error saving cache: {e}")
            raise

def get_final_reward(intro, dialog):
    global api_num, prompt_cache
    dialog = '\n'.join(dialog)
    history = intro + dialog
    # print(dialog)

    template = """{history}{schema}"""
    with open('sotopia_utils/output_schema.txt', 'r') as f:
        schema = f.read()

    prompt = template.format(history=history, schema=schema)
    
    if prompt in prompt_cache:
        try:
            prompt_cache[prompt] = prompt_cache[prompt].replace('Score":', 'Score",').replace('\n', '').replace('  ', '').replace(']]}', ']}')
            return eval(prompt_cache[prompt]), prompt
        except Exception as e:
            print(f"Error evaluating output: {e}")
            print(f"Problematic output: {prompt_cache[prompt]}")
            del prompt_cache[prompt]
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="./gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=0,
        top_p=1,
        n=1,
        max_completion_tokens=500000
    )
    print(f"Completion took {time.time() - start_time} seconds.")
    output_text = response.choices[0].message.content
    # print(output_text)
    output_text = output_text.replace('```json\n', '')
    output_text = output_text.replace('```', '')
    output_text = output_text.replace('Score":', 'Score",').replace('\n', '').replace('  ', '').replace(']]}', ']}')
    
    prompt_cache[prompt] = output_text
    save_cache()
    try:
        result = eval(output_text)
        return result, prompt
    except Exception as e:
        print(f"Error evaluating output: {e}")
        print(f"Problematic output: {output_text}")
        return None, prompt

load_cache()

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
        combs.append((m1, m2))

seeds = [0, 1, 2, 3, 4]
print(combs)

for model_name in combs:
    for seed in seeds:
        if not isinstance(model_name, str):
            model_1_name, model_2_name = model_name
            model_name = f"{model_1_name}_{model_2_name}"
            
        print(model_name, seed)
        
        p = f'sotopia_results/dialogs/{model_name}'
        create_folder_if_not_there(f'{p}/eval_oss')

        suf = ""
        for i in tqdm(range(450)):
            eval_file_path = f'{p}/eval_oss/{i}_temp0.7_seed{seed}.json'
            if os.path.exists(eval_file_path) and os.path.getsize(eval_file_path) > 0:
                # Examine if the content is not NoneType
                try:
                    with open(eval_file_path, 'r', encoding='utf-8') as eval_file:
                        content = json.load(eval_file)
                        if content.get("result") is not None:
                            continue
                except Exception as e:
                    print(f"Error reading {eval_file_path}: {e}, re-evaluating.")

            with open(f'{p}/{i}_temp0.7_seed{seed}{suf}.csv', mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)
                
                for row in reader:
                    intro, dialog, intended_dialog = row
                    intended_dialog = eval(intended_dialog)
                    result, prompt = get_final_reward(intro, intended_dialog)

                    record = {
                        "result": result,
                        "prompt": prompt
                    }
                    with open(eval_file_path, mode='w', encoding='utf-8') as eval_file:
                        json.dump(record, eval_file, indent=2)
