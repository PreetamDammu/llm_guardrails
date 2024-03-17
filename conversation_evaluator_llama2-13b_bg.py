import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
import random
random.seed(42)

from utils.evaluator_helpers import get_evaluation_prompt

import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "Llama-2-13b-chat-hf"
fname = 'conv_db_eight_models_combined_TestSplit'

df = pd.read_csv(f'outputs/{fname}.csv')

df

def save_to_csv(df, res, filename, temp=False):
    df[f'evaluated_result_{model}'] = res
    if temp:
        df.to_csv(f'outputs/temps/{filename}', index=False)
    else:
        df.to_csv(f'outputs/evaluations/{filename}', index=False)

def write_to_log(text, modelname):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")  # Format the date and time

    # Create the log entry with the timestamp
    log_entry = f"{timestamp} - {text}\n"

    # Write to the log file
    with open(f"logs/log_evaluator_{modelname}_{fname}.txt", "a") as log_file:
        log_file.write(log_entry)

    return

print('Started running...')
write_to_log('Started running...', model)


list_evaluated = ["TBD"] * len(df)

for i in tqdm(range(len(df))): #len(conv_db)

    prompt = df.iloc[i]['prompt']
    conv = df.iloc[i]['generated_text']

    eval_prompt = get_evaluation_prompt(prompt, conv)

    completion = openai.completions.create(model=model, prompt=eval_prompt, max_tokens=1024, temperature=0.2)
    
    evaluatedConv = completion.choices[0].text
    
    list_evaluated[i] = evaluatedConv

    if (i+1) % 10 == 0:
        save_to_csv(df, list_evaluated, f'evaluated_conversations_{model}_{fname}_temp.csv', temp=True)
        print(f'Saved Intermediate [{i}/{len(df)}]')
        write_to_log(f'Saved Intermediate [{i}/{len(df)}]', model) 

save_to_csv(df, list_evaluated, f'evaluated_conversations_{model}_{fname}_final.csv')
print('Saved Final!')
write_to_log('Saved Final!', model)

