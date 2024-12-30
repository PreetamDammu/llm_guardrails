import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
import random
random.seed(42)
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.openai_helpers import generate_gpt_response
from utils.file_helpers import concatenate_csv_files

# df = concatenate_csv_files('input/gpt4_prompts_for_preetam')

# df
# fname = 'sbic_gpt4_study1_prompts_cot'
fname = 'conv_db_eight_models_combined_abstain_prompts'
df = pd.read_csv(f'outputs/{fname}.csv')

# model = 'gpt-4-0125-preview'
model = 'gpt4-preview'
saveInterval = 20

def save_to_csv(df, res, filename, temp=False):
    df[f'evaluated_result_{model}'] = res
    if temp:
        df.to_csv(f'outputs/temps/{filename}', index=False)
    else:
        df.to_csv(f'outputs/evaluations/{filename}', index=False)
    return

def write_to_log(text, modelname, fname):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")  # Format the date and time

    # Create the log entry with the timestamp
    log_entry = f"{timestamp} - {text}\n"

    # Write to the log file
    with open(f"logs/log_evaluator_{modelname}_{fname}.txt", "a") as log_file:
        log_file.write(log_entry)

    return


# Assuming df, query_evaluator_openai_model, save_to_csv, write_to_log are defined
list_evaluated = ["TBD"] * len(df)

def evaluate_prompt(index, prompt):
    try:
        evaluation = generate_gpt_response(model, prompt)
        return index, evaluation
    except Exception as e:
        print(f'Error evaluating prompt at index {index}: {e}')
        return index, "Error"

max_threads = 5 # Adjust based on your system's capabilities

print('Started running...')
write_to_log('Started running...', model, fname)

print(f'Processing file: {fname}')
write_to_log(f'Processing file: {fname}', model, fname)


with ThreadPoolExecutor(max_workers=max_threads) as executor:
    # Map index and prompt to future tasks
    future_to_index = {executor.submit(evaluate_prompt, i, df.iloc[i]['prompt_abstain']): i for i in range(len(df))}

    # Iterate over future results as they are completed
    for future in tqdm(as_completed(future_to_index), total=len(df)):
        index = future_to_index[future]
        try:
            _, evaluation = future.result()
            list_evaluated[index] = evaluation
        except Exception as exc:
            print(f'Generated an exception: {exc}')
            write_to_log(f'Generated an exception: {exc}', model)

        # Save periodically after every 10 evaluations
        if (index + 1) % saveInterval == 0 or index + 1 == len(df):
            save_to_csv(df, list_evaluated, f'evaluated_conversations_{model}_{fname}_temp.csv', temp=True)
            print(f'Saved Intermediate [{index+1}/{len(df)}]')
            write_to_log(f'Saved Intermediate [{index+1}/{len(df)}]', model, fname)

# Save final results
save_to_csv(df, list_evaluated, f'evaluated_conversations_{model}_{fname}_final.csv')
print('Saved Final!')
write_to_log('Saved Final!', model, fname)




