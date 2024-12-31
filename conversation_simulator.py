import os
import argparse
import datetime
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from datasetGenerator import getDataset
import openai

random.seed(42)

def save_to_csv(res, filename):
    """Save results to a CSV file."""
    df = pd.DataFrame(res, columns=['id', 'concept', 'job', 'prompt', 'model', 'generated_text'])
    df.to_csv(f'outputs/{filename}', index=False)

def write_to_log(text, modelname):
    """Write log entry to a file with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {text}\n"
    log_path = f"logs/log_{modelname}.txt"
    with open(log_path, "a") as log_file:
        log_file.write(log_entry)

def main():
    """Main function to run the dataset generation and completion pipeline."""
    parser = argparse.ArgumentParser(description="Run dataset generation and prompt completions.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use for completions.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/", help="Base URL for the locally loaded model.")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing the dataset.")
    parser.add_argument("--temp_file", type=str, default="temp.csv", help="Temporary output file name.")
    parser.add_argument("--final_file", type=str, default="final.csv", help="Final output file name.")

    args = parser.parse_args()

    openai.api_key = args.api_key
    openai.base_url = args.base_url
    
    model = args.model
    conv_db = getDataset()

    print('Started running...')
    write_to_log('Started running...', model)

    for i in tqdm(range(args.start_index, len(conv_db))):
        prompt = conv_db[i][3]

        # Create a completion
        try:
            completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=512)
            generated_conv = completion.choices[0].text
        except Exception as e:
            write_to_log(f"Error processing index {i}: {str(e)}", model)
            generated_conv = ""

        conv_db[i] = conv_db[i] + [model, generated_conv]

        if (i + 1) % 10 == 0:
            save_to_csv(conv_db, f'conv_db_conversations_{args.temp_file}')
            print(f'Saved Intermediate [{i}/{len(conv_db)}]')
            write_to_log(f'Saved Intermediate [{i}/{len(conv_db)}]', model)

    save_to_csv(conv_db, f'conv_db_conversations_{args.final_file}')
    print('Saved Final!')
    write_to_log('Saved Final!', model)

if __name__ == "__main__":
    main()
