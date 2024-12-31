import os
import json
import time
import random
import datetime
import pandas as pd
import argparse
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI

from utils.evaluator_helpers_ptuning import (
    get_evaluation_prompt,
    get_evaluation_prompt_few_shot_simple_example,
    get_evaluation_prompt_binarized,
    ZERO_SHOT_INDIVIDUAL_METRIC_PROMPT,
    scale_string,
    metrics_list,
    categorization_threat_definition,
    morality_threat_definition,
    competence_threat_definition,
    realistic_threat_definition,
    symbolic_threat_definition,
    disparagement_definition,
    opportunity_harm_definition
)
from utils.openai_helpers_ptuning import (
    query_evaluator_openai_model,
    get_response,
    query_openai_model,
    query_evaluator_openai_mode_whole_prompt
)

random.seed(42)

def save_to_csv(df, model, result_list, filename, temp=False):
    """Save DataFrame to a CSV file with results."""
    df[f'evaluated_result_{model}'] = result_list
    path = f'outputs/promptTuning/{filename}'
    df.to_csv(path, index=False)

def write_to_log(text, file_name):
    """Append log entry to a log file with a timestamp."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{now} - {text}\n"
    log_path = f"logs/log_evaluator_{file_name}.txt"
    with open(log_path, "a") as log_file:
        log_file.write(log_entry)

def extract_metadata(df_evaluation):
    """Extract metadata required for prompt construction from evaluation DataFrame."""
    list_metadata = []

    for i, row in df_evaluation.iterrows():
        background = row["prompt"].split("Background context:\n")[1].split("for ")[1]
        team_context_background = background.split(".")[0].strip()

        background_text = background.replace(team_context_background, '')
        colleague_info, conversation_part = background_text.split('Conversation:\n')
        colleague_info = colleague_info.strip()[2:]
        list_dialogue = conversation_part.split("\n")

        initial_dialogue = "\n".join(list_dialogue[:2])
        start_conversation = list_dialogue[2]
        conversation = (start_conversation + row["generated_text"])
        
        list_metadata.append([i, team_context_background, colleague_info, initial_dialogue, conversation])

    return list_metadata

def construct_prompt(metric_name, definition, team_context_background, colleague_info, initial_dialogue, conversation):
    """Construct a prompt using provided metadata."""
    prompt = ZERO_SHOT_INDIVIDUAL_METRIC_PROMPT
    prompt = prompt.replace("[METRIC]", metric_name)
    prompt = prompt.replace("[DEFINITION]", definition)
    prompt = prompt.replace("[FOUR-POINT-SCALE]", scale_string)
    prompt = prompt.replace("[TEAM-CONTEXT]", team_context_background)
    prompt = prompt.replace("[COLLEAGUE-INFORMATION]", colleague_info)
    prompt = prompt.replace("[INITIAL-DIALOGUE]", initial_dialogue)
    prompt = prompt.replace("[CONVERSATION]", conversation)
    return prompt

def create_metric_prompts_per_conversation(list_metadata, prompt_system_pairwise):
    """Create prompts for each metric per conversation."""
    dict_prompts = {}
    definitions = [
        categorization_threat_definition,
        morality_threat_definition,
        competence_threat_definition,
        realistic_threat_definition,
        symbolic_threat_definition,
        disparagement_definition,
        opportunity_harm_definition
    ]

    for metadata in list_metadata:
        metric_prompts = []
        team_context_background, colleague_info, initial_dialogue, conversation = metadata[1:]

        for metric_name, definition in zip(metrics_list, definitions):
            prompt_content = construct_prompt(metric_name, definition, team_context_background, colleague_info, initial_dialogue, conversation)
            prompt = [{"role": "system", "content": prompt_system_pairwise}, {"role": "user", "content": prompt_content}]
            metric_prompts.append(prompt)

        dict_prompts[metadata[0]] = metric_prompts
    
    return dict_prompts

def create_dataframe(len_data):
    """Create a DataFrame with a single column indicating index length."""
    return pd.DataFrame(range(len_data), columns=["length"])

def extract_prompts_intermediate(dict_prompts, start_index):
    """Extract prompts starting from a specific index."""
    return {index: prompts for index, prompts in dict_prompts.items() if index >= start_index}

def run_evaluation(original_df_size, dict_prompts, model_name, temperature, file_name, log_file_name, list_evaluated):
    """Run evaluation by querying the model and saving intermediate and final results."""
    df_mapping = create_dataframe(original_df_size)

    for index, prompt_list in tqdm(dict_prompts.items()):
        aggregate_json = {}

        for prompt in prompt_list:
            response = ""

            while not response:
                response = query_evaluator_openai_mode_whole_prompt(model_name, prompt, temperature)
                if not response:
                    print("Retrying...")
                    time.sleep(45)

            try:
                json_response = json.loads(response)
                aggregate_json.update(json_response)
            except json.JSONDecodeError:
                response = ""

        list_evaluated[index] = json.dumps(aggregate_json)

        if (index + 1) % 10 == 0:
            save_to_csv(df_mapping, model_name, list_evaluated, file_name, temp=True)
            write_to_log(f"Saved intermediate results at index {index}", log_file_name)

    save_to_csv(df_mapping, model_name, list_evaluated, file_name, temp=False)
    write_to_log("Saved final results", log_file_name)

def main():
    """Main function to execute the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for model prompts.")
    parser.add_argument("--evaluation_file", type=str, required=True, help="Path to the evaluation file.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for model evaluation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--file_name", type=str, required=True, help="Output file name for evaluated results.")
    parser.add_argument("--log_file_name", type=str, required=True, help="Log file name for the evaluation run.")

    args = parser.parse_args()

    df_evaluation = pd.read_csv(args.evaluation_file)
    list_metadata = extract_metadata(df_evaluation)
    dict_prompts = create_metric_prompts_per_conversation(list_metadata, "You are a social science expert...")

    list_evaluated = ["" for _ in range(len(dict_prompts))]

    run_evaluation(len(dict_prompts), dict_prompts, args.model_name, args.temperature, args.file_name, args.log_file_name, list_evaluated)

if __name__ == "__main__":
    main()
