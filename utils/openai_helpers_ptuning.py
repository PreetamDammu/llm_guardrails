import os
from openai import AzureOpenAI
from dotenv import dotenv_values
import time

secrets = dotenv_values(".env")
personal_api_key = secrets['AZURE_OPENAI_KEY']
azure_endpoint = secrets['AZURE_OPENAI_ENDPOINT']

client = AzureOpenAI(
    azure_endpoint = azure_endpoint, 
    api_key=personal_api_key,  
    api_version="2024-02-15-preview"
    )


def query_openai_model(model_name, prompt):
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        temperature = 0.7,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    return response.choices[0].message.content

def query_evaluator_openai_model(model_name, prompt, temperature=0):
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=temperature,  # Control the randomness of the output
        response_format={"type": "json_object"}  # Ensure output is in JSON format
    )

    return response.choices[0].message.content

def query_evaluator_openai_mode_whole_prompt(model_name, prompt, temperature=0):
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        temperature=temperature,  # Control the randomness of the output
        response_format={"type": "json_object"}  # Ensure output is in JSON format
    )

    return response.choices[0].message.content


def get_response(model_name, prompt, temperature, retries=5, wait_time=2):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    while ct < 3:
        ct += 1
        try:
            response = client.chat.completions.create(
                model=model_name, # model = "deployment_name".
                messages=prompt, 
                temperature=temperature,  # Control the randomness of the output
                response_format={"type": "json_object"}  # Ensure output is in JSON format
            )
            # print(response)
            num_tokens += response.usage.total_tokens
            num_completion_tokens += response.usage.completion_tokens
            num_prompt_tokens += response.usage.prompt_tokens
            
            return response, num_tokens, num_completion_tokens, num_prompt_tokens #.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait_time * attempt} seconds...")
                time.sleep(wait_time * attempt)
            else:
                print("All attempts failed. Returning None.")
                return None
        finally:
            attempt += 1


    return None, num_tokens, num_completion_tokens, num_prompt_tokens


# def query_evaluator_openai_model(client, model_name, prompt, temperature=0):
#     response = client.chat.completions.create(
#         model=model_name,  # Specify the model, e.g., "gpt-3.5-turbo"
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=temperature,  # Control the randomness of the output
#         response_format={"type": "json_object"}  # Ensure output is in JSON format
#     )

#     return response.choices[0].message['content']  # Return the JSON-formatted content
