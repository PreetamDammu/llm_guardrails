import os
from openai import AzureOpenAI
from dotenv import dotenv_values

secrets = dotenv_values(".env")
personal_api_key = secrets['AZURE_OPENAI_KEY']
azure_endpoint = secrets['AZURE_OPENAI_ENDPOINT']

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

def query_evaluator_openai_model(client, model_name, system, prompt, temperature=0):
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=[
            {"role": "system", "content": f"{system}"},
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=temperature,  # Control the randomness of the output
        response_format={"type": "json_object"}  # Ensure output is in JSON format
    )

    return response.choices[0].message.content


def get_response(client, messages, model_name, temperature):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    while ct < 3:
        ct += 1
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},  # Ensure output is in JSON format
                max_tokens=512,
                frequency_penalty=0,
                presence_penalty=0
            )
            num_tokens += response.usage.total_tokens
            num_completion_tokens += response.usage.completion_tokens
            num_prompt_tokens += response.usage.prompt_tokens
            
            return response, num_tokens, num_completion_tokens, num_prompt_tokens
        except Exception as e:
            print("Error")
            #logging.error(traceback.format_exc())
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
