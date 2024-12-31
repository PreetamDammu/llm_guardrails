import os
from openai import AzureOpenAI
from dotenv import dotenv_values

secrets = dotenv_values(".env")
personal_api_key = secrets['AZURE_OPENAI_KEY']
azure_endpoint = secrets['AZURE_OPENAI_ENDPOINT']

client = AzureOpenAI(
    azure_endpoint = azure_endpoint, 
    api_key=personal_api_key,  
    api_version="2023-05-15"
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
