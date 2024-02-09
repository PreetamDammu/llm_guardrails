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
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    return response.choices[0].message.content

# import openai
# import time
# from dotenv import dotenv_values

# secrets = dotenv_values(".env")
# personal_api_key = secrets['OPENAI_API_KEY']

# from openai import OpenAI

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key=personal_api_key,
# )

# def query_chat_gpt(prompt, model_name, num_retries=10):
#     for _ in range(num_retries):
#         try:
#             response = client.chat.completions.create(
#                 model= model_name, 
#                 messages=[{"role": "user", "content": prompt}]
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(e)
#             print("Retrying...")
#             time.sleep(10)
#             continue
#     print(f"Failed after {num_retries} retries")
#     return None