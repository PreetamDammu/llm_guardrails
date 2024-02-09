Running LLM models locally requires at least 16GB GPU memory (some 13b models may require more depending on context length).
The following commands work well for a VM with 2 GPUs of 16GB each (Tesla V100)
Driver Version: 535.129.03   CUDA Version: 12.2

Steps to run LLMs locally.
1) Create a new conda environment to install dependencies
#conda create -n fc_env python=3.9
2) Install Requirements
# pip install requirements
3) Follow instructions to install FastChat
#pip3 install "fschat[model_worker,webui]"
4) Steps for FastChat LLM model local run
    a) Start the serve controller
    # python3 -m fastchat.serve.controller &

    b) Pick any one of the models and run the corresponding command

        Vicuna-7b-1.5
        # python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2 --max-gpu-memory 14GiB &

        Vicuna-13b-1.5
        # python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5 --num-gpus 2 --max-gpu-memory 14.5GiB &

        Microsoft/Orca-2-7b
        # python3 -m fastchat.serve.model_worker --model-path Microsoft/Orca-2-7b --num-gpus 2 --max-gpu-memory 14.5GiB &

        mosaicml/mpt-7b-chat
        # python3 -m fastchat.serve.model_worker --model-path mosaicml/mpt-7b-chat --num-gpus 2 --max-gpu-memory 14.5GiB &

        meta-llama/Llama-2-7b-chat-hf
        # python3 -m fastchat.serve.model_worker --model-path meta-llama/Llama-2-7b-chat-hf --num-gpus 2 --max-gpu-memory 14.5GiB &

        meta-llama/Llama-2-13b-chat-hf
        # python3 -m fastchat.serve.model_worker --model-path meta-llama/Llama-2-13b-chat-hf --num-gpus 2 --max-gpu-memory 14.5GiB &

    c) Make the loaded model available for querying at port 8000
    # python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &

5) Now, the local models can be queried directly from the .ipynb notebooks


For running Openai Models (GPT-3.5 and GPT-4)
1) Setup Azure Openai environment and deploy models
Link: https://learn.microsoft.com/en-gb/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython&pivots=programming-language-python

2) Place access credentials in .env file

3) Now, the openai model cans be queried directly from the .ipynb notebooks