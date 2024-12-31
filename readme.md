
# Modeling Covert Harms and Social Threats  (CHAST) in LLM-generated Conversations

**Paper Link**: https://aclanthology.org/2024.emnlp-main.1134.pdf

**Citation**: Dammu, Preetam Prabhu Srikar, et al. "" They are uncultured": Unveiling Covert Harms and Social Threats in LLM Generated Conversations." arXiv preprint arXiv:2405.05378 (2024).

### **System Requirements**
- **Driver Version**: 535.129.03
- **CUDA Version**: 12.2
- **Python Version**: 3.9


---

## **Table of Contents**
1. [Setting Up the Environment (Steps 1–4)](#setting-up-the-environment)
2. [Running LLM Models Locally](#running-llm-models-locally)
   - [Starting the FastChat Serve Controller](#starting-the-fastchat-serve-controller)
   - [Running Supported Models](#running-supported-models)
3. [Querying Local Models](#querying-local-models)
4. [Running OpenAI Models (GPT-3.5 and GPT-4)](#running-openai-models)
5. [Notes and Troubleshooting](#notes-and-troubleshooting)

---

## **Setting Up the Environment**

### **Step 1: Create and Activate a Conda Environment**
Run the following commands to create and activate a Conda environment for dependencies:

```bash
conda create -n fc_env python=3.9
conda activate fc_env
```

---

### **Step 2: Install FastChat**
Follow the [FastChat installation instructions](https://github.com/lm-sys/FastChat), or simply run:

```bash
pip install "fschat[model_worker,webui]"
```

---

### **Step 3: Install Additional Requirements**
Install the dependencies specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### **Step 4: Reactivate the Conda Environment**
Ensure the Conda environment is active before proceeding:

```bash
conda activate fc_env
```

---

## **Running LLM Models Locally**

### **Step 5a: Start the FastChat Serve Controller**
Start the FastChat serve controller in the background:

```bash
python -m fastchat.serve.controller &
```

---

### **Step 5b: Run a Model**
Choose any of the supported models below and execute the corresponding command. Ensure the `--model-path` points to the correct model directory.

#### Supported Models:

- **Vicuna-7b-1.5**
  ```bash
  python -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2 --max-gpu-memory 14GiB &
  ```

- **Vicuna-13b-1.5**
  ```bash
  python -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5 --num-gpus 2 --max-gpu-memory 14.5GiB &
  ```

- **Microsoft/Orca-2-7b**
  ```bash
  python -m fastchat.serve.model_worker --model-path Microsoft/Orca-2-7b --num-gpus 2 --max-gpu-memory 14.5GiB &
  ```

- **mosaicml/mpt-7b-chat**
  ```bash
  python -m fastchat.serve.model_worker --model-path mosaicml/mpt-7b-chat --num-gpus 2 --max-gpu-memory 14.5GiB &
  ```

- **meta-llama/Llama-2-7b-chat-hf**
  ```bash
  python -m fastchat.serve.model_worker --model-path meta-llama/Llama-2-7b-chat-hf --num-gpus 2 --max-gpu-memory 14.5GiB &
  ```

- **meta-llama/Llama-2-13b-chat-hf**
  ```bash
  python -m fastchat.serve.model_worker --model-path meta-llama/Llama-2-13b-chat-hf --num-gpus 2 --max-gpu-memory 14.5GiB &
  ```

---

### **Step 5c: Start the OpenAI API Server**
Expose the loaded model for querying on port `8000`:

```bash
python -m fastchat.serve.openai_api_server --host localhost --port 8000 &
```

---

## **Querying Local Models**

Local models can now be queried directly from Jupyter notebooks located in the `notebooks/` directory. Ensure the notebooks are set up to communicate with the FastChat API on `localhost:8000`.

---

## **Running OpenAI Models**

If you wish to run OpenAI models (GPT-3.5 or GPT-4), follow these steps:

### **Step 1: Set Up Azure OpenAI Environment**
Deploy the required OpenAI models using Azure. Refer to the [Azure OpenAI Quickstart Guide](https://learn.microsoft.com/en-gb/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython&pivots=programming-language-python).

---

### **Step 2: Store Access Credentials**
Place your Azure credentials in a `.env` file in the root directory of the project. An example `.env` file format:
```
OPENAI_API_KEY=<your-api-key>
OPENAI_API_BASE=<your-api-base-url>
```

---

### **Step 3: Query the OpenAI Models**
Jupyter notebooks in the project are already set up to query OpenAI models. Ensure your `.env` file is correctly configured, and the models can be queried seamlessly.

---

## **Notes and Troubleshooting**

- **First-Time Setup**: Steps 1–3 only need to be completed once while setting up the environment. For subsequent runs, start from Step 4.
- **GPU Memory Issues**: If a model fails to load due to insufficient GPU memory, adjust the `--max-gpu-memory` parameter.
- **Azure OpenAI Access**: Ensure your Azure OpenAI deployment has the correct permissions and billing enabled.

---

## **Future Improvements**
1. Add automated scripts for environment setup and model loading.
2. Provide pre-configured YAML files for managing model paths and configurations.
3. Expand support for other LLMs and frameworks.

---
