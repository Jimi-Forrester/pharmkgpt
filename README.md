# PharmKGPT

**PharmKGPT: Accelerating Delirium and Alzheimer's Research through AI-Driven Knowledge Discovery**

To combat the issue of large language model "hallucinations" in specialized fields like medicine, this platform builds a bioinformatics-focused knowledge Q&A system for specific diseases such as Delirium and Alzheimer's Disease, ensuring responses are derived from curated scientific knowledge rather than generalized training data.

![Overview of the PharmKGPT Platform](image-1.png)

### Key Features

*   **Builds Specialized Knowledge Bases:** Automatically extracts entities (**Genes, Proteins, Pathways, Processes, Metabolites**) and their relationships from targeted literature (e.g., analyzing 153,000+ PubMed abstracts relevant to **Delirium and Alzheimer's Disease**) to create custom Knowledge Graphs (KGs) focused on these specific neurological conditions.

*   **Advanced Q&A System:** Employs an innovative hybrid Retrieval-Augmented Generation (RAG) approach, combining semantic search with Knowledge Graph queries, to provide high-precision answers to complex biological questions related to **Delirium and Alzheimer's Disease**.

*   **Visual Insights:** Presents visualized, relevant subgraph snippets alongside answers to enhance understanding and transparency of connections within the knowledge domain.

*   **Goal:** To empower researchers to efficiently explore disease mechanisms for **Delirium and Alzheimer's Disease**, identify potential drug targets, and accelerate the development of effective therapies.

## Table of Contents

1.  [Prerequisites](#1-prerequisites)
    *   [1.1 Install Ollama & Models](#11-install-ollama--models)
    *   [1.2 Download Reranker Model](#12-download-reranker-model)
    *   [1.3 Prepare Data](#13-prepare-data)
2.  [Setup & Run](#2-setup--run)
    *   [2.1 Docker (Recommended)](#21-docker-recommended)
    *   [2.2 Local (Conda)](#22-local-conda)
3.  [Test](#3-test)
4.  [Data Version History](#4-data-version-history)

## 1. Prerequisites

### 1.1 Install Ollama & Models

Install [ollama](https://ollama.com/).

```sh
# Pull required models
ollama pull deepseek-r1:32b
ollama pull nomic-embed-text:latest
ollama pull gemma3:27b
ollama pull Qwen2.5:0.5b

# Start the Ollama server in the background
ollama serve &
```

### 1.2 Download Reranker Model

```sh
# Download helper script (if you don't have it)
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh BAAI/bge-reranker-large

# Extract the model archive (assuming it downloads as a tar.gz)
tar -zxvf bge-reranker-large.tar.gz
```
*Adjust paths and download methods if necessary.*

### 1.3 Prepare Data

```sh
# Extract the delirium data (latest version)
tar -zxvf Data_v5.tar.gz
```
*Use `Data_v5` for the full dataset*

## 2. Setup & Run

Choose either Docker or Local setup.

### 2.1 Docker (Recommended)

1.  **Build Image:**
    ```sh
    docker build -t mindresilience:v1 .
    ```

2.  **Run Container (using Full Data):**
    ```sh
    # Adjust volume paths (-v) to match your local machine
    docker run -d -p 5000:5000 \
        -v "/path/to/your/Data_v5:/app/data" \
        -v "/path/to/your/bge-reranker-large:/app/bge-reranker-large" \
        -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
        --name mindresilience-app mindresilience:v1
    ```

3.  **Configure Model (Optional, sets defaults):**
    ```sh
    curl -X GET http://localhost:5000/

    curl -X POST http://localhost:5000/api/update_config \
         -H "Content-Type: application/json" \
         -d '{
               "model_type": "gemma3",
               "api_key": "NA",
               "top_k": 5,
               "hops": 1
             }'

    curl -X GET http://localhost:5001
    ```

4.  **Query:**
    ```sh
    curl -X POST http://localhost:5000/api/query \
         -H "Content-Type: application/json" \
         -d '{"question": "What are the main symptoms of delirium?"}' \
         -N
    ```

### 2.2 Local (Conda)

1.  **Create Environment & Install Dependencies:**
    ```sh
    conda create -n rag python=3.10 -y
    conda activate rag

    pip install -r requirements.txt

    pip install llama-index-llms-gemini==0.4.13
    pip install itext2kg==0.0.7
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the project root with:
    ```dotenv
    # Path to the downloaded BGE reranker model directory
    RERANKER_PATH=/path/to/your/bge-reranker-large

    # Path to the root directory of the dataset you want to use
    # Example: /path/to/your/Data_v5 or /path/to/your/Data_test_v4
    DATA_ROOT=/path/to/your/Data_v5
    ```

3.  **Run the Application Servers:**
    You need to start two components: the backend API server and the web interface builder.

    *   **Start the Backend API Server:**
        ```bash
        # Runs the Flask/FastAPI application in the background, logging output to app.log
        nohup python app.py > app.log 2>&1 &
        ```
        *Note: The API server typically runs on `http://localhost:5000`.*

    *   **Start the Web Interface:**
        ```bash
        # Runs the Gradio/Streamlit web interface builder
        python builder.py
        ```
        *Note: The web interface will be accessible by default at `http://127.0.0.1:7860/`.*

4.  **Querying the System:**
    You can interact with the system in two main ways:

    *   **Via Web Interface:** Open your browser and navigate to `http://127.0.0.1:7860/`.
    *   **Via API (Terminal):** Send a POST request to the backend API endpoint:
        ```bash
        curl -X POST http://localhost:5000/api/query \
             -H "Content-Type: application/json" \
             -d '{"question": "How does kynurenic acid contribute to delirium?"}' \
             -N
        ```
    *(Optional: Mention `run.py` here if it provides a distinct, important way to interact, e.g., "For single-script execution or specific tasks, you might use `run.py` [add brief explanation if needed].")*

## Running Tests

To run the automated tests:

1.  **Ensure Test Data:** Make sure you have the test dataset (e.g., `Data_test_v4`) downloaded and extracted.
2.  **Configure Environment:** Verify that the `DATA_ROOT` variable in your `.env` file points to the *test* dataset directory (or ensure it's correctly mounted if using Docker).
3.  **Execute Pytest:**
    ```bash
    pytest test/test_QA.py
    ```

## Data Version History

This project uses different versions of datasets. Here's a summary:

*   **Data_v1:** Initial version used for basic demo functionality.
*   **Data_v2:** Version with entries containing empty abstracts removed.
*   **Data_v3:** Refactored data structure. (Deprecated due to retrieval issues).
*   **Data_v4:** Rebuilt based on the v2 structure; serves as the primary dataset for QA functionality.
*   **Data_test_v4:** A smaller dataset matching the `Data_v4` structure, intended for testing purposes.
*   **Data_v5:** Contains approximately 150,000 literature abstracts focused on Alzheimer's Disease (AD) and delirium, primarily intended for Knowledge Graph (KG) construction. Last updated: April 9, 2025.

## Release History

*   **v0.1.0** (2025-04-11): Initial public pre-release. Functionality may be unstable.