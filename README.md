# InqIIITB
# RAG System using Langchain

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using Langchain. The system leverages a combination of document retrieval and language generation to provide accurate and contextually relevant responses.

## Features
- **Language Model**: HuggingFaceEndpoint model from the repository `mistralai/Mistral-7B-Instruct-v0.2`.
- **Document Loading**: PDF documents are loaded using PyMuPDF.
- **Text Splitting**: RecursiveTextSplitter is used to split the document into manageable chunks.
- **Vector Store**: Chroma vectorstore is used to store the split document chunks.
- **Embeddings**: Huggingface embeddings are used for vector representation.
- **Contextualized Prompts**: Chat history is maintained to create contextualized prompts for the chat retriever.

## Components

### 1. Language Model
The language model used in this project is the HuggingFaceEndpoint model from the repository `mistralai/Mistral-7B-Instruct-v0.2`. This model is designed to generate high-quality text based on the input it receives.

### 2. Document Loading
PDF documents are loaded into the system using PyMuPDF. This library allows for efficient extraction of text from PDF files, which is essential for the retrieval process.

### 3. Text Splitting
To handle large documents, the RecursiveTextSplitter is employed. This tool splits the document into smaller, more manageable chunks, making it easier to process and retrieve relevant information.

### 4. Vector Store
The Chroma vectorstore is used to store the split document chunks. This vector store allows for efficient retrieval of document segments based on their vector representations.

### 5. Embeddings
Huggingface embeddings are utilized to convert text into vector representations. These embeddings are crucial for comparing and retrieving relevant document chunks.

### 6. Contextualized Prompts
To maintain the flow of conversation and provide relevant responses, the system uses contextualized prompts. The chat retriever is designed to consider the chat history when generating prompts, ensuring that responses are coherent and contextually appropriate.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt

