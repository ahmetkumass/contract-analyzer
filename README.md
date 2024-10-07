# Contract Analyzer

This package provides a RAG system for contract analysis using the Meta-Llama-3-8B-Instruct model.

## Installation

You can install the package using pip:

```
pip install contract-analyzer
```

## Usage

Here's a basic example of how to use the ContractAnalysisRAG system:

```python
from huggingface_hub import login
from contract_analyzer import ContractAnalysisRAG

# Authenticate with Hugging Face
token = "your_huggingface_token_here"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
login(token=token)

# Initialize the RAG system
rag_system = ContractAnalysisRAG(
    max_chunk_size=100,
    overlap=50,
    min_chunk_size=50,
    top_k=3,
    use_semantic_search=True,
    model_id=model_name,
    max_new_tokens=512,
    temperature=0.7
)

# Add your PDF documents
pdf_paths = ['path/to/your/contract.pdf']
rag_system.add_pdf_documents(pdf_paths)

# Example query
query = "Who is the employee named in the contract?"

# Retrieve the most relevant chunk and generate a response
most_related_chunk = rag_system.retrieve_and_print(query)
if most_related_chunk:
    response = rag_system.generate(query, [most_related_chunk])
    print(f"Query: {query}")
    print(f"Generated Response: {response}")
else:
    print("No relevant information found.")
```

Note: Replace 'your_huggingface_token_here' with your actual Hugging Face token to access the Meta-Llama-3-8B-Instruct model.

## Features

- PDF document processing
- Semantic search for relevant information retrieval
- Integration with Meta-Llama-3-8B-Instruct model for response generation
- Customizable chunking and retrieval parameters

## Requirements

- Python 3.7+
- Hugging Face account with access to the Meta-Llama-3-8B-Instruct model
