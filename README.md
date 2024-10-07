# Contract Analysis RAG

This package provides a Retrieval-Augmented Generation (RAG) system for contract analysis. It uses the Meta-Llama-3-8B-Instruct model for generating responses.

## Installation

You can install the package using pip:

```
pip install contract-analyzer
```

## Usage

Here's a basic example of how to use the ContractAnalysisRAG system:

```python
from contract_analyzer import ContractAnalysisRAG

# Initialize the RAG system
rag_system = ContractAnalysisRAG(
    max_chunk_size=100,
    overlap=50,
    min_chunk_size=50,
    top_k=3,
    use_semantic_search=True,
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
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
