import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import pdfplumber
from transformers import pipeline
import torch
from .chunking_system import ChunkingSystem
from .chunk import Chunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContractAnalysisRAG:
    def __init__(self, 
                 max_chunk_size: int = 200,
                 overlap: int = 50,
                 min_chunk_size: int = 100,
                 top_k: int = 5,
                 use_semantic_search: bool = True,
                 model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 max_new_tokens: int = 150,
                 temperature: float = 0.7):
        self.chunking_system = ChunkingSystem(max_chunk_size, overlap, min_chunk_size)
        self.top_k = top_k
        self.use_semantic_search = use_semantic_search
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

        logger.info(f"Initializing Contract Analysis RAG System with model: {model_id}")
        
        if use_semantic_search:
            logger.info("Using semantic search for retrieval")
            self.encoder = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        else:
            logger.info("Using TF-IDF for retrieval")
            self.vectorizer = TfidfVectorizer()

        try:
            self.pipe = pipeline(
                "text-generation", 
                model=model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or pipeline: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        logger.info(f"Adding {len(documents)} documents to the system")
        try:
            new_chunks = self.chunking_system.create_chunks(documents)
            self.chunks.extend(new_chunks)
            self._update_index()
            logger.info(f"Added {len(new_chunks)} chunks to the system")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def add_pdf_documents(self, pdf_paths: List[str]) -> None:
        logger.info(f"Adding {len(pdf_paths)} PDF documents to the system")
        documents = []
        for path in pdf_paths:
            try:
                with pdfplumber.open(path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    documents.append({
                        'text': text,
                        'metadata': {'source': path, 'type': 'pdf'}
                    })
                logger.info(f"Successfully processed PDF: {path}")
            except Exception as e:
                logger.error(f"Error processing PDF {path}: {e}")
        self.add_documents(documents)

    def _update_index(self) -> None:
        logger.info("Updating search index")
        try:
            chunk_texts = [chunk.text for chunk in self.chunks]
            if self.use_semantic_search:
                self.chunk_embeddings = self.encoder.encode(chunk_texts, show_progress_bar=True)
            else:
                self.chunk_embeddings = self.vectorizer.fit_transform(chunk_texts)
            logger.info("Search index updated successfully")
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            raise

    def retrieve_and_print(self, query: str) -> Chunk:
        logger.info(f"Retrieving chunks for query: {query}")
    
        if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            logger.error("No chunks or embeddings available for retrieval.")
            return None
    
        try:
            if self.use_semantic_search:
                query_embedding = self.encoder.encode([query])
                similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            else:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.chunk_embeddings).flatten()
    
            top_index = similarities.argsort()[-1]
            most_related_chunk = self.chunks[top_index]
    
            print(f"Most Related Chunk (ID: {most_related_chunk.metadata['chunk_id']}):\n")
            print(f"Text: {most_related_chunk.text[:500]}...\n")
            print(f"Metadata: {most_related_chunk.metadata}")
    
            return most_related_chunk
    
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return None

    def generate(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        logger.info("Generating response")
        try:
            context = retrieved_chunks[0].text if retrieved_chunks else "Not found."
    
            prompt = f"""
    You are a contract analysis expert. Based on the provided context, answer the following query as briefly and concisely as possible. Only answer if the context contains the information. If it doesn't, say "Not found."
    
    Context: {context}
    
    Query: {query}
    
    Answer:
    """
            response = self.pipe(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)[0]['generated_text']
            
            cleaned_response = response.strip()
    
            if "Answer:" in cleaned_response:
                cleaned_response = cleaned_response.split("Answer:")[-1].strip()
            
            logger.info("Response generated successfully")
            return cleaned_response
    
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return "An error occurred while generating the response."