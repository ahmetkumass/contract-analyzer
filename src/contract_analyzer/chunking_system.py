import re
from typing import List, Dict, Any
from .chunk import Chunk

class ChunkingSystem:
    def __init__(self, max_chunk_size: int = 200, overlap: int = 50, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def create_chunks(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc['text'], doc['metadata'])
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for i, para in enumerate(paragraphs):
            para_size = len(self._word_tokenize(para))
            
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(chunk_text, {**metadata, 'chunk_id': len(chunks)}))
                
                # Handle overlap
                overlap_text = ' '.join(current_chunk[-self._sentences_for_overlap(current_chunk):])
                current_chunk = [overlap_text]
                current_size = len(self._word_tokenize(overlap_text))

            current_chunk.append(para)
            current_size += para_size

        # Handle the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(chunk_text, {**metadata, 'chunk_id': len(chunks)}))

        return self._handle_small_chunks(chunks)

    def _split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def _word_tokenize(self, text: str) -> List[str]:
        return text.split()

    def _sentences_for_overlap(self, chunk: List[str]) -> int:
        overlap_text = ' '.join(chunk[-1:])
        sentences = re.split(r'(?<=[.!?]) +', overlap_text)
        overlap_sentences = []
        overlap_size = 0
        for sent in reversed(sentences):
            sent_size = len(self._word_tokenize(sent))
            if overlap_size + sent_size > self.overlap:
                break
            overlap_sentences.insert(0, sent)
            overlap_size += sent_size
        return len(overlap_sentences)

    def _handle_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        result = []
        for i, chunk in enumerate(chunks):
            if len(self._word_tokenize(chunk.text)) < self.min_chunk_size:
                if i > 0:
                    prev_chunk = result[-1]
                    combined_text = prev_chunk.text + ' ' + chunk.text
                    combined_metadata = {**prev_chunk.metadata, **chunk.metadata}
                    result[-1] = Chunk(combined_text, combined_metadata)
                elif i < len(chunks) - 1:
                    next_chunk = chunks[i+1]
                    combined_text = chunk.text + ' ' + next_chunk.text
                    combined_metadata = {**chunk.metadata, **next_chunk.metadata}
                    result.append(Chunk(combined_text, combined_metadata))
                else:
                    result.append(chunk)
            else:
                result.append(chunk)
        return result