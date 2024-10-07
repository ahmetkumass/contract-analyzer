from typing import Dict, Any

class Chunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(text='{self.text[:50]}...', metadata={self.metadata})"