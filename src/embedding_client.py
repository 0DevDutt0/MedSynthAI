import os
import ollama
import numpy as np
from typing import List, Union

class EmbeddingClient:
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        self.model_name = model_name
        # Verify the model is available
        try:
            ollama.show(model_name)
            print(f"Model {model_name} is available")
        except Exception as e:
            print(f"Model {model_name} not found: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response['embedding']

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        # We can get a sample embedding to determine dimension
        sample_embedding = self.get_embedding("sample")
        return len(sample_embedding)

# Example usage
if __name__ == "__main__":
    client = EmbeddingClient()
    print(f"Embedding dimension: {client.get_embedding_dimension()}")