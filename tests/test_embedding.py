import asyncio
import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.embedding_client import EmbeddingClient, embed_batch, embed_batch_async

def test_embedding_client_initialization():
    """Test that EmbeddingClient initializes correctly."""
    client = EmbeddingClient()
    assert client is not None
    assert client.cache_db_path is not None

def test_cache_functionality():
    """Test cache functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.db")
        client = EmbeddingClient(cache_path)

        # Test cache miss
        text = "Hello world"
        embedding = np.array([0.1, 0.2, 0.3])

        # Save embedding
        client._save_embedding_to_cache(text, embedding)

        # Retrieve embedding
        cached = client._get_cached_embedding(text)
        assert cached is not None
        np.testing.assert_array_equal(cached, embedding)

        # Test cache miss for different text
        cached = client._get_cached_embedding("different text")
        assert cached is None

@pytest.mark.asyncio
async def test_async_embedding():
    """Test async embedding functionality with mocked Ollama."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.db")
        client = EmbeddingClient(cache_path)

        # Mock Ollama client
        with patch('ollama.Client') as mock_client:
            mock_response = MagicMock()
            mock_response['embedding'] = [0.1, 0.2, 0.3]
            mock_client.return_value.embeddings.return_value = mock_response

            # Test async embedding
            texts = ["Hello world", "Test text"]
            result = await client.embed_batch_async(texts)
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2
            assert result.shape[1] == 3  # 3D embedding

def test_sync_embedding():
    """Test sync embedding functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.db")
        client = EmbeddingClient(cache_path)

        # Test sync embedding
        texts = ["Hello world", "Test text"]
        result = client.embed_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

def test_embedding_with_fallback():
    """Test that embedding falls back to TF-IDF when Ollama is not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.db")
        client = EmbeddingClient(cache_path)

        # Mock Ollama to fail
        with patch('ollama.Client') as mock_client:
            mock_client.side_effect = Exception("Ollama not available")

            # Test that fallback works
            texts = ["Hello world", "Test text"]
            result = client.embed_batch(texts)
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])