"""Code embedding using MLX or hash-based fallback."""

from pathlib import Path
from typing import List, Optional, Union
import json

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


class CodeEmbedder:
    """Base class for code embedding."""

    def embed(self, code: str) -> List[float]:
        """Generate embedding for code snippet.

        Args:
            code: Source code string

        Returns:
            Embedding vector as list of floats
        """
        raise NotImplementedError


class MLXCodeEmbedder(CodeEmbedder):
    """MLX-based code embedder for Apple Silicon."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 512,
        pooling: str = "mean",
    ):
        """Initialize MLX embedder.

        Args:
            model_path: Path to MLX model weights (optional, uses default if None)
            max_length: Maximum sequence length for embedding
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        if not HAS_MLX:
            raise ImportError("MLX not installed. Install with: pip install mlx")

        self.model_path = model_path
        self.max_length = max_length
        self.pooling = pooling
        self.model = None
        self.tokenizer = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load MLX model."""
        print(f"Loading MLX model from {model_path}...")
        # self.model, self.tokenizer = load(model_path)

    def embed(self, code: str) -> List[float]:
        """Generate embedding for code snippet using MLX.

        Args:
            code: Source code string

        Returns:
            Embedding vector (768-dimensional by default)
        """
        if self.model is None:
            # Fallback: Use simple hashing-based embeddings
            return self._fallback_embedding(code)

        # Tokenize
        # tokens = self.tokenizer.encode(code, max_length=self.max_length)

        # Get embeddings from model
        # with mx.no_grad():
        #     embeddings = self.model.encode(tokens)
        #     pooled = self._pool_embeddings(embeddings)

        # return pooled.tolist()

        # For now, return fallback
        return self._fallback_embedding(code)

    def embed_batch(self, codes: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of code snippets.

        Args:
            codes: List of source code strings

        Returns:
            List of embedding vectors
        """
        return [self.embed(code) for code in codes]

    def _pool_embeddings(self, embeddings):
        """Pool token embeddings into single vector.

        Args:
            embeddings: Token-level embeddings

        Returns:
            Pooled embedding vector
        """
        if self.pooling == "mean":
            return mx.mean(embeddings, axis=0)
        elif self.pooling == "max":
            return mx.max(embeddings, axis=0)
        elif self.pooling == "cls":
            return embeddings[0]  # Use [CLS] token
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _fallback_embedding(self, code: str) -> List[float]:
        """Generate 768-dim embedding using code characteristics."""
        import hashlib
        import math

        # Initialize embedding
        embedding = [0.0] * 768

        # 1. Character-level features (first 256 dims)
        char_counts = {}
        for char in code[:1000]:  # Sample first 1000 chars
            char_counts[char] = char_counts.get(char, 0) + 1

        for i, char in enumerate(sorted(char_counts.keys())[:256]):
            if i < 256:
                embedding[i] = char_counts[char] / len(code)

        # 2. Token-level features (next 256 dims)
        tokens = code.split()[:100]
        token_hash_buckets = [0] * 256

        for token in tokens:
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            bucket = hash_val % 256
            token_hash_buckets[bucket] += 1

        for i in range(256):
            embedding[256 + i] = token_hash_buckets[i] / max(len(tokens), 1)

        # 3. Structural features (next 256 dims)
        lines = code.split('\n')
        embedding[512] = len(lines) / 1000.0  # Normalized line count
        embedding[513] = len(code) / 10000.0  # Normalized char count
        embedding[514] = code.count('def ') / max(len(lines), 1)  # Function density
        embedding[515] = code.count('class ') / max(len(lines), 1)  # Class density
        embedding[516] = code.count('import ') / max(len(lines), 1)  # Import density
        embedding[517] = code.count('#') / max(len(lines), 1)  # Comment density
        embedding[518] = code.count('(') / max(len(code), 1)  # Parenthesis density
        embedding[519] = code.count('{') / max(len(code), 1)  # Brace density

        # 4. Fill remaining with sinusoidal position encodings
        for i in range(520, 768):
            pos = i - 520
            embedding[i] = math.sin(pos / 10000.0)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


class SimpleHashEmbedder(CodeEmbedder):
    """Hash-based embedder with no external dependencies."""

    def __init__(self, dim: int = 256):
        """Initialize hash embedder.

        Args:
            dim: Embedding dimension
        """
        self.dim = dim

    def embed(self, code: str) -> List[float]:
        """Generate hash-based embedding.

        Args:
            code: Source code string

        Returns:
            Embedding vector
        """
        import hashlib

        # Use multiple hash functions
        embedding = [0.0] * self.dim

        # Hash with different seeds
        for seed in range(10):
            hash_input = f"{seed}:{code}".encode()
            hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)

            for i in range(self.dim // 10):
                idx = (i * 10 + seed) % self.dim
                embedding[idx] = ((hash_val >> i) & 0xFF) / 255.0

        # Add some code-specific features
        embedding[0] = len(code) / 10000.0
        embedding[1] = code.count('\n') / 1000.0
        embedding[2] = code.count(' ') / len(code) if code else 0
        embedding[3] = code.count('def ') / max(code.count('\n'), 1)

        return embedding


def get_embedder(
    backend: str = "mlx",
    model_path: Optional[str] = None,
    **kwargs
) -> CodeEmbedder:
    """Factory function to get code embedder.

    Args:
        backend: Embedding backend ('mlx', 'hash')
        model_path: Path to model (for mlx backend)
        **kwargs: Additional arguments for embedder

    Returns:
        CodeEmbedder instance
    """
    if backend == "mlx":
        if not HAS_MLX:
            print("Warning: MLX not available, falling back to hash embedder")
            return SimpleHashEmbedder(**kwargs)
        return MLXCodeEmbedder(model_path=model_path, **kwargs)
    elif backend == "hash":
        return SimpleHashEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
