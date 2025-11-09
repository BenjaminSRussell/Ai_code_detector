"""Learned classifier for AI code detection (Phase 2).

Replaces heuristic aggregator with trained classifier using embeddings + features.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pickle

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class MLClassifier:
    """Machine learning classifier for AI code detection.

    Combines code embeddings with extracted features to predict AI probability.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        embedding_dim: int = 768,
        feature_dim: int = 34,  # 12 stylometry + 11 structural + 11 history
    ):
        """Initialize classifier.

        Args:
            model_path: Path to saved model weights
            embedding_dim: Dimension of code embeddings
            feature_dim: Number of extracted features
        """
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None

        if model_path and model_path.exists():
            self.load(model_path)
        else:
            # Initialize with simple linear model
            self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize default linear model."""
        if not HAS_NUMPY:
            # Simple fallback without numpy
            self.weights = {
                'embedding': [0.001] * self.embedding_dim,
                'features': [0.01] * self.feature_dim,
                'bias': 0.0,
            }
        else:
            # Random initialization (would be replaced by trained weights)
            np.random.seed(42)
            self.weights = {
                'embedding': np.random.randn(self.embedding_dim) * 0.01,
                'features': np.random.randn(self.feature_dim) * 0.1,
                'bias': 0.0,
            }

    def predict(
        self,
        embedding: List[float],
        features: List[float],
    ) -> float:
        """Predict AI probability for code snippet.

        Args:
            embedding: Code embedding vector
            features: Extracted feature vector

        Returns:
            AI probability (0-1)
        """
        if not HAS_NUMPY:
            return self._predict_simple(embedding, features)

        # Convert to numpy
        emb = np.array(embedding)
        feat = np.array(features)

        # Linear combination
        score = (
            np.dot(emb, self.weights['embedding']) +
            np.dot(feat, self.weights['features']) +
            self.weights['bias']
        )

        # Sigmoid activation
        prob = 1.0 / (1.0 + np.exp(-score))

        return float(prob)

    def _predict_simple(self, embedding: List[float], features: List[float]) -> float:
        """Simple prediction without numpy.

        Args:
            embedding: Code embedding vector
            features: Feature vector

        Returns:
            AI probability (0-1)
        """
        import math

        # Dot product manually
        score = 0.0

        for i, val in enumerate(embedding):
            score += val * self.weights['embedding'][i]

        for i, val in enumerate(features):
            score += val * self.weights['features'][i]

        score += self.weights['bias']

        # Sigmoid
        prob = 1.0 / (1.0 + math.exp(-max(-10, min(10, score))))

        return prob

    def predict_batch(
        self,
        embeddings: List[List[float]],
        features: List[List[float]],
    ) -> List[float]:
        """Predict AI probabilities for batch of snippets.

        Args:
            embeddings: List of embedding vectors
            features: List of feature vectors

        Returns:
            List of AI probabilities
        """
        return [
            self.predict(emb, feat)
            for emb, feat in zip(embeddings, features)
        ]

    def train(
        self,
        train_embeddings: List[List[float]],
        train_features: List[List[float]],
        train_labels: List[int],
        val_embeddings: Optional[List[List[float]]] = None,
        val_features: Optional[List[List[float]]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> Dict[str, List[float]]:
        """Train classifier on labeled data.

        Args:
            train_embeddings: Training embeddings
            train_features: Training features
            train_labels: Training labels (0=human, 1=AI)
            val_embeddings: Validation embeddings (optional)
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training history dict
        """
        if not HAS_NUMPY:
            raise ImportError("Numpy required for training")

        X_emb = np.array(train_embeddings)
        X_feat = np.array(train_features)
        y = np.array(train_labels)

        # Initialize weights if not already done
        if self.weights is None:
            self._initialize_default_model()

        # Convert weights to numpy arrays
        w_emb = np.array(self.weights['embedding'])
        w_feat = np.array(self.weights['features'])
        bias = self.weights['bias']

        history = {'loss': [], 'accuracy': []}

        # Simple gradient descent
        for epoch in range(epochs):
            # Forward pass
            scores = X_emb @ w_emb + X_feat @ w_feat + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -10, 10)))

            # Loss (binary cross-entropy)
            loss = -np.mean(y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8))

            # Accuracy
            preds = (probs > 0.5).astype(int)
            accuracy = np.mean(preds == y)

            # Backward pass
            grad = probs - y

            grad_w_emb = X_emb.T @ grad / len(y)
            grad_w_feat = X_feat.T @ grad / len(y)
            grad_bias = np.mean(grad)

            # Update
            w_emb -= learning_rate * grad_w_emb
            w_feat -= learning_rate * grad_w_feat
            bias -= learning_rate * grad_bias

            history['loss'].append(float(loss))
            history['accuracy'].append(float(accuracy))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}")

        # Save weights
        self.weights = {
            'embedding': w_emb,
            'features': w_feat,
            'bias': float(bias),
        }

        return history

    def save(self, path: Path):
        """Save model weights to file.

        Args:
            path: Path to save model
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_NUMPY and isinstance(self.weights['embedding'], np.ndarray):
            # Convert numpy arrays to lists for JSON serialization
            weights_serializable = {
                'embedding': self.weights['embedding'].tolist(),
                'features': self.weights['features'].tolist(),
                'bias': float(self.weights['bias']),
            }
        else:
            weights_serializable = self.weights

        with open(path, 'w') as f:
            json.dump({
                'weights': weights_serializable,
                'embedding_dim': self.embedding_dim,
                'feature_dim': self.feature_dim,
            }, f)

        print(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model weights from file.

        Args:
            path: Path to model file
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self.embedding_dim = data['embedding_dim']
        self.feature_dim = data['feature_dim']

        if HAS_NUMPY:
            self.weights = {
                'embedding': np.array(data['weights']['embedding']),
                'features': np.array(data['weights']['features']),
                'bias': data['weights']['bias'],
            }
        else:
            self.weights = data['weights']

        print(f"Model loaded from {path}")


class EnsembleClassifier:
    """Ensemble of heuristic + ML classifiers.

    Combines traditional heuristic scoring with learned classifier
    for more robust predictions.
    """

    def __init__(
        self,
        ml_classifier: MLClassifier,
        heuristic_weight: float = 0.3,
        ml_weight: float = 0.7,
    ):
        """Initialize ensemble.

        Args:
            ml_classifier: Trained ML classifier
            heuristic_weight: Weight for heuristic score
            ml_weight: Weight for ML score
        """
        self.ml_classifier = ml_classifier
        self.heuristic_weight = heuristic_weight
        self.ml_weight = ml_weight

    def predict(
        self,
        embedding: List[float],
        features: List[float],
        heuristic_score: float,
    ) -> float:
        """Predict AI probability using ensemble.

        Args:
            embedding: Code embedding
            features: Extracted features
            heuristic_score: Score from heuristic aggregator

        Returns:
            Ensemble AI probability
        """
        ml_score = self.ml_classifier.predict(embedding, features)

        ensemble_score = (
            self.heuristic_weight * heuristic_score +
            self.ml_weight * ml_score
        )

        return ensemble_score
