"""
Abstract Base Classes for Research Compass core components.

Provides clean interfaces and contracts for:
- Graph Managers
- Vector Databases
- LLM Providers
- GNN Models

Benefits:
- Better testing (mock implementations)
- Easier to extend (new implementations)
- Clear contracts (interface documentation)
- Loose coupling (dependency injection)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np


class AbstractGraphManager(ABC):
    """
    Abstract base class for graph database managers.

    Implementations: Neo4jGraphManager, NetworkXGraphManager
    """

    @abstractmethod
    def create_entity(self, entity_name: str, entity_type: str, properties: Dict = None):
        """Create an entity node in the graph."""
        pass

    @abstractmethod
    def create_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict = None
    ):
        """Create a relationship between two entities."""
        pass

    @abstractmethod
    def query_neighbors(self, entity_name: str, max_depth: int = 1) -> List[Dict]:
        """Query neighboring entities up to a given depth."""
        pass

    @abstractmethod
    def get_entity_context(self, entity_name: str, max_depth: int = 2) -> str:
        """Get textual context about an entity from the graph."""
        pass

    @abstractmethod
    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the graph."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection."""
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass


class AbstractVectorDatabase(ABC):
    """
    Abstract base class for vector database providers.

    Implementations: FAISSVectorDB, PineconeVectorDB, ChromaVectorDB
    """

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add texts to the vector database.

        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            ids: Optional IDs for each text
            batch_size: Batch size for uploads

        Returns:
            List of IDs for added texts
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts.

        Args:
            query: Query string
            top_k: Number of results
            filter: Optional metadata filter

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all vectors from the database."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection."""
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass


class AbstractLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations: OllamaProvider, OpenAIProvider, etc.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text completion.

        Args:
            system_prompt: System instruction
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Generate text completion with streaming.

        Args:
            system_prompt: System instruction
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks as they're generated
        """
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test LLM provider connection."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, str]]:
        """List available models."""
        pass


class AbstractGNNModel(ABC):
    """
    Abstract base class for GNN models.

    Implementations: NodeClassifier, LinkPredictor, GraphEmbedder
    """

    @abstractmethod
    def train(
        self,
        data,
        epochs: int = 100,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the GNN model.

        Args:
            data: Training data (PyG Data object)
            epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional training parameters

        Returns:
            Training metrics dictionary
        """
        pass

    @abstractmethod
    def predict(self, data, **kwargs) -> Any:
        """
        Make predictions with the model.

        Args:
            data: Input data
            **kwargs: Additional prediction parameters

        Returns:
            Predictions (format depends on model type)
        """
        pass

    @abstractmethod
    def evaluate(self, data, **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            data: Evaluation data
            **kwargs: Additional evaluation parameters

        Returns:
            Evaluation metrics dictionary
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass


class AbstractEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations: HuggingFaceEmbeddings, OllamaEmbeddings
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            Numpy array embedding
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class AbstractCacheManager(ABC):
    """
    Abstract base class for cache managers.

    Implementations: MemoryCache, DiskCache, RedisCache
    """

    @abstractmethod
    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """Get cached value."""
        pass

    @abstractmethod
    def set(
        self,
        namespace: str,
        value: Any,
        *args,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ):
        """Set cached value."""
        pass

    @abstractmethod
    def invalidate(self, namespace: str, *args, **kwargs):
        """Invalidate specific cache entry."""
        pass

    @abstractmethod
    def clear_all(self):
        """Clear all caches."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class AbstractRecommendationEngine(ABC):
    """
    Abstract base class for recommendation engines.

    Implementations: GNNRecommender, CollaborativeFilteringRecommender
    """

    @abstractmethod
    def recommend_papers(
        self,
        user_interests: List[str],
        viewed_papers: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend papers based on user interests.

        Args:
            user_interests: List of interest keywords/topics
            viewed_papers: Papers already viewed by user
            top_k: Number of recommendations
            **kwargs: Additional parameters

        Returns:
            List of recommended papers with scores
        """
        pass

    @abstractmethod
    def explain_recommendation(
        self,
        paper_id: str,
        user_context: Dict[str, Any]
    ) -> str:
        """
        Explain why a paper was recommended.

        Args:
            paper_id: Paper identifier
            user_context: User context information

        Returns:
            Explanation text
        """
        pass


# Utility function for dependency injection
class DependencyContainer:
    """
    Simple dependency injection container.

    Usage:
        container = DependencyContainer()
        container.register('graph_manager', Neo4jGraphManager(...))
        container.register('vector_db', PineconeProvider(...))

        graph_manager = container.resolve('graph_manager')
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, instance: Any):
        """Register a service instance."""
        self._services[name] = instance

    def resolve(self, name: str) -> Any:
        """Resolve a service instance."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        return self._services[name]

    def is_registered(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def clear(self):
        """Clear all registered services."""
        self._services.clear()


# Example usage
if __name__ == '__main__':
    # Demonstration of abstract base classes

    class MockGraphManager(AbstractGraphManager):
        """Mock implementation for testing."""

        def create_entity(self, entity_name, entity_type, properties=None):
            print(f"Creating entity: {entity_name} ({entity_type})")

        def create_relationship(self, source, target, relation_type, properties=None):
            print(f"Creating relationship: {source} -> {target} ({relation_type})")

        def query_neighbors(self, entity_name, max_depth=1):
            return [{'name': 'neighbor1'}, {'name': 'neighbor2'}]

        def get_entity_context(self, entity_name, max_depth=2):
            return f"Context for {entity_name}"

        def get_graph_stats(self):
            return {'nodes': 100, 'edges': 200}

        def test_connection(self):
            return True, "Mock connection OK"

        def close(self):
            print("Closing mock connection")

    # Dependency injection example
    container = DependencyContainer()
    container.register('graph_manager', MockGraphManager())

    graph_manager = container.resolve('graph_manager')
    graph_manager.create_entity("Paper1", "Paper")
    print(graph_manager.get_graph_stats())
