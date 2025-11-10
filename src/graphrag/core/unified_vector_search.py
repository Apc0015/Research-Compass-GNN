"""
Unified Vector Search Module
Provides a unified interface for different vector database providers (FAISS, Pinecone, Chroma).
"""

from typing import List, Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class UnifiedVectorSearch:
    """Unified interface for vector search across multiple providers."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the unified vector search engine.

        Args:
            config: Optional configuration dictionary
        """
        # Import unified configuration
        from ...config.config_manager import get_config_manager

        if config is None:
            config_manager = get_config_manager()
            self.vector_db_config = config_manager.config.vector_db
            self.db_config = config_manager.config.database
            self.embedding_config = config_manager.config.embedding
        else:
            # Use provided config
            self.vector_db_config = config.get('vector_db', {})
            self.db_config = config.get('database', {})
            self.embedding_config = config.get('embedding', {})

        # Determine provider
        provider = getattr(self.vector_db_config, 'provider', 'faiss')
        use_pinecone = getattr(self.vector_db_config, 'use_pinecone', False)
        use_faiss = getattr(self.vector_db_config, 'use_faiss', True)

        # Override provider based on shortcuts
        if use_pinecone:
            provider = 'pinecone'
        elif use_faiss:
            provider = 'faiss'

        self.provider = provider.lower()
        self.backend = None

        # Initialize the appropriate backend
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the vector database backend."""
        logger.info(f"Initializing vector search with provider: {self.provider}")

        if self.provider == 'pinecone':
            self._initialize_pinecone()
        elif self.provider == 'faiss':
            self._initialize_faiss()
        elif self.provider == 'chroma':
            self._initialize_chroma()
        else:
            logger.warning(f"Unknown provider '{self.provider}', falling back to FAISS")
            self._initialize_faiss()

    def _initialize_pinecone(self):
        """Initialize Pinecone backend."""
        try:
            from .pinecone_provider import PineconeProvider

            # Get Pinecone configuration
            api_key = getattr(self.db_config, 'pinecone_api_key', None)
            environment = getattr(self.db_config, 'pinecone_environment', 'gcp-starter')
            index_name = getattr(self.db_config, 'pinecone_index_name', 'research-compass')
            dimension = getattr(self.db_config, 'pinecone_dimension', 384)
            metric = getattr(self.db_config, 'pinecone_metric', 'cosine')
            use_local = getattr(self.db_config, 'pinecone_use_local', False)

            embedding_model = getattr(self.embedding_config, 'model_name', 'all-MiniLM-L6-v2')

            self.backend = PineconeProvider(
                api_key=api_key,
                environment=environment,
                index_name=index_name,
                dimension=dimension,
                metric=metric,
                use_local=use_local,
                embedding_model=embedding_model
            )

            logger.info("✅ Pinecone backend initialized successfully")

        except ImportError as e:
            logger.error(f"❌ Failed to import Pinecone: {e}")
            logger.info("Falling back to FAISS")
            self._initialize_faiss()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            logger.info("Falling back to FAISS")
            self._initialize_faiss()

    def _initialize_faiss(self):
        """Initialize FAISS backend."""
        try:
            from .vector_search import VectorSearch

            model_name = getattr(self.embedding_config, 'model_name', 'all-MiniLM-L6-v2')
            provider = getattr(self.embedding_config, 'provider', 'huggingface')
            base_url = getattr(self.embedding_config, 'base_url', 'http://localhost:11434')

            self.backend = VectorSearch(
                model_name=model_name,
                provider=provider,
                base_url=base_url
            )

            logger.info("✅ FAISS backend initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS: {e}")
            raise

    def _initialize_chroma(self):
        """Initialize Chroma backend (placeholder for future implementation)."""
        logger.warning("⚠️ Chroma backend not yet implemented, falling back to FAISS")
        self._initialize_faiss()

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
            metadata: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            batch_size: Batch size for uploads

        Returns:
            List of IDs for the added texts
        """
        if hasattr(self.backend, 'add_texts'):
            return self.backend.add_texts(texts, metadata, ids, batch_size)
        else:
            # For FAISS (VectorSearch) which uses different API
            self.backend.add_documents(texts)
            return [f"doc_{i}" for i in range(len(texts))]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the vector database.

        Args:
            query: Query string
            top_k: Number of results to return
            filter: Optional metadata filter (Pinecone only)

        Returns:
            List of search results with scores and metadata
        """
        if self.provider == 'pinecone':
            return self.backend.search(query, top_k, filter)
        else:
            # FAISS backend
            results = self.backend.search(query, top_k=top_k)
            # Format results to match unified interface
            formatted_results = []
            for i, (chunk, score) in enumerate(results):
                formatted_results.append({
                    "id": f"doc_{i}",
                    "score": float(score),
                    "text": chunk,
                    "metadata": {"text": chunk}
                })
            return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dictionary with database statistics
        """
        if self.provider == 'pinecone':
            return self.backend.get_index_stats()
        else:
            # FAISS backend
            return {
                "total_vectors": len(self.backend.chunks),
                "provider": self.provider
            }

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test vector database connection.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if hasattr(self.backend, 'test_connection'):
            return self.backend.test_connection()
        else:
            # FAISS doesn't need connection testing
            stats = self.get_stats()
            return True, (
                f"✓ Using {self.provider.upper()} vector search\n"
                f"  Total vectors: {stats.get('total_vectors', 0)}"
            )

    def clear(self) -> bool:
        """
        Clear all vectors from the database.

        Returns:
            True if successful
        """
        if self.provider == 'pinecone':
            return self.backend.clear_index()
        else:
            # FAISS backend
            self.backend.index = None
            self.backend.chunks = []
            self.backend.documents = []
            logger.info("✅ Cleared FAISS index")
            return True

    def save(self, path: str) -> bool:
        """
        Save the vector database to disk (FAISS only).

        Args:
            path: Path to save the database

        Returns:
            True if successful
        """
        if hasattr(self.backend, 'save_index'):
            self.backend.save_index(path)
            return True
        else:
            logger.warning(f"⚠️ Save not supported for provider: {self.provider}")
            return False

    def load(self, path: str) -> bool:
        """
        Load the vector database from disk (FAISS only).

        Args:
            path: Path to load the database from

        Returns:
            True if successful
        """
        if hasattr(self.backend, 'load_index'):
            self.backend.load_index(path)
            return True
        else:
            logger.warning(f"⚠️ Load not supported for provider: {self.provider}")
            return False

    def close(self):
        """Close the vector database connection."""
        if hasattr(self.backend, 'close'):
            self.backend.close()
        logger.info(f"Closed {self.provider} vector search backend")
