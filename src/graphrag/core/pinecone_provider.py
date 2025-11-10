"""
Pinecone Vector Database Provider
Manages vector embeddings and similarity search using Pinecone (cloud and local).
"""

from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PineconeProvider:
    """Manages vector embeddings and similarity search using Pinecone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "gcp-starter",
        index_name: str = "research-compass",
        dimension: int = 384,
        metric: str = "cosine",
        use_local: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[Dict] = None
    ):
        """
        Initialize the Pinecone provider.

        Args:
            api_key: Pinecone API key (required for cloud mode)
            environment: Pinecone environment (e.g., 'gcp-starter', 'us-east-1-aws')
            index_name: Name of the Pinecone index
            dimension: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
            use_local: Use Pinecone Lite (local mode) instead of cloud
            embedding_model: Embedding model to use
            config: Optional unified configuration dictionary
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.use_local = use_local
        self.embedding_model_name = embedding_model

        # Import configuration if not provided
        if config is None:
            try:
                from ...config.config_manager import get_config_manager
                config_manager = get_config_manager()
                pinecone_config = config_manager.config.database

                # Override with config values if available
                if hasattr(pinecone_config, 'pinecone_api_key') and pinecone_config.pinecone_api_key:
                    self.api_key = pinecone_config.pinecone_api_key
                if hasattr(pinecone_config, 'pinecone_environment') and pinecone_config.pinecone_environment:
                    self.environment = pinecone_config.pinecone_environment
                if hasattr(pinecone_config, 'pinecone_index_name') and pinecone_config.pinecone_index_name:
                    self.index_name = pinecone_config.pinecone_index_name
                if hasattr(pinecone_config, 'pinecone_use_local'):
                    self.use_local = pinecone_config.pinecone_use_local
            except Exception as e:
                logger.warning(f"Could not load config, using defaults: {e}")

        self.pinecone = None
        self.index = None
        self.embedding_model = None

        # Initialize Pinecone
        self._initialize_pinecone()

        # Initialize embedding model
        self._initialize_embedding_model()

        # Storage for metadata
        self.documents = []
        self.chunks = []

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            import pinecone

            if self.use_local:
                # Use Pinecone Lite (local mode)
                logger.info("🏠 Initializing Pinecone Lite (local mode)")
                try:
                    # Pinecone 3.0+ API
                    from pinecone import Pinecone
                    self.pinecone = Pinecone()  # Local mode doesn't need API key

                    # Create or connect to local index
                    if self.index_name not in self.pinecone.list_indexes().names():
                        logger.info(f"Creating local Pinecone index: {self.index_name}")
                        self.pinecone.create_index(
                            name=self.index_name,
                            dimension=self.dimension,
                            metric=self.metric,
                            spec={
                                "local": {}  # Specify local deployment
                            }
                        )

                    self.index = self.pinecone.Index(self.index_name)
                    logger.info(f"✅ Connected to local Pinecone index: {self.index_name}")

                except ImportError:
                    logger.warning("⚠️ Pinecone 3.0+ not available, trying legacy API")
                    # Legacy Pinecone 2.x API
                    pinecone.init(api_key="local", environment="local")

                    # Check if index exists
                    if self.index_name not in pinecone.list_indexes():
                        logger.info(f"Creating local Pinecone index: {self.index_name}")
                        pinecone.create_index(
                            name=self.index_name,
                            dimension=self.dimension,
                            metric=self.metric
                        )

                    self.index = pinecone.Index(self.index_name)
                    logger.info(f"✅ Connected to local Pinecone index: {self.index_name}")

            else:
                # Use Pinecone Cloud
                if not self.api_key:
                    raise ValueError(
                        "Pinecone API key is required for cloud mode. "
                        "Set PINECONE_API_KEY in environment or config file."
                    )

                logger.info(f"☁️ Initializing Pinecone Cloud (environment: {self.environment})")

                try:
                    # Pinecone 3.0+ API
                    from pinecone import Pinecone, ServerlessSpec

                    self.pinecone = Pinecone(api_key=self.api_key)

                    # Create or connect to cloud index
                    if self.index_name not in self.pinecone.list_indexes().names():
                        logger.info(f"Creating Pinecone cloud index: {self.index_name}")
                        self.pinecone.create_index(
                            name=self.index_name,
                            dimension=self.dimension,
                            metric=self.metric,
                            spec=ServerlessSpec(
                                cloud='aws',  # or 'gcp', 'azure'
                                region=self.environment
                            )
                        )

                    self.index = self.pinecone.Index(self.index_name)
                    logger.info(f"✅ Connected to Pinecone cloud index: {self.index_name}")

                except ImportError:
                    logger.warning("⚠️ Pinecone 3.0+ not available, trying legacy API")
                    # Legacy Pinecone 2.x API
                    pinecone.init(api_key=self.api_key, environment=self.environment)

                    # Check if index exists
                    if self.index_name not in pinecone.list_indexes():
                        logger.info(f"Creating Pinecone cloud index: {self.index_name}")
                        pinecone.create_index(
                            name=self.index_name,
                            dimension=self.dimension,
                            metric=self.metric
                        )

                    self.index = pinecone.Index(self.index_name)
                    logger.info(f"✅ Connected to Pinecone cloud index: {self.index_name}")

        except ImportError:
            logger.error(
                "❌ Pinecone library not installed. Install with: pip install pinecone-client"
            )
            raise ImportError(
                "Pinecone library not found. Install with: pip install pinecone-client"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✅ Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings
        """
        return self.embedding_model.encode(texts, show_progress_bar=True)

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add texts to the Pinecone index.

        Args:
            texts: List of text strings
            metadata: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            batch_size: Batch size for uploads

        Returns:
            List of IDs for the added texts
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(texts))]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embed_texts(texts)

        # Prepare metadata
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        else:
            # Ensure text is in metadata
            for i, meta in enumerate(metadata):
                if "text" not in meta:
                    meta["text"] = texts[i]

        # Store documents and chunks for backward compatibility
        self.documents.extend(texts)
        self.chunks.extend(texts)

        # Upload to Pinecone in batches
        logger.info(f"Uploading {len(texts)} vectors to Pinecone...")
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end].tolist()
            batch_metadata = metadata[i:batch_end]

            # Create vectors list
            vectors = [
                {
                    "id": vid,
                    "values": embedding,
                    "metadata": meta
                }
                for vid, embedding, meta in zip(batch_ids, batch_embeddings, batch_metadata)
            ]

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

        logger.info(f"✅ Successfully added {len(texts)} texts to Pinecone")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the Pinecone index.

        Args:
            query: Query string
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of search results with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embed_texts([query])[0].tolist()

        # Search Pinecone
        search_kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }

        if filter:
            search_kwargs["filter"] = filter

        results = self.index.query(**search_kwargs)

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })

        return formatted_results

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if successful
        """
        try:
            self.index.delete(ids=ids)
            logger.info(f"✅ Deleted {len(ids)} vectors from Pinecone")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete vectors: {e}")
            return False

    def clear_index(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            True if successful
        """
        try:
            self.index.delete(delete_all=True)
            self.documents = []
            self.chunks = []
            logger.info("✅ Cleared Pinecone index")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear index: {e}")
            return False

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test Pinecone connection.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            stats = self.get_index_stats()
            mode = "Local" if self.use_local else "Cloud"
            return True, (
                f"✓ Connected to Pinecone {mode}\n"
                f"  Index: {self.index_name}\n"
                f"  Total vectors: {stats.get('total_vectors', 0)}\n"
                f"  Dimension: {stats.get('dimension', self.dimension)}"
            )
        except Exception as e:
            return False, f"✗ Pinecone connection failed: {str(e)}"

    def close(self):
        """Clean up resources."""
        # Pinecone doesn't require explicit connection closing
        logger.info("Pinecone provider closed")
