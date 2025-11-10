"""
Lazy GNN Manager - Deferred model loading for faster startup.

This wrapper implements lazy initialization for GNN models, loading them only when
first accessed rather than at startup. This significantly improves application startup
time and memory footprint.

Part of Phase 3 optimization (OPT-014: Lazy Loading for GNN Models).
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class LazyGNNManager:
    """
    Lazy-loading wrapper for GNNManager.

    Benefits:
    - Faster application startup (no model loading at init)
    - Lower initial memory footprint
    - Models loaded only when actually needed
    - Transparent to calling code (same interface as GNNManager)

    Usage:
        # Create lazy manager (doesn't load models yet)
        manager = LazyGNNManager(uri, user, password)

        # Models loaded automatically on first use
        results = manager.predict_links(data)  # Triggers model loading
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        model_dir: str = "models/gnn",
        auto_initialize: bool = False
    ):
        """
        Initialize lazy GNN manager.

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            model_dir: Directory for model storage
            auto_initialize: If True, initialize immediately (default: False for lazy)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.model_dir = Path(model_dir)
        self.auto_initialize = auto_initialize

        # Lazy initialization flag
        self._initialized = False
        self._manager = None

        logger.info(f"LazyGNNManager created (lazy={'not ' if auto_initialize else ''}enabled)")

        # If auto_initialize is True, load immediately (backwards compatibility)
        if auto_initialize:
            self._ensure_initialized()

    def _ensure_initialized(self):
        """
        Ensure GNN manager is initialized. Loads models on first call.

        This is the core of lazy loading - models are loaded here, not in __init__.
        """
        if not self._initialized:
            logger.info("🔄 Lazy loading GNN models (first access)...")

            try:
                # Import here to avoid loading GNN dependencies at module import time
                from .gnn_manager import GNNManager

                # Create actual GNN manager
                self._manager = GNNManager(
                    uri=self.uri,
                    user=self.user,
                    password=self.password,
                    model_dir=str(self.model_dir)
                )

                # Initialize models (load from disk or create new)
                self._manager.initialize_models()

                self._initialized = True
                logger.info("✅ GNN models loaded successfully")

            except Exception as e:
                logger.error(f"❌ Failed to initialize GNN models: {e}")
                logger.warning("GNN functionality will be disabled")
                self._manager = None
                # Don't set _initialized to True on failure - allow retry

    def is_initialized(self) -> bool:
        """Check if models are currently loaded."""
        return self._initialized

    def force_initialize(self):
        """
        Force initialization of models.

        Useful for:
        - Pre-loading models before they're needed
        - Testing model initialization
        - Warming up the system
        """
        if not self._initialized:
            self._ensure_initialized()

    # Proxy all methods to underlying GNN manager (with lazy initialization)

    def train(self, *args, **kwargs) -> Dict:
        """Train GNN model (loads models if not initialized)."""
        self._ensure_initialized()
        if self._manager:
            return self._manager.train(*args, **kwargs)
        return {"error": "GNN manager not initialized"}

    def predict_links(self, *args, **kwargs):
        """Predict citation links (loads models if not initialized)."""
        self._ensure_initialized()
        if self._manager:
            return self._manager.predict_links(*args, **kwargs)
        return []

    def predict_node_labels(self, *args, **kwargs):
        """Predict node classifications (loads models if not initialized)."""
        self._ensure_initialized()
        if self._manager:
            return self._manager.predict_node_labels(*args, **kwargs)
        return []

    def generate_embeddings(self, *args, **kwargs):
        """Generate graph embeddings (loads models if not initialized)."""
        self._ensure_initialized()
        if self._manager:
            return self._manager.generate_embeddings(*args, **kwargs)
        return None

    def save_models(self, *args, **kwargs):
        """Save trained models to disk."""
        if self._initialized and self._manager:
            return self._manager.save_models(*args, **kwargs)
        logger.warning("No models to save (not initialized)")

    def load_models(self, *args, **kwargs):
        """Load models from disk."""
        self._ensure_initialized()
        if self._manager:
            return self._manager.load_models(*args, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about models.

        Returns info even if not initialized (indicates lazy state).
        """
        if self._initialized and self._manager:
            return {
                "initialized": True,
                "models_loaded": True,
                "model_dir": str(self.model_dir),
                "device": str(self._manager.device) if hasattr(self._manager, 'device') else "unknown"
            }
        else:
            return {
                "initialized": False,
                "models_loaded": False,
                "model_dir": str(self.model_dir),
                "note": "Models will be loaded on first use (lazy loading)"
            }

    def __getattr__(self, name):
        """
        Proxy all other attributes to underlying manager (with lazy init).

        This allows LazyGNNManager to be a drop-in replacement for GNNManager.
        Any method not explicitly defined above will be proxied automatically.
        """
        if name.startswith('_'):
            # Don't proxy private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Ensure initialized before proxying
        self._ensure_initialized()

        if self._manager:
            return getattr(self._manager, name)
        else:
            raise RuntimeError("GNN manager failed to initialize")

    def __repr__(self):
        status = "initialized" if self._initialized else "not initialized (lazy)"
        return f"LazyGNNManager(uri={self.uri}, status={status})"


# Factory function for easy migration
def create_gnn_manager(
    uri: str,
    user: str,
    password: str,
    model_dir: str = "models/gnn",
    lazy: bool = True
) -> LazyGNNManager:
    """
    Factory function to create GNN manager with optional lazy loading.

    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
        model_dir: Model directory
        lazy: Enable lazy loading (default: True for better performance)

    Returns:
        LazyGNNManager instance

    Example:
        # Lazy loading (default - faster startup)
        manager = create_gnn_manager(uri, user, password)

        # Eager loading (backwards compatible)
        manager = create_gnn_manager(uri, user, password, lazy=False)
    """
    return LazyGNNManager(
        uri=uri,
        user=user,
        password=password,
        model_dir=model_dir,
        auto_initialize=not lazy
    )


if __name__ == '__main__':
    # Example usage
    print("Creating lazy GNN manager...")
    manager = LazyGNNManager(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="password"
    )

    print(f"Manager created: {manager}")
    print(f"Model info: {manager.get_model_info()}")

    # Models will be loaded here on first access
    # print("Training model (will trigger lazy loading)...")
    # manager.train(data)
