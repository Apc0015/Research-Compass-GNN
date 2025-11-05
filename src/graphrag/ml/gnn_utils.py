"""
GNN Utilities - Helper functions for GNN components
Includes dependency checking, validation, and error handling
"""

import logging
from typing import Tuple, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


def check_gnn_dependencies() -> Tuple[bool, str]:
    """
    Check if all GNN dependencies are installed.

    Returns:
        Tuple of (success: bool, message: str)
    """
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import torch_geometric
    except ImportError:
        missing.append("torch-geometric")

    try:
        import torch_scatter
    except ImportError:
        missing.append("torch-scatter")

    try:
        import torch_sparse
    except ImportError:
        missing.append("torch-sparse")

    if missing:
        install_cmd_cpu = "pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"
        install_cmd_gpu = "pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"

        error_msg = f"""
❌ Missing GNN Dependencies: {', '.join(missing)}

📦 Installation Required:

For CPU:
{install_cmd_cpu}

For GPU (CUDA 11.8):
{install_cmd_gpu}

📖 See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
        """
        return False, error_msg.strip()

    return True, "✅ All GNN dependencies are installed"


def safe_parse_embedding(embedding: Any, default_dim: int = 384) -> list:
    """
    Safely parse embedding from various formats.

    Args:
        embedding: Embedding in various formats (list, str, None)
        default_dim: Default dimension for random embeddings

    Returns:
        List of floats representing the embedding
    """
    import numpy as np

    if embedding is None:
        # Create random embedding
        return np.random.randn(default_dim).tolist()

    if isinstance(embedding, (list, tuple)):
        return list(embedding)

    if isinstance(embedding, str):
        try:
            # SAFE: Use json.loads instead of eval()
            parsed = json.loads(embedding)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse embedding string, using random")

    # Fallback
    return np.random.randn(default_dim).tolist()


def detect_embedding_dimension(graph_manager) -> int:
    """
    Auto-detect embedding dimension from graph.

    Args:
        graph_manager: Graph manager instance (Neo4j or NetworkX)

    Returns:
        Detected embedding dimension (default: 384)
    """
    try:
        # Try Neo4j
        if hasattr(graph_manager, 'driver'):
            with graph_manager.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.embedding IS NOT NULL
                    RETURN n.embedding LIMIT 1
                """)
                record = result.single()
                if record and record['embedding']:
                    embedding = safe_parse_embedding(record['embedding'])
                    dim = len(embedding)
                    logger.info(f"Detected embedding dimension: {dim}")
                    return dim

        # Try NetworkX
        elif hasattr(graph_manager, 'graph'):
            for node, data in graph_manager.graph.nodes(data=True):
                if 'embedding' in data and data['embedding']:
                    embedding = safe_parse_embedding(data['embedding'])
                    dim = len(embedding)
                    logger.info(f"Detected embedding dimension: {dim}")
                    return dim

    except Exception as e:
        logger.warning(f"Could not detect embedding dimension: {e}")

    # Default fallback
    logger.info("Using default embedding dimension: 384")
    return 384


def validate_graph_for_training(graph_data, min_nodes: int = 10, min_edges: int = 10) -> Tuple[bool, str]:
    """
    Validate graph data before training.

    Args:
        graph_data: PyTorch Geometric Data object
        min_nodes: Minimum number of nodes required
        min_edges: Minimum number of edges required

    Returns:
        Tuple of (valid: bool, message: str)
    """
    import torch

    # Check nodes exist
    if graph_data is None:
        return False, "❌ No graph data available. Upload and process documents first."

    if graph_data.num_nodes == 0:
        return False, "❌ Graph has no nodes. Upload and process documents first."

    if graph_data.num_nodes < min_nodes:
        return False, f"❌ Graph too small. Need at least {min_nodes} nodes, have {graph_data.num_nodes}.\n💡 Upload more papers to build a larger graph."

    # Check edges exist
    if graph_data.num_edges == 0:
        return False, "❌ Graph has no connections. Build relationships between documents first."

    if graph_data.num_edges < min_edges:
        return False, f"❌ Graph too sparse. Need at least {min_edges} edges, have {graph_data.num_edges}.\n💡 Upload more related papers or enable relationship extraction."

    # Check node features exist
    if not hasattr(graph_data, 'x') or graph_data.x is None:
        return False, "❌ Nodes are missing embeddings.\n💡 Process documents with 'Build Knowledge Graph' enabled."

    if graph_data.x.shape[0] == 0:
        return False, "❌ Node features are empty. Check embedding extraction."

    if graph_data.x.shape[1] == 0:
        return False, "❌ Node feature dimension is zero.\n💡 Rebuild graph with embeddings enabled."

    # Check for NaN or Inf values
    if torch.isnan(graph_data.x).any():
        return False, "❌ Node features contain NaN values. Rebuild graph."

    if torch.isinf(graph_data.x).any():
        return False, "❌ Node features contain infinite values. Rebuild graph."

    # Check edge index validity
    max_node_idx = graph_data.x.shape[0] - 1
    if graph_data.edge_index.max() > max_node_idx:
        return False, f"❌ Invalid edge index. Max node index is {max_node_idx} but edge references {graph_data.edge_index.max()}."

    # Success
    return True, f"✅ Graph validation passed: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges"


def get_user_friendly_error(exception: Exception) -> str:
    """
    Convert technical exceptions to user-friendly messages.

    Args:
        exception: The exception that occurred

    Returns:
        User-friendly error message with actionable advice
    """
    error_str = str(exception)

    # CUDA out of memory
    if "CUDA" in error_str and "out of memory" in error_str:
        return """
❌ GPU ran out of memory!

💡 Try these fixes:
1. Reduce epochs: Use 20 instead of 50
2. Use CPU instead: Set device to 'cpu' in settings
3. Reduce batch size (if available)
4. Close other GPU applications
5. Use a smaller model type (try GCN instead of Transformer)
        """

    # Dimension mismatch
    if "mat1 and mat2" in error_str or "size mismatch" in error_str:
        return """
❌ Embedding dimension mismatch!

💡 This happens when:
- You changed the embedding model after creating the graph
- Node features have inconsistent dimensions

Fix:
1. Delete existing graph data
2. Re-process all documents with current embedding model
3. Rebuild knowledge graph
4. Try training again
        """

    # Connection errors
    if "Connection" in error_str or "connection" in error_str:
        return f"""
❌ Database connection lost!

Error: {error_str}

💡 Fix:
1. Check Neo4j is running: Open http://localhost:7474
2. Verify credentials in .env file
3. Check network connection
4. Restart Neo4j service
        """

    # Import errors
    if "No module named" in error_str:
        module = error_str.split("'")[1] if "'" in error_str else "unknown"
        return f"""
❌ Missing Python package: {module}

💡 Install with:
pip install {module}

Or reinstall all dependencies:
pip install -r requirements.txt
        """

    # Graph too small
    if "too small" in error_str.lower() or "not enough" in error_str.lower():
        return """
❌ Not enough data for training!

💡 You need:
- At least 10 nodes (papers/documents)
- At least 10 connections (citations/relationships)

Actions:
1. Upload more papers
2. Enable relationship extraction
3. Build citation network
        """

    # Generic fallback
    return f"""
❌ Training failed: {error_str}

💡 Common solutions:
1. Check graph has enough data (10+ nodes, 10+ edges)
2. Verify embeddings are generated
3. Try reducing epochs
4. Check system logs for details
5. Ensure Neo4j is running and accessible
    """


def estimate_training_time(num_nodes: int, num_edges: int, epochs: int, model_type: str) -> str:
    """
    Estimate training time based on graph size and parameters.

    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph
        epochs: Number of training epochs
        model_type: Type of GNN model (GAT, GCN, etc.)

    Returns:
        Human-readable time estimate
    """
    # Base time per epoch (seconds)
    base_times = {
        'gat': 0.5,
        'gcn': 0.3,
        'transformer': 1.0,
        'hetero': 0.8
    }

    base_time = base_times.get(model_type.lower(), 0.5)

    # Scale by graph size (rough approximation)
    size_factor = (num_nodes / 100) * (num_edges / 500)
    time_per_epoch = base_time * max(size_factor, 1.0)

    total_seconds = time_per_epoch * epochs

    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"~{minutes} minutes"
    else:
        hours = total_seconds / 3600
        return f"~{hours:.1f} hours"


def create_checkpoint(model, optimizer, epoch: int, loss: float, save_path: str) -> bool:
    """
    Save training checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        save_path: Path to save checkpoint

    Returns:
        True if successful, False otherwise
    """
    import torch
    from pathlib import Path

    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint(model, optimizer, load_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        load_path: Path to load checkpoint from

    Returns:
        Checkpoint dict if successful, None otherwise
    """
    import torch
    from pathlib import Path

    try:
        load_path = Path(load_path)
        if not load_path.exists():
            logger.warning(f"Checkpoint not found: {load_path}")
            return None

        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded: {load_path}")
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None
