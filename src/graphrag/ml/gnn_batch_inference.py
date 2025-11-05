#!/usr/bin/env python3
"""
GNN Batch Inference - Efficient batch prediction for deployed models

Provides:
- Batch node classification
- Batch link prediction
- Batch embedding generation
- Inference optimization and caching
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """
    Efficient batch inference engine for GNN models
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_cache: bool = True
    ):
        """
        Initialize batch inference engine

        Args:
            model: Trained GNN model
            device: Device for inference (CPU/GPU)
            batch_size: Batch size for predictions
            use_cache: Enable prediction caching
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.use_cache = use_cache

        self.model.to(self.device)
        self.model.eval()

        self._cache = {} if use_cache else None

        logger.info(f"Batch inference engine initialized on {self.device}")

    def predict_nodes_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: List[int],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple nodes

        Args:
            node_features: Node feature tensor [num_nodes, feat_dim]
            edge_index: Edge index tensor [2, num_edges]
            node_indices: List of node indices to predict
            top_k: Number of top predictions per node

        Returns:
            List of prediction dicts for each node
        """
        results = []

        # Move to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            # Single forward pass for all nodes
            logits = self.model(node_features, edge_index)

            # Get predictions for requested nodes
            for node_idx in node_indices:
                # Check cache
                cache_key = f"node_{node_idx}"
                if self.use_cache and cache_key in self._cache:
                    results.append(self._cache[cache_key])
                    continue

                node_logits = logits[node_idx]
                probs = torch.softmax(node_logits, dim=0)

                # Get top-k
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

                prediction = {
                    'node_idx': node_idx,
                    'predictions': [
                        {
                            'class_idx': idx.item(),
                            'probability': prob.item()
                        }
                        for idx, prob in zip(top_indices, top_probs)
                    ]
                }

                # Cache result
                if self.use_cache:
                    self._cache[cache_key] = prediction

                results.append(prediction)

        return results

    def predict_edges_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple edges (link prediction)

        Args:
            node_features: Node feature tensor
            edge_index: Edge index tensor
            edge_pairs: List of (source, target) node pairs to predict

        Returns:
            List of link prediction results
        """
        results = []

        # Move to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            # Encode nodes
            if hasattr(self.model, 'encode'):
                node_embeddings = self.model.encode(node_features, edge_index)
            else:
                # Use model forward pass
                node_embeddings = self.model(node_features, edge_index)

            # Process edge pairs in batches
            for i in range(0, len(edge_pairs), self.batch_size):
                batch_pairs = edge_pairs[i:i + self.batch_size]

                batch_results = []
                for src, dst in batch_pairs:
                    # Check cache
                    cache_key = f"edge_{src}_{dst}"
                    if self.use_cache and cache_key in self._cache:
                        batch_results.append(self._cache[cache_key])
                        continue

                    # Compute edge score
                    if hasattr(self.model, 'decode'):
                        # Use model's decode method
                        edge_label_index = torch.tensor([[src], [dst]], device=self.device)
                        score = self.model.decode(node_embeddings, edge_label_index)
                        score = torch.sigmoid(score).item()
                    else:
                        # Compute similarity
                        src_emb = node_embeddings[src]
                        dst_emb = node_embeddings[dst]
                        score = torch.cosine_similarity(src_emb, dst_emb, dim=0).item()

                    prediction = {
                        'source': src,
                        'target': dst,
                        'score': score,
                        'exists': score > 0.5
                    }

                    # Cache result
                    if self.use_cache:
                        self._cache[cache_key] = prediction

                    batch_results.append(prediction)

                results.extend(batch_results)

        return results

    def generate_embeddings_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Generate embeddings for nodes in batch

        Args:
            node_features: Node feature tensor
            edge_index: Edge index tensor
            node_indices: Specific nodes to generate embeddings for (None = all)

        Returns:
            Numpy array of embeddings [num_nodes, embedding_dim]
        """
        # Move to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            # Generate embeddings
            if hasattr(self.model, 'encode'):
                embeddings = self.model.encode(node_features, edge_index)
            else:
                embeddings = self.model(node_features, edge_index)

            # Select specific nodes if requested
            if node_indices is not None:
                embeddings = embeddings[node_indices]

            return embeddings.cpu().numpy()

    def predict_with_confidence(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict with confidence filtering

        Args:
            node_features: Node features
            edge_index: Edge index
            node_idx: Node to predict
            confidence_threshold: Minimum confidence threshold

        Returns:
            Prediction dict with high-confidence results
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            logits = self.model(node_features, edge_index)
            probs = torch.softmax(logits[node_idx], dim=0)

            # Filter by confidence
            confident_indices = (probs > confidence_threshold).nonzero(as_tuple=True)[0]

            predictions = []
            for idx in confident_indices:
                predictions.append({
                    'class_idx': idx.item(),
                    'probability': probs[idx].item()
                })

            # Sort by probability
            predictions.sort(key=lambda x: x['probability'], reverse=True)

            return {
                'node_idx': node_idx,
                'num_confident_predictions': len(predictions),
                'predictions': predictions,
                'max_confidence': predictions[0]['probability'] if predictions else 0.0
            }

    def clear_cache(self):
        """Clear prediction cache"""
        if self.use_cache:
            self._cache.clear()
            logger.info("Prediction cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        if not self.use_cache:
            return {'enabled': False}

        return {
            'enabled': True,
            'size': len(self._cache),
            'node_predictions': len([k for k in self._cache if k.startswith('node_')]),
            'edge_predictions': len([k for k in self._cache if k.startswith('edge_')])
        }


class AsyncBatchPredictor:
    """
    Asynchronous batch predictor with parallel processing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 4
    ):
        """
        Initialize async predictor

        Args:
            model: Trained model
            device: Inference device
            max_workers: Number of parallel workers
        """
        self.engine = BatchInferenceEngine(model, device)
        self.max_workers = max_workers

    def predict_nodes_async(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: List[int],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous batch node prediction

        Args:
            node_features: Node features
            edge_index: Edge index
            node_indices: Nodes to predict
            top_k: Top predictions

        Returns:
            List of predictions
        """
        # Split into chunks for parallel processing
        chunk_size = max(1, len(node_indices) // self.max_workers)
        chunks = [
            node_indices[i:i + chunk_size]
            for i in range(0, len(node_indices), chunk_size)
        ]

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.engine.predict_nodes_batch,
                    node_features,
                    edge_index,
                    chunk,
                    top_k
                ): chunk
                for chunk in chunks
            }

            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Async prediction failed: {e}")

        return results


def save_predictions_to_file(
    predictions: List[Dict[str, Any]],
    output_path: str,
    format: str = 'json'
):
    """
    Save predictions to file

    Args:
        predictions: List of prediction dicts
        output_path: Output file path
        format: Output format ('json' or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")

    elif format == 'csv':
        import csv

        with open(output_path, 'w', newline='') as f:
            if not predictions:
                return

            # Determine fields
            if 'node_idx' in predictions[0]:
                # Node predictions
                writer = csv.writer(f)
                writer.writerow(['node_idx', 'class_idx', 'probability'])

                for pred in predictions:
                    node_idx = pred['node_idx']
                    for p in pred['predictions']:
                        writer.writerow([node_idx, p['class_idx'], p['probability']])

            elif 'source' in predictions[0]:
                # Edge predictions
                writer = csv.writer(f)
                writer.writerow(['source', 'target', 'score', 'exists'])

                for pred in predictions:
                    writer.writerow([pred['source'], pred['target'], pred['score'], pred['exists']])

        logger.info(f"Predictions saved to {output_path}")

    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == "__main__":
    print("GNN Batch Inference Utilities")
    print("=" * 50)
    print("\nFeatures:")
    print("- Batch node classification")
    print("- Batch link prediction")
    print("- Batch embedding generation")
    print("- Prediction caching")
    print("- Async parallel processing")
    print("- Export predictions to JSON/CSV")
    print("\nSee function docstrings for usage examples")
