"""
Mini-Batch Training for Large Graphs

Uses PyTorch Geometric's NeighborLoader for efficient
training on graphs with thousands of nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from typing import Dict, List, Optional, Tuple
import time


class MiniBatchTrainer:
    """
    Mini-batch trainer using neighborhood sampling

    Scales to large graphs (10K+ nodes) by sampling neighborhoods
    instead of using the full graph for each forward pass.

    Args:
        model: GNN model to train
        data: Full graph data
        optimizer: PyTorch optimizer
        device: Device to train on
        batch_size: Number of nodes per batch (default: 32)
        num_neighbors: Number of neighbors to sample per layer (default: [10, 5])
        num_workers: Number of data loading workers (default: 0)

    Example:
        >>> from models import GCNModel
        >>> model = GCNModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = MiniBatchTrainer(model, data, optimizer, batch_size=32)
        >>> metrics = trainer.train_epoch()
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        batch_size: int = 32,
        num_neighbors: List[int] = None,
        num_workers: int = 0
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [10, 5]
        self.num_workers = num_workers

        # Move model to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        # Create data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._create_loaders()

    def _create_loaders(self):
        """Create neighborhood sampling loaders"""

        # Training loader
        if hasattr(self.data, 'train_mask'):
            train_idx = self.data.train_mask.nonzero(as_tuple=True)[0]
            self.train_loader = NeighborLoader(
                self.data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=train_idx,
                shuffle=True,
                num_workers=self.num_workers
            )

        # Validation loader
        if hasattr(self.data, 'val_mask'):
            val_idx = self.data.val_mask.nonzero(as_tuple=True)[0]
            self.val_loader = NeighborLoader(
                self.data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=val_idx,
                shuffle=False,
                num_workers=self.num_workers
            )

        # Test loader
        if hasattr(self.data, 'test_mask'):
            test_idx = self.data.test_mask.nonzero(as_tuple=True)[0]
            self.test_loader = NeighborLoader(
                self.data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=test_idx,
                shuffle=False,
                num_workers=self.num_workers
            )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using mini-batches

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_examples = 0
        num_batches = 0

        start_time = time.time()

        for batch in self.train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(batch.x, batch.edge_index)

            # Compute loss only on target nodes (batch_size)
            # The sampled subgraph includes neighbors, but we only
            # compute loss on the original batch nodes
            batch_size = batch.batch_size
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred = out[:batch_size].argmax(dim=1)
            total_correct += (pred == batch.y[:batch_size]).sum().item()
            total_examples += batch_size
            num_batches += 1

        epoch_time = time.time() - start_time

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_examples,
            'time': epoch_time
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate using mini-batches

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_examples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass
                out = self.model(batch.x, batch.edge_index)

                # Compute metrics on target nodes
                batch_size = batch.batch_size
                loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])

                total_loss += loss.item()
                pred = out[:batch_size].argmax(dim=1)
                total_correct += (pred == batch.y[:batch_size]).sum().item()
                total_examples += batch_size
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_examples
        }

    def test(self) -> Dict[str, float]:
        """
        Test using mini-batches

        Returns:
            Dictionary with test metrics
        """
        self.model.eval()

        total_correct = 0
        total_examples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)

                # Forward pass
                out = self.model(batch.x, batch.edge_index)

                # Predictions on target nodes
                batch_size = batch.batch_size
                pred = out[:batch_size].argmax(dim=1)
                total_correct += (pred == batch.y[:batch_size]).sum().item()
                total_examples += batch_size

                all_preds.append(pred)
                all_labels.append(batch.y[:batch_size])

        # Concatenate all predictions
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return {
            'accuracy': total_correct / total_examples,
            'predictions': all_preds,
            'labels': all_labels
        }

    @staticmethod
    def should_use_minibatch(num_nodes: int, threshold: int = 1000) -> bool:
        """
        Determine if mini-batch training should be used

        Args:
            num_nodes: Number of nodes in graph
            threshold: Node count threshold for mini-batch (default: 1000)

        Returns:
            True if mini-batch should be used
        """
        return num_nodes >= threshold


class FullBatchTrainer:
    """
    Traditional full-batch trainer for small graphs

    Faster for small graphs (<1000 nodes) since it avoids
    sampling overhead.

    Args:
        model: GNN model
        data: Full graph data
        optimizer: PyTorch optimizer
        device: Device to train on

    Example:
        >>> model = GCNModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = FullBatchTrainer(model, data, optimizer)
        >>> metrics = trainer.train_epoch()
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch on full graph"""
        self.model.train()

        start_time = time.time()

        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Metrics
        pred = out[self.data.train_mask].argmax(dim=1)
        accuracy = (pred == self.data.y[self.data.train_mask]).float().mean()

        epoch_time = time.time() - start_time

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'time': epoch_time
        }

    def validate(self) -> Dict[str, float]:
        """Validate on full graph"""
        self.model.eval()

        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.val_mask], self.data.y[self.data.val_mask])

            pred = out[self.data.val_mask].argmax(dim=1)
            accuracy = (pred == self.data.y[self.data.val_mask]).float().mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

    def test(self) -> Dict[str, float]:
        """Test on full graph"""
        self.model.eval()

        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out[self.data.test_mask].argmax(dim=1)
            accuracy = (pred == self.data.y[self.data.test_mask]).float().mean()

        return {
            'accuracy': accuracy.item(),
            'predictions': pred,
            'labels': self.data.y[self.data.test_mask],
            'probabilities': F.softmax(out[self.data.test_mask], dim=1)
        }


def create_trainer(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    device: torch.device = None,
    force_minibatch: bool = False,
    minibatch_threshold: int = 1000,
    **minibatch_kwargs
):
    """
    Factory function to create appropriate trainer

    Automatically chooses between full-batch and mini-batch
    based on graph size.

    Args:
        model: GNN model
        data: Graph data
        optimizer: Optimizer
        device: Device
        force_minibatch: Force mini-batch even for small graphs
        minibatch_threshold: Node count threshold for mini-batch
        **minibatch_kwargs: Additional arguments for MiniBatchTrainer

    Returns:
        Trainer instance (either FullBatchTrainer or MiniBatchTrainer)

    Example:
        >>> trainer = create_trainer(model, data, optimizer)
        >>> # Automatically chooses based on graph size
    """
    num_nodes = data.num_nodes

    if force_minibatch or MiniBatchTrainer.should_use_minibatch(num_nodes, minibatch_threshold):
        print(f"Using MiniBatchTrainer (graph has {num_nodes} nodes)")
        return MiniBatchTrainer(model, data, optimizer, device, **minibatch_kwargs)
    else:
        print(f"Using FullBatchTrainer (graph has {num_nodes} nodes)")
        return FullBatchTrainer(model, data, optimizer, device)
