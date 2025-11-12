"""
Training utilities with learning rate scheduling and multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Callable
import time
from pathlib import Path


class BaseTrainer:
    """
    Base trainer with learning rate scheduling

    Features:
    - Automatic learning rate reduction on plateau
    - Training/validation loss tracking
    - Best model checkpointing
    - Learning rate history logging

    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        device: Device to train on (cuda/cpu)
        scheduler_config: Configuration for ReduceLROnPlateau

    Example:
        >>> model = GCNModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = BaseTrainer(model, optimizer)
        >>> metrics = trainer.train_epoch(data)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        scheduler_config: Optional[Dict] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Learning rate scheduler
        scheduler_config = scheduler_config or {}
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'max'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6),
            verbose=scheduler_config.get('verbose', True)
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.lr_history = []
        self.epoch_times = []

        # Best model tracking
        self.best_val_metric = -float('inf')
        self.best_model_state = None
        self.best_epoch = 0

    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(
        self,
        data: Data,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            data: PyTorch Geometric Data object
            loss_fn: Loss function callable(model, data) -> loss

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        start_time = time.time()

        # Forward pass
        loss = loss_fn(self.model, data)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        epoch_time = time.time() - start_time

        return {
            'loss': loss.item(),
            'time': epoch_time,
            'lr': self.get_current_lr()
        }

    def validate(
        self,
        data: Data,
        loss_fn: Callable,
        metric_fn: Callable
    ) -> Dict[str, float]:
        """
        Validate model

        Args:
            data: PyTorch Geometric Data object
            loss_fn: Loss function
            metric_fn: Metric function callable(model, data) -> metric

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        with torch.no_grad():
            loss = loss_fn(self.model, data)
            metric = metric_fn(self.model, data)

        return {
            'loss': loss.item(),
            'metric': metric
        }

    def step_scheduler(self, val_metric: float):
        """
        Step the learning rate scheduler

        Args:
            val_metric: Validation metric to use for scheduling
        """
        self.scheduler.step(val_metric)
        self.lr_history.append(self.get_current_lr())

    def save_best_model(self, val_metric: float, epoch: int):
        """
        Save model if it's the best so far

        Args:
            val_metric: Current validation metric
            epoch: Current epoch number

        Returns:
            True if model was saved, False otherwise
        """
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_model_state = self.model.state_dict().copy()
            self.best_epoch = epoch
            return True
        return False

    def load_best_model(self):
        """Load the best model state"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """
        Save training checkpoint

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'lr_history': self.lr_history,
            **kwargs
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint

        Args:
            filepath: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_metric = checkpoint['best_val_metric']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.lr_history = checkpoint.get('lr_history', [])
        return checkpoint


class GCNTrainer(BaseTrainer):
    """
    Trainer for GCN models (node classification)

    Example:
        >>> from models import GCNModel
        >>> model = GCNModel(output_dim=5)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        >>> trainer = GCNTrainer(model, optimizer)
        >>> for epoch in range(100):
        >>>     train_metrics = trainer.train_epoch(data, data.train_mask)
        >>>     val_metrics = trainer.validate(data, data.val_mask)
        >>>     trainer.step_scheduler(val_metrics['accuracy'])
    """

    def train_epoch(
        self,
        data: Data,
        train_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Train one epoch for node classification"""

        def loss_fn(model, data):
            out = model(data.x, data.edge_index)
            return F.cross_entropy(out[train_mask], data.y[train_mask])

        return super().train_epoch(data, loss_fn)

    def validate(
        self,
        data: Data,
        val_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Validate for node classification"""

        def loss_fn(model, data):
            out = model(data.x, data.edge_index)
            return F.cross_entropy(out[val_mask], data.y[val_mask])

        def metric_fn(model, data):
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            accuracy = (pred[val_mask] == data.y[val_mask]).float().mean()
            return accuracy.item()

        return super().validate(data, loss_fn, metric_fn)


class MultiTaskGATTrainer(BaseTrainer):
    """
    Multi-Task Trainer for GAT models

    Trains GAT on both link prediction and node classification simultaneously.

    Args:
        model: GAT model
        optimizer: Optimizer
        link_weight: Weight for link prediction loss (default: 0.7)
        node_weight: Weight for node classification loss (default: 0.3)

    Example:
        >>> from models import GATModel
        >>> model = GATModel(output_dim=5)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = MultiTaskGATTrainer(model, optimizer, link_weight=0.7, node_weight=0.3)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        scheduler_config: Optional[Dict] = None,
        link_weight: float = 0.7,
        node_weight: float = 0.3
    ):
        super().__init__(model, optimizer, device, scheduler_config)
        self.link_weight = link_weight
        self.node_weight = node_weight

        # Track task-specific metrics
        self.link_losses = []
        self.node_losses = []
        self.link_accs = []
        self.node_accs = []

    def train_epoch(
        self,
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        train_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train one epoch with multi-task learning

        Args:
            data: Graph data
            pos_edge_index: Positive edges (existing citations)
            neg_edge_index: Negative edges (non-existing citations)
            train_mask: Training node mask

        Returns:
            Dictionary with loss breakdown
        """
        self.model.train()
        self.optimizer.zero_grad()

        start_time = time.time()

        # Encode nodes
        z = self.model.encode(data.x, data.edge_index)

        # Task 1: Link Prediction
        pos_pred = self.model.decode(z, pos_edge_index)
        neg_pred = self.model.decode(z, neg_edge_index)

        pos_labels = torch.ones(pos_pred.size(0), device=self.device)
        neg_labels = torch.zeros(neg_pred.size(0), device=self.device)

        link_pred = torch.cat([pos_pred, neg_pred])
        link_labels = torch.cat([pos_labels, neg_labels])

        loss_link = F.binary_cross_entropy_with_logits(link_pred, link_labels)

        # Task 2: Node Classification
        node_pred = self.model.node_classifier(z)
        loss_node = F.cross_entropy(node_pred[train_mask], data.y[train_mask])

        # Combined loss
        total_loss = self.link_weight * loss_link + self.node_weight * loss_node

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        epoch_time = time.time() - start_time

        # Compute accuracies
        with torch.no_grad():
            link_acc = ((torch.sigmoid(link_pred) > 0.5) == link_labels).float().mean().item()
            node_acc = (node_pred[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean().item()

        return {
            'total_loss': total_loss.item(),
            'link_loss': loss_link.item(),
            'node_loss': loss_node.item(),
            'link_acc': link_acc,
            'node_acc': node_acc,
            'time': epoch_time,
            'lr': self.get_current_lr()
        }

    def validate(
        self,
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        val_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate with multi-task metrics

        Returns:
            Dictionary with validation metrics for both tasks
        """
        self.model.eval()

        with torch.no_grad():
            # Encode
            z = self.model.encode(data.x, data.edge_index)

            # Link prediction
            pos_pred = self.model.decode(z, pos_edge_index)
            neg_pred = self.model.decode(z, neg_edge_index)

            pos_labels = torch.ones(pos_pred.size(0), device=self.device)
            neg_labels = torch.zeros(neg_pred.size(0), device=self.device)

            link_pred = torch.cat([pos_pred, neg_pred])
            link_labels = torch.cat([pos_labels, neg_labels])

            loss_link = F.binary_cross_entropy_with_logits(link_pred, link_labels)
            link_acc = ((torch.sigmoid(link_pred) > 0.5) == link_labels).float().mean().item()

            # Node classification
            node_pred = self.model.node_classifier(z)
            loss_node = F.cross_entropy(node_pred[val_mask], data.y[val_mask])
            node_acc = (node_pred[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()

            # Combined
            total_loss = self.link_weight * loss_link + self.node_weight * loss_node

        return {
            'total_loss': total_loss.item(),
            'link_loss': loss_link.item(),
            'node_loss': loss_node.item(),
            'link_acc': link_acc,
            'node_acc': node_acc,
            'metric': link_acc  # Use link accuracy for scheduler
        }


class HANTrainer(BaseTrainer):
    """
    Trainer for Heterogeneous Attention Network (HAN)

    Features:
    - Handles heterogeneous graph data with multiple node and edge types
    - Supports node classification on target node type (typically 'paper')
    - Tracks attention weights for analysis
    - Automatic learning rate scheduling

    Args:
        model: HAN model
        optimizer: PyTorch optimizer
        device: Device to train on
        scheduler_config: Configuration for learning rate scheduler
        target_node_type: Node type to predict on (default: 'paper')

    Example:
        >>> from models import create_han_model
        >>> from data import convert_to_heterogeneous
        >>> hetero_data = convert_to_heterogeneous(data)
        >>> model = create_han_model(hetero_data, task='classification')
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> trainer = HANTrainer(model, optimizer, target_node_type='paper')
        >>> metrics = trainer.train_epoch(hetero_data)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        scheduler_config: Optional[Dict] = None,
        target_node_type: str = 'paper'
    ):
        super().__init__(model, optimizer, device, scheduler_config)
        self.target_node_type = target_node_type
        self.attention_history = []

    def train_epoch(
        self,
        hetero_data,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch on heterogeneous graph

        Args:
            hetero_data: HeteroData object with x_dict, edge_index_dict, masks
            loss_fn: Optional custom loss function

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        start_time = time.time()

        # Move data to device
        x_dict = {k: v.to(self.device) for k, v in hetero_data.x_dict.items()}
        edge_index_dict = {
            k: v.to(self.device) for k, v in hetero_data.edge_index_dict.items()
        }

        # Forward pass
        out_dict = self.model(x_dict, edge_index_dict)

        # Compute loss (only on target node type with train mask)
        out = out_dict[self.target_node_type]
        y = hetero_data[self.target_node_type].y.to(self.device)
        train_mask = hetero_data[self.target_node_type].train_mask.to(self.device)

        if loss_fn is not None:
            loss = loss_fn(out[train_mask], y[train_mask])
        else:
            loss = F.cross_entropy(out[train_mask], y[train_mask])

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        epoch_time = time.time() - start_time

        # Compute training accuracy
        with torch.no_grad():
            pred = out[train_mask].argmax(dim=1)
            train_acc = (pred == y[train_mask]).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': train_acc,
            'time': epoch_time,
            'lr': self.get_current_lr()
        }

    def validate(
        self,
        hetero_data,
        loss_fn: Optional[Callable] = None,
        return_attention: bool = False
    ) -> Dict[str, float]:
        """
        Validate model on heterogeneous graph

        Args:
            hetero_data: HeteroData object
            loss_fn: Optional custom loss function
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            x_dict = {k: v.to(self.device) for k, v in hetero_data.x_dict.items()}
            edge_index_dict = {
                k: v.to(self.device) for k, v in hetero_data.edge_index_dict.items()
            }

            # Forward pass
            if return_attention:
                out_dict, attention_weights = self.model(
                    x_dict, edge_index_dict, return_attention=True
                )
                self.attention_history.append(attention_weights)
            else:
                out_dict = self.model(x_dict, edge_index_dict)

            # Compute metrics on validation set
            out = out_dict[self.target_node_type]
            y = hetero_data[self.target_node_type].y.to(self.device)
            val_mask = hetero_data[self.target_node_type].val_mask.to(self.device)

            # Loss
            if loss_fn is not None:
                loss = loss_fn(out[val_mask], y[val_mask])
            else:
                loss = F.cross_entropy(out[val_mask], y[val_mask])

            # Accuracy
            pred = out[val_mask].argmax(dim=1)
            val_acc = (pred == y[val_mask]).float().mean().item()

        result = {
            'loss': loss.item(),
            'accuracy': val_acc,
            'metric': val_acc  # Use accuracy for scheduler
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result

    def get_attention_weights(self):
        """Get attention weights from last validation"""
        if len(self.attention_history) > 0:
            return self.attention_history[-1]
        return None
