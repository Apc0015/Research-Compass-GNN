#!/usr/bin/env python3
"""
Node Classifier - GNN for paper topic classification
Uses Graph Convolutional Networks for multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class PaperClassifier(nn.Module):
    """
    Graph Convolutional Network for paper topic classification

    Architecture:
    - 3 GCN layers
    - Hidden dimension: 256
    - Dropout for regularization
    - Multi-label output
    """

    def __init__(
        self,
        input_dim: int = 384,  # Sentence transformer embedding size
        hidden_dim: int = 256,
        output_dim: int = 10,  # Number of topics
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize paper classifier

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes/topics
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super(PaperClassifier, self).__init__()

        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            raise ImportError("torch_geometric not installed. Run: pip install torch-geometric")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build GCN layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, x, edge_index):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Output logits [num_nodes, output_dim]
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)

        return x

    def predict_topics(
        self,
        x,
        edge_index,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k topics for nodes

        Args:
            x: Node features
            edge_index: Edge indices
            top_k: Number of top topics to return
            threshold: Probability threshold

        Returns:
            Tuple of (topic indices, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = torch.sigmoid(logits)  # Multi-label

            # Get top-k
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

            # Filter by threshold
            mask = top_probs > threshold

            return top_indices, top_probs


def train_node_classifier(
    model: PaperClassifier,
    data: 'Data',
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 10,
    save_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
    progress_callback: Optional[callable] = None,
    resume_from_checkpoint: bool = True
) -> Dict:
    """
    Train node classifier with early stopping, checkpointing, and progress reporting

    Args:
        model: PaperClassifier model
        data: PyG Data object with train/val/test masks
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        save_path: Path to save best model
        checkpoint_path: Path to save checkpoints (for crash recovery)
        checkpoint_interval: Save checkpoint every N epochs
        progress_callback: Callback function(epoch, total_epochs, metrics)
        resume_from_checkpoint: Resume from checkpoint if exists

    Returns:
        Training history dictionary
    """
    from .gnn_utils import create_checkpoint, load_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Try to resume from checkpoint
    start_epoch = 0
    if resume_from_checkpoint and checkpoint_path and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(model, optimizer, str(checkpoint_path))
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")

    # Loss function
    if hasattr(data, 'y'):
        num_classes = data.y.max().item() + 1
        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()  # Multi-label default

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # GPU memory management
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        if hasattr(data, 'train_mask'):
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)

            if hasattr(data, 'val_mask'):
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])

                # Calculate accuracy
                pred = val_out[data.val_mask].argmax(dim=1)
                train_pred = out[data.train_mask].argmax(dim=1)

                train_acc = (train_pred == data.y[data.train_mask]).float().mean()
                val_acc = (pred == data.y[data.val_mask]).float().mean()
            else:
                val_loss = loss
                train_acc = 0.0
                val_acc = 0.0

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        history['val_acc'].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)

        # Progress callback
        if progress_callback:
            try:
                progress_callback(
                    epoch + 1,
                    epochs,
                    {
                        'train_loss': loss.item(),
                        'val_loss': val_loss.item(),
                        'train_acc': train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc,
                        'val_acc': val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc
                    }
                )
            except Exception as e:
                print(f"Progress callback error: {e}")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d}: '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}')

        # Periodic checkpoint (for crash recovery)
        if checkpoint_path and (epoch + 1) % checkpoint_interval == 0:
            create_checkpoint(model, optimizer, epoch, loss.item(), str(checkpoint_path))
            print(f"  Checkpoint saved at epoch {epoch + 1}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'val_loss': val_loss
                }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if save_path and save_path.exists():
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    return history


def evaluate_classifier(
    model: PaperClassifier,
    data: 'Data',
    mask_name: str = 'test_mask'
) -> Dict[str, float]:
    """
    Evaluate classifier on test set

    Args:
        model: Trained model
        data: PyG Data object
        mask_name: Name of mask to use ('test_mask', 'val_mask', etc.)

    Returns:
        Dictionary of evaluation metrics
    """
    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        if hasattr(data, mask_name):
            mask = getattr(data, mask_name)
            logits = out[mask]
            labels = data.y[mask]
        else:
            logits = out
            labels = data.y

        # Predictions
        pred = logits.argmax(dim=1)

        # Accuracy
        accuracy = (pred == labels).float().mean().item()

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels).item()

        # Per-class accuracy
        unique_labels = labels.unique()
        class_acc = {}
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                class_acc[label.item()] = (pred[mask] == labels[mask]).float().mean().item()

        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'num_samples': len(labels),
            'class_accuracy': class_acc
        }

        return metrics


# Main execution for testing
if __name__ == "__main__":
    from .graph_converter import Neo4jToTorchGeometric
    import os

    print("=" * 80)
    print("Node Classifier Test")
    print("=" * 80)

    # Get configuration
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Load graph
    converter = Neo4jToTorchGeometric(neo4j_uri, neo4j_user, neo4j_password)

    try:
        data = converter.export_papers_graph()
        data = converter.create_train_val_test_split(data)
        data, label_map = converter.add_node_labels(data)

        print(f"\nGraph loaded:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Classes: {len(label_map)}")

        # Create model
        model = PaperClassifier(
            input_dim=data.x.shape[1],
            output_dim=len(label_map),
            hidden_dim=256,
            num_layers=3,
            dropout=0.5
        )

        print(f"\nModel created:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        print("\nTraining...")
        history = train_node_classifier(
            model,
            data,
            epochs=50,
            lr=0.01,
            patience=10,
            save_path=Path("models/node_classifier_best.pt")
        )

        # Evaluate
        print("\nEvaluating...")
        metrics = evaluate_classifier(model, data, 'test_mask')
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test Loss: {metrics['loss']:.4f}")

    finally:
        converter.close()
