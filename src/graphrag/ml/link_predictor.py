#!/usr/bin/env python3
"""
Link Predictor - GNN for citation prediction
Predicts potential future citations between papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class CitationPredictor(nn.Module):
    """
    Graph Neural Network for link prediction (citation prediction)

    Uses GAT (Graph Attention Network) encoder + edge predictor
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize citation predictor

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CitationPredictor, self).__init__()

        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError("torch_geometric not installed")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # GAT encoder layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Output layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))

        # Edge predictor (MLP)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        """
        Encode nodes using GAT

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)

        return x

    def decode(self, z, edge_index):
        """
        Decode edges from node embeddings

        Args:
            z: Node embeddings
            edge_index: Edge indices to predict

        Returns:
            Edge predictions (logits)
        """
        # Get source and target embeddings
        src = z[edge_index[0]]
        dst = z[edge_index[1]]

        # Concatenate
        edge_features = torch.cat([src, dst], dim=1)

        # Predict
        return self.edge_predictor(edge_features).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        """
        Forward pass

        Args:
            x: Node features
            edge_index: Training edge indices
            edge_label_index: Edges to predict

        Returns:
            Edge predictions
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def predict_citations(
        self,
        x,
        edge_index,
        paper_id: int,
        candidate_papers: Optional[List[int]] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k papers this paper will cite

        Args:
            x: Node features
            edge_index: Existing edges
            paper_id: Source paper node ID
            candidate_papers: Candidate target papers (None = all)
            top_k: Number of predictions

        Returns:
            Tuple of (paper indices, prediction scores)
        """
        self.eval()
        with torch.no_grad():
            # Encode all nodes
            z = self.encode(x, edge_index)

            if candidate_papers is None:
                # Predict to all other papers
                candidate_papers = list(range(len(x)))
                candidate_papers.remove(paper_id)

            # Create candidate edge indices
            edge_label_index = torch.tensor(
                [[paper_id] * len(candidate_papers), candidate_papers],
                dtype=torch.long
            )

            # Predict
            scores = torch.sigmoid(self.decode(z, edge_label_index))

            # Get top-k
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

            # Convert to original paper IDs
            top_papers = torch.tensor([candidate_papers[i] for i in top_indices])

            return top_papers, top_scores


def create_negative_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Create negative edge samples for training

    Args:
        edge_index: Positive edge indices
        num_nodes: Number of nodes in graph
        num_neg_samples: Number of negative samples (None = same as positives)

    Returns:
        Negative edge indices
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)

    # Convert positive edges to set
    positive_edges = set(
        tuple(edge_index[:, i].tolist()) for i in range(edge_index.size(1))
    )

    negative_edges = []

    while len(negative_edges) < num_neg_samples:
        # Random source and target
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)

        # Skip if self-loop or exists
        if src != dst and (src, dst) not in positive_edges:
            negative_edges.append([src, dst])

    return torch.tensor(negative_edges, dtype=torch.long).t()


def train_link_predictor(
    model: CitationPredictor,
    data: 'Data',
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: Optional[int] = None,
    patience: int = 10,
    save_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
    progress_callback: Optional[callable] = None,
    resume_from_checkpoint: bool = True
) -> Dict:
    """
    Train link predictor with checkpointing and progress reporting

    Args:
        model: CitationPredictor model
        data: PyG Data object
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size (None = full batch)
        patience: Early stopping patience
        save_path: Path to save model
        checkpoint_path: Path to save checkpoints (for crash recovery)
        checkpoint_interval: Save checkpoint every N epochs
        progress_callback: Callback function(epoch, total_epochs, metrics)
        resume_from_checkpoint: Resume from checkpoint if exists

    Returns:
        Training history
    """
    from .gnn_utils import create_checkpoint, load_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Try to resume from checkpoint
    start_epoch = 0
    if resume_from_checkpoint and checkpoint_path and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(model, optimizer, str(checkpoint_path))
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")

    # Split edges for training/validation
    num_edges = data.edge_index.size(1)
    num_train = int(num_edges * 0.8)
    num_val = num_edges - num_train

    # Random permutation
    perm = torch.randperm(num_edges)
    train_edges = data.edge_index[:, perm[:num_train]]
    val_edges = data.edge_index[:, perm[num_train:]]

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device}")
    print(f"Train edges: {train_edges.size(1)}, Val edges: {val_edges.size(1)}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # GPU memory management
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Training
        model.train()
        optimizer.zero_grad()

        # Create negative samples
        neg_train_edges = create_negative_samples(
            train_edges,
            data.num_nodes,
            num_neg_samples=train_edges.size(1)
        ).to(device)

        # Combine positive and negative
        edge_label_index = torch.cat([train_edges, neg_train_edges], dim=1)
        edge_labels = torch.cat([
            torch.ones(train_edges.size(1)),
            torch.zeros(neg_train_edges.size(1))
        ]).to(device)

        # Forward
        out = model(data.x, train_edges, edge_label_index)
        loss = criterion(out, edge_labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            neg_val_edges = create_negative_samples(
                val_edges,
                data.num_nodes,
                num_neg_samples=val_edges.size(1)
            ).to(device)

            val_edge_label_index = torch.cat([val_edges, neg_val_edges], dim=1)
            val_edge_labels = torch.cat([
                torch.ones(val_edges.size(1)),
                torch.zeros(neg_val_edges.size(1))
            ]).to(device)

            val_out = model(data.x, train_edges, val_edge_label_index)
            val_loss = criterion(val_out, val_edge_labels)

            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(
                edge_labels.cpu().numpy(),
                torch.sigmoid(out).detach().cpu().numpy()
            )
            val_auc = roc_auc_score(
                val_edge_labels.cpu().numpy(),
                torch.sigmoid(val_out).cpu().numpy()
            )

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Progress callback
        if progress_callback:
            try:
                progress_callback(
                    epoch + 1,
                    epochs,
                    {
                        'train_loss': loss.item(),
                        'val_loss': val_loss.item(),
                        'train_auc': train_auc,
                        'val_auc': val_auc
                    }
                )
            except Exception as e:
                print(f"Progress callback error: {e}")

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d}: '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, '
                  f'Train AUC: {train_auc:.4f}, '
                  f'Val AUC: {val_auc:.4f}')

        # Periodic checkpoint (for crash recovery)
        if checkpoint_path and (epoch + 1) % checkpoint_interval == 0:
            create_checkpoint(model, optimizer, epoch, loss.item(), str(checkpoint_path))
            print(f"  Checkpoint saved at epoch {epoch + 1}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'val_loss': val_loss,
                    'val_auc': val_auc
                }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history


# Main execution for testing
if __name__ == "__main__":
    from .graph_converter import Neo4jToTorchGeometric
    import os

    print("=" * 80)
    print("Link Predictor Test")
    print("=" * 80)

    # Configuration
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Load graph
    converter = Neo4jToTorchGeometric(neo4j_uri, neo4j_user, neo4j_password)

    try:
        data = converter.export_citation_network(min_citations=1)

        print(f"\nCitation network loaded:")
        print(f"  Papers: {data.num_nodes}")
        print(f"  Citations: {data.num_edges}")

        # Create model
        model = CitationPredictor(
            input_dim=data.x.shape[1],
            hidden_dim=128,
            num_layers=2,
            heads=4
        )

        print(f"\nModel created:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        print("\nTraining...")
        history = train_link_predictor(
            model,
            data,
            epochs=50,
            lr=0.001,
            patience=10,
            save_path=Path("models/link_predictor_best.pt")
        )

        print(f"\nFinal Val AUC: {history['val_auc'][-1]:.4f}")

        # Test prediction
        print("\nTesting predictions...")
        paper_id = 0
        top_papers, scores = model.predict_citations(
            data.x,
            data.edge_index,
            paper_id,
            top_k=5
        )

        print(f"\nTop 5 citation predictions for paper {paper_id}:")
        for i, (paper, score) in enumerate(zip(top_papers, scores), 1):
            print(f"  {i}. Paper {paper.item()}: {score.item():.4f}")

    finally:
        converter.close()
