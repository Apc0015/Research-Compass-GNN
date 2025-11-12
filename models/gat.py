"""
GAT (Graph Attention Network) Model
For link prediction and node classification (multi-task learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class GATModel(nn.Module):
    """
    Graph Attention Network for Link Prediction and Node Classification

    This model supports multi-task learning:
    - Primary task: Link prediction (citation prediction)
    - Secondary task: Node classification (paper topic prediction)

    Architecture:
    - Multi-head attention layers
    - Separate heads for link prediction and node classification
    - Attention weight extraction for visualization

    Args:
        input_dim (int): Input feature dimension (default: 384)
        hidden_dim (int): Hidden dimension per attention head (default: 128)
        num_layers (int): Number of GAT layers (default: 2)
        heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout probability (default: 0.3)
        output_dim (int): Number of classes for node classification (default: 5)

    Example:
        >>> model = GATModel(input_dim=384, hidden_dim=128, output_dim=5)
        >>> # Link prediction
        >>> link_pred = model(x, edge_index, edge_label_index)
        >>> # Node classification
        >>> node_pred = model.forward_node_classification(x, edge_index)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        output_dim: int = 5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.output_dim = output_dim

        # Attention layers for encoding
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout
                )
            )

        # Final encoding layer (single head)
        self.convs.append(
            GATConv(
                hidden_dim * heads if num_layers > 1 else hidden_dim,
                hidden_dim,
                heads=1,
                dropout=dropout
            )
        )

        # Link prediction head (edge predictor)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Node classification head
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Store attention weights for visualization
        self.attention_weights = None

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Encode nodes into latent representations

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            return_attention_weights: Whether to store attention weights

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            if return_attention_weights and i == len(self.convs) - 1:
                # Extract attention weights from final layer
                x, (edge_idx, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                self.attention_weights = (edge_idx, alpha)
            else:
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)

        return x

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode edge existence from node embeddings (link prediction)

        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices to predict [2, num_edges_to_predict]

        Returns:
            Edge prediction scores [num_edges_to_predict]
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=1)
        return self.edge_predictor(edge_features).squeeze()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for link prediction

        Args:
            x: Node features
            edge_index: Training edges
            edge_label_index: Edges to predict
            return_attention_weights: Whether to store attention

        Returns:
            Link prediction scores
        """
        z = self.encode(x, edge_index, return_attention_weights)
        return self.decode(z, edge_label_index)

    def forward_node_classification(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for node classification (multi-task learning)

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node class predictions [num_nodes, output_dim]
        """
        z = self.encode(x, edge_index)
        return self.node_classifier(z)

    def multi_task_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both tasks (multi-task learning)

        Args:
            x: Node features
            edge_index: Training edges
            edge_label_index: Edges to predict

        Returns:
            Tuple of (link_predictions, node_predictions)
        """
        z = self.encode(x, edge_index)
        link_pred = self.decode(z, edge_label_index)
        node_pred = self.node_classifier(z)
        return link_pred, node_pred

    def get_attention_weights(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get stored attention weights from last forward pass

        Returns:
            Tuple of (edge_index, attention_weights) or None
        """
        return self.attention_weights

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"heads={self.heads}, "
            f"dropout={self.dropout}, "
            f"output_dim={self.output_dim}, "
            f"params={self.count_parameters():,})"
        )
