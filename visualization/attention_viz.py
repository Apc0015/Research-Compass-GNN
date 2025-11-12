"""
Attention Visualization for GAT Models

Visualizes attention weights to understand what the model learns:
- Attention heatmaps
- Top-K attention weights
- Per-head attention patterns
- Interactive network visualization with attention
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class AttentionVisualizer:
    """
    Visualizer for GAT attention weights

    Example:
        >>> viz = AttentionVisualizer()
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     out = model(data.x, data.edge_index, return_attention_weights=True)
        >>> attn_weights = model.get_attention_weights()
        >>> viz.plot_attention_heatmap(attn_weights, paper_titles)
    """

    def __init__(self):
        self.figures = {}

    def plot_attention_heatmap(
        self,
        attention_weights: Tuple[torch.Tensor, torch.Tensor],
        node_names: Optional[List[str]] = None,
        top_k: int = 20,
        title: str = 'GAT Attention Weights'
    ) -> plt.Figure:
        """
        Plot attention heatmap for top-K nodes

        Args:
            attention_weights: Tuple of (edge_index, attention_values)
            node_names: Optional list of node names
            top_k: Number of top nodes to show
            title: Plot title

        Returns:
            Matplotlib figure
        """
        edge_index, alpha = attention_weights

        # Convert to numpy
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy().squeeze()

        # Get unique nodes and their attention statistics
        unique_nodes = np.unique(edge_index)
        num_nodes = len(unique_nodes)

        # Compute average attention for each node
        node_attention = {}
        for node in unique_nodes:
            # Incoming attention
            incoming_mask = edge_index[1] == node
            if incoming_mask.sum() > 0:
                avg_attn = alpha[incoming_mask].mean()
                node_attention[node] = avg_attn

        # Select top-K nodes by average attention
        sorted_nodes = sorted(node_attention.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected_nodes = [n for n, _ in sorted_nodes]

        # Create attention matrix
        attn_matrix = np.zeros((top_k, top_k))

        for i, src_node in enumerate(selected_nodes):
            for j, dst_node in enumerate(selected_nodes):
                # Find edge from src to dst
                edge_mask = (edge_index[0] == src_node) & (edge_index[1] == dst_node)
                if edge_mask.sum() > 0:
                    attn_matrix[i, j] = alpha[edge_mask].mean()

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Node labels
        if node_names:
            labels = [node_names[n] if n < len(node_names) else f"Node {n}"
                     for n in selected_nodes]
        else:
            labels = [f"Node {n}" for n in selected_nodes]

        # Plot heatmap
        sns.heatmap(
            attn_matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )

        ax.set_xlabel('Target Node', fontsize=12)
        ax.set_ylabel('Source Node', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        self.figures['attention_heatmap'] = fig
        return fig

    def plot_attention_distribution(
        self,
        attention_weights: Tuple[torch.Tensor, torch.Tensor],
        title: str = 'Attention Weight Distribution'
    ) -> plt.Figure:
        """
        Plot distribution of attention weights

        Args:
            attention_weights: Tuple of (edge_index, attention_values)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        edge_index, alpha = attention_weights

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy().squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(alpha, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Attention Weight', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Attention Weight Histogram', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(alpha, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[1].set_ylabel('Attention Weight', fontsize=12)
        axes[1].set_title('Attention Weight Statistics', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add statistics
        stats_text = f"Mean: {alpha.mean():.4f}\n"
        stats_text += f"Median: {np.median(alpha):.4f}\n"
        stats_text += f"Std: {alpha.std():.4f}\n"
        stats_text += f"Min: {alpha.min():.4f}\n"
        stats_text += f"Max: {alpha.max():.4f}"

        axes[1].text(1.15, 0.5, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        self.figures['attention_distribution'] = fig
        return fig

    def plot_top_k_attention(
        self,
        attention_weights: Tuple[torch.Tensor, torch.Tensor],
        node_names: Optional[List[str]] = None,
        k: int = 10,
        title: str = 'Top-K Attention Edges'
    ) -> plt.Figure:
        """
        Plot top-K edges by attention weight

        Args:
            attention_weights: Tuple of (edge_index, attention_values)
            node_names: Optional list of node names
            k: Number of top edges to show
            title: Plot title

        Returns:
            Matplotlib figure
        """
        edge_index, alpha = attention_weights

        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy().squeeze()

        # Get top-K edges
        top_k_idx = np.argsort(alpha)[-k:][::-1]
        top_edges = edge_index[:, top_k_idx]
        top_weights = alpha[top_k_idx]

        # Create labels
        edge_labels = []
        for i in range(k):
            src, dst = top_edges[0, i], top_edges[1, i]
            if node_names:
                src_name = node_names[src] if src < len(node_names) else f"Node {src}"
                dst_name = node_names[dst] if dst < len(node_names) else f"Node {dst}"
                label = f"{src_name}\n→ {dst_name}"
            else:
                label = f"Node {src}\n→ Node {dst}"
            edge_labels.append(label)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(k)
        bars = ax.barh(y_pos, top_weights, color='coral', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(edge_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add values on bars
        for i, (bar, weight) in enumerate(zip(bars, top_weights)):
            ax.text(weight + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{weight:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        self.figures['top_k_attention'] = fig
        return fig

    def create_interactive_attention_graph(
        self,
        data,
        attention_weights: Tuple[torch.Tensor, torch.Tensor],
        node_names: Optional[List[str]] = None,
        top_k_nodes: int = 50,
        min_attention: float = 0.1
    ) -> go.Figure:
        """
        Create interactive Plotly graph with attention-weighted edges

        Args:
            data: PyG Data object
            attention_weights: Tuple of (edge_index, attention_values)
            node_names: Optional list of node names
            top_k_nodes: Number of nodes to show
            min_attention: Minimum attention weight to display

        Returns:
            Plotly figure
        """
        edge_index, alpha = attention_weights

        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy().squeeze()

        # Select top-K nodes by degree
        degrees = np.bincount(edge_index[0], minlength=data.num_nodes)
        top_nodes = np.argsort(degrees)[-top_k_nodes:]

        # Filter edges
        mask = np.isin(edge_index[0], top_nodes) & np.isin(edge_index[1], top_nodes)
        mask = mask & (alpha >= min_attention)

        filtered_edges = edge_index[:, mask]
        filtered_alpha = alpha[mask]

        # Create NetworkX graph
        G = nx.DiGraph()
        for i in range(filtered_edges.shape[1]):
            src, dst = filtered_edges[0, i], filtered_edges[1, i]
            weight = filtered_alpha[i]
            G.add_edge(src, dst, weight=weight)

        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Create edge traces
        edge_traces = []

        # Normalize attention weights for visualization
        max_alpha = filtered_alpha.max()
        min_alpha = filtered_alpha.min()

        for edge in G.edges(data=True):
            src, dst, attr = edge
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            weight = attr['weight']

            # Color based on attention weight
            normalized_weight = (weight - min_alpha) / (max_alpha - min_alpha + 1e-8)

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=1 + 3 * normalized_weight,
                    color=f'rgba(255, {int(100 + 155 * (1-normalized_weight))}, 100, 0.6)'
                ),
                hoverinfo='text',
                text=f'Attention: {weight:.4f}',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            if node_names and node < len(node_names):
                name = node_names[node]
            else:
                name = f"Node {node}"

            # Compute average attention (incoming)
            incoming_edges = [e for e in G.in_edges(node, data=True)]
            if incoming_edges:
                avg_attn = np.mean([e[2]['weight'] for e in incoming_edges])
            else:
                avg_attn = 0

            node_text.append(f"{name}<br>Avg Attention: {avg_attn:.4f}")
            node_color.append(avg_attn)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=15,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Avg Attention',
                    thickness=15,
                    len=0.7
                ),
                line=dict(width=2, color='white')
            )
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title='Interactive Attention-Weighted Citation Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        return fig

    def save_all(self, output_dir: str):
        """Save all generated figures"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in self.figures.items():
            if isinstance(fig, plt.Figure):
                filepath = output_path / f"{name}.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved {filepath}")

    def close_all(self):
        """Close all figures"""
        for fig in self.figures.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)
        self.figures.clear()


def analyze_attention_patterns(
    attention_weights: Tuple[torch.Tensor, torch.Tensor],
    data,
    node_names: Optional[List[str]] = None
) -> Dict:
    """
    Analyze attention patterns and return statistics

    Args:
        attention_weights: Tuple of (edge_index, attention_values)
        data: PyG Data object
        node_names: Optional node names

    Returns:
        Dictionary with analysis results
    """
    edge_index, alpha = attention_weights

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.cpu().numpy().squeeze()

    analysis = {}

    # Overall statistics
    analysis['mean_attention'] = float(alpha.mean())
    analysis['median_attention'] = float(np.median(alpha))
    analysis['std_attention'] = float(alpha.std())
    analysis['min_attention'] = float(alpha.min())
    analysis['max_attention'] = float(alpha.max())

    # Per-node statistics
    node_attention = {}
    for node in range(data.num_nodes):
        incoming_mask = edge_index[1] == node
        if incoming_mask.sum() > 0:
            node_attention[node] = {
                'avg_incoming_attention': float(alpha[incoming_mask].mean()),
                'max_incoming_attention': float(alpha[incoming_mask].max()),
                'num_incoming': int(incoming_mask.sum())
            }

    # Top nodes by attention
    sorted_nodes = sorted(node_attention.items(),
                         key=lambda x: x[1]['avg_incoming_attention'],
                         reverse=True)[:10]

    analysis['top_10_nodes'] = [
        {
            'node_id': int(node),
            'node_name': node_names[node] if node_names and node < len(node_names) else f"Node {node}",
            'avg_attention': stats['avg_incoming_attention'],
            'max_attention': stats['max_incoming_attention'],
            'num_incoming': stats['num_incoming']
        }
        for node, stats in sorted_nodes
    ]

    # Attention concentration (Gini coefficient)
    sorted_alpha = np.sort(alpha)
    n = len(sorted_alpha)
    cumsum = np.cumsum(sorted_alpha)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_alpha)) / (n * cumsum[-1]) - (n + 1) / n
    analysis['attention_gini'] = float(gini)

    return analysis
