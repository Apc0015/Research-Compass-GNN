#!/usr/bin/env python3
"""
GNN Explainer - Visualization and explainability for Graph Neural Networks
Provides insights into GNN decision-making and recommendation reasoning.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GNNAttentionVisualizer:
    """
    Visualizes attention weights from GNN models.
    
    Shows which nodes and edges are most important
    for GNN predictions and recommendations.
    """
    
    def __init__(self, gnn_model):
        """
        Initialize attention visualizer.
        
        Args:
            gnn_model: Trained GNN model with attention mechanisms
        """
        self.gnn_model = gnn_model
        self.attention_weights = {}
        self.node_importance = {}
        
    def extract_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_nodes: Optional[List[int]] = None
    ) -> Dict:
        """
        Extract attention weights from GNN layers.
        
        Args:
            x: Node features
            edge_index: Edge indices
            target_nodes: Specific nodes to analyze
            
        Returns:
            Dictionary with attention weights and importance scores
        """
        attention_data = {
            'layer_attention': [],
            'edge_attention': {},
            'node_importance': {},
            'attention_paths': []
        }
        
        # Hook into GNN layers to capture attention
        hooks = []
        attention_layers = self._find_attention_layers(self.gnn_model)
        
        for layer in attention_layers:
            hook = self._create_attention_hook(layer)
            hooks.append(hook)
        
        # Forward pass to capture attention
        with torch.no_grad():
            output = self.gnn_model(x, edge_index)
        
        # Extract attention from hooks
        for i, hook in enumerate(hooks):
            if hasattr(hook, 'attention_weights'):
                attention_data['layer_attention'].append({
                    'layer': i,
                    'weights': hook.attention_weights.detach().cpu().numpy(),
                    'shape': hook.attention_weights.shape
                })
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute node importance
        attention_data['node_importance'] = self._compute_node_importance(
            attention_data['layer_attention'], target_nodes
        )
        
        # Extract important paths
        attention_data['attention_paths'] = self._extract_attention_paths(
            attention_data['layer_attention'], edge_index
        )
        
        return attention_data
    
    def _find_attention_layers(self, model: nn.Module) -> List[nn.Module]:
        """Find layers with attention mechanisms."""
        attention_layers = []
        
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'gat' in name.lower():
                attention_layers.append(module)
            elif hasattr(module, 'attention_weights'):
                attention_layers.append(module)
        
        return attention_layers
    
    def _create_attention_hook(self, layer: nn.Module):
        """Create hook to capture attention weights."""
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                hook.attention_weights = module.attention_weights
            elif hasattr(output, 'attention_weights'):
                hook.attention_weights = output.attention_weights
        
        hook = layer.register_forward_hook(hook_fn)
        return hook
    
    def _compute_node_importance(
        self,
        layer_attention: List[Dict],
        target_nodes: Optional[List[int]]
    ) -> Dict[int, float]:
        """Compute node importance from attention weights."""
        importance = {}
        
        for layer_data in layer_attention:
            weights = layer_data['weights']
            
            # Average attention across heads and edges
            if len(weights.shape) == 3:  # [heads, nodes, nodes]
                node_attention = np.mean(weights, axis=(0, 1))
            elif len(weights.shape) == 2:  # [nodes, nodes]
                node_attention = np.mean(weights, axis=1)
            else:
                node_attention = np.mean(weights)
            
            # Accumulate importance
            for i, attn in enumerate(node_attention):
                if i not in importance:
                    importance[i] = 0.0
                importance[i] += attn
        
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            importance = {k: v / max_importance for k, v in importance.items()}
        
        return importance
    
    def _extract_attention_paths(
        self,
        layer_attention: List[Dict],
        edge_index: torch.Tensor
    ) -> List[Dict]:
        """Extract important attention paths."""
        paths = []
        
        if not layer_attention:
            return paths
        
        # Get attention from final layer
        final_attention = layer_attention[-1]['weights']
        
        # Find top attention paths
        if len(final_attention.shape) >= 2:
            edge_attention = np.mean(final_attention, axis=0) if len(final_attention.shape) == 3 else final_attention
            
            # Get top-k edges by attention
            top_k = min(10, edge_index.size(1))
            top_indices = np.argsort(edge_attention.flatten())[-top_k:]
            
            for idx in top_indices:
                edge_idx = idx % edge_index.size(1)
                source = edge_index[0, edge_idx].item()
                target = edge_index[1, edge_idx].item()
                attention_score = edge_attention.flatten()[idx]
                
                paths.append({
                    'source': source,
                    'target': target,
                    'attention_weight': attention_score,
                    'path_type': 'attention'
                })
        
        return paths
    
    def create_attention_visualization(
        self,
        attention_data: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create HTML visualization of attention weights.
        
        Args:
            attention_data: Attention analysis results
            output_path: Optional file path to save visualization
            
        Returns:
            HTML string for visualization
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Attention Visualization</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .node { stroke: #333; stroke-width: 2px; }
                .link { stroke: #999; stroke-opacity: 0.6; }
                .important { stroke: #ff6b6b; stroke-width: 3px; }
                .attention { stroke: #4ecdc4; stroke-width: 2px; }
                .tooltip { position: absolute; text-align: center; padding: 8px; 
                          font: 12px sans-serif; background: rgba(0,0,0,0.8); color: white; 
                          border-radius: 4px; pointer-events: none; }
            </style>
        </head>
        <body>
            <h2>GNN Attention Analysis</h2>
            <div id="graph"></div>
            <div id="legend">
                <h3>Legend</h3>
                <svg width="200" height="60">
                    <line x1="10" y1="20" x2="50" y2="20" class="important"/>
                    <text x="60" y="25">High Importance</text>
                    <line x1="10" y1="40" x2="50" y2="40" class="attention"/>
                    <text x="60" y="45">Attention Path</text>
                </svg>
            </div>
            
            <script>
                // Data from attention analysis
                const attentionData = """ + json.dumps(attention_data) + """;
                
                // Create force-directed graph
                const width = 800;
                const height = 600;
                
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Create simulation
                const simulation = d3.forceSimulation(attentionData.attention_paths || [])
                    .force("link", d3.forceLink().id(d => d.id).distance(100))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2));
                
                // Add links
                const link = svg.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(attentionData.attention_paths || [])
                    .enter().append("line")
                    .attr("class", d => d.path_type === "attention" ? "attention" : "link")
                    .attr("stroke-width", d => Math.max(1, d.attention_weight * 5));
                
                // Add nodes
                const node = svg.append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(Object.entries(attentionData.node_importance || {}))
                    .enter().append("circle")
                    .attr("class", d => d[1] > 0.7 ? "important" : "node")
                    .attr("r", d => 5 + d[1] * 15)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                // Add labels
                const label = svg.append("g")
                    .attr("class", "labels")
                    .selectAll("text")
                    .data(Object.entries(attentionData.node_importance || {}))
                    .enter().append("text")
                    .text(d => "Node " + d[0])
                    .attr("font-size", 10)
                    .attr("dx", 12)
                    .attr("dy", 4);
                
                // Update positions on simulation tick
                simulation.on("tick", () => {
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);
                    
                    label
                        .attr("x", d => d.x + 15)
                        .attr("y", d => d.y);
                });
                
                // Drag functions
                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }
                
                function dragged(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }
                
                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }
            </script>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"Attention visualization saved to {output_path}")
        
        return html


class GNNDecisionExplainer:
    """
    Explains GNN decisions and recommendations.
    
    Provides human-readable explanations for why
    the GNN made specific predictions or recommendations.
    """
    
    def __init__(self, gnn_model, graph_data):
        """
        Initialize decision explainer.
        
        Args:
            gnn_model: Trained GNN model
            graph_data: Graph structure for context
        """
        self.gnn_model = gnn_model
        self.graph_data = graph_data
        self.explanation_cache = {}
        
    def explain_recommendation(
        self,
        user_id: str,
        paper_id: int,
        recommendation_score: float,
        user_profile: Dict,
        top_k_explanations: int = 3
    ) -> Dict:
        """
        Explain why a paper was recommended to a user.
        
        Args:
            user_id: User identifier
            paper_id: Recommended paper ID
            recommendation_score: GNN recommendation score
            user_profile: User's reading history and interests
            top_k_explanations: Number of top explanations to return
            
        Returns:
            Dictionary with detailed explanations
        """
        explanations = {
            'user_id': user_id,
            'paper_id': paper_id,
            'recommendation_score': recommendation_score,
            'primary_explanation': '',
            'supporting_evidence': [],
            'confidence': 0.0,
            'explanation_type': 'gnn_based'
        }
        
        # 1. Content-based explanation
        content_explanation = self._explain_content_similarity(
            paper_id, user_profile
        )
        explanations['supporting_evidence'].append(content_explanation)
        
        # 2. Collaborative explanation
        collab_explanation = self._explain_collaborative_filtering(
            paper_id, user_profile
        )
        explanations['supporting_evidence'].append(collab_explanation)
        
        # 3. Graph structure explanation
        structure_explanation = self._explain_graph_structure(paper_id, user_profile)
        explanations['supporting_evidence'].append(structure_explanation)
        
        # 4. Diversity explanation
        diversity_explanation = self._explain_diversity_contribution(
            paper_id, user_profile
        )
        explanations['supporting_evidence'].append(diversity_explanation)
        
        # 5. Select primary explanation
        primary = self._select_primary_explanation(
            explanations['supporting_evidence']
        )
        explanations['primary_explanation'] = primary['text']
        explanations['confidence'] = primary['confidence']
        explanations['explanation_type'] = primary['type']
        
        return explanations
    
    def _explain_content_similarity(
        self,
        paper_id: int,
        user_profile: Dict
    ) -> Dict:
        """Explain recommendation based on content similarity."""
        read_papers = user_profile.get('read_papers', [])
        
        # Find similar papers in user history
        similar_papers = self._find_similar_papers(paper_id, read_papers)
        
        if similar_papers:
            return {
                'type': 'content_based',
                'text': f"Recommended because it's similar to papers you've read: {', '.join(similar_papers[:3])}",
                'confidence': 0.8,
                'evidence': similar_papers
            }
        else:
            return {
                'type': 'content_based',
                'text': "Recommended based on content analysis using GNN embeddings",
                'confidence': 0.5,
                'evidence': []
            }
    
    def _explain_collaborative_filtering(
        self,
        paper_id: int,
        user_profile: Dict
    ) -> Dict:
        """Explain recommendation based on collaborative filtering."""
        # Find users with similar reading patterns
        similar_users = self._find_similar_users(user_profile)
        
        if similar_users:
            return {
                'type': 'collaborative',
                'text': f"Recommended because users with similar interests also read this paper",
                'confidence': 0.7,
                'evidence': similar_users[:5]
            }
        else:
            return {
                'type': 'collaborative',
                'text': "Recommended based on collaborative filtering patterns",
                'confidence': 0.4,
                'evidence': []
            }
    
    def _explain_graph_structure(
        self,
        paper_id: int,
        user_profile: Dict
    ) -> Dict:
        """Explain recommendation based on graph structure."""
        # Analyze graph position of paper
        graph_features = self._analyze_graph_position(paper_id)
        
        if graph_features['centrality'] > 0.7:
            return {
                'type': 'graph_structure',
                'text': f"Recommended because it's a highly central paper in the research network (centrality: {graph_features['centrality']:.2f})",
                'confidence': 0.8,
                'evidence': graph_features
            }
        elif graph_features['community_importance'] > 0.6:
            return {
                'type': 'graph_structure',
                'text': f"Recommended because it's important in your research communities",
                'confidence': 0.7,
                'evidence': graph_features
            }
        else:
            return {
                'type': 'graph_structure',
                'text': "Recommended based on its position in the research network",
                'confidence': 0.5,
                'evidence': graph_features
            }
    
    def _explain_diversity_contribution(
        self,
        paper_id: int,
        user_profile: Dict
    ) -> Dict:
        """Explain recommendation based on diversity contribution."""
        # Analyze how this paper adds diversity
        diversity_score = self._compute_diversity_contribution(paper_id, user_profile)
        
        if diversity_score > 0.7:
            return {
                'type': 'diversity',
                'text': f"Recommended to introduce diverse perspectives and new research areas",
                'confidence': 0.6,
                'evidence': {'diversity_score': diversity_score}
            }
        else:
            return {
                'type': 'diversity',
                'text': "Recommended to broaden your research perspective",
                'confidence': 0.4,
                'evidence': {'diversity_score': diversity_score}
            }
    
    def _find_similar_papers(
        self,
        paper_id: int,
        read_papers: List[int]
    ) -> List[str]:
        """Find papers similar to the recommended paper."""
        # Simplified similarity check
        similar = []
        
        for read_paper in read_papers:
            # Check if papers are connected in graph
            if self._are_papers_connected(paper_id, read_paper):
                similar.append(f"Paper {read_paper}")
        
        return similar[:3]
    
    def _find_similar_users(self, user_profile: Dict) -> List[str]:
        """Find users with similar reading patterns."""
        # Simplified user similarity
        return [f"User {i}" for i in range(5)]  # Mock similar users
    
    def _analyze_graph_position(self, paper_id: int) -> Dict:
        """Analyze graph position of a paper."""
        # Compute graph metrics
        degree = self._compute_degree(paper_id)
        centrality = self._compute_centrality(paper_id)
        community_importance = self._compute_community_importance(paper_id)
        
        return {
            'degree': degree,
            'centrality': centrality,
            'community_importance': community_importance
        }
    
    def _compute_diversity_contribution(
        self,
        paper_id: int,
        user_profile: Dict
    ) -> float:
        """Compute how much this paper contributes to diversity."""
        read_papers = user_profile.get('read_papers', [])
        
        if not read_papers:
            return 0.5
        
        # Compute diversity based on graph distance
        total_distance = 0
        for read_paper in read_papers:
            distance = self._compute_graph_distance(paper_id, read_paper)
            total_distance += distance
        
        # Normalize diversity score
        avg_distance = total_distance / len(read_papers)
        diversity_score = min(1.0, avg_distance / 5.0)  # Normalize to [0,1]
        
        return diversity_score
    
    def _are_papers_connected(self, paper1: int, paper2: int) -> bool:
        """Check if two papers are connected in graph."""
        if self.graph_data is None:
            return False
        
        edge_index = self.graph_data.edge_index
        connections = set()
        
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            connections.add((source, target))
            connections.add((target, source))
        
        return (paper1, paper2) in connections or (paper2, paper1) in connections
    
    def _compute_degree(self, paper_id: int) -> int:
        """Compute degree of a paper node."""
        if self.graph_data is None:
            return 0
        
        edge_index = self.graph_data.edge_index
        return (edge_index[0] == paper_id).sum().item()
    
    def _compute_centrality(self, paper_id: int) -> float:
        """Compute centrality score for a paper."""
        degree = self._compute_degree(paper_id)
        
        if self.graph_data is None:
            return 0.0
        
        # Normalize by max degree
        max_degree = max(
            (self.graph_data.edge_index[0] == i).sum().item()
            for i in range(self.graph_data.num_nodes)
        )
        
        return degree / max(max_degree, 1)
    
    def _compute_community_importance(self, paper_id: int) -> float:
        """Compute importance within community."""
        # Simplified community importance
        return 0.6  # Mock value
    
    def _compute_graph_distance(self, paper1: int, paper2: int) -> int:
        """Compute graph distance between two papers."""
        # Simplified distance computation
        return abs(paper1 - paper2) % 10  # Mock distance
    
    def _select_primary_explanation(
        self,
        explanations: List[Dict]
    ) -> Dict:
        """Select the best explanation from candidates."""
        if not explanations:
            return {
                'text': "No explanation available",
                'confidence': 0.0,
                'type': 'unknown'
            }
        
        # Sort by confidence
        sorted_explanations = sorted(
            explanations, key=lambda x: x['confidence'], reverse=True
        )
        
        return sorted_explanations[0]
    
    def explain_node_classification(
        self,
        node_id: int,
        predicted_class: int,
        class_probabilities: torch.Tensor,
        true_class: Optional[int] = None
    ) -> Dict:
        """
        Explain GNN node classification decision.
        
        Args:
            node_id: Node being classified
            predicted_class: Predicted class label
            class_probabilities: Class probability distribution
            true_class: True class label (if available)
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'node_id': node_id,
            'predicted_class': predicted_class,
            'true_class': true_class,
            'confidence': 0.0,
            'feature_importance': {},
            'neighborhood_influence': {},
            'reasoning_path': []
        }
        
        # Extract top-k predictions
        top_probs, top_classes = torch.topk(class_probabilities, k=3)
        
        explanation['confidence'] = top_probs[0].item()
        explanation['top_predictions'] = [
            {'class': cls.item(), 'probability': prob.item()}
            for cls, prob in zip(top_classes, top_probs)
        ]
        
        # Analyze feature importance
        explanation['feature_importance'] = self._analyze_feature_importance(
            node_id, predicted_class
        )
        
        # Analyze neighborhood influence
        explanation['neighborhood_influence'] = self._analyze_neighborhood_influence(
            node_id, predicted_class
        )
        
        # Generate reasoning path
        explanation['reasoning_path'] = self._generate_reasoning_path(
            node_id, predicted_class, explanation
        )
        
        return explanation
    
    def _analyze_feature_importance(
        self,
        node_id: int,
        predicted_class: int
    ) -> Dict[str, float]:
        """Analyze which features were most important."""
        # Simplified feature importance analysis
        # In practice, use gradient-based or attention-based importance
        
        feature_names = [
            'title_similarity', 'abstract_similarity', 'author_overlap',
            'venue_similarity', 'citation_pattern', 'temporal_proximity'
        ]
        
        # Mock importance scores
        importance = {
            name: np.random.random() for name in feature_names
        }
        
        # Normalize
        total = sum(importance.values())
        importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _analyze_neighborhood_influence(
        self,
        node_id: int,
        predicted_class: int
    ) -> Dict:
        """Analyze how neighborhood influenced classification."""
        # Get neighbors of the node
        neighbors = self._get_node_neighbors(node_id)
        
        # Analyze neighbor class distribution
        neighbor_classes = {}
        for neighbor in neighbors:
            # Mock neighbor class
            neighbor_class = neighbor % 5  # Mock class assignment
            neighbor_classes[neighbor_class] = neighbor_classes.get(neighbor_class, 0) + 1
        
        return {
            'neighbors': neighbors,
            'neighbor_class_distribution': neighbor_classes,
            'dominant_neighbor_class': max(neighbor_classes, key=neighbor_classes.get) if neighbor_classes else None
        }
    
    def _generate_reasoning_path(
        self,
        node_id: int,
        predicted_class: int,
        explanation: Dict
    ) -> List[str]:
        """Generate step-by-step reasoning path."""
        reasoning = []
        
        # Step 1: Initial observation
        reasoning.append(f"Node {node_id} analyzed with features")
        
        # Step 2: Feature analysis
        important_features = [
            feat for feat, imp in explanation['feature_importance'].items()
            if imp > 0.1
        ]
        if important_features:
            reasoning.append(f"Key features identified: {', '.join(important_features[:3])}")
        
        # Step 3: Neighborhood influence
        neighborhood = explanation['neighborhood_influence']
        if neighborhood['dominant_neighbor_class'] is not None:
            reasoning.append(
                f"Neighborhood dominated by class {neighborhood['dominant_neighbor_class']}"
            )
        
        # Step 4: Final decision
        reasoning.append(f"Classified as {predicted_class} with confidence {explanation['confidence']:.2f}")
        
        return reasoning
    
    def _get_node_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node."""
        if self.graph_data is None:
            return []
        
        edge_index = self.graph_data.edge_index
        neighbors = set()
        
        # Find outgoing edges
        outgoing = edge_index[1][edge_index[0] == node_id].tolist()
        neighbors.update(outgoing)
        
        # Find incoming edges
        incoming = edge_index[0][edge_index[1] == node_id].tolist()
        neighbors.update(incoming)
        
        return list(neighbors)
    
    def create_explanation_report(
        self,
        explanations: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive explanation report.
        
        Args:
            explanations: List of explanation dictionaries
            output_path: Optional file path to save report
            
        Returns:
            HTML report string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Explanation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .explanation { border: 1px solid #ddd; margin: 10px 0; padding: 15px; 
                             border-radius: 5px; background: #f9f9f9; }
                .confidence { color: #2ecc71; font-weight: bold; }
                .evidence { background: #ecf0f1; padding: 10px; margin: 10px 0; 
                           border-radius: 3px; }
                .feature-importance { margin: 10px 0; }
                .feature-bar { height: 20px; background: #3498db; margin: 2px 0; }
                .reasoning-path { background: #fff3cd; padding: 10px; border-left: 4px solid #f39c12; }
            </style>
        </head>
        <body>
            <h1>GNN Decision Explanations</h1>
        """
        
        for i, explanation in enumerate(explanations, 1):
            html += f"""
            <div class="explanation">
                <h2>Explanation {i}</h2>
                
                <p><strong>Paper ID:</strong> {explanation.get('paper_id', 'N/A')}</p>
                <p><strong>Predicted Class:</strong> {explanation.get('predicted_class', 'N/A')}</p>
                <p><strong>Confidence:</strong> 
                    <span class="confidence">{explanation.get('confidence', 0):.2f}</span>
                </p>
                
                <h3>Feature Importance</h3>
                <div class="feature-importance">
            """
            
            # Add feature importance
            feature_importance = explanation.get('feature_importance', {})
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                percentage = importance * 100
                html += f"""
                <div>
                    <strong>{feature}:</strong> {percentage:.1f}%
                    <div class="feature-bar" style="width: {percentage}%"></div>
                </div>
                """
            
            html += "</div>"
            
            # Add neighborhood influence
            neighborhood = explanation.get('neighborhood_influence', {})
            if neighborhood:
                html += f"""
                <h3>Neighborhood Influence</h3>
                <div class="evidence">
                    <p><strong>Neighbors:</strong> {', '.join(map(str, neighborhood.get('neighbors', [])))}</p>
                    <p><strong>Dominant Neighbor Class:</strong> {neighborhood.get('dominant_neighbor_class', 'N/A')}</p>
                </div>
                """
            
            # Add reasoning path
            reasoning_path = explanation.get('reasoning_path', [])
            if reasoning_path:
                html += """
                <h3>Reasoning Path</h3>
                <div class="reasoning-path">
                    <ol>
                """
                
                for step in reasoning_path:
                    html += f"<li>{step}</li>"
                
                html += """
                    </ol>
                </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"Explanation report saved to {output_path}")
        
        return html


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Explainer Test")
    print("=" * 80)
    
    # Mock GNN model for testing
    class MockGNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_weights = None
        
        def forward(self, x, edge_index):
            # Mock attention weights
            self.attention_weights = torch.randn(x.size(0), x.size(0))
            return x
    
    # Mock graph data
    class MockGraphData:
        def __init__(self):
            self.num_nodes = 10
            self.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    
    # Test attention visualization
    print("\n1. Testing Attention Visualization...")
    model = MockGNNModel()
    visualizer = GNNAttentionVisualizer(model)
    
    x = torch.randn(10, 128)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    attention_data = visualizer.extract_attention_weights(x, edge_index)
    print(f"  Attention layers: {len(attention_data['layer_attention'])}")
    print(f"  Node importance: {len(attention_data['node_importance'])}")
    print(f"  Attention paths: {len(attention_data['attention_paths'])}")
    
    # Test decision explanation
    print("\n2. Testing Decision Explanation...")
    explainer = GNNDecisionExplainer(model, MockGraphData())
    
    user_profile = {
        'user_id': 'test_user',
        'read_papers': [1, 2, 3],
        'interests': ['machine learning']
    }
    
    explanation = explainer.explain_recommendation(
        user_id='test_user',
        paper_id=5,
        recommendation_score=0.85,
        user_profile=user_profile
    )
    
    print(f"  Primary explanation: {explanation['primary_explanation']}")
    print(f"  Confidence: {explanation['confidence']:.2f}")
    print(f"  Evidence types: {[e['type'] for e in explanation['supporting_evidence']]}")
    
    # Test node classification explanation
    print("\n3. Testing Node Classification Explanation...")
    class_probs = torch.tensor([0.1, 0.7, 0.15, 0.05])
    node_explanation = explainer.explain_node_classification(
        node_id=5,
        predicted_class=1,
        class_probabilities=class_probs,
        true_class=2
    )
    
    print(f"  Predicted class: {node_explanation['predicted_class']}")
    print(f"  Confidence: {node_explanation['confidence']:.2f}")
    print(f"  Top features: {list(node_explanation['feature_importance'].keys())[:3]}")
    
    print("\n✓ GNN Explainer test complete")