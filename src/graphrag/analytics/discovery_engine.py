#!/usr/bin/env python3
"""
Serendipitous Discovery Engine - Enable tangential research discovery using GNN embeddings.

Helps researchers discover unexpected connections and explore research beyond
their immediate field using graph structure and learned embeddings.
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class SerendipitousDiscoveryEngine:
    """
    Enable tangential research discovery using GNN embeddings.
    
    Uses graph structure and GNN embeddings to help researchers discover
    unexpected but relevant papers outside their immediate field.
    
    Example:
        >>> engine = SerendipitousDiscoveryEngine(graph_manager, gnn_manager)
        >>> similar = engine.find_similar_by_embedding("paper123", n=5)
        >>> for paper, score in similar[:3]:
        ...     print(f"{paper}: {score:.3f}")
    """
    
    def __init__(self, graph_manager, gnn_manager):
        """
        Initialize discovery engine.
        
        Args:
            graph_manager: GraphManager for graph operations
            gnn_manager: GNNManager for embeddings and predictions
        """
        self.graph_manager = graph_manager
        self.gnn_manager = gnn_manager
    
    def find_similar_by_embedding(
        self, 
        paper_id: str, 
        n: int = 10, 
        method: str = "cosine"
    ) -> List[Dict]:
        """
        Find similar papers using GNN-learned embeddings (not just citations).
        
        Uses structural similarity learned by GNN rather than just citation overlap.
        
        Args:
            paper_id: Source paper ID
            n: Number of similar papers to return
            method: 'cosine|euclidean|dot_product'
            
        Returns:
            [
                {
                    'paper_id': str,
                    'similarity_score': float,
                    'similarity_explanation': str,
                    'shared_concepts': [concepts],
                    'connection_type': 'structural|semantic|hybrid'
                },
                ...
            ]
        """
        results = []
        
        try:
            if not self.gnn_manager or not hasattr(self.gnn_manager, 'embedder'):
                logger.warning("No GNN embedder available")
                return results
            
            embedder = self.gnn_manager.embedder
            source_emb = embedder.get_embedding(paper_id)
            
            if source_emb is None:
                logger.warning(f"No embedding for {paper_id}")
                return results
            
            # Find similar by embedding
            all_embeddings = embedder.embeddings
            similarities = []
            
            for other_id, other_emb in all_embeddings.items():
                if other_id == paper_id:
                    continue
                
                if method == "cosine":
                    sim = np.dot(source_emb, other_emb) / (
                        np.linalg.norm(source_emb) * np.linalg.norm(other_emb) + 1e-8
                    )
                elif method == "euclidean":
                    sim = 1.0 / (1.0 + np.linalg.norm(source_emb - other_emb))
                else:  # dot_product
                    sim = np.dot(source_emb, other_emb)
                
                similarities.append((other_id, sim))
            
            # Sort and format
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for other_id, score in similarities[:n]:
                results.append({
                    'paper_id': other_id,
                    'similarity_score': float(score),
                    'similarity_explanation': f"Structurally similar in graph (method: {method})",
                    'shared_concepts': [],
                    'connection_type': 'structural'
                })
        
        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
        
        return results
    
    def discover_cross_disciplinary_connections(
        self, 
        paper_id: str, 
        n: int = 5
    ) -> List[Dict]:
        """
        Find papers in different fields that might be relevant.
        
        Identifies papers that are similar in the embedding space but belong
        to different research areas, enabling cross-pollination of ideas.
        
        Args:
            paper_id: Source paper ID
            n: Number of cross-disciplinary papers to find
            
        Returns:
            [
                {
                    'paper_id': str,
                    'field': str,
                    'connection_strength': float,
                    'bridging_concepts': [concepts],
                    'novelty_score': float,  # How unexpected
                    'explanation': str
                },
                ...
            ]
        """
        results = []
        
        try:
            # Get similar papers
            similar_papers = self.find_similar_by_embedding(paper_id, n=50)
            
            # Get source paper's topics
            source_topics = set()
            if self.gnn_manager:
                try:
                    topic_predictions = self.gnn_manager.predict_paper_topics(paper_id, top_k=3)
                    source_topics = set(t[0] for t in topic_predictions)
                except:
                    pass
            
            # Find papers with different topics but high similarity
            for similar in similar_papers:
                other_id = similar['paper_id']
                
                # Get topics for this paper
                other_topics = set()
                if self.gnn_manager:
                    try:
                        topic_predictions = self.gnn_manager.predict_paper_topics(other_id, top_k=3)
                        other_topics = set(t[0] for t in topic_predictions)
                    except:
                        pass
                
                # Calculate topic overlap
                overlap = len(source_topics & other_topics)
                total = len(source_topics | other_topics)
                
                # Prefer papers with low overlap but high similarity (novelty)
                if total > 0:
                    topic_diversity = 1.0 - (overlap / total)
                    
                    if topic_diversity > 0.5:  # At least 50% different
                        novelty_score = similar['similarity_score'] * topic_diversity
                        
                        results.append({
                            'paper_id': other_id,
                            'field': list(other_topics - source_topics)[0] if other_topics - source_topics else 'Unknown',
                            'connection_strength': similar['similarity_score'],
                            'bridging_concepts': list(source_topics & other_topics),
                            'novelty_score': novelty_score,
                            'explanation': f"Similar structure but explores {topic_diversity:.0%} different topics"
                        })
            
            # Sort by novelty score
            results.sort(key=lambda x: x['novelty_score'], reverse=True)
        
        except Exception as e:
            logger.error(f"Error finding cross-disciplinary connections: {e}")
        
        return results[:n]
    
    def explore_tangential_research(
        self, 
        starting_paper: str, 
        exploration_depth: int = 2,
        surprise_weight: float = 0.5
    ) -> Dict:
        """
        Random walk with restart on graph for serendipitous discovery.
        
        Performs a guided random walk that balances relevance with exploration,
        discovering papers along interesting but unexpected paths.
        
        Args:
            starting_paper: Paper to start exploration from
            exploration_depth: How far to explore (number of steps)
            surprise_weight: Balance between relevance and surprise (0-1)
            
        Returns:
            {
                'discovered_papers': [
                    {
                        'paper_id': str,
                        'discovery_path': [node_ids],
                        'surprise_score': float,
                        'relevance_score': float,
                        'why_interesting': str
                    },
                    ...
                ],
                'exploration_map': str  # HTML visualization
            }
        """
        result = {
            'discovered_papers': [],
            'exploration_map': ''
        }
        
        try:
            discoveries = []
            visited = set([starting_paper])
            current = starting_paper
            path = [starting_paper]
            
            for step in range(exploration_depth * 10):  # Allow multiple explorations
                # Get neighbors
                try:
                    neighbors = self.graph_manager.query_neighbors(current, max_depth=1)
                except:
                    break
                
                if not neighbors:
                    break
                
                # Choose next node balancing exploration and exploitation
                candidates = []
                for neighbor in neighbors:
                    neighbor_id = neighbor['name']
                    
                    if neighbor_id in visited:
                        continue
                    
                    # Calculate relevance (embedding similarity to start)
                    relevance = self._calculate_relevance(starting_paper, neighbor_id)
                    
                    # Calculate surprise (distance from expected path)
                    surprise = 1.0 / (neighbor['distance'] + 1)
                    
                    # Combined score
                    score = (1 - surprise_weight) * relevance + surprise_weight * surprise
                    candidates.append((neighbor_id, score, relevance, surprise))
                
                if not candidates:
                    break
                
                # Select next node (probabilistic based on scores)
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Sometimes pick a less obvious choice for serendipity
                if random.random() < 0.3 and len(candidates) > 1:
                    next_node, score, relevance, surprise = random.choice(candidates[1:min(5, len(candidates))])
                else:
                    next_node, score, relevance, surprise = candidates[0]
                
                # Record discovery
                visited.add(next_node)
                path.append(next_node)
                
                # If we've reached exploration depth, record this as a discovery
                if len(path) >= exploration_depth:
                    discoveries.append({
                        'paper_id': next_node,
                        'discovery_path': path.copy(),
                        'surprise_score': surprise,
                        'relevance_score': relevance,
                        'why_interesting': self._generate_interest_explanation(
                            starting_paper, next_node, path, surprise, relevance
                        )
                    })
                    
                    # Start new path
                    path = [starting_paper]
                    current = starting_paper
                else:
                    current = next_node
            
            result['discovered_papers'] = discoveries[:10]  # Top 10 discoveries
            result['exploration_map'] = self._create_exploration_viz(starting_paper, discoveries)
        
        except Exception as e:
            logger.error(f"Error in tangential exploration: {e}")
        
        return result
    
    def _calculate_relevance(self, source_id: str, target_id: str) -> float:
        """Calculate relevance between two papers."""
        if not self.gnn_manager or not hasattr(self.gnn_manager, 'embedder'):
            return 0.5
        
        source_emb = self.gnn_manager.embedder.get_embedding(source_id)
        target_emb = self.gnn_manager.embedder.get_embedding(target_id)
        
        if source_emb is None or target_emb is None:
            return 0.5
        
        sim = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
        )
        
        return float(sim)
    
    def _generate_interest_explanation(
        self, 
        start: str, 
        end: str, 
        path: List[str], 
        surprise: float, 
        relevance: float
    ) -> str:
        """Generate explanation for why discovery is interesting."""
        if surprise > 0.7:
            return f"Unexpected connection {len(path)} steps away, but {relevance:.0%} relevant"
        elif relevance > 0.7:
            return f"Highly relevant ({relevance:.0%}) via {len(path)}-step path"
        else:
            return f"Balanced discovery: {relevance:.0%} relevant, {surprise:.0%} novel"
    
    def _create_exploration_viz(self, start: str, discoveries: List[Dict]) -> str:
        """Create visualization of exploration paths."""
        html = f"""
        <html>
        <head>
            <style>
                .exploration {{ padding: 20px; }}
                .path {{ margin: 10px 0; padding: 10px; background: #e3f2fd; }}
                .node {{ display: inline-block; padding: 5px 10px; margin: 2px; 
                        background: #2196F3; color: white; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="exploration">
                <h2>Exploration from: {start}</h2>
        """
        
        for i, disc in enumerate(discoveries, 1):
            html += f"""
                <div class="path">
                    <strong>Discovery {i}</strong> (Surprise: {disc['surprise_score']:.2%}, 
                    Relevance: {disc['relevance_score']:.2%})<br>
                    <div class="path-viz">
            """
            
            for node in disc['discovery_path']:
                html += f'<span class="node">{node[:10]}</span> → '
            
            html += f"""
                    </div>
                    <em>{disc['why_interesting']}</em>
                </div>
            """
        
        html += "</div></body></html>"
        return html


if __name__ == "__main__":
    print("Serendipitous Discovery Engine module loaded successfully")
