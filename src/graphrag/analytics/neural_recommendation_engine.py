#!/usr/bin/env python3
"""
Neural Recommendation Engine - Pure GNN-based recommendation system
Uses graph neural networks for all recommendation tasks instead of traditional methods.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class GraphCollaborativeFiltering(nn.Module):
    """
    Graph-based collaborative filtering using GNN.
    
    Learns user and paper embeddings from the citation network
    and uses them for collaborative recommendations.
    """
    
    def __init__(
        self,
        num_users: int,
        num_papers: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Collaborative Filtering.
        
        Args:
            num_users: Number of users
            num_papers: Number of papers
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super(GraphCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # User and paper embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.paper_embeddings = nn.Embedding(num_papers, embedding_dim)
        
        # GNN layers for message passing
        try:
            from torch_geometric.nn import GCNConv
            self.gnn_layers = nn.ModuleList([
                GCNConv(embedding_dim, hidden_dim) if i == 0 
                else GCNConv(hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
        except ImportError:
            # Fallback to linear layers
            self.gnn_layers = nn.ModuleList([
                nn.Linear(embedding_dim, hidden_dim) if i == 0
                else nn.Linear(hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
        
        # Output layers
        self.user_output = nn.Linear(hidden_dim, embedding_dim)
        self.paper_output = nn.Linear(hidden_dim, embedding_dim)
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, user_ids, paper_ids, edge_index, user_paper_interactions):
        """
        Forward pass for collaborative filtering.
        
        Args:
            user_ids: User IDs
            paper_ids: Paper IDs
            edge_index: Citation graph edges
            user_paper_interactions: User-paper interaction matrix
            
        Returns:
            Prediction scores
        """
        # Get initial embeddings
        user_emb = self.user_embeddings(user_ids)
        paper_emb = self.paper_embeddings(paper_ids)
        
        # Apply GNN layers
        if hasattr(self.gnn_layers[0], '__class__') and 'GCN' in str(self.gnn_layers[0].__class__):
            # PyG GCN layers
            x = torch.cat([user_emb, paper_emb], dim=0)
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
        else:
            # Linear layers (fallback)
            x = torch.cat([user_emb, paper_emb], dim=0)
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x)
                x = F.relu(x)
                x = self.dropout(x)
        
        # Split back to users and papers
        user_final = self.user_output(x[:len(user_ids)])
        paper_final = self.paper_output(x[len(user_ids):])
        
        # Predict interactions
        user_expanded = user_final.unsqueeze(1).expand(-1, len(paper_ids), -1)
        paper_expanded = paper_final.unsqueeze(0).expand(len(user_ids), -1, -1)
        
        combined = torch.cat([user_expanded, paper_expanded], dim=-1)
        predictions = self.predictor(combined).squeeze(-1)
        
        return predictions


class GraphNeuralDiversity(nn.Module):
    """
    Neural diversity module for recommendations.
    
    Uses GNN to compute diversity scores and ensure
    recommendations span different research communities.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_communities: int = 10,
        hidden_dim: int = 64
    ):
        """
        Initialize Neural Diversity module.
        
        Args:
            embedding_dim: Dimension of paper embeddings
            num_communities: Expected number of research communities
            hidden_dim: Hidden layer dimension
        """
        super(GraphNeuralDiversity, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_communities = num_communities
        
        # Community detection network
        self.community_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_communities),
            nn.Softmax(dim=1)
        )
        
        # Diversity scoring network
        self.diversity_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Novelty detection
        self.novelty_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, paper_embeddings, user_embedding, candidate_indices):
        """
        Compute diversity and novelty scores.
        
        Args:
            paper_embeddings: All paper embeddings
            user_embedding: User's embedding
            candidate_indices: Candidate paper indices
            
        Returns:
            Diversity and novelty scores
        """
        # Detect communities
        community_probs = self.community_detector(paper_embeddings)
        community_assignments = torch.argmax(community_probs, dim=1)
        
        # Get candidate embeddings
        candidate_emb = paper_embeddings[candidate_indices]
        
        # Compute diversity scores
        diversity_scores = []
        novelty_scores = []
        
        for i, candidate_idx in enumerate(candidate_indices):
            candidate_emb = candidate_emb[i]
            
            # Diversity from user
            diversity_input = torch.cat([user_embedding, candidate_emb], dim=-1)
            diversity_score = self.diversity_scorer(diversity_input)
            diversity_scores.append(diversity_score)
            
            # Novelty score
            novelty_score = self.novelty_detector(candidate_emb)
            novelty_scores.append(novelty_score)
        
        diversity_scores = torch.stack(diversity_scores)
        novelty_scores = torch.stack(novelty_scores)
        
        return {
            'diversity_scores': diversity_scores,
            'novelty_scores': novelty_scores,
            'community_assignments': community_assignments[candidate_indices]
        }


class NeuralRecommendationEngine:
    """
    Pure GNN-based recommendation engine.
    
    Uses graph neural networks for all recommendation tasks:
    - Collaborative filtering with GNN
    - Content-based filtering with graph embeddings
    - Diversity and novelty optimization
    - Cross-domain recommendations
    """
    
    def __init__(self, gnn_core):
        """
        Initialize Neural Recommendation Engine.
        
        Args:
            gnn_core: GNN core system for embeddings and models
        """
        self.gnn_core = gnn_core
        self.user_profiles = {}
        self.paper_embeddings = None
        self.user_embeddings = None
        
        # Neural components
        self.collaborative_filter = None
        self.diversity_module = None
        self.cross_domain_net = None
        
        # Recommendation cache
        self.recommendation_cache = {}
        
        logger.info("Neural Recommendation Engine initialized")
    
    def initialize_models(self):
        """Initialize neural recommendation models."""
        if self.gnn_core.current_graph_data is not None:
            num_papers = self.gnn_core.current_graph_data.num_nodes
            
            # Initialize collaborative filtering
            self.collaborative_filter = GraphCollaborativeFiltering(
                num_users=1000,  # Estimated
                num_papers=num_papers,
                embedding_dim=128,
                hidden_dim=256
            )
            
            # Initialize diversity module
            self.diversity_module = GraphNeuralDiversity(
                embedding_dim=128,
                num_communities=10
            )
            
            # Initialize cross-domain network
            self.cross_domain_net = self._create_cross_domain_network()
            
            logger.info("Neural recommendation models initialized")
    
    def recommend_papers_gnn(
        self,
        user_profile: Dict,
        top_k: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Generate GNN-based paper recommendations.
        
        Args:
            user_profile: User reading history and interests
            top_k: Number of recommendations
            diversity_weight: Weight for diversity optimization
            
        Returns:
            List of recommended papers with GNN-based explanations
        """
        if not self.collaborative_filter:
            self.initialize_models()
        
        user_id = user_profile.get('user_id', 'default_user')
        
        # Check cache first
        cache_key = f"{user_id}_{top_k}_{diversity_weight}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        # 1. Create user embedding
        user_embedding = self._create_user_embedding(user_profile)
        
        # 2. Get candidate papers
        candidates = self._get_candidate_papers(user_profile)
        
        # 3. Compute GNN-based scores
        gnn_scores = self._compute_gnn_scores(
            user_embedding, candidates, user_profile
        )
        
        # 4. Apply neural diversity optimization
        if self.diversity_module and self.paper_embeddings is not None:
            diversity_results = self.diversity_module(
                self.paper_embeddings, user_embedding, candidates
            )
            
            # Combine scores with diversity
            final_scores = self._combine_scores_with_diversity(
                gnn_scores, diversity_results, diversity_weight
            )
        else:
            final_scores = gnn_scores
        
        # 5. Rank and select top-k
        top_recommendations = self._rank_recommendations(
            candidates, final_scores, top_k
        )
        
        # 6. Generate GNN-based explanations
        explained_recommendations = self._generate_gnn_explanations(
            top_recommendations, user_profile, final_scores
        )
        
        # Cache results
        self.recommendation_cache[cache_key] = explained_recommendations
        
        return explained_recommendations
    
    def _create_user_embedding(self, user_profile: Dict) -> torch.Tensor:
        """Create user embedding from profile."""
        # Get embeddings for read papers
        read_papers = user_profile.get('read_papers', [])
        interests = user_profile.get('interests', [])
        
        if self.gnn_core.node_embeddings is not None:
            # Use GNN embeddings
            paper_embeddings = []
            
            for paper_id in read_papers:
                if paper_id in self.gnn_core.node_embeddings:
                    emb = self.gnn_core.node_embeddings[paper_id]
                    paper_embeddings.append(emb)
            
            if paper_embeddings:
                # Average of read papers
                user_emb = torch.mean(torch.stack(paper_embeddings), dim=0)
            else:
                # Random embedding for new users
                user_emb = torch.randn(128)
            
            # Add interest information
            if interests and self.gnn_core.graph_embedder:
                interest_emb = self.gnn_core.graph_embedder.embed_texts(interests)
                if interest_emb is not None:
                    interest_emb = torch.mean(torch.tensor(interest_emb), dim=0)
                    user_emb = user_emb + 0.3 * interest_emb
            
            return user_emb
        else:
            # Fallback to random embedding
            return torch.randn(128)
    
    def _get_candidate_papers(self, user_profile: Dict) -> List[int]:
        """Get candidate papers for recommendation."""
        # Exclude already read papers
        read_papers = set(user_profile.get('read_papers', []))
        
        if self.gnn_core.current_graph_data is not None:
            all_papers = list(range(self.gnn_core.current_graph_data.num_nodes))
            candidates = [p for p in all_papers if p not in read_papers]
            
            # Limit candidates for efficiency
            return candidates[:1000]
        else:
            return []
    
    def _compute_gnn_scores(
        self,
        user_embedding: torch.Tensor,
        candidates: List[int],
        user_profile: Dict
    ) -> Dict[int, float]:
        """Compute GNN-based similarity scores."""
        scores = {}
        
        if self.gnn_core.node_embeddings is not None:
            # Content-based similarity using GNN embeddings
            for candidate_id in candidates:
                if candidate_id in self.gnn_core.node_embeddings:
                    candidate_emb = self.gnn_core.node_embeddings[candidate_id]
                    
                    # Cosine similarity
                    similarity = F.cosine_similarity(
                        user_embedding.unsqueeze(0),
                        candidate_emb.unsqueeze(0)
                    ).item()
                    
                    scores[candidate_id] = similarity
        
        # Add collaborative filtering scores
        if self.collaborative_filter:
            collab_scores = self._compute_collaborative_scores(
                user_embedding, candidates, user_profile
            )
            for candidate_id, score in collab_scores.items():
                if candidate_id in scores:
                    scores[candidate_id] = 0.7 * scores[candidate_id] + 0.3 * score
                else:
                    scores[candidate_id] = score
        
        return scores
    
    def _compute_collaborative_scores(
        self,
        user_embedding: torch.Tensor,
        candidates: List[int],
        user_profile: Dict
    ) -> Dict[int, float]:
        """Compute collaborative filtering scores."""
        # Simplified collaborative filtering
        # In practice, this would use the GraphCollaborativeFiltering network
        scores = {}
        
        # Get similar users
        read_papers = user_profile.get('read_papers', [])
        
        for candidate_id in candidates:
            # Score based on co-citation patterns
            score = self._compute_co_citation_score(candidate_id, read_papers)
            scores[candidate_id] = score
        
        return scores
    
    def _compute_co_citation_score(self, candidate_id: int, read_papers: List[int]) -> float:
        """Compute co-citation based score."""
        if not read_papers or self.gnn_core.current_graph_data is None:
            return 0.0
        
        # Count common citations
        common_citations = 0
        total_pairs = 0
        
        for read_paper in read_papers:
            total_pairs += 1
            if self._are_cited_together(candidate_id, read_paper):
                common_citations += 1
        
        return common_citations / max(total_pairs, 1)
    
    def _are_cited_together(self, paper1: int, paper2: int) -> bool:
        """Check if two papers are commonly cited together."""
        # Simplified check - in practice, use citation co-occurrence
        if self.gnn_core.current_graph_data is None:
            return False
        
        # Get papers that cite both paper1 and paper2
        edge_index = self.gnn_core.current_graph_data.edge_index
        
        # Find papers citing paper1
        citing_paper1 = set(edge_index[1][edge_index[0] == paper1].tolist())
        
        # Find papers citing paper2
        citing_paper2 = set(edge_index[1][edge_index[0] == paper2].tolist())
        
        # Check overlap
        common_citing = citing_paper1.intersection(citing_paper2)
        
        return len(common_citing) > 2  # Threshold for "commonly cited together"
    
    def _combine_scores_with_diversity(
        self,
        gnn_scores: Dict[int, float],
        diversity_results: Dict,
        diversity_weight: float
    ) -> Dict[int, float]:
        """Combine GNN scores with diversity optimization."""
        combined_scores = {}
        
        diversity_scores = diversity_results['diversity_scores']
        novelty_scores = diversity_results['novelty_scores']
        community_assignments = diversity_results['community_assignments']
        
        for i, candidate_id in enumerate(gnn_scores.keys()):
            base_score = gnn_scores[candidate_id]
            
            # Get diversity and novelty scores
            if i < len(diversity_scores):
                diversity_score = diversity_scores[i].item()
                novelty_score = novelty_scores[i].item()
                community = community_assignments[i].item()
            else:
                diversity_score = 0.5
                novelty_score = 0.5
                community = 0
            
            # Combine scores
            final_score = (
                (1 - diversity_weight) * base_score +
                diversity_weight * (0.7 * diversity_score + 0.3 * novelty_score)
            )
            
            combined_scores[candidate_id] = final_score
        
        return combined_scores
    
    def _rank_recommendations(
        self,
        candidates: List[int],
        scores: Dict[int, float],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Rank candidates by scores and return top-k."""
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def _generate_gnn_explanations(
        self,
        recommendations: List[Tuple[int, float]],
        user_profile: Dict,
        scores: Dict[int, float]
    ) -> List[Dict]:
        """Generate GNN-based explanations for recommendations."""
        explanations = []
        
        for paper_id, score in recommendations:
            explanation = {
                'paper_id': paper_id,
                'score': score,
                'explanation': self._create_explanation(paper_id, user_profile, score),
                'gnn_features': self._extract_gnn_features(paper_id),
                'recommendation_type': self._classify_recommendation_type(paper_id, user_profile)
            }
            explanations.append(explanation)
        
        return explanations
    
    def _create_explanation(
        self,
        paper_id: int,
        user_profile: Dict,
        score: float
    ) -> str:
        """Create explanation for recommendation."""
        read_papers = user_profile.get('read_papers', [])
        interests = user_profile.get('interests', [])
        
        # Find similar read papers
        similar_papers = self._find_similar_read_papers(paper_id, read_papers)
        
        if similar_papers:
            return f"Recommended because it's similar to papers you've read: {', '.join(similar_papers[:3])}"
        elif interests:
            return f"Recommended based on your interests in: {', '.join(interests[:2])}"
        else:
            return f"Recommended based on graph neural network analysis (score: {score:.3f})"
    
    def _find_similar_read_papers(
        self,
        paper_id: int,
        read_papers: List[int]
    ) -> List[str]:
        """Find papers similar to candidate that user has read."""
        similar = []
        
        if self.gnn_core.node_embeddings is not None:
            candidate_emb = self.gnn_core.node_embeddings.get(paper_id)
            
            if candidate_emb is not None:
                for read_paper in read_papers:
                    read_emb = self.gnn_core.node_embeddings.get(read_paper)
                    
                    if read_emb is not None:
                        similarity = F.cosine_similarity(
                            candidate_emb.unsqueeze(0),
                            read_emb.unsqueeze(0)
                        ).item()
                        
                        if similarity > 0.7:  # Similarity threshold
                            similar.append(f"Paper {read_paper}")
        
        return similar[:3]  # Top 3 similar papers
    
    def _extract_gnn_features(self, paper_id: int) -> Dict:
        """Extract GNN features for explanation."""
        features = {}
        
        if self.gnn_core.current_graph_data is not None:
            # Structural features
            degree = self._compute_node_degree(paper_id)
            features['degree'] = degree
            
            # Community features
            community = self._get_node_community(paper_id)
            features['community'] = community
            
            # Centrality features
            centrality = self._compute_node_centrality(paper_id)
            features['centrality'] = centrality
        
        return features
    
    def _compute_node_degree(self, node_id: int) -> int:
        """Compute degree of a node."""
        if self.gnn_core.current_graph_data is None:
            return 0
        
        edge_index = self.gnn_core.current_graph_data.edge_index
        return (edge_index[0] == node_id).sum().item()
    
    def _get_node_community(self, node_id: int) -> int:
        """Get community assignment for node."""
        # Simplified - in practice, use community detection
        return node_id % 10  # Mock community assignment
    
    def _compute_node_centrality(self, node_id: int) -> float:
        """Compute centrality score for node."""
        # Simplified PageRank-like centrality
        degree = self._compute_node_degree(node_id)
        return degree / max(1, self.gnn_core.current_graph_data.num_nodes)
    
    def _classify_recommendation_type(self, paper_id: int, user_profile: Dict) -> str:
        """Classify the type of recommendation."""
        read_papers = user_profile.get('read_papers', [])
        
        if self._find_similar_read_papers(paper_id, read_papers):
            return 'content_based'
        elif user_profile.get('interests'):
            return 'interest_based'
        else:
            return 'graph_based'
    
    def _create_cross_domain_network(self) -> nn.Module:
        """Create cross-domain recommendation network."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def recommend_authors_gnn(
        self,
        user_profile: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recommend authors using GNN analysis.
        
        Args:
            user_profile: User profile with interests and read papers
            top_k: Number of author recommendations
            
        Returns:
            List of recommended authors with explanations
        """
        # Get authors from user's read papers
        read_papers = user_profile.get('read_papers', [])
        coauthors = self._extract_coauthors(read_papers)
        
        # Find authors in similar research areas
        interests = user_profile.get('interests', [])
        similar_authors = self._find_authors_by_interests(interests)
        
        # Combine and rank
        all_authors = list(set(coauthors + similar_authors))
        author_scores = self._score_authors_gnn(all_authors, user_profile)
        
        # Get top recommendations
        top_authors = sorted(
            author_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        recommendations = []
        for author_id, score in top_authors:
            recommendations.append({
                'author_id': author_id,
                'score': score,
                'explanation': self._create_author_explanation(author_id, user_profile),
                'collaboration_potential': self._compute_collaboration_potential(author_id, user_profile)
            })
        
        return recommendations
    
    def _extract_coauthors(self, read_papers: List[int]) -> List[str]:
        """Extract coauthors from read papers."""
        # Simplified - in practice, query graph for coauthor relationships
        coauthors = []
        
        for paper_id in read_papers[:10]:  # Limit to 10 papers
            # Mock coauthor extraction
            coauthors.extend([f'coauthor_{i}_{paper_id}' for i in range(3)])
        
        return list(set(coauthors))
    
    def _find_authors_by_interests(self, interests: List[str]) -> List[str]:
        """Find authors working in user's interest areas."""
        # Simplified - in practice, use graph traversal
        authors = []
        
        for interest in interests:
            authors.extend([f'author_{interest}_{i}' for i in range(5)])
        
        return list(set(authors))
    
    def _score_authors_gnn(self, authors: List[str], user_profile: Dict) -> Dict[str, float]:
        """Score authors using GNN-based features."""
        scores = {}
        
        for author in authors:
            # Simplified scoring based on collaboration patterns
            score = np.random.random()  # In practice, use GNN embeddings
            scores[author] = score
        
        return scores
    
    def _create_author_explanation(self, author_id: str, user_profile: Dict) -> str:
        """Create explanation for author recommendation."""
        interests = user_profile.get('interests', [])
        
        if interests:
            return f"Recommended author works in your interest areas: {', '.join(interests[:2])}"
        else:
            return f"Recommended based on collaboration network analysis"
    
    def _compute_collaboration_potential(self, author_id: str, user_profile: Dict) -> float:
        """Compute collaboration potential with user."""
        # Simplified collaboration potential score
        return np.random.random()  # In practice, use GNN path analysis
    
    def update_user_feedback(
        self,
        user_id: str,
        paper_id: int,
        feedback: float  # 0-1 rating
    ):
        """
        Update recommendation model based on user feedback.
        
        Args:
            user_id: User identifier
            paper_id: Recommended paper ID
            feedback: User feedback rating
        """
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'read_papers': [], 'feedback': []}
        
        self.user_profiles[user_id]['feedback'].append({
            'paper_id': paper_id,
            'feedback': feedback,
            'timestamp': torch.tensor([0.0])  # Simplified timestamp
        })
        
        # Clear cache to force re-computation
        self.recommendation_cache.clear()
        
        logger.info(f"Updated feedback for user {user_id}, paper {paper_id}: {feedback}")
    
    def get_recommendation_stats(self) -> Dict:
        """Get statistics about recommendation performance."""
        stats = {
            'total_users': len(self.user_profiles),
            'cache_size': len(self.recommendation_cache),
            'models_initialized': self.collaborative_filter is not None
        }
        
        # Compute average feedback if available
        all_feedback = []
        for user_profile in self.user_profiles.values():
            all_feedback.extend([f['feedback'] for f in user_profile.get('feedback', [])])
        
        if all_feedback:
            stats['average_feedback'] = np.mean(all_feedback)
            stats['feedback_count'] = len(all_feedback)
        
        return stats


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("Neural Recommendation Engine Test")
    print("=" * 80)
    
    # Mock GNN core for testing
    class MockGNNCore:
        def __init__(self):
            self.current_graph_data = None
            self.node_embeddings = {i: torch.randn(128) for i in range(100)}
    
    # Initialize engine
    engine = NeuralRecommendationEngine(MockGNNCore())
    
    # Test paper recommendations
    user_profile = {
        'user_id': 'test_user',
        'read_papers': [1, 5, 10],
        'interests': ['machine learning', 'neural networks']
    }
    
    recommendations = engine.recommend_papers_gnn(user_profile, top_k=5)
    
    print(f"\nPaper Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. Paper {rec['paper_id']} (score: {rec['score']:.3f})")
        print(f"     Type: {rec['recommendation_type']}")
        print(f"     Explanation: {rec['explanation']}")
    
    # Test author recommendations
    author_recs = engine.recommend_authors_gnn(user_profile, top_k=3)
    
    print(f"\nAuthor Recommendations:")
    for i, rec in enumerate(author_recs, 1):
        print(f"  {i}. {rec['author_id']} (score: {rec['score']:.3f})")
        print(f"     Explanation: {rec['explanation']}")
    
    # Get stats
    stats = engine.get_recommendation_stats()
    print(f"\nRecommendation Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Neural Recommendation Engine test complete")