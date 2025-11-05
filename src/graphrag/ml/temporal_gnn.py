#!/usr/bin/env python3
"""
Temporal GNN - Graph Neural Networks for research evolution analysis
Models how research networks evolve over time and predicts future trends.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class TemporalGraphConvolution(nn.Module):
    """
    Temporal Graph Convolution layer.
    
    Combines spatial graph convolution with temporal dynamics
    to model evolving research networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_timesteps: int,
        dropout: float = 0.1
    ):
        """
        Initialize Temporal Graph Convolution.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_timesteps: Number of time steps to model
            dropout: Dropout probability
        """
        super(TemporalGraphConvolution, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Spatial convolution
        try:
            from torch_geometric.nn import GCNConv
            self.spatial_conv = GCNConv(input_dim, hidden_dim)
        except ImportError:
            # Fallback to linear layer
            self.spatial_conv = nn.Linear(input_dim, hidden_dim)
        
        # Temporal modeling
        self.temporal_rnn = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index_sequence: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through temporal graph convolution.
        
        Args:
            x_sequence: List of node features for each timestep
            edge_index_sequence: List of edge indices for each timestep
            
        Returns:
            Temporal node embeddings
        """
        # Spatial convolution for each timestep
        spatial_embeddings = []
        
        for t in range(min(len(x_sequence), self.num_timesteps)):
            if hasattr(self.spatial_conv, '__class__') and 'GCN' in str(self.spatial_conv.__class__):
                # PyG GCN
                spatial_emb = self.spatial_conv(x_sequence[t], edge_index_sequence[t])
            else:
                # Linear fallback
                spatial_emb = self.spatial_conv(x_sequence[t])
            
            spatial_emb = F.relu(spatial_emb)
            spatial_emb = self.dropout(spatial_emb)
            spatial_embeddings.append(spatial_emb)
        
        # Stack temporal sequence
        temporal_tensor = torch.stack(spatial_embeddings, dim=1)  # [num_nodes, timesteps, hidden_dim]
        
        # Temporal RNN processing
        rnn_output, _ = self.temporal_rnn(temporal_tensor)
        
        # Temporal attention
        attn_output, _ = self.temporal_attention(
            rnn_output, rnn_output, rnn_output
        )
        
        # Output projection
        output = self.output_proj(attn_output)
        
        return output


class ResearchEvolutionPredictor(nn.Module):
    """
    Predicts research evolution and future trends.
    
    Uses temporal GNN to model how research topics,
    citations, and collaborations evolve over time.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_timesteps: int = 5,
        prediction_horizon: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Research Evolution Predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_timesteps: Number of historical timesteps
            prediction_horizon: Number of future timesteps to predict
            dropout: Dropout probability
        """
        super(ResearchEvolutionPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.prediction_horizon = prediction_horizon
        
        # Temporal GNN encoder
        self.temporal_encoder = TemporalGraphConvolution(
            input_dim, hidden_dim, num_timesteps, dropout
        )
        
        # Evolution prediction heads
        self.citation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_horizon)
        )
        
        self.topic_evolution_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),  # 10 topic categories
            nn.Softmax(dim=1)
        )
        
        self.collaboration_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Pairwise features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Trend classifier
        self.trend_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # Growing, stable, declining
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index_sequence: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for evolution prediction.
        
        Args:
            x_sequence: Historical node features
            edge_index_sequence: Historical edge indices
            
        Returns:
            Dictionary with predictions
        """
        # Encode temporal evolution
        temporal_embeddings = self.temporal_encoder(
            x_sequence, edge_index_sequence
        )
        
        # Get final timestep embeddings for prediction
        final_embeddings = temporal_embeddings[:, -1, :]  # [num_nodes, hidden_dim]
        
        # Predict citation evolution
        citation_predictions = self.citation_predictor(final_embeddings)
        
        # Predict topic evolution
        topic_predictions = self.topic_evolution_predictor(final_embeddings)
        
        # Classify trends
        trend_predictions = self.trend_classifier(final_embeddings)
        
        # Predict future collaborations
        collaboration_predictions = self._predict_collaborations(
            final_embeddings, edge_index_sequence[-1]
        )
        
        return {
            'temporal_embeddings': temporal_embeddings,
            'citation_predictions': citation_predictions,
            'topic_predictions': topic_predictions,
            'trend_predictions': trend_predictions,
            'collaboration_predictions': collaboration_predictions
        }
    
    def _predict_collaborations(
        self,
        embeddings: torch.Tensor,
        current_edges: torch.Tensor
    ) -> torch.Tensor:
        """Predict future collaboration edges."""
        # Create all possible pairs (sample for efficiency)
        num_nodes = embeddings.size(0)
        max_pairs = min(1000, num_nodes * (num_nodes - 1) // 2)
        
        # Sample node pairs
        pairs = []
        for _ in range(max_pairs):
            i, j = np.random.choice(num_nodes, 2, replace=False)
            pairs.append([i, j])
        
        pairs = torch.tensor(pairs, dtype=torch.long)
        
        # Compute pairwise features
        node_i_emb = embeddings[pairs[:, 0]]
        node_j_emb = embeddings[pairs[:, 1]]
        pairwise_features = torch.cat([node_i_emb, node_j_emb], dim=1)
        
        # Predict collaboration probability
        collaboration_probs = self.collaboration_predictor(pairwise_features)
        
        return collaboration_probs.squeeze()


class TemporalResearchAnalyzer:
    """
    Analyzes research evolution using temporal GNN.
    
    Provides insights into how research topics evolve,
    citation patterns change, and collaborations form over time.
    """
    
    def __init__(self, gnn_core):
        """
        Initialize Temporal Research Analyzer.
        
        Args:
            gnn_core: GNN core system for graph processing
        """
        self.gnn_core = gnn_core
        self.evolution_predictor = None
        self.temporal_data = {}
        self.evolution_cache = {}
        
    def initialize_temporal_models(self):
        """Initialize temporal GNN models."""
        if self.gnn_core.current_graph_data is not None:
            input_dim = self.gnn_core.current_graph_data.x.shape[1]
            
            self.evolution_predictor = ResearchEvolutionPredictor(
                input_dim=input_dim,
                hidden_dim=256,
                num_timesteps=5,
                prediction_horizon=2
            )
            
            logger.info("Temporal GNN models initialized")
    
    def analyze_research_evolution(
        self,
        topic: str,
        years: List[int],
        prediction_years: List[int] = None
    ) -> Dict:
        """
        Analyze research evolution for a specific topic.
        
        Args:
            topic: Research topic to analyze
            years: Historical years to analyze
            prediction_years: Future years to predict
            
        Returns:
            Evolution analysis with predictions
        """
        if not self.evolution_predictor:
            self.initialize_temporal_models()
        
        # Build temporal graphs
        temporal_graphs = self._build_temporal_graphs(topic, years)
        
        if not temporal_graphs:
            return {'error': f'No data found for topic: {topic}'}
        
        # Extract temporal features
        x_sequence, edge_index_sequence = self._extract_temporal_features(
            temporal_graphs
        )
        
        # Run evolution prediction
        with torch.no_grad():
            predictions = self.evolution_predictor(x_sequence, edge_index_sequence)
        
        # Analyze results
        evolution_analysis = self._analyze_evolution_results(
            predictions, topic, years, prediction_years
        )
        
        return evolution_analysis
    
    def _build_temporal_graphs(self, topic: str, years: List[int]) -> List[Dict]:
        """Build temporal graphs for specific topic and years."""
        temporal_graphs = []
        
        for year in years:
            # Get papers for this year and topic
            year_graph = self._extract_topic_year_graph(topic, year)
            
            if year_graph:
                temporal_graphs.append(year_graph)
        
        return temporal_graphs
    
    def _extract_topic_year_graph(self, topic: str, year: int) -> Optional[Dict]:
        """Extract graph for specific topic and year."""
        # Simplified graph extraction
        # In practice, query database for papers from year with topic
        
        # Mock graph data
        num_papers = np.random.randint(10, 50)
        
        graph_data = {
            'year': year,
            'topic': topic,
            'num_papers': num_papers,
            'papers': [f'paper_{year}_{i}' for i in range(num_papers)],
            'citations': self._generate_mock_citations(num_papers),
            'features': torch.randn(num_papers, 384)
        }
        
        return graph_data
    
    def _generate_mock_citations(self, num_papers: int) -> torch.Tensor:
        """Generate mock citation edges."""
        # Create random citation network
        num_edges = min(num_papers * 2, 100)
        edges = []
        
        for _ in range(num_edges):
            source = np.random.randint(0, num_papers)
            target = np.random.randint(0, num_papers)
            if source != target:
                edges.append([source, target])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.zeros((2, 1), dtype=torch.long)
    
    def _extract_temporal_features(
        self,
        temporal_graphs: List[Dict]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract features from temporal graphs."""
        x_sequence = []
        edge_index_sequence = []
        
        for graph_data in temporal_graphs:
            x_sequence.append(graph_data['features'])
            edge_index_sequence.append(graph_data['citations'])
        
        return x_sequence, edge_index_sequence
    
    def _analyze_evolution_results(
        self,
        predictions: Dict[str, torch.Tensor],
        topic: str,
        years: List[int],
        prediction_years: List[int]
    ) -> Dict:
        """Analyze evolution prediction results."""
        analysis = {
            'topic': topic,
            'historical_years': years,
            'prediction_years': prediction_years or [],
            'citation_evolution': {},
            'topic_evolution': {},
            'trend_analysis': {},
            'collaboration_predictions': {},
            'insights': []
        }
        
        # Analyze citation evolution
        citation_preds = predictions['citation_predictions']
        if citation_preds is not None:
            analysis['citation_evolution'] = {
                'predicted_growth': torch.mean(citation_preds).item(),
                'growth_variance': torch.var(citation_preds).item(),
                'top_growing_papers': self._get_top_growing_papers(citation_preds)
            }
        
        # Analyze topic evolution
        topic_preds = predictions['topic_predictions']
        if topic_preds is not None:
            analysis['topic_evolution'] = {
                'dominant_topics': self._get_dominant_topics(topic_preds),
                'topic_diversity': self._compute_topic_diversity(topic_preds),
                'topic_shift': self._compute_topic_shift(topic_preds)
            }
        
        # Analyze trends
        trend_preds = predictions['trend_predictions']
        if trend_preds is not None:
            analysis['trend_analysis'] = {
                'trend_distribution': self._get_trend_distribution(trend_preds),
                'growing_papers': self._get_papers_by_trend(trend_preds, 'growing'),
                'declining_papers': self._get_papers_by_trend(trend_preds, 'declining')
            }
        
        # Analyze collaboration predictions
        collab_preds = predictions['collaboration_predictions']
        if collab_preds is not None:
            analysis['collaboration_predictions'] = {
                'new_collaborations': self._get_top_collaborations(collab_preds),
                'collaboration_density': torch.mean(collab_preds).item()
            }
        
        # Generate insights
        analysis['insights'] = self._generate_evolution_insights(analysis)
        
        return analysis
    
    def _get_top_growing_papers(self, citation_preds: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Get papers with highest predicted citation growth."""
        growth_scores = torch.mean(citation_preds, dim=1)
        top_indices = torch.topk(growth_scores, k=min(top_k, len(growth_scores))).indices
        
        return [
            {'paper_id': idx.item(), 'growth_score': growth_scores[idx].item()}
            for idx in top_indices
        ]
    
    def _get_dominant_topics(self, topic_preds: torch.Tensor) -> List[Dict]:
        """Get dominant topics from predictions."""
        topic_probs = torch.mean(topic_preds, dim=0)
        top_topics = torch.topk(topic_probs, k=3)
        
        return [
            {'topic_id': idx.item(), 'probability': prob.item()}
            for idx, prob in zip(top_topics.indices, top_topics.values)
        ]
    
    def _compute_topic_diversity(self, topic_preds: torch.Tensor) -> float:
        """Compute topic diversity (entropy)."""
        topic_probs = torch.mean(topic_preds, dim=0)
        entropy = -torch.sum(topic_probs * torch.log(topic_probs + 1e-8))
        return entropy.item()
    
    def _compute_topic_shift(self, topic_preds: torch.Tensor) -> float:
        """Compute how much topics are shifting."""
        # Simplified topic shift computation
        topic_variance = torch.var(topic_preds, dim=0)
        return torch.mean(topic_variance).item()
    
    def _get_trend_distribution(self, trend_preds: torch.Tensor) -> Dict[str, float]:
        """Get distribution of trend predictions."""
        trend_probs = torch.mean(trend_preds, dim=0)
        
        return {
            'growing': trend_probs[0].item(),
            'stable': trend_probs[1].item(),
            'declining': trend_probs[2].item()
        }
    
    def _get_papers_by_trend(
        self,
        trend_preds: torch.Tensor,
        trend_type: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Get papers by trend type."""
        trend_map = {'growing': 0, 'stable': 1, 'declining': 2}
        trend_idx = trend_map.get(trend_type, 1)
        
        # Get papers with highest probability for this trend
        trend_probs = trend_preds[:, trend_idx]
        top_indices = torch.topk(trend_probs, k=min(top_k, len(trend_probs))).indices
        
        return [
            {'paper_id': idx.item(), 'confidence': trend_probs[idx].item()}
            for idx in top_indices
        ]
    
    def _get_top_collaborations(
        self,
        collab_preds: torch.Tensor,
        top_k: int = 10
    ) -> List[Dict]:
        """Get top predicted collaborations."""
        top_probs, top_indices = torch.topk(collab_preds, k=min(top_k, len(collab_preds)))
        
        return [
            {'collaboration_id': idx.item(), 'probability': prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ]
    
    def _generate_evolution_insights(self, analysis: Dict) -> List[str]:
        """Generate insights from evolution analysis."""
        insights = []
        
        # Citation growth insights
        citation_growth = analysis['citation_evolution'].get('predicted_growth', 0)
        if citation_growth > 0.5:
            insights.append(f"Research in {analysis['topic']} shows strong citation growth potential")
        elif citation_growth < 0.2:
            insights.append(f"Research in {analysis['topic']} may be reaching saturation")
        
        # Topic diversity insights
        topic_diversity = analysis['topic_evolution'].get('topic_diversity', 0)
        if topic_diversity > 2.0:
            insights.append(f"High topic diversity suggests interdisciplinary growth in {analysis['topic']}")
        
        # Trend insights
        trend_dist = analysis['trend_analysis'].get('trend_distribution', {})
        if trend_dist.get('growing', 0) > 0.5:
            insights.append(f"Most papers in {analysis['topic']} are on a growing trend")
        
        # Collaboration insights
        collab_density = analysis['collaboration_predictions'].get('collaboration_density', 0)
        if collab_density > 0.3:
            insights.append(f"High collaboration potential detected in {analysis['topic']}")
        
        return insights
    
    def predict_future_impact(
        self,
        paper_ids: List[int],
        prediction_horizon: int = 2
    ) -> Dict:
        """
        Predict future impact of specific papers.
        
        Args:
            paper_ids: Papers to analyze
            prediction_horizon: Years ahead to predict
            
        Returns:
            Impact predictions for each paper
        """
        if not self.evolution_predictor:
            self.initialize_temporal_models()
        
        # Get temporal data for these papers
        temporal_data = self._get_paper_temporal_data(paper_ids)
        
        if not temporal_data:
            return {'error': 'No temporal data available for papers'}
        
        # Predict future impact
        impact_predictions = {}
        
        for paper_id in paper_ids:
            paper_temporal = temporal_data.get(paper_id, {})
            
            if paper_temporal:
                # Simple impact prediction based on citation trends
                citation_trend = paper_temporal.get('citation_trend', [0, 0, 0, 0, 0])
                
                # Predict future citations
                future_citations = self._predict_future_citations(
                    citation_trend, prediction_horizon
                )
                
                impact_predictions[paper_id] = {
                    'future_citations': future_citations,
                    'impact_score': np.mean(future_citations),
                    'confidence': self._compute_prediction_confidence(citation_trend)
                }
        
        return impact_predictions
    
    def _get_paper_temporal_data(self, paper_ids: List[int]) -> Dict:
        """Get temporal data for specific papers."""
        # Simplified temporal data extraction
        temporal_data = {}
        
        for paper_id in paper_ids:
            # Mock temporal data
            temporal_data[paper_id] = {
                'citation_trend': [
                    np.random.randint(1, 10) for _ in range(5)
                ],
                'topic_evolution': np.random.randn(5, 10).tolist()
            }
        
        return temporal_data
    
    def _predict_future_citations(
        self,
        historical_citations: List[int],
        prediction_horizon: int
    ) -> List[int]:
        """Predict future citations based on historical trend."""
        if len(historical_citations) < 2:
            return [0] * prediction_horizon
        
        # Simple linear trend prediction
        recent_growth = historical_citations[-1] - historical_citations[-2]
        
        future_citations = []
        current_citations = historical_citations[-1]
        
        for _ in range(prediction_horizon):
            # Apply growth with some decay
            growth = recent_growth * 0.8  # Decay factor
            current_citations = max(0, current_citations + growth)
            future_citations.append(int(current_citations))
            recent_growth = growth * 0.8  # Further decay
        
        return future_citations
    
    def _compute_prediction_confidence(self, historical_data: List[float]) -> float:
        """Compute confidence in prediction based on data quality."""
        if len(historical_data) < 3:
            return 0.3  # Low confidence with little data
        
        # Compute variance as confidence indicator
        variance = np.var(historical_data)
        confidence = max(0.1, 1.0 - variance / max(historical_data))
        
        return confidence
    
    def analyze_topic_evolution_timeline(
        self,
        topic: str,
        start_year: int,
        end_year: int
    ) -> Dict:
        """
        Analyze topic evolution over a timeline.
        
        Args:
            topic: Research topic
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            Timeline analysis with key events and trends
        """
        years = list(range(start_year, end_year + 1))
        
        # Analyze evolution for each year
        evolution_analysis = self.analyze_research_evolution(topic, years)
        
        # Create timeline
        timeline = {
            'topic': topic,
            'start_year': start_year,
            'end_year': end_year,
            'yearly_data': {},
            'key_events': [],
            'trend_changes': []
        }
        
        # Process yearly data
        for year in years:
            year_analysis = self.analyze_research_evolution(topic, [year])
            timeline['yearly_data'][year] = year_analysis
        
        # Identify key events and trend changes
        timeline['key_events'] = self._identify_key_events(timeline['yearly_data'])
        timeline['trend_changes'] = self._identify_trend_changes(timeline['yearly_data'])
        
        return timeline
    
    def _identify_key_events(self, yearly_data: Dict) -> List[Dict]:
        """Identify key events in topic evolution."""
        key_events = []
        
        for year, data in yearly_data.items():
            # Look for significant citation growth
            citation_growth = data.get('citation_evolution', {}).get('predicted_growth', 0)
            if citation_growth > 0.8:
                key_events.append({
                    'year': year,
                    'type': 'citation_growth',
                    'description': f'Significant citation growth detected',
                    'magnitude': citation_growth
                })
            
            # Look for topic shifts
            topic_shift = data.get('topic_evolution', {}).get('topic_shift', 0)
            if topic_shift > 0.5:
                key_events.append({
                    'year': year,
                    'type': 'topic_shift',
                    'description': f'Major topic shift detected',
                    'magnitude': topic_shift
                })
        
        return key_events
    
    def _identify_trend_changes(self, yearly_data: Dict) -> List[Dict]:
        """Identify changes in research trends."""
        trend_changes = []
        
        years = sorted(yearly_data.keys())
        prev_trend = None
        
        for year in years:
            data = yearly_data[year]
            trend_dist = data.get('trend_analysis', {}).get('trend_distribution', {})
            
            # Get dominant trend
            dominant_trend = max(trend_dist.items(), key=lambda x: x[1])[0] if trend_dist else 'stable'
            
            if prev_trend and dominant_trend != prev_trend:
                trend_changes.append({
                    'year': year,
                    'from_trend': prev_trend,
                    'to_trend': dominant_trend,
                    'description': f'Trend changed from {prev_trend} to {dominant_trend}'
                })
            
            prev_trend = dominant_trend
        
        return trend_changes
    
    def get_temporal_analysis_stats(self) -> Dict:
        """Get statistics about temporal analysis."""
        return {
            'evolution_predictor_available': self.evolution_predictor is not None,
            'temporal_data_cached': len(self.temporal_data),
            'evolution_cache_size': len(self.evolution_cache)
        }


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("Temporal GNN Test")
    print("=" * 80)
    
    # Mock GNN core for testing
    class MockGNNCore:
        def __init__(self):
            self.current_graph_data = None
    
    # Initialize analyzer
    analyzer = TemporalResearchAnalyzer(MockGNNCore())
    
    # Test research evolution analysis
    print("\n1. Testing Research Evolution Analysis...")
    evolution = analyzer.analyze_research_evolution(
        topic="machine learning",
        years=[2018, 2019, 2020, 2021, 2022],
        prediction_years=[2023, 2024]
    )
    
    print(f"  Topic: {evolution.get('topic')}")
    print(f"  Citation growth: {evolution.get('citation_evolution', {}).get('predicted_growth', 0):.3f}")
    print(f"  Topic diversity: {evolution.get('topic_evolution', {}).get('topic_diversity', 0):.3f}")
    print(f"  Insights: {len(evolution.get('insights', []))}")
    
    # Test future impact prediction
    print("\n2. Testing Future Impact Prediction...")
    impact = analyzer.predict_future_impact(
        paper_ids=[1, 2, 3],
        prediction_horizon=3
    )
    
    for paper_id, prediction in impact.items():
        if 'error' not in prediction:
            print(f"  Paper {paper_id}: Impact score {prediction['impact_score']:.2f}")
    
    # Test timeline analysis
    print("\n3. Testing Timeline Analysis...")
    timeline = analyzer.analyze_topic_evolution_timeline(
        topic="deep learning",
        start_year=2018,
        end_year=2022
    )
    
    print(f"  Timeline: {timeline['start_year']}-{timeline['end_year']}")
    print(f"  Key events: {len(timeline['key_events'])}")
    print(f"  Trend changes: {len(timeline['trend_changes'])}")
    
    # Get stats
    stats = analyzer.get_temporal_analysis_stats()
    print(f"\nTemporal Analysis Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Temporal GNN test complete")