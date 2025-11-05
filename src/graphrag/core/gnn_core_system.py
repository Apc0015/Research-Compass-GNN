#!/usr/bin/env python3
"""
GNN Core System - Complete GNN-first architecture for Research Compass
Integrates all GNN components into a unified system optimized for graph neural networks.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import all GNN components
from .gnn_data_pipeline import GNNDataPipeline
from ..analytics.neural_recommendation_engine import NeuralRecommendationEngine
from ..visualization.gnn_explainer import GNNAttentionVisualizer, GNNDecisionExplainer
from ..ml.temporal_gnn import TemporalResearchAnalyzer
from ..query.gnn_search_engine import GNNSearchEngine
from ..evaluation.gnn_evaluator import GNNEvaluator


@dataclass
class GNNSystemConfig:
    """Configuration for GNN Core System."""
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model-specific configs
    transformer_heads: int = 8
    heterogeneous_types: List[str] = None
    temporal_timesteps: int = 5
    
    # System configs
    cache_size: int = 1000
    enable_explainability: bool = True
    enable_temporal_analysis: bool = True
    enable_search: bool = True


class GNNCoreSystem:
    """
    Complete GNN-first core system for Research Compass.
    
    Integrates all GNN components into a unified architecture
    optimized for graph neural network processing.
    """
    
    def __init__(self, config: Optional[GNNSystemConfig] = None):
        """
        Initialize GNN Core System.
        
        Args:
            config: System configuration
        """
        self.config = config or GNNSystemConfig()
        self.device = torch.device(self.config.device)
        
        # Core components
        self.data_pipeline = None
        self.recommendation_engine = None
        self.temporal_analyzer = None
        self.search_engine = None
        self.evaluator = None
        
        # Visualization and explainability
        self.attention_visualizer = None
        self.decision_explainer = None
        
        # Graph data and models
        self.current_graph_data = None
        self.node_embeddings = None
        self.edge_embeddings = None
        self.graph_embedder = None
        
        # System state
        self.is_initialized = False
        self.system_stats = {
            'total_documents_processed': 0,
            'total_queries_processed': 0,
            'total_recommendations_generated': 0,
            'system_uptime': time.time()
        }
        
        logger.info("GNN Core System initialized")
    
    def initialize_system(self, documents: Optional[List[Union[str, Path]]] = None):
        """
        Initialize the complete GNN system.
        
        Args:
            documents: Optional initial documents to process
        """
        logger.info("Initializing GNN Core System components...")
        
        # 1. Initialize data pipeline
        self.data_pipeline = GNNDataPipeline(self)
        
        # 2. Initialize recommendation engine
        self.recommendation_engine = NeuralRecommendationEngine(self)
        
        # 3. Initialize temporal analyzer
        if self.config.enable_temporal_analysis:
            self.temporal_analyzer = TemporalResearchAnalyzer(self)
        
        # 4. Initialize search engine
        if self.config.enable_search:
            self.search_engine = GNNSearchEngine(self)
        
        # 5. Initialize evaluator
        self.evaluator = GNNEvaluator()
        
        # 6. Initialize explainability components
        if self.config.enable_explainability:
            self.attention_visualizer = GNNAttentionVisualizer(None)  # Will be set when model is available
            self.decision_explainer = GNNDecisionExplainer(None, None)  # Will be set when model is available
        
        # 7. Process initial documents if provided
        if documents:
            self.process_documents(documents)
        
        # 8. Initialize models
        self._initialize_models()
        
        self.is_initialized = True
        logger.info("GNN Core System fully initialized")
    
    def process_documents(
        self,
        documents: List[Union[str, Path]],
        build_temporal_graphs: bool = False
    ) -> Dict:
        """
        Process documents using GNN-first pipeline.
        
        Args:
            documents: Documents to process
            build_temporal_graphs: Whether to build temporal graphs
            
        Returns:
            Processing results
        """
        if not self.is_initialized:
            self.initialize_system()
        
        logger.info(f"Processing {len(documents)} documents with GNN pipeline...")
        
        # Process documents through GNN pipeline
        graph_data, features = self.data_pipeline.process_documents_to_graph(documents)
        
        if graph_data is not None:
            self.current_graph_data = graph_data
            self.node_embeddings = features
            
            # Initialize recommendation models
            self.recommendation_engine.initialize_models()
            
            # Initialize search models
            if self.search_engine:
                self.search_engine.initialize_search_models()
            
            # Build temporal graphs if requested
            if build_temporal_graphs and self.temporal_analyzer:
                temporal_graphs = self.data_pipeline.create_temporal_graphs(
                    documents, [2018, 2019, 2020, 2021, 2022]
                )
                logger.info(f"Created {len(temporal_graphs)} temporal graphs")
        
        # Update stats
        self.system_stats['total_documents_processed'] += len(documents)
        
        return {
            'num_documents': len(documents),
            'graph_nodes': graph_data.num_nodes if graph_data else 0,
            'graph_edges': graph_data.num_edges if graph_data else 0,
            'embedding_dim': graph_data.x.shape[1] if graph_data else 0,
            'processing_time': time.time() - self.system_stats['system_uptime']
        }
    
    def get_recommendations(
        self,
        user_profile: Dict,
        top_k: int = 10,
        diversity_weight: float = 0.3,
        explain: bool = False
    ) -> List[Dict]:
        """
        Get GNN-powered recommendations.
        
        Args:
            user_profile: User profile with reading history
            top_k: Number of recommendations
            diversity_weight: Weight for diversity optimization
            explain: Whether to generate explanations
            
        Returns:
            List of recommendations
        """
        if not self.recommendation_engine:
            raise RuntimeError("Recommendation engine not initialized")
        
        # Get recommendations
        recommendations = self.recommendation_engine.recommend_papers_gnn(
            user_profile=user_profile,
            top_k=top_k,
            diversity_weight=diversity_weight
        )
        
        # Add explanations if requested
        if explain and self.decision_explainer:
            for rec in recommendations:
                explanation = self.decision_explainer.explain_recommendation(
                    user_id=user_profile.get('user_id', 'unknown'),
                    paper_id=rec['paper_id'],
                    recommendation_score=rec['score'],
                    user_profile=user_profile
                )
                rec['explanation'] = explanation
        
        # Update stats
        self.system_stats['total_recommendations_generated'] += len(recommendations)
        
        return recommendations
    
    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = 'semantic',
        use_graph_context: bool = True,
        explain: bool = False
    ) -> List[Dict]:
        """
        Search documents using GNN-powered search.
        
        Args:
            query: Search query
            top_k: Number of results
            search_type: Type of search
            use_graph_context: Whether to use graph context
            explain: Whether to generate explanations
            
        Returns:
            List of search results
        """
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        
        # Perform search
        results = self.search_engine.search(
            query=query,
            top_k=top_k,
            search_type=search_type,
            use_graph_context=use_graph_context
        )
        
        # Add explanations if requested
        if explain and self.decision_explainer:
            for result in results:
                # Create mock explanation for search results
                result['explanation'] = {
                    'primary_explanation': f"Found based on {search_type} search",
                    'confidence': result['score'],
                    'explanation_type': 'search_based'
                }
        
        # Update stats
        self.system_stats['total_queries_processed'] += 1
        
        return results
    
    def analyze_research_evolution(
        self,
        topic: str,
        years: List[int],
        prediction_years: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze research evolution using temporal GNN.
        
        Args:
            topic: Research topic
            years: Historical years
            prediction_years: Future years to predict
            
        Returns:
            Evolution analysis
        """
        if not self.temporal_analyzer:
            raise RuntimeError("Temporal analyzer not initialized")
        
        return self.temporal_analyzer.analyze_research_evolution(
            topic=topic,
            years=years,
            prediction_years=prediction_years
        )
    
    def explain_prediction(
        self,
        prediction_type: str,
        **kwargs
    ) -> Dict:
        """
        Explain GNN predictions.
        
        Args:
            prediction_type: Type of prediction to explain
            **kwargs: Additional arguments for explanation
            
        Returns:
            Explanation dictionary
        """
        if not self.decision_explainer:
            raise RuntimeError("Decision explainer not initialized")
        
        if prediction_type == 'recommendation':
            return self.decision_explainer.explain_recommendation(**kwargs)
        elif prediction_type == 'node_classification':
            return self.decision_explainer.explain_node_classification(**kwargs)
        else:
            return {'error': f'Unknown prediction type: {prediction_type}'}
    
    def visualize_attention(
        self,
        node_ids: Optional[List[int]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create attention visualization.
        
        Args:
            node_ids: Specific nodes to visualize
            output_path: Output path for visualization
            
        Returns:
            HTML visualization string
        """
        if not self.attention_visualizer:
            raise RuntimeError("Attention visualizer not initialized")
        
        # Mock attention data for visualization
        attention_data = {
            'layer_attention': [
                {
                    'layer': 0,
                    'weights': np.random.randn(10, 10),
                    'shape': (10, 10)
                }
            ],
            'node_importance': {i: np.random.random() for i in range(10)},
            'attention_paths': [
                {
                    'source': i,
                    'target': i+1,
                    'attention_weight': np.random.random(),
                    'path_type': 'attention'
                }
                for i in range(9)
            ]
        }
        
        return self.attention_visualizer.create_attention_visualization(
            attention_data, output_path
        )
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: 'Data',
        task_type: str,
        evaluation_config: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate GNN model performance.
        
        Args:
            model: Model to evaluate
            test_data: Test data
            task_type: Type of task
            evaluation_config: Evaluation configuration
            
        Returns:
            Evaluation results
        """
        if not self.evaluator:
            raise RuntimeError("Evaluator not initialized")
        
        return self.evaluator.evaluate_model(
            model=model,
            test_data=test_data,
            task_type=task_type,
            evaluation_config=evaluation_config
        )
    
    def benchmark_models(
        self,
        models: List[nn.Module],
        test_data: 'Data',
        task_type: str,
        benchmark_config: Optional[Dict] = None
    ) -> Dict:
        """
        Benchmark multiple GNN models.
        
        Args:
            models: Models to benchmark
            test_data: Test data
            task_type: Type of task
            benchmark_config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        if not self.evaluator:
            raise RuntimeError("Evaluator not initialized")
        
        return self.evaluator.benchmark_models(
            models=models,
            test_data=test_data,
            task_type=task_type,
            benchmark_config=benchmark_config
        )
    
    def _initialize_models(self):
        """Initialize GNN models based on current graph data."""
        if self.current_graph_data is None:
            logger.warning("No graph data available for model initialization")
            return
        
        logger.info("Initializing GNN models...")
        
        # Initialize graph embedder
        self.graph_embedder = self._create_graph_embedder()
        
        # Generate initial embeddings
        if self.current_graph_data.x is not None:
            self.node_embeddings = self.graph_embedder(self.current_graph_data)
        
        logger.info("GNN models initialized successfully")
    
    def _create_graph_embedder(self) -> nn.Module:
        """Create graph embedding model."""
        # Simple graph embedding model
        class GraphEmbedder(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                super().__init__()
                self.conv1 = nn.Linear(input_dim, hidden_dim)
                self.conv2 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, graph_data):
                x = graph_data.x
                x = F.relu(self.conv1(x))
                x = self.dropout(x)
                x = self.conv2(x)
                return x
        
        if self.current_graph_data is not None:
            input_dim = self.current_graph_data.x.shape[1]
            return GraphEmbedder(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.embedding_dim
            )
        else:
            return GraphEmbedder(128, 256, 128)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {
            'is_initialized': self.is_initialized,
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'device': self.config.device,
                'enable_explainability': self.config.enable_explainability,
                'enable_temporal_analysis': self.config.enable_temporal_analysis,
                'enable_search': self.config.enable_search
            },
            'components': {
                'data_pipeline': self.data_pipeline is not None,
                'recommendation_engine': self.recommendation_engine is not None,
                'temporal_analyzer': self.temporal_analyzer is not None,
                'search_engine': self.search_engine is not None,
                'evaluator': self.evaluator is not None,
                'attention_visualizer': self.attention_visualizer is not None,
                'decision_explainer': self.decision_explainer is not None
            },
            'graph_data': {
                'has_graph_data': self.current_graph_data is not None,
                'num_nodes': self.current_graph_data.num_nodes if self.current_graph_data else 0,
                'num_edges': self.current_graph_data.num_edges if self.current_graph_data else 0,
                'embedding_dim': self.current_graph_data.x.shape[1] if self.current_graph_data and self.current_graph_data.x is not None else 0
            },
            'system_stats': self.system_stats.copy()
        }
        
        # Add component-specific stats
        if self.data_pipeline:
            status['data_pipeline_stats'] = self.data_pipeline.get_pipeline_stats()
        
        if self.recommendation_engine:
            status['recommendation_stats'] = self.recommendation_engine.get_recommendation_stats()
        
        if self.search_engine:
            status['search_stats'] = self.search_engine.get_search_stats()
        
        if self.temporal_analyzer:
            status['temporal_stats'] = self.temporal_analyzer.get_temporal_analysis_stats()
        
        if self.evaluator:
            status['evaluation_stats'] = self.evaluator.get_evaluation_stats()
        
        return status
    
    def save_system_state(self, output_path: str):
        """Save system state to file."""
        state = {
            'config': self.config.__dict__,
            'system_stats': self.system_stats,
            'system_status': self.get_system_status(),
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"System state saved to {output_path}")
    
    def load_system_state(self, input_path: str):
        """Load system state from file."""
        with open(input_path, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        if 'config' in state:
            self.config = GNNSystemConfig(**state['config'])
        
        # Restore stats
        if 'system_stats' in state:
            self.system_stats = state['system_stats']
        
        logger.info(f"System state loaded from {input_path}")
    
    def create_system_report(self, output_path: Optional[str] = None) -> str:
        """
        Create comprehensive system report.
        
        Args:
            output_path: Optional output path for report
            
        Returns:
            HTML report string
        """
        status = self.get_system_status()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Core System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .status-ok {{ color: #27ae60; font-weight: bold; }}
                .status-warning {{ color: #f39c12; font-weight: bold; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #ecf0f1; border-radius: 3px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 GNN Core System Report</h1>
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>System Status</h2>
                <p class="{'status-ok' if status['is_initialized'] else 'status-warning'}">
                    System: {'✅ Initialized' if status['is_initialized'] else '❌ Not Initialized'}
                </p>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <div class="metric">
                    <div class="metric-value">{status['config']['embedding_dim']}</div>
                    <div class="metric-label">Embedding Dim</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['config']['hidden_dim']}</div>
                    <div class="metric-label">Hidden Dim</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['config']['device']}</div>
                    <div class="metric-label">Device</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Components</h2>
        """
        
        for component, is_active in status['components'].items():
            status_class = 'status-ok' if is_active else 'status-warning'
            status_text = '✅ Active' if is_active else '❌ Inactive'
            html += f'<p class="{status_class}">{component}: {status_text}</p>'
        
        html += f"""
            </div>
            
            <div class="section">
                <h2>Graph Data</h2>
                <div class="metric">
                    <div class="metric-value">{status['graph_data']['num_nodes']}</div>
                    <div class="metric-label">Nodes</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['graph_data']['num_edges']}</div>
                    <div class="metric-label">Edges</div>
                </div>
            </div>
            
            <div class="section">
                <h2>System Statistics</h2>
                <div class="metric">
                    <div class="metric-value">{status['system_stats']['total_documents_processed']}</div>
                    <div class="metric-label">Documents Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['system_stats']['total_queries_processed']}</div>
                    <div class="metric-label">Queries Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['system_stats']['total_recommendations_generated']}</div>
                    <div class="metric-label">Recommendations Generated</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"System report saved to {output_path}")
        
        return html


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Core System Test")
    print("=" * 80)
    
    # Initialize system
    config = GNNSystemConfig(
        embedding_dim=128,
        hidden_dim=256,
        enable_explainability=True,
        enable_temporal_analysis=True,
        enable_search=True
    )
    
    system = GNNCoreSystem(config)
    
    # Test system initialization
    print("\n1. Testing System Initialization...")
    system.initialize_system()
    
    status = system.get_system_status()
    print(f"  System initialized: {status['is_initialized']}")
    print(f"  Active components: {sum(status['components'].values())}")
    
    # Test document processing
    print("\n2. Testing Document Processing...")
    test_documents = [
        "Machine learning is a method of data analysis.",
        "Deep learning uses neural networks with multiple layers.",
        "Graph neural networks process graph-structured data."
    ]
    
    processing_results = system.process_documents(test_documents)
    print(f"  Documents processed: {processing_results['num_documents']}")
    print(f"  Graph nodes: {processing_results['graph_nodes']}")
    print(f"  Graph edges: {processing_results['graph_edges']}")
    
    # Test recommendations
    print("\n3. Testing Recommendations...")
    user_profile = {
        'user_id': 'test_user',
        'read_papers': [0, 1],
        'interests': ['machine learning', 'neural networks']
    }
    
    recommendations = system.get_recommendations(
        user_profile=user_profile,
        top_k=5,
        explain=True
    )
    
    print(f"  Recommendations generated: {len(recommendations)}")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"    {i}. Paper {rec['paper_id']} (score: {rec['score']:.3f})")
    
    # Test search
    print("\n4. Testing Search...")
    search_results = system.search_documents(
        query="machine learning",
        top_k=5,
        search_type='semantic',
        explain=True
    )
    
    print(f"  Search results: {len(search_results)}")
    for i, result in enumerate(search_results[:3], 1):
        print(f"    {i}. Document {result['document_id']} (score: {result['score']:.3f})")
    
    # Test temporal analysis
    print("\n5. Testing Temporal Analysis...")
    evolution = system.analyze_research_evolution(
        topic="machine learning",
        years=[2018, 2019, 2020, 2021, 2022]
    )
    
    print(f"  Topic: {evolution.get('topic')}")
    print(f"  Insights: {len(evolution.get('insights', []))}")
    
    # Test visualization
    print("\n6. Testing Visualization...")
    viz_html = system.visualize_attention()
    print(f"  Visualization generated: {len(viz_html)} characters")
    
    # Test system report
    print("\n7. Testing System Report...")
    report_html = system.create_system_report("gnn_system_report.html")
    print(f"  Report generated: {len(report_html)} characters")
    
    # Final status
    print("\n8. Final System Status...")
    final_status = system.get_system_status()
    print(f"  Total documents processed: {final_status['system_stats']['total_documents_processed']}")
    print(f"  Total queries processed: {final_status['system_stats']['total_queries_processed']}")
    print(f"  Total recommendations generated: {final_status['system_stats']['total_recommendations_generated']}")
    
    print("\n✓ GNN Core System test complete")
    print("🎉 All GNN components integrated successfully!")