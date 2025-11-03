#!/usr/bin/env python3
"""
GNN Manager - Orchestrate all ML models
Manages training, prediction, and model lifecycle
"""

import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .graph_converter import Neo4jToTorchGeometric
from .node_classifier import PaperClassifier, train_node_classifier, evaluate_classifier
from .link_predictor import CitationPredictor, train_link_predictor
from .embeddings_generator import GraphEmbedder


class GNNManager:
    """
    Manage all GNN models and predictions
    Provides unified interface for ML capabilities
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        model_dir: str = "models/gnn"
    ):
        """
        Initialize GNN Manager

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            model_dir: Directory to save/load models
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.converter = Neo4jToTorchGeometric(uri, user, password)
        self.embedder = GraphEmbedder(uri, user, password)

        # Models
        self.node_classifier = None
        self.link_predictor = None

        # Data
        self.graph_data = None
        self.label_map = None

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_models(self, force_retrain: bool = False):
        """
        Initialize all models (load or create new)

        Args:
            force_retrain: Force retraining even if models exist
        """
        print("Initializing GNN models...")

        # Load graph data
        if self.graph_data is None:
            print("  Loading graph from Neo4j...")
            self.graph_data = self.converter.export_papers_graph()
            self.graph_data = self.converter.create_train_val_test_split(self.graph_data)
            self.graph_data, self.label_map = self.converter.add_node_labels(self.graph_data)

        # Node classifier
        nc_path = self.model_dir / "node_classifier.pt"
        if nc_path.exists() and not force_retrain:
            print("  Loading node classifier...")
            self.node_classifier = self._load_node_classifier(nc_path)
        else:
            print("  Creating new node classifier...")
            self.node_classifier = PaperClassifier(
                input_dim=self.graph_data.x.shape[1],
                output_dim=len(self.label_map),
                hidden_dim=256,
                num_layers=3,
                dropout=0.5
            ).to(self.device)

        # Link predictor
        lp_path = self.model_dir / "link_predictor.pt"
        if lp_path.exists() and not force_retrain:
            print("  Loading link predictor...")
            self.link_predictor = self._load_link_predictor(lp_path)
        else:
            print("  Creating new link predictor...")
            self.link_predictor = CitationPredictor(
                input_dim=self.graph_data.x.shape[1],
                hidden_dim=128,
                num_layers=2,
                heads=4,
                dropout=0.3
            ).to(self.device)

        print("✓ Models initialized")

    def train_all_models(
        self,
        epochs: int = 100,
        patience: int = 10
    ) -> Dict:
        """
        Train all models

        Args:
            epochs: Training epochs
            patience: Early stopping patience

        Returns:
            Training history
        """
        history = {}

        # Ensure models are initialized
        if self.node_classifier is None or self.link_predictor is None:
            self.initialize_models()

        # Train node classifier
        print("\n" + "=" * 80)
        print("Training Node Classifier...")
        print("=" * 80)

        nc_history = train_node_classifier(
            self.node_classifier,
            self.graph_data,
            epochs=epochs,
            patience=patience,
            save_path=self.model_dir / "node_classifier.pt"
        )
        history['node_classifier'] = nc_history

        # Train link predictor
        print("\n" + "=" * 80)
        print("Training Link Predictor...")
        print("=" * 80)

        lp_history = train_link_predictor(
            self.link_predictor,
            self.graph_data,
            epochs=epochs,
            patience=patience,
            save_path=self.model_dir / "link_predictor.pt"
        )
        history['link_predictor'] = lp_history

        # Train embeddings
        print("\n" + "=" * 80)
        print("Training Graph Embeddings...")
        print("=" * 80)

        embeddings = self.embedder.train_node2vec()
        self.embedder.store_embeddings_in_neo4j()
        history['embeddings'] = {'num_embeddings': len(embeddings)}

        # Save history
        with open(self.model_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        print("\n✓ All models trained successfully")
        return history

    def predict_paper_topics(
        self,
        paper_id: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict topics for a paper

        Args:
            paper_id: Paper node ID
            top_k: Number of top topics

        Returns:
            List of (topic_name, confidence) tuples
        """
        if self.node_classifier is None:
            self.initialize_models()

        # Find paper in graph
        try:
            node_idx = self.graph_data.node_ids.index(paper_id)
        except ValueError:
            return []

        # Predict
        self.node_classifier.eval()
        with torch.no_grad():
            data = self.graph_data.to(self.device)
            logits = self.node_classifier(data.x, data.edge_index)
            probs = torch.softmax(logits[node_idx], dim=0)

            top_probs, top_indices = torch.topk(probs, k=top_k)

            # Map back to topic names
            idx_to_label = {v: k for k, v in self.label_map.items()}
            results = [
                (idx_to_label.get(idx.item(), f"Topic_{idx}"), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]

        return results

    def predict_paper_citations(
        self,
        paper_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Predict potential citations

        Args:
            paper_id: Source paper ID
            top_k: Number of predictions

        Returns:
            List of (paper_id, score) tuples
        """
        if self.link_predictor is None:
            self.initialize_models()

        # Find paper in graph
        try:
            node_idx = self.graph_data.node_ids.index(paper_id)
        except ValueError:
            return []

        # Predict
        data = self.graph_data.to(self.device)
        top_papers, scores = self.link_predictor.predict_citations(
            data.x,
            data.edge_index,
            node_idx,
            top_k=top_k
        )

        # Map back to paper IDs
        results = [
            (self.graph_data.node_ids[idx.item()], score.item())
            for idx, score in zip(top_papers, scores)
        ]

        return results

    def get_paper_recommendations(
        self,
        paper_id: str,
        method: str = 'hybrid',
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get paper recommendations

        Args:
            paper_id: Source paper ID
            method: 'embedding', 'citation', or 'hybrid'
            top_k: Number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        if method in ['embedding', 'hybrid']:
            # Embedding-based
            similar = self.embedder.find_similar_by_embedding(paper_id, top_k=top_k)
            for sim_id, score in similar:
                recommendations.append({
                    'paper_id': sim_id,
                    'score': score,
                    'method': 'embedding'
                })

        if method in ['citation', 'hybrid']:
            # Citation-based
            citations = self.predict_paper_citations(paper_id, top_k=top_k)
            for cite_id, score in citations:
                recommendations.append({
                    'paper_id': cite_id,
                    'score': score,
                    'method': 'citation'
                })

        # Sort and deduplicate
        seen = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda x: x['score'], reverse=True):
            if rec['paper_id'] not in seen:
                seen.add(rec['paper_id'])
                unique_recs.append(rec)

        return unique_recs[:top_k]

    def get_collaboration_suggestions(
        self,
        author_id: str,
        top_k: int = 10
    ) -> List[str]:
        """
        Suggest potential collaborators

        Args:
            author_id: Author node ID
            top_k: Number of suggestions

        Returns:
            List of author IDs
        """
        # Use embedding similarity
        similar = self.embedder.find_similar_by_embedding(author_id, top_k=top_k)
        return [author_id for author_id, _ in similar]

    def retrain_on_new_data(self):
        """Retrain models with new data"""
        print("Retraining models on updated graph...")

        # Reload graph
        self.graph_data = None
        self.initialize_models(force_retrain=False)

        # Retrain
        self.train_all_models()

    def get_model_metrics(self) -> Dict:
        """Get current model performance metrics"""
        metrics = {}

        if self.node_classifier and self.graph_data:
            nc_metrics = evaluate_classifier(
                self.node_classifier,
                self.graph_data,
                'test_mask'
            )
            metrics['node_classifier'] = nc_metrics

        return metrics

    def _load_node_classifier(self, path: Path) -> PaperClassifier:
        """Load node classifier from file"""
        checkpoint = torch.load(path, map_location=self.device)

        model = PaperClassifier(
            input_dim=self.graph_data.x.shape[1],
            output_dim=len(self.label_map),
            hidden_dim=256,
            num_layers=3,
            dropout=0.5
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _load_link_predictor(self, path: Path) -> CitationPredictor:
        """Load link predictor from file"""
        checkpoint = torch.load(path, map_location=self.device)

        model = CitationPredictor(
            input_dim=self.graph_data.x.shape[1],
            hidden_dim=128,
            num_layers=2,
            heads=4,
            dropout=0.3
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def close(self):
        """Cleanup resources"""
        self.converter.close()
        self.embedder.close()


# Main execution for testing
if __name__ == "__main__":
    import os

    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    print("=" * 80)
    print("GNN Manager Test")
    print("=" * 80)

    manager = GNNManager(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Initialize
        manager.initialize_models()

        # Train (quick test)
        history = manager.train_all_models(epochs=10, patience=5)

        # Test predictions
        if manager.graph_data and manager.graph_data.node_ids:
            paper_id = manager.graph_data.node_ids[0]

            print(f"\nTesting predictions for paper: {paper_id}")

            topics = manager.predict_paper_topics(paper_id, top_k=3)
            print(f"\nPredicted topics:")
            for topic, conf in topics:
                print(f"  - {topic}: {conf:.4f}")

            citations = manager.predict_paper_citations(paper_id, top_k=5)
            print(f"\nPredicted citations:")
            for cite_id, score in citations:
                print(f"  - {cite_id}: {score:.4f}")

        print("\n✓ GNN Manager test complete")

    finally:
        manager.close()
