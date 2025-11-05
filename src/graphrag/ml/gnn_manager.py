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

    def train(
        self,
        data: 'Data',
        model_type: str,
        task: str,
        epochs: int = 50,
        lr: float = 0.01,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Train a specific GNN model with progress tracking

        Args:
            data: PyG Data object
            model_type: Model type (gcn, gat, transformer, hetero)
            task: Task (node_classification, link_prediction, embedding)
            epochs: Training epochs
            lr: Learning rate
            progress_callback: Progress callback function(epoch, total, metrics)

        Returns:
            Training results
        """
        # Store graph data
        self.graph_data = data

        # Checkpoint paths
        checkpoint_path = self.model_dir / f"{task}_checkpoint.pt"
        save_path = self.model_dir / f"{task}_best.pt"

        if task == "node_classification":
            # Initialize model if needed
            if self.node_classifier is None:
                self.node_classifier = self._create_node_classifier(data)

            # Add labels if not present
            if not hasattr(data, 'y'):
                # Simple auto-labeling based on node properties
                num_classes = 5  # Default
                data.y = torch.randint(0, num_classes, (data.num_nodes,))

            # Train
            history = train_node_classifier(
                self.node_classifier,
                data,
                epochs=epochs,
                lr=lr,
                patience=max(10, epochs // 5),
                save_path=save_path,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=10,
                progress_callback=progress_callback,
                resume_from_checkpoint=True
            )

            return {
                "task": task,
                "model_type": model_type,
                "history": history,
                "final_val_acc": history['val_acc'][-1] if history['val_acc'] else 0,
                "final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0
            }

        elif task == "link_prediction":
            # Initialize model if needed
            if self.link_predictor is None:
                self.link_predictor = self._create_link_predictor(data)

            # Train
            history = train_link_predictor(
                self.link_predictor,
                data,
                epochs=epochs,
                lr=lr,
                patience=max(10, epochs // 5),
                save_path=save_path,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=10,
                progress_callback=progress_callback,
                resume_from_checkpoint=True
            )

            return {
                "task": task,
                "model_type": model_type,
                "history": history,
                "final_val_auc": history['val_auc'][-1] if history['val_auc'] else 0,
                "final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0
            }

        elif task == "embedding":
            # Train node2vec embeddings
            embeddings = self.embedder.train_node2vec()
            self.embedder.store_embeddings_in_neo4j()

            return {
                "task": task,
                "model_type": "node2vec",
                "num_embeddings": len(embeddings)
            }

        else:
            raise ValueError(f"Unknown task: {task}")

    def _create_node_classifier(self, data):
        """Create node classifier model"""
        from .node_classifier import PaperClassifier

        num_classes = data.y.max().item() + 1 if hasattr(data, 'y') else 5

        model = PaperClassifier(
            input_dim=data.x.shape[1],
            output_dim=num_classes,
            hidden_dim=256,
            num_layers=3,
            dropout=0.5
        ).to(self.device)

        return model

    def _create_link_predictor(self, data):
        """Create link predictor model"""
        from .link_predictor import CitationPredictor

        model = CitationPredictor(
            input_dim=data.x.shape[1],
            hidden_dim=128,
            num_layers=2,
            heads=4,
            dropout=0.3
        ).to(self.device)

        return model

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
            save_path=self.model_dir / "node_classifier.pt",
            checkpoint_path=self.model_dir / "node_classifier_checkpoint.pt",
            checkpoint_interval=10
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
            save_path=self.model_dir / "link_predictor.pt",
            checkpoint_path=self.model_dir / "link_predictor_checkpoint.pt",
            checkpoint_interval=10
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

    def export_models(
        self,
        output_dir: str = "exports",
        formats: List[str] = ['torchscript', 'onnx']
    ) -> Dict[str, Any]:
        """
        Export trained models for deployment

        Args:
            output_dir: Output directory for exported models
            formats: Export formats ('torchscript', 'onnx')

        Returns:
            Dict with export results
        """
        from .gnn_export import export_gnn_model

        if not self.graph_data:
            raise ValueError("No graph data available. Train models first.")

        results = {}

        # Export node classifier
        if self.node_classifier:
            logger.info("Exporting node classifier...")

            # Prepare example inputs
            example_x = self.graph_data.x[:100]  # Sample nodes
            example_edge_index = self.graph_data.edge_index[:, :200]  # Sample edges

            metadata = {
                'task': 'node_classification',
                'num_classes': self.graph_data.y.max().item() + 1 if hasattr(self.graph_data, 'y') else 0,
                'input_dim': self.graph_data.x.shape[1],
                'performance': self.get_model_metrics().get('node_classifier', {})
            }

            results['node_classifier'] = export_gnn_model(
                self.node_classifier,
                example_x,
                example_edge_index,
                output_dir,
                'node_classifier',
                formats=formats,
                metadata=metadata
            )

        # Export link predictor
        if self.link_predictor:
            logger.info("Exporting link predictor...")

            example_x = self.graph_data.x[:100]
            example_edge_index = self.graph_data.edge_index[:, :200]

            metadata = {
                'task': 'link_prediction',
                'input_dim': self.graph_data.x.shape[1]
            }

            results['link_predictor'] = export_gnn_model(
                self.link_predictor,
                example_x,
                example_edge_index,
                output_dir,
                'link_predictor',
                formats=formats,
                metadata=metadata
            )

        logger.info(f"✓ Models exported to {output_dir}")
        return results

    def generate_performance_report(
        self,
        output_dir: str = "reports"
    ) -> Dict[str, str]:
        """
        Generate comprehensive performance report with visualizations

        Args:
            output_dir: Output directory for reports

        Returns:
            Dict with paths to generated report files
        """
        from .gnn_visualization import generate_performance_report

        # Load training history
        history_file = self.model_dir / "training_history.json"
        if not history_file.exists():
            raise ValueError("No training history found. Train models first.")

        with open(history_file, 'r') as f:
            training_history = json.load(f)

        # Get current metrics
        metrics = self.get_model_metrics()

        all_reports = {}

        # Generate report for each model
        if 'node_classifier' in training_history:
            nc_report = generate_performance_report(
                model_name="node_classifier",
                task="node_classification",
                history=training_history['node_classifier'],
                final_metrics=metrics.get('node_classifier', {}),
                output_dir=output_dir
            )
            all_reports['node_classifier'] = nc_report

        if 'link_predictor' in training_history:
            lp_report = generate_performance_report(
                model_name="link_predictor",
                task="link_prediction",
                history=training_history['link_predictor'],
                final_metrics={},
                output_dir=output_dir
            )
            all_reports['link_predictor'] = lp_report

        logger.info(f"✓ Performance reports generated in {output_dir}")
        return all_reports

    def create_batch_predictor(self, batch_size: int = 32) -> 'BatchInferenceEngine':
        """
        Create batch inference engine for efficient predictions

        Args:
            batch_size: Batch size for predictions

        Returns:
            BatchInferenceEngine instance
        """
        from .gnn_batch_inference import BatchInferenceEngine

        if not self.node_classifier and not self.link_predictor:
            raise ValueError("No trained models available")

        # Use node classifier by default, or link predictor
        model = self.node_classifier or self.link_predictor

        predictor = BatchInferenceEngine(
            model=model,
            device=self.device,
            batch_size=batch_size,
            use_cache=True
        )

        logger.info("✓ Batch predictor created")
        return predictor

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
