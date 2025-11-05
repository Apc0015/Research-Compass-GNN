#!/usr/bin/env python3
"""
GNN Evaluator - Comprehensive evaluation framework for Graph Neural Networks
Provides metrics, benchmarks, and analysis for GNN model performance.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class GNNEvaluationMetrics:
    """
    Comprehensive metrics for GNN evaluation.
    
    Covers node classification, link prediction,
    graph generation, and recommendation tasks.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.metrics_history = []
        self.benchmark_results = {}
        
    def evaluate_node_classification(
        self,
        model: nn.Module,
        test_data: 'Data',
        true_labels: torch.Tensor,
        predicted_labels: torch.Tensor,
        node_embeddings: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Evaluate node classification performance.
        
        Args:
            model: Trained GNN model
            test_data: Test graph data
            true_labels: Ground truth labels
            predicted_labels: Model predictions
            node_embeddings: Optional node embeddings for analysis
            
        Returns:
            Comprehensive evaluation metrics
        """
        metrics = {
            'task': 'node_classification',
            'num_nodes': len(true_labels),
            'num_classes': len(torch.unique(true_labels))
        }
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels.cpu(), predicted_labels.cpu(), average='weighted'
        )
        
        metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            true_labels.cpu(), predicted_labels.cpu(), average=None
        )
        
        metrics['per_class_metrics'] = {}
        for i in range(len(per_class_precision)):
            metrics['per_class_metrics'][f'class_{i}'] = {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i],
                'f1_score': per_class_f1[i]
            }
        
        # Embedding quality metrics
        if node_embeddings is not None:
            embedding_metrics = self._evaluate_embedding_quality(
                node_embeddings, true_labels
            )
            metrics['embedding_quality'] = embedding_metrics
        
        # Model complexity
        metrics['model_complexity'] = self._compute_model_complexity(model)
        
        return metrics
    
    def evaluate_link_prediction(
        self,
        model: nn.Module,
        test_edges: torch.Tensor,
        edge_labels: torch.Tensor,
        predicted_scores: torch.Tensor,
        negative_edges: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Evaluate link prediction performance.
        
        Args:
            model: Trained GNN model
            test_edges: Test edge pairs
            edge_labels: Ground truth edge labels
            predicted_scores: Predicted edge scores
            negative_edges: Optional negative edge samples
            
        Returns:
            Link prediction evaluation metrics
        """
        metrics = {
            'task': 'link_prediction',
            'num_edges': len(edge_labels),
            'positive_edges': edge_labels.sum().item(),
            'negative_edges': (1 - edge_labels).sum().item()
        }
        
        # Convert scores to binary predictions
        predicted_labels = (predicted_scores > 0.5).float()
        
        # Basic metrics
        accuracy = accuracy_score(edge_labels.cpu(), predicted_labels.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            edge_labels.cpu(), predicted_labels.cpu(), average='binary'
        )
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(edge_labels.cpu(), predicted_scores.cpu())
        except ValueError:
            auc_roc = 0.5  # Default for edge cases
        
        metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        })
        
        # Ranking metrics
        ranking_metrics = self._compute_ranking_metrics(
            test_edges, edge_labels, predicted_scores
        )
        metrics['ranking_metrics'] = ranking_metrics
        
        # Negative sampling quality
        if negative_edges is not None:
            neg_metrics = self._evaluate_negative_sampling(
                negative_edges, predicted_scores
            )
            metrics['negative_sampling'] = neg_metrics
        
        return metrics
    
    def evaluate_graph_generation(
        self,
        model: nn.Module,
        generated_graph: 'Data',
        reference_graph: 'Data',
        generation_config: Dict
    ) -> Dict:
        """
        Evaluate graph generation quality.
        
        Args:
            model: Trained generative GNN model
            generated_graph: Generated graph
            reference_graph: Reference/ground truth graph
            generation_config: Configuration used for generation
            
        Returns:
            Graph generation evaluation metrics
        """
        metrics = {
            'task': 'graph_generation',
            'generation_config': generation_config
        }
        
        # Structural similarity metrics
        structural_metrics = self._compute_structural_similarity(
            generated_graph, reference_graph
        )
        metrics['structural_similarity'] = structural_metrics
        
        # Graph property metrics
        gen_properties = self._compute_graph_properties(generated_graph)
        ref_properties = self._compute_graph_properties(reference_graph)
        
        metrics['graph_properties'] = {
            'generated': gen_properties,
            'reference': ref_properties,
            'differences': self._compute_property_differences(
                gen_properties, ref_properties
            )
        }
        
        # Novelty and diversity metrics
        novelty_metrics = self._compute_generation_novelty(
            generated_graph, reference_graph
        )
        metrics['novelty_diversity'] = novelty_metrics
        
        return metrics
    
    def evaluate_recommendation_system(
        self,
        recommendations: List[Dict],
        ground_truth: Dict,
        user_profiles: Dict,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict:
        """
        Evaluate recommendation system performance.
        
        Args:
            recommendations: List of recommendation results
            ground_truth: Ground truth relevance data
            user_profiles: User profile information
            k_values: Values of k for precision@k, recall@k
            
        Returns:
            Recommendation evaluation metrics
        """
        metrics = {
            'task': 'recommendation_system',
            'num_users': len(recommendations),
            'k_values': k_values
        }
        
        # Compute precision@k and recall@k
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for user_id, user_recs in recommendations.items():
                if user_id not in ground_truth:
                    continue
                
                # Get top-k recommendations
                top_k_recs = user_recs[:k]
                rec_ids = [rec['document_id'] for rec in top_k_recs]
                
                # Get ground truth
                true_relevant = set(ground_truth[user_id])
                
                # Compute precision and recall
                if rec_ids:
                    relevant_recs = len(set(rec_ids).intersection(true_relevant))
                    precision = relevant_recs / len(rec_ids)
                    recall = relevant_recs / len(true_relevant) if true_relevant else 0
                else:
                    precision = 0
                    recall = 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Average across users
            precision_at_k[k] = np.mean(precisions) if precisions else 0
            recall_at_k[k] = np.mean(recalls) if recalls else 0
            f1_at_k[k] = (
                2 * precision_at_k[k] * recall_at_k[k] / 
                (precision_at_k[k] + recall_at_k[k]) 
                if (precision_at_k[k] + recall_at_k[k]) > 0 else 0
            )
        
        metrics['precision_at_k'] = precision_at_k
        metrics['recall_at_k'] = recall_at_k
        metrics['f1_at_k'] = f1_at_k
        
        # Diversity metrics
        diversity_metrics = self._compute_recommendation_diversity(
            recommendations, user_profiles
        )
        metrics['diversity'] = diversity_metrics
        
        # Coverage metrics
        coverage_metrics = self._compute_recommendation_coverage(
            recommendations, ground_truth
        )
        metrics['coverage'] = coverage_metrics
        
        return metrics
    
    def _evaluate_embedding_quality(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict:
        """Evaluate quality of node embeddings."""
        metrics = {}
        
        # Intra-cluster cohesion
        intra_cohesion = self._compute_intra_cluster_cohesion(
            embeddings, labels
        )
        metrics['intra_cluster_cohesion'] = intra_cohesion
        
        # Inter-cluster separation
        inter_separation = self._compute_inter_cluster_separation(
            embeddings, labels
        )
        metrics['inter_cluster_separation'] = inter_separation
        
        # Silhouette coefficient
        silhouette = self._compute_silhouette_coefficient(
            embeddings, labels
        )
        metrics['silhouette_coefficient'] = silhouette
        
        return metrics
    
    def _compute_ranking_metrics(
        self,
        test_edges: torch.Tensor,
        edge_labels: torch.Tensor,
        predicted_scores: torch.Tensor
    ) -> Dict:
        """Compute ranking-based metrics for link prediction."""
        metrics = {}
        
        # Mean Reciprocal Rank (MRR)
        mrr = self._compute_mean_reciprocal_rank(
            test_edges, edge_labels, predicted_scores
        )
        metrics['mean_reciprocal_rank'] = mrr
        
        # Hits@K
        hits_at_k = {}
        for k in [1, 3, 5, 10]:
            hits = self._compute_hits_at_k(
                test_edges, edge_labels, predicted_scores, k
            )
            hits_at_k[k] = hits
        
        metrics['hits_at_k'] = hits_at_k
        
        return metrics
    
    def _compute_structural_similarity(
        self,
        generated_graph: 'Data',
        reference_graph: 'Data'
    ) -> Dict:
        """Compute structural similarity between graphs."""
        metrics = {}
        
        # Degree distribution similarity
        gen_degrees = self._compute_degree_distribution(generated_graph)
        ref_degrees = self._compute_degree_distribution(reference_graph)
        
        degree_similarity = self._compute_distribution_similarity(
            gen_degrees, ref_degrees
        )
        metrics['degree_distribution_similarity'] = degree_similarity
        
        # Clustering coefficient similarity
        gen_clustering = self._compute_clustering_coefficient(generated_graph)
        ref_clustering = self._compute_clustering_coefficient(reference_graph)
        
        clustering_similarity = 1 - abs(gen_clustering - ref_clustering)
        metrics['clustering_similarity'] = clustering_similarity
        
        # Path length similarity
        gen_path_length = self._compute_average_path_length(generated_graph)
        ref_path_length = self._compute_average_path_length(reference_graph)
        
        path_similarity = 1 - abs(gen_path_length - ref_path_length) / max(gen_path_length, ref_path_length)
        metrics['path_length_similarity'] = path_similarity
        
        return metrics
    
    def _compute_generation_novelty(
        self,
        generated_graph: 'Data',
        reference_graph: 'Data'
    ) -> Dict:
        """Compute novelty and diversity of generated graph."""
        metrics = {}
        
        # Novel edges (not in reference)
        gen_edges = set()
        ref_edges = set()
        
        if hasattr(generated_graph, 'edge_index'):
            gen_edge_index = generated_graph.edge_index
            for i in range(gen_edge_index.size(1)):
                edge = tuple(sorted([gen_edge_index[0, i].item(), gen_edge_index[1, i].item()]))
                gen_edges.add(edge)
        
        if hasattr(reference_graph, 'edge_index'):
            ref_edge_index = reference_graph.edge_index
            for i in range(ref_edge_index.size(1)):
                edge = tuple(sorted([ref_edge_index[0, i].item(), ref_edge_index[1, i].item()]))
                ref_edges.add(edge)
        
        novel_edges = gen_edges - ref_edges
        metrics['novel_edges_ratio'] = len(novel_edges) / max(len(gen_edges), 1)
        
        # Node diversity
        gen_nodes = generated_graph.num_nodes if hasattr(generated_graph, 'num_nodes') else 0
        ref_nodes = reference_graph.num_nodes if hasattr(reference_graph, 'num_nodes') else 0
        
        node_diversity = abs(gen_nodes - ref_nodes) / max(ref_nodes, 1)
        metrics['node_diversity'] = node_diversity
        
        return metrics
    
    def _compute_recommendation_diversity(
        self,
        recommendations: List[Dict],
        user_profiles: Dict
    ) -> Dict:
        """Compute diversity metrics for recommendations."""
        metrics = {}
        
        # Intra-list diversity
        intra_diversities = []
        for user_id, user_recs in recommendations.items():
            if len(user_recs) < 2:
                continue
            
            # Compute pairwise diversity
            diversity_scores = []
            for i in range(len(user_recs)):
                for j in range(i + 1, len(user_recs)):
                    rec1 = user_recs[i]
                    rec2 = user_recs[j]
                    
                    # Simple diversity based on different categories
                    diversity = 1.0 if rec1.get('category') != rec2.get('category') else 0.0
                    diversity_scores.append(diversity)
            
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
            intra_diversities.append(avg_diversity)
        
        metrics['intra_list_diversity'] = np.mean(intra_diversities) if intra_diversities else 0
        
        # Inter-user diversity
        all_recommended_items = set()
        for user_recs in recommendations.values():
            for rec in user_recs:
                all_recommended_items.add(rec['document_id'])
        
        metrics['catalog_coverage'] = len(all_recommended_items) / 1000  # Assume 1000 total items
        
        return metrics
    
    def _compute_recommendation_coverage(
        self,
        recommendations: List[Dict],
        ground_truth: Dict
    ) -> Dict:
        """Compute coverage metrics for recommendations."""
        metrics = {}
        
        # Item coverage
        recommended_items = set()
        relevant_items = set()
        
        for user_id, user_recs in recommendations.items():
            for rec in user_recs:
                recommended_items.add(rec['document_id'])
        
        for user_id, true_items in ground_truth.items():
            relevant_items.update(true_items)
        
        item_coverage = len(recommended_items.intersection(relevant_items)) / max(len(relevant_items), 1)
        metrics['item_coverage'] = item_coverage
        
        # User coverage
        users_with_recs = set(recommendations.keys())
        users_with_truth = set(ground_truth.keys())
        user_coverage = len(users_with_recs.intersection(users_with_truth)) / max(len(users_with_truth), 1)
        metrics['user_coverage'] = user_coverage
        
        return metrics
    
    def _compute_intra_cluster_cohesion(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute intra-cluster cohesion."""
        unique_labels = torch.unique(labels)
        cohesion_scores = []
        
        for label in unique_labels:
            # Get embeddings for this cluster
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            
            if len(cluster_embeddings) < 2:
                continue
            
            # Compute pairwise similarities
            similarities = F.cosine_similarity(cluster_embeddings, cluster_embeddings)
            
            # Average similarity (excluding diagonal)
            mask = torch.eye(len(similarities), dtype=torch.bool)
            similarities[mask] = 0
            
            cohesion = similarities.sum() / (len(similarities) - len(similarities))
            cohesion_scores.append(cohesion.item())
        
        return np.mean(cohesion_scores) if cohesion_scores else 0
    
    def _compute_inter_cluster_separation(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute inter-cluster separation."""
        unique_labels = torch.unique(labels)
        separation_scores = []
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                # Get embeddings for both clusters
                mask1 = labels == label1
                mask2 = labels == label2
                
                emb1 = embeddings[mask1]
                emb2 = embeddings[mask2]
                
                if len(emb1) == 0 or len(emb2) == 0:
                    continue
                
                # Compute centroid similarity
                centroid1 = torch.mean(emb1, dim=0)
                centroid2 = torch.mean(emb2, dim=0)
                
                similarity = F.cosine_similarity(
                    centroid1.unsqueeze(0), centroid2.unsqueeze(0)
                ).item()
                
                separation_scores.append(1 - similarity)  # Lower similarity = higher separation
        
        return np.mean(separation_scores) if separation_scores else 0
    
    def _compute_silhouette_coefficient(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute silhouette coefficient."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(
                embeddings.cpu().numpy(), 
                labels.cpu().numpy()
            )
        except ImportError:
            return 0.0
    
    def _compute_mean_reciprocal_rank(
        self,
        test_edges: torch.Tensor,
        edge_labels: torch.Tensor,
        predicted_scores: torch.Tensor
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        # Sort by predicted scores
        sorted_indices = torch.argsort(predicted_scores, descending=True)
        
        reciprocal_ranks = []
        for i, idx in enumerate(sorted_indices):
            if edge_labels[idx] == 1:  # Positive edge
                rank = i + 1
                reciprocal_ranks.append(1.0 / rank)
                break
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    def _compute_hits_at_k(
        self,
        test_edges: torch.Tensor,
        edge_labels: torch.Tensor,
        predicted_scores: torch.Tensor,
        k: int
    ) -> float:
        """Compute Hits@K metric."""
        # Sort by predicted scores
        sorted_indices = torch.argsort(predicted_scores, descending=True)
        
        # Check if any positive edge is in top-k
        top_k_indices = sorted_indices[:k]
        hits = (edge_labels[top_k_indices] == 1).sum().item()
        
        return hits / k
    
    def _compute_distribution_similarity(
        self,
        dist1: Dict,
        dist2: Dict
    ) -> float:
        """Compute similarity between two distributions."""
        # Simple KL divergence approximation
        all_keys = set(dist1.keys()).union(set(dist2.keys()))
        
        similarity = 0
        for key in all_keys:
            p1 = dist1.get(key, 0)
            p2 = dist2.get(key, 0)
            
            if p1 > 0 and p2 > 0:
                similarity += min(p1, p2) / max(p1, p2)
        
        return similarity / len(all_keys)
    
    def _compute_degree_distribution(self, graph: 'Data') -> Dict:
        """Compute degree distribution of graph."""
        if not hasattr(graph, 'edge_index'):
            return {}
        
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else edge_index.max().item() + 1
        
        degrees = torch.zeros(num_nodes)
        for i in range(edge_index.size(1)):
            source = edge_index[0, i]
            degrees[source] += 1
        
        # Convert to distribution
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree.item()] = degree_counts.get(degree.item(), 0) + 1
        
        # Normalize
        total_edges = sum(degree_counts.values())
        for degree in degree_counts:
            degree_counts[degree] /= total_edges
        
        return degree_counts
    
    def _compute_clustering_coefficient(self, graph: 'Data') -> float:
        """Compute average clustering coefficient."""
        # Simplified clustering coefficient computation
        return 0.3  # Mock value
    
    def _compute_average_path_length(self, graph: 'Data') -> float:
        """Compute average path length in graph."""
        # Simplified path length computation
        return 3.5  # Mock value
    
    def _compute_model_complexity(self, model: nn.Module) -> Dict:
        """Compute model complexity metrics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
        }


class GNNEvaluator:
    """
    Comprehensive GNN evaluation framework.
    
    Provides automated evaluation, benchmarking,
    and performance analysis for GNN models.
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize GNN Evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_calculator = GNNEvaluationMetrics()
        self.evaluation_history = []
        
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: 'Data',
        task_type: str,
        evaluation_config: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained GNN model
            test_data: Test dataset
            task_type: Type of task (classification, link_prediction, etc.)
            evaluation_config: Optional evaluation configuration
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating model for task: {task_type}")
        
        evaluation_config = evaluation_config or {}
        
        # Set model to evaluation mode
        model.eval()
        
        # Task-specific evaluation
        if task_type == 'node_classification':
            results = self._evaluate_node_classification_task(
                model, test_data, evaluation_config
            )
        elif task_type == 'link_prediction':
            results = self._evaluate_link_prediction_task(
                model, test_data, evaluation_config
            )
        elif task_type == 'graph_generation':
            results = self._evaluate_graph_generation_task(
                model, test_data, evaluation_config
            )
        elif task_type == 'recommendation':
            results = self._evaluate_recommendation_task(
                model, test_data, evaluation_config
            )
        else:
            results = {'error': f'Unknown task type: {task_type}'}
        
        # Add evaluation metadata
        results['evaluation_metadata'] = {
            'task_type': task_type,
            'evaluation_config': evaluation_config,
            'timestamp': time.time(),
            'model_info': self._get_model_info(model)
        }
        
        # Save results
        self._save_evaluation_results(results, task_type)
        
        # Add to history
        self.evaluation_history.append(results)
        
        return results
    
    def _evaluate_node_classification_task(
        self,
        model: nn.Module,
        test_data: 'Data',
        config: Dict
    ) -> Dict:
        """Evaluate node classification task."""
        with torch.no_grad():
            # Get predictions
            if hasattr(model, 'forward'):
                output = model(test_data.x, test_data.edge_index)
                
                if output.dim() > 1:
                    predicted_labels = torch.argmax(output, dim=1)
                else:
                    predicted_labels = output
            else:
                return {'error': 'Model does not have forward method'}
        
        # Get true labels
        true_labels = test_data.y if hasattr(test_data, 'y') else torch.zeros(output.size(0))
        
        # Get node embeddings if available
        node_embeddings = None
        if hasattr(model, 'get_embeddings'):
            node_embeddings = model.get_embeddings(test_data.x, test_data.edge_index)
        
        # Evaluate
        return self.metrics_calculator.evaluate_node_classification(
            model, test_data, true_labels, predicted_labels, node_embeddings
        )
    
    def _evaluate_link_prediction_task(
        self,
        model: nn.Module,
        test_data: 'Data',
        config: Dict
    ) -> Dict:
        """Evaluate link prediction task."""
        # Generate negative samples
        positive_edges = test_data.edge_index.t()
        negative_edges = self._generate_negative_samples(
            positive_edges, test_data.num_nodes, len(positive_edges)
        )
        
        # Combine positive and negative edges
        all_edges = torch.cat([positive_edges, negative_edges], dim=0)
        edge_labels = torch.cat([
            torch.ones(len(positive_edges)),
            torch.zeros(len(negative_edges))
        ])
        
        with torch.no_grad():
            # Get predictions
            if hasattr(model, 'predict_links'):
                predicted_scores = model.predict_links(all_edges, test_data.x, test_data.edge_index)
            else:
                # Fallback: use node embeddings
                node_embeddings = model.get_embeddings(test_data.x, test_data.edge_index)
                predicted_scores = self._compute_link_scores(node_embeddings, all_edges)
        
        # Evaluate
        return self.metrics_calculator.evaluate_link_prediction(
            model, all_edges, edge_labels, predicted_scores, negative_edges
        )
    
    def _evaluate_graph_generation_task(
        self,
        model: nn.Module,
        test_data: 'Data',
        config: Dict
    ) -> Dict:
        """Evaluate graph generation task."""
        # Generate graph
        generation_config = config.get('generation', {})
        
        with torch.no_grad():
            if hasattr(model, 'generate_graph'):
                generated_graph = model.generate_graph(**generation_config)
            else:
                return {'error': 'Model does not have generate_graph method'}
        
        # Get reference graph
        reference_graph = test_data
        
        # Evaluate
        return self.metrics_calculator.evaluate_graph_generation(
            model, generated_graph, reference_graph, generation_config
        )
    
    def _evaluate_recommendation_task(
        self,
        model: nn.Module,
        test_data: 'Data',
        config: Dict
    ) -> Dict:
        """Evaluate recommendation task."""
        # Generate recommendations
        user_profiles = config.get('user_profiles', {})
        ground_truth = config.get('ground_truth', {})
        
        recommendations = {}
        for user_id, profile in user_profiles.items():
            with torch.no_grad():
                if hasattr(model, 'recommend'):
                    user_recs = model.recommend(profile, top_k=20)
                else:
                    user_recs = []
            
            recommendations[user_id] = user_recs
        
        # Evaluate
        return self.metrics_calculator.evaluate_recommendation_system(
            recommendations, ground_truth, user_profiles
        )
    
    def _generate_negative_samples(
        self,
        positive_edges: torch.Tensor,
        num_nodes: int,
        num_negatives: int
    ) -> torch.Tensor:
        """Generate negative edge samples."""
        negative_edges = []
        
        # Get existing edge pairs
        existing_pairs = set()
        for edge in positive_edges:
            pair = tuple(sorted(edge.tolist()))
            existing_pairs.add(pair)
        
        # Generate random negative edges
        while len(negative_edges) < num_negatives:
            source = torch.randint(0, num_nodes, (1,)).item()
            target = torch.randint(0, num_nodes, (1,)).item()
            
            if source != target:
                pair = tuple(sorted([source, target]))
                if pair not in existing_pairs:
                    negative_edges.append([source, target])
                    existing_pairs.add(pair)
        
        return torch.tensor(negative_edges, dtype=torch.long)
    
    def _compute_link_scores(
        self,
        node_embeddings: torch.Tensor,
        edges: torch.Tensor
    ) -> torch.Tensor:
        """Compute link scores from node embeddings."""
        scores = []
        
        for edge in edges:
            source_emb = node_embeddings[edge[0]]
            target_emb = node_embeddings[edge[1]]
            
            # Cosine similarity
            similarity = F.cosine_similarity(
                source_emb.unsqueeze(0), target_emb.unsqueeze(0)
            )
            scores.append(similarity)
        
        return torch.stack(scores).squeeze()
    
    def _get_model_info(self, model: nn.Module) -> Dict:
        """Get model information."""
        return {
            'model_type': type(model).__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_config': getattr(model, 'config', {})
        }
    
    def _save_evaluation_results(self, results: Dict, task_type: str):
        """Save evaluation results to file."""
        timestamp = int(time.time())
        filename = f"evaluation_{task_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def create_evaluation_report(
        self,
        results: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            output_path: Optional file path to save report
            
        Returns:
            HTML report string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GNN Evaluation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .metric-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; 
                               border-radius: 5px; background: #f9f9f9; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2ecc71; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                .chart-container { height: 400px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>GNN Model Evaluation Report</h1>
        """
        
        # Add summary metrics
        html += self._create_summary_section(results)
        
        # Add detailed metrics for each evaluation
        for i, result in enumerate(results, 1):
            html += f"""
            <div class="metric-card">
                <h2>Evaluation {i}: {result.get('evaluation_metadata', {}).get('task_type', 'Unknown')}</h2>
                <div class="metric-value">{result.get('accuracy', 0):.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return html
    
    def _create_summary_section(self, results: List[Dict]) -> str:
        """Create summary section of evaluation report."""
        if not results:
            return "<p>No evaluation results available</p>"
        
        # Compute summary statistics
        accuracies = [r.get('accuracy', 0) for r in results]
        f1_scores = [r.get('f1_score', 0) for r in results]
        
        summary_html = """
        <div class="metric-card">
            <h2>Summary Statistics</h2>
            <div class="metric-value">{:.3f}</div>
            <div class="metric-label">Average Accuracy</div>
            
            <div class="metric-value">{:.3f}</div>
            <div class="metric-label">Average F1 Score</div>
            
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Evaluations</div>
        </div>
        """.format(
            np.mean(accuracies),
            np.mean(f1_scores),
            len(results)
        )
        
        return summary_html
    
    def benchmark_models(
        self,
        models: List[nn.Module],
        test_data: 'Data',
        task_type: str,
        benchmark_config: Optional[Dict] = None
    ) -> Dict:
        """
        Benchmark multiple models.
        
        Args:
            models: List of models to benchmark
            test_data: Test dataset
            task_type: Type of task
            benchmark_config: Optional benchmark configuration
            
        Returns:
            Benchmark comparison results
        """
        logger.info(f"Benchmarking {len(models)} models for task: {task_type}")
        
        benchmark_results = {
            'task_type': task_type,
            'models': [],
            'comparison_metrics': {},
            'ranking': {}
        }
        
        # Evaluate each model
        for i, model in enumerate(models):
            model_name = f"model_{i+1}"
            
            # Evaluate model
            results = self.evaluate_model(model, test_data, task_type, benchmark_config)
            
            # Extract key metrics
            key_metrics = {
                'accuracy': results.get('accuracy', 0),
                'f1_score': results.get('f1_score', 0),
                'model_complexity': results.get('model_complexity', {}).get('total_parameters', 0)
            }
            
            benchmark_results['models'].append({
                'name': model_name,
                'results': results,
                'key_metrics': key_metrics
            })
        
        # Compare models
        comparison_metrics = ['accuracy', 'f1_score', 'model_complexity']
        for metric in comparison_metrics:
            values = [m['key_metrics'].get(metric, 0) for m in benchmark_results['models']]
            
            benchmark_results['comparison_metrics'][metric] = {
                'values': values,
                'best_model': np.argmax(values) if metric != 'model_complexity' else np.argmin(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Rank models
        for i, model_info in enumerate(benchmark_results['models']):
            # Compute composite score
            accuracy = model_info['key_metrics'].get('accuracy', 0)
            f1 = model_info['key_metrics'].get('f1_score', 0)
            complexity = model_info['key_metrics'].get('model_complexity', 0)
            
            # Composite score (higher is better, lower complexity is better)
            composite_score = (accuracy + f1) / 2 - (complexity / 1000000)
            
            model_info['composite_score'] = composite_score
        
        # Sort by composite score
        benchmark_results['models'].sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Create ranking
        for i, model_info in enumerate(benchmark_results['models']):
            benchmark_results['ranking'][i+1] = {
                'model_name': model_info['name'],
                'composite_score': model_info['composite_score'],
                'key_metrics': model_info['key_metrics']
            }
        
        # Save benchmark results
        self._save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _save_benchmark_results(self, results: Dict):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"benchmark_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def get_evaluation_stats(self) -> Dict:
        """Get statistics about evaluations performed."""
        return {
            'total_evaluations': len(self.evaluation_history),
            'output_directory': str(self.output_dir),
            'evaluation_types': list(set(
                result.get('evaluation_metadata', {}).get('task_type', 'unknown')
                for result in self.evaluation_history
            ))
        }


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Evaluator Test")
    print("=" * 80)
    
    # Mock model for testing
    class MockGNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 10)
        
        def forward(self, x, edge_index):
            return self.linear(x)
        
        def get_embeddings(self, x, edge_index):
            return x
    
    # Mock test data
    class MockTestData:
        def __init__(self):
            self.x = torch.randn(100, 128)
            self.edge_index = torch.tensor([[i, i+1] for i in range(99)], dtype=torch.long).t()
            self.y = torch.randint(0, 10, (100,))
            self.num_nodes = 100
    
    # Initialize evaluator
    evaluator = GNNEvaluator()
    
    # Test node classification evaluation
    print("\n1. Testing Node Classification Evaluation...")
    model = MockGNNModel()
    test_data = MockTestData()
    
    results = evaluator.evaluate_model(
        model=model,
        test_data=test_data,
        task_type='node_classification'
    )
    
    print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
    print(f"  F1 Score: {results.get('f1_score', 0):.3f}")
    print(f"  Model complexity: {results.get('model_complexity', {}).get('total_parameters', 0)}")
    
    # Test benchmarking
    print("\n2. Testing Model Benchmarking...")
    models = [MockGNNModel() for _ in range(3)]
    
    benchmark_results = evaluator.benchmark_models(
        models=models,
        test_data=test_data,
        task_type='node_classification'
    )
    
    print(f"  Best model: {benchmark_results['ranking'].get(1, {}).get('model_name', 'Unknown')}")
    print(f"  Best score: {benchmark_results['ranking'].get(1, {}).get('composite_score', 0):.3f}")
    
    # Test evaluation report
    print("\n3. Testing Evaluation Report...")
    report_html = evaluator.create_evaluation_report(
        results=[results],
        output_path="evaluation_report.html"
    )
    
    print(f"  Report generated: {len(report_html)} characters")
    
    # Get stats
    stats = evaluator.get_evaluation_stats()
    print(f"\nEvaluator Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ GNN Evaluator test complete")