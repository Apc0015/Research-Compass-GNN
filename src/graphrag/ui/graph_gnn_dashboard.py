#!/usr/bin/env python3
"""
Graph & GNN Dashboard - Interactive visualization and management for GNN models.

This module provides a comprehensive interface for:
- Visualizing the knowledge graph
- Training and managing GNN models
- Viewing graph statistics and metrics
- Exploring graph structure interactively
- Monitoring GNN predictions and explanations
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)


class GraphGNNDashboard:
    """Dashboard for graph visualization and GNN model management."""

    def __init__(self, system):
        """
        Initialize the dashboard.

        Args:
            system: AcademicRAGSystem instance
        """
        self.system = system

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        try:
            stats = {
                "status": "success",
                "graph_size": {},
                "node_types": {},
                "edge_types": {},
                "metrics": {},
                "gnn_status": {}
            }

            # Get basic graph info
            if hasattr(self.system.graph, 'get_all_nodes'):
                all_nodes = self.system.graph.get_all_nodes()
                stats["graph_size"]["total_nodes"] = len(all_nodes)

                # Count by node type
                node_counts = {}
                for node in all_nodes:
                    node_type = node.get('type', 'Unknown')
                    node_counts[node_type] = node_counts.get(node_type, 0) + 1
                stats["node_types"] = node_counts

            # Get edge info
            if hasattr(self.system.graph, 'get_all_relationships'):
                all_edges = self.system.graph.get_all_relationships()
                stats["graph_size"]["total_edges"] = len(all_edges)

                # Count by edge type
                edge_counts = {}
                for edge in all_edges:
                    edge_type = edge.get('type', 'Unknown')
                    edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
                stats["edge_types"] = edge_counts

            # Get academic-specific stats
            if hasattr(self.system, 'academic'):
                try:
                    papers = self.system.academic.get_all_papers()
                    authors = self.system.academic.get_all_authors()
                    stats["academic"] = {
                        "papers": len(papers) if papers else 0,
                        "authors": len(authors) if authors else 0
                    }

                    # Get year distribution
                    if papers:
                        year_dist = {}
                        for paper in papers:
                            year = paper.get('year', 'Unknown')
                            year_dist[str(year)] = year_dist.get(str(year), 0) + 1
                        stats["year_distribution"] = dict(sorted(year_dist.items()))

                except Exception as e:
                    logger.warning(f"Could not get academic stats: {e}")

            # Get GNN status
            try:
                if hasattr(self.system, 'gnn_manager') and self.system.gnn_manager:
                    gnn_mgr = self.system.gnn_manager
                    stats["gnn_status"] = {
                        "available": True,
                        "models_trained": len(gnn_mgr.models) if hasattr(gnn_mgr, 'models') else 0,
                        "device": str(getattr(gnn_mgr, 'device', 'cpu'))
                    }
                else:
                    stats["gnn_status"] = {
                        "available": False,
                        "message": "GNN manager not initialized"
                    }
            except Exception as e:
                stats["gnn_status"] = {
                    "available": False,
                    "error": str(e)
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def visualize_full_graph(self, max_nodes: int = 100, layout: str = "spring") -> str:
        """
        Visualize the entire knowledge graph.

        Args:
            max_nodes: Maximum number of nodes to display
            layout: Layout algorithm (spring, circular, hierarchical)

        Returns:
            HTML string for visualization
        """
        try:
            from src.graphrag.visualization.enhanced_viz import EnhancedGraphVisualizer

            # Get all nodes (limited)
            all_nodes = self.system.graph.get_all_nodes()
            if len(all_nodes) > max_nodes:
                all_nodes = all_nodes[:max_nodes]
                logger.info(f"Limiting visualization to {max_nodes} nodes")

            # Get all edges
            all_edges = self.system.graph.get_all_relationships()

            # Filter edges to only include nodes in our sample
            node_ids = {node.get('id') for node in all_nodes}
            filtered_edges = [
                edge for edge in all_edges
                if edge.get('source') in node_ids and edge.get('target') in node_ids
            ]

            # Create visualization
            visualizer = EnhancedGraphVisualizer()

            # Add nodes
            for node in all_nodes:
                node_type = node.get('type', 'Unknown')
                label = node.get('name', node.get('title', node.get('id', 'Unknown')))
                visualizer.add_node(
                    node['id'],
                    label=label,
                    node_type=node_type,
                    properties=node
                )

            # Add edges
            for edge in filtered_edges:
                visualizer.add_edge(
                    edge['source'],
                    edge['target'],
                    relationship_type=edge.get('type', 'RELATED'),
                    properties=edge
                )

            # Generate HTML
            html = visualizer.generate_html(
                title=f"Knowledge Graph ({len(all_nodes)} nodes, {len(filtered_edges)} edges)",
                physics_enabled=True
            )

            return html

        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return f"<html><body><h2>Error</h2><p>{str(e)}</p></body></html>"

    def get_node_details(self, node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node."""
        try:
            node = self.system.graph.get_node(node_id)
            if not node:
                return {"status": "error", "message": "Node not found"}

            # Get neighbors
            neighbors = self.system.graph.get_neighbors(node_id)

            return {
                "status": "success",
                "node": node,
                "neighbors": neighbors,
                "neighbor_count": len(neighbors) if neighbors else 0
            }

        except Exception as e:
            logger.error(f"Error getting node details: {e}")
            return {"status": "error", "message": str(e)}

    def train_gnn_model(
        self,
        model_type: str,
        task: str,
        epochs: int = 50,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """
        Train a GNN model.

        Args:
            model_type: Type of GNN (gcn, gat, transformer, hetero)
            task: Task type (node_classification, link_prediction, embedding)
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training results and metrics
        """
        try:
            if not hasattr(self.system, 'gnn_manager') or not self.system.gnn_manager:
                return {
                    "status": "error",
                    "message": "GNN manager not available. Install PyTorch Geometric."
                }

            gnn_mgr = self.system.gnn_manager

            # Prepare training data
            logger.info(f"Preparing data for {model_type} model, task: {task}")

            # Convert graph to PyTorch Geometric format
            if hasattr(self.system, 'gnn_data_pipeline'):
                graph_data = self.system.gnn_data_pipeline.build_pyg_graph()
            else:
                return {
                    "status": "error",
                    "message": "GNN data pipeline not available"
                }

            # Train model
            logger.info(f"Training {model_type} for {epochs} epochs...")
            results = gnn_mgr.train_model(
                model_type=model_type,
                task=task,
                data=graph_data,
                epochs=epochs,
                lr=learning_rate
            )

            return {
                "status": "success",
                "message": f"Model trained successfully",
                "metrics": results
            }

        except Exception as e:
            logger.error(f"Error training GNN model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_gnn_predictions(
        self,
        prediction_type: str,
        node_id: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get GNN predictions.

        Args:
            prediction_type: Type (node_classification, link_prediction, similar_nodes)
            node_id: Optional node ID for focused predictions
            top_k: Number of top predictions to return

        Returns:
            Predictions with scores
        """
        try:
            if not hasattr(self.system, 'gnn_manager') or not self.system.gnn_manager:
                return {
                    "status": "error",
                    "message": "GNN manager not available"
                }

            gnn_mgr = self.system.gnn_manager

            if prediction_type == "link_prediction" and node_id:
                # Predict likely connections for this node
                predictions = gnn_mgr.predict_links(node_id, top_k=top_k)
                return {
                    "status": "success",
                    "predictions": predictions
                }

            elif prediction_type == "node_classification" and node_id:
                # Classify node type/category
                predictions = gnn_mgr.classify_node(node_id)
                return {
                    "status": "success",
                    "predictions": predictions
                }

            elif prediction_type == "similar_nodes" and node_id:
                # Find similar nodes using GNN embeddings
                predictions = gnn_mgr.find_similar_nodes(node_id, top_k=top_k)
                return {
                    "status": "success",
                    "predictions": predictions
                }

            else:
                return {
                    "status": "error",
                    "message": f"Invalid prediction type or missing node_id"
                }

        except Exception as e:
            logger.error(f"Error getting GNN predictions: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def visualize_gnn_attention(
        self,
        node_id: str,
        model_type: str = "gat"
    ) -> str:
        """
        Visualize GNN attention weights for a node.

        Args:
            node_id: Node to visualize
            model_type: GNN model type with attention

        Returns:
            HTML visualization
        """
        try:
            if not hasattr(self.system, 'gnn_explainer'):
                return "<html><body><h2>GNN Explainer not available</h2></body></html>"

            explainer = self.system.gnn_explainer

            # Get attention visualization
            html = explainer.visualize_attention(node_id, model_type=model_type)

            return html

        except Exception as e:
            logger.error(f"Error visualizing attention: {e}")
            return f"<html><body><h2>Error</h2><p>{str(e)}</p></body></html>"

    def export_graph_data(self, format: str = "json") -> str:
        """
        Export graph data.

        Args:
            format: Export format (json, csv, graphml)

        Returns:
            Exported data as string
        """
        try:
            all_nodes = self.system.graph.get_all_nodes()
            all_edges = self.system.graph.get_all_relationships()

            if format == "json":
                data = {
                    "nodes": all_nodes,
                    "edges": all_edges,
                    "metadata": {
                        "node_count": len(all_nodes),
                        "edge_count": len(all_edges)
                    }
                }
                return json.dumps(data, indent=2)

            elif format == "csv":
                # Simple CSV export
                csv_data = "type,id,properties\n"
                for node in all_nodes:
                    props = json.dumps(node).replace(',', ';')
                    csv_data += f"node,{node['id']},{props}\n"
                for edge in all_edges:
                    props = json.dumps(edge).replace(',', ';')
                    csv_data += f"edge,{edge['source']}->{edge['target']},{props}\n"
                return csv_data

            else:
                return f"Format {format} not supported yet"

        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            return f"Error: {str(e)}"
