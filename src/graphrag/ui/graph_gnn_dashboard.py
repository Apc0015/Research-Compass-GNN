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
from datetime import datetime

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

            # Get basic graph stats using the correct API
            if hasattr(self.system.graph, 'get_graph_stats'):
                graph_stats = self.system.graph.get_graph_stats()
                stats["graph_size"] = {
                    "total_nodes": graph_stats.get("node_count", 0),
                    "total_edges": graph_stats.get("relationship_count", 0)
                }

            # Query Neo4j or NetworkX for node type counts
            try:
                if self.system.graph._use_neo4j:
                    # Neo4j query for node labels
                    with self.system.graph.driver.session() as session:
                        result = session.run("""
                            CALL db.labels() YIELD label
                            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                            YIELD value
                            RETURN label, value.count as count
                        """)
                        node_counts = {record["label"]: record["count"] for record in result}

                        # Fallback if APOC not available
                        if not node_counts:
                            result = session.run("""
                                MATCH (n)
                                RETURN labels(n)[0] as label, count(n) as count
                                """)
                            node_counts = {record["label"]: record["count"] for record in result if record["label"]}

                        stats["node_types"] = node_counts
                else:
                    # NetworkX: count by 'type' property
                    node_counts = {}
                    for node_id in self.system.graph._graph.nodes():
                        props = self.system.graph._node_props.get(node_id, {})
                        node_type = props.get('type', 'Unknown')
                        node_counts[node_type] = node_counts.get(node_type, 0) + 1
                    stats["node_types"] = node_counts
            except Exception as e:
                logger.warning(f"Could not get node type counts: {e}")
                stats["node_types"] = {"error": str(e)}

            # Query for edge type counts
            try:
                if self.system.graph._use_neo4j:
                    with self.system.graph.driver.session() as session:
                        result = session.run("""
                            MATCH ()-[r]->()
                            RETURN type(r) as rel_type, count(r) as count
                        """)
                        edge_counts = {record["rel_type"]: record["count"] for record in result}
                        stats["edge_types"] = edge_counts
                else:
                    # NetworkX: count by edge type attribute
                    edge_counts = {}
                    for u, v, data in self.system.graph._graph.edges(data=True):
                        edge_type = data.get('type', 'RELATED')
                        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
                    stats["edge_types"] = edge_counts
            except Exception as e:
                logger.warning(f"Could not get edge type counts: {e}")
                stats["edge_types"] = {"error": str(e)}

            # Get GNN status
            try:
                if hasattr(self.system, 'gnn_manager') and self.system.gnn_manager:
                    gnn_mgr = self.system.gnn_manager
                    stats["gnn_status"] = {
                        "available": True,
                        "models_trained": len(getattr(gnn_mgr, 'models', {})),
                        "device": str(getattr(gnn_mgr, 'device', 'cpu'))
                    }
                else:
                    stats["gnn_status"] = {
                        "available": False,
                        "message": "GNN manager not initialized. Install PyTorch Geometric."
                    }
            except Exception as e:
                stats["gnn_status"] = {
                    "available": False,
                    "error": str(e)
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            import traceback
            traceback.print_exc()
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
            from pyvis.network import Network

            # Create network
            net = Network(height="750px", width="100%", directed=True)
            net.barnes_hut()

            nodes_data = []
            edges_data = []

            # Query nodes from graph
            if self.system.graph._use_neo4j:
                with self.system.graph.driver.session() as session:
                    # Get nodes
                    result = session.run(f"""
                        MATCH (n)
                        RETURN id(n) as id, labels(n) as labels, properties(n) as props
                        LIMIT {max_nodes}
                    """)
                    for record in result:
                        node_id = str(record["id"])
                        labels = record["labels"]
                        props = record["props"]
                        nodes_data.append({
                            'id': node_id,
                            'label': props.get('name', props.get('title', node_id[:20])),
                            'type': labels[0] if labels else 'Unknown',
                            'props': props
                        })

                    # Get edges between these nodes
                    result = session.run(f"""
                        MATCH (n)-[r]->(m)
                        WHERE id(n) IN [node.id for node in $nodes]
                        AND id(m) IN [node.id for node in $nodes]
                        RETURN id(n) as source, id(m) as target, type(r) as rel_type
                        LIMIT {max_nodes * 2}
                    """, nodes=[{'id': int(n['id'])} for n in nodes_data])
                    for record in result:
                        edges_data.append({
                            'source': str(record["source"]),
                            'target': str(record["target"]),
                            'type': record["rel_type"]
                        })
            else:
                # NetworkX fallback
                count = 0
                for node_id in self.system.graph._graph.nodes():
                    if count >= max_nodes:
                        break
                    props = self.system.graph._node_props.get(node_id, {})
                    nodes_data.append({
                        'id': str(node_id),
                        'label': props.get('name', props.get('text', str(node_id)[:20])),
                        'type': props.get('type', 'Unknown'),
                        'props': props
                    })
                    count += 1

                # Get edges
                node_ids_set = {n['id'] for n in nodes_data}
                for u, v, data in self.system.graph._graph.edges(data=True):
                    if str(u) in node_ids_set and str(v) in node_ids_set:
                        edges_data.append({
                            'source': str(u),
                            'target': str(v),
                            'type': data.get('type', 'RELATED')
                        })

            # Add nodes to visualization
            color_map = {
                'Paper': '#3498db',
                'Author': '#2ecc71',
                'Topic': '#f39c12',
                'Venue': '#9b59b6',
                'Entity': '#e74c3c',
                'Unknown': '#95a5a6'
            }

            for node in nodes_data:
                color = color_map.get(node['type'], '#95a5a6')
                net.add_node(
                    node['id'],
                    label=node['label'],
                    color=color,
                    title=f"{node['type']}: {node['label']}"
                )

            # Add edges
            for edge in edges_data:
                net.add_edge(edge['source'], edge['target'], title=edge['type'])

            # Generate HTML
            html = net.generate_html()

            return html

        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            import traceback
            traceback.print_exc()
            return f"<html><body><h2>Error</h2><p>{str(e)}</p><pre>{traceback.format_exc()}</pre></body></html>"

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
        learning_rate: float = 0.01,
        progress=None
    ) -> Dict[str, Any]:
        """
        Train a GNN model with comprehensive validation, error handling, and progress reporting.

        Args:
            model_type: Type of GNN (gcn, gat, transformer, hetero)
            task: Task type (node_classification, link_prediction, embedding)
            epochs: Number of training epochs
            learning_rate: Learning rate
            progress: Gradio progress object for real-time updates

        Returns:
            Training results and metrics
        """
        try:
            # STEP 1: Check GNN dependencies
            from src.graphrag.ml.gnn_utils import (
                check_gnn_dependencies,
                validate_graph_for_training,
                get_user_friendly_error,
                estimate_training_time
            )

            deps_ok, deps_msg = check_gnn_dependencies()
            if not deps_ok:
                return {
                    "status": "error",
                    "message": deps_msg
                }

            # STEP 2: Check if GNN manager is available or initialize it
            try:
                if hasattr(self.system, 'gnn_manager'):
                    gnn_mgr = self.system.gnn_manager
                else:
                    # Check if Neo4j is available
                    if not hasattr(self.system.graph, 'uri'):
                        return {
                            "status": "error",
                            "message": """
❌ GNN requires Neo4j database connection.

Current setup: NetworkX (in-memory graph)

💡 To use GNN features:
1. Install Neo4j (https://neo4j.com/download/)
2. Configure connection in Settings tab
3. Rebuild your knowledge graph with Neo4j

Note: NetworkX fallback support coming soon!
                            """
                        }

                    # Try to initialize GNN manager
                    from src.graphrag.ml.gnn_manager import GNNManager
                    gnn_mgr = GNNManager(
                        uri=self.system.graph.uri,
                        user=self.system.graph.user,
                        password=self.system.graph.password
                    )
            except ImportError as e:
                deps_ok, deps_msg = check_gnn_dependencies()
                return {
                    "status": "error",
                    "message": deps_msg
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"❌ Failed to initialize GNN manager: {str(e)}\n\n💡 Check Neo4j connection in Settings."
                }

            # STEP 3: Prepare training data - convert graph to PyG format
            logger.info(f"Converting graph to PyTorch Geometric format...")

            try:
                from src.graphrag.ml.graph_converter import Neo4jToTorchGeometric

                # Pass Neo4j credentials from graph manager
                converter = Neo4jToTorchGeometric(
                    uri=self.system.graph.uri,
                    user=self.system.graph.user,
                    password=self.system.graph.password
                )

                # Export graph
                graph_data = converter.export_graph_to_pyg(use_cache=False)

                if graph_data is None or graph_data.num_nodes == 0:
                    converter.close()
                    return {
                        "status": "error",
                        "message": "❌ No graph data available.\n\n💡 Upload and process documents first."
                    }

                # STEP 4: Validate graph data before training
                valid, validation_msg = validate_graph_for_training(graph_data)
                if not valid:
                    converter.close()
                    return {
                        "status": "error",
                        "message": validation_msg
                    }

                # Add train/val/test splits for training
                graph_data = converter.create_train_val_test_split(graph_data)

                # Estimate training time
                time_estimate = estimate_training_time(
                    graph_data.num_nodes,
                    graph_data.num_edges,
                    epochs,
                    model_type
                )

                # Close converter connection
                converter.close()

                logger.info(f"✓ Graph validation passed. Estimated training time: {time_estimate}")

            except Exception as e:
                logger.error(f"Graph conversion failed: {e}")
                error_msg = get_user_friendly_error(e)
                return {
                    "status": "error",
                    "message": error_msg
                }

            # STEP 5: Train model with comprehensive error handling and progress tracking
            logger.info(f"Training {model_type} for {task} task ({epochs} epochs)...")
            logger.info(f"Estimated training time: {time_estimate}")

            # Create progress callback for Gradio
            def training_progress_callback(current_epoch, total_epochs, metrics):
                """Update Gradio progress bar during training"""
                if progress:
                    progress_pct = current_epoch / total_epochs
                    desc = f"Epoch {current_epoch}/{total_epochs} - Loss: {metrics.get('train_loss', 0):.4f}"
                    progress(progress_pct, desc=desc)

            # Show initial progress
            if progress:
                progress(0, desc="Initializing training...")

            try:
                results = gnn_mgr.train(
                    data=graph_data,
                    model_type=model_type,
                    task=task,
                    epochs=epochs,
                    lr=learning_rate,
                    progress_callback=training_progress_callback
                )

                success_msg = f"""
✅ {model_type.upper()} model trained successfully!

📊 Training Results:
- Task: {task}
- Epochs: {epochs}
- Nodes: {graph_data.num_nodes}
- Edges: {graph_data.num_edges}
- Training Time: {time_estimate}

💡 Next Steps:
1. Go to "GNN Predictions" tab to test your model
2. Try different prediction types
3. Evaluate model performance on your data
                """

                return {
                    "status": "success",
                    "message": success_msg.strip(),
                    "metrics": results
                }

            except RuntimeError as e:
                logger.error(f"Training failed with RuntimeError: {e}")
                error_msg = get_user_friendly_error(e)
                return {
                    "status": "error",
                    "message": error_msg
                }

            except Exception as e:
                logger.error(f"Training failed: {e}")
                import traceback
                traceback.print_exc()
                error_msg = get_user_friendly_error(e)
                return {
                    "status": "error",
                    "message": error_msg
                }

        except Exception as e:
            logger.error(f"Error in train_gnn_model: {e}")
            import traceback
            traceback.print_exc()
            error_msg = get_user_friendly_error(e)
            return {
                "status": "error",
                "message": error_msg
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
            # Validate inputs
            if not node_id or not node_id.strip():
                return {
                    "status": "error",
                    "message": "Please provide a node ID (paper title or ID)"
                }

            # Check if GNN manager is available
            try:
                if hasattr(self.system, 'gnn_manager'):
                    gnn_mgr = self.system.gnn_manager
                else:
                    from src.graphrag.ml.gnn_manager import GNNManager
                    gnn_mgr = GNNManager(self.system.graph)
            except ImportError as e:
                logger.warning(f"GNN Manager not available: {e}")
                return {
                    "status": "error",
                    "message": "PyTorch Geometric not installed. Install with: pip install torch torch-geometric"
                }

            # Make sure model is trained
            if not hasattr(gnn_mgr, 'models') or not gnn_mgr.models:
                return {
                    "status": "error",
                    "message": "No trained GNN models available. Train a model first in the 'Train GNN Models' tab."
                }

            logger.info(f"Getting {prediction_type} predictions for node: {node_id}")

            try:
                if prediction_type == "link_prediction":
                    # Predict likely connections
                    try:
                        predictions = gnn_mgr.predict_links(node_id, top_k=top_k)
                        return {
                            "status": "success",
                            "prediction_type": "link_prediction",
                            "node_id": node_id,
                            "predictions": predictions
                        }
                    except AttributeError:
                        # Fallback: use link predictor if available
                        if hasattr(self.system, 'link_predictor'):
                            predictions = self.system.link_predictor.predict(node_id, top_k=top_k)
                            return {
                                "status": "success",
                                "predictions": predictions
                            }
                        else:
                            return {
                                "status": "error",
                                "message": "Link prediction model not available. Train a link_prediction model first."
                            }

                elif prediction_type == "node_classification":
                    # Classify node
                    try:
                        predictions = gnn_mgr.classify_node(node_id)
                        return {
                            "status": "success",
                            "prediction_type": "node_classification",
                            "node_id": node_id,
                            "predictions": predictions
                        }
                    except AttributeError:
                        return {
                            "status": "error",
                            "message": "Node classification model not available. Train a node_classification model first."
                        }

                elif prediction_type == "similar_nodes":
                    # Find similar nodes
                    try:
                        predictions = gnn_mgr.find_similar_nodes(node_id, top_k=top_k)
                        return {
                            "status": "success",
                            "prediction_type": "similar_nodes",
                            "node_id": node_id,
                            "predictions": predictions
                        }
                    except AttributeError:
                        # Fallback: use graph-based similarity
                        return {
                            "status": "error",
                            "message": "Similarity model not available. Train an embedding model first."
                        }

                else:
                    return {
                        "status": "error",
                        "message": f"Unknown prediction type: {prediction_type}. Use: link_prediction, node_classification, or similar_nodes"
                    }

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": f"Prediction failed: {str(e)}"
                }

        except Exception as e:
            logger.error(f"Error in get_gnn_predictions: {e}")
            import traceback
            traceback.print_exc()
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
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        try:
            nodes_data = []
            edges_data = []

            # Query nodes and edges
            if self.system.graph._use_neo4j:
                with self.system.graph.driver.session() as session:
                    # Get nodes
                    result = session.run("""
                        MATCH (n)
                        RETURN id(n) as id, labels(n) as labels, properties(n) as props
                        LIMIT 10000
                    """)
                    for record in result:
                        nodes_data.append({
                            'id': str(record["id"]),
                            'labels': record["labels"],
                            'properties': dict(record["props"])
                        })

                    # Get edges
                    result = session.run("""
                        MATCH (n)-[r]->(m)
                        RETURN id(n) as source, id(m) as target,
                               type(r) as rel_type, properties(r) as props
                        LIMIT 10000
                    """)
                    for record in result:
                        edges_data.append({
                            'source': str(record["source"]),
                            'target': str(record["target"]),
                            'type': record["rel_type"],
                            'properties': dict(record["props"])
                        })
            else:
                # NetworkX fallback
                for node_id in self.system.graph._graph.nodes():
                    props = self.system.graph._node_props.get(node_id, {})
                    nodes_data.append({
                        'id': str(node_id),
                        'properties': props
                    })

                for u, v, data in self.system.graph._graph.edges(data=True):
                    edges_data.append({
                        'source': str(u),
                        'target': str(v),
                        'type': data.get('type', 'RELATED'),
                        'properties': data
                    })

            if format == "json":
                data = {
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "metadata": {
                        "node_count": len(nodes_data),
                        "edge_count": len(edges_data),
                        "exported_at": json.dumps(datetime.now().isoformat())
                    }
                }
                return json.dumps(data, indent=2, default=str)

            elif format == "csv":
                # CSV export
                csv_lines = []
                csv_lines.append("type,id,label,properties")

                for node in nodes_data:
                    node_id = node['id']
                    label = node.get('labels', ['Unknown'])[0] if 'labels' in node else node.get('properties', {}).get('type', 'Unknown')
                    props = json.dumps(node.get('properties', {})).replace('"', '""')
                    csv_lines.append(f'node,{node_id},{label},"{props}"')

                csv_lines.append("\ntype,source,target,rel_type,properties")
                for edge in edges_data:
                    source = edge['source']
                    target = edge['target']
                    rel_type = edge.get('type', 'RELATED')
                    props = json.dumps(edge.get('properties', {})).replace('"', '""')
                    csv_lines.append(f'edge,{source},{target},{rel_type},"{props}"')

                return "\n".join(csv_lines)

            else:
                return f"Format '{format}' not supported. Use 'json' or 'csv'."

        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}\n\n{traceback.format_exc()}"

    def export_gnn_models(
        self,
        formats: str = "torchscript,onnx"
    ) -> Dict[str, Any]:
        """
        Export trained GNN models for deployment

        Args:
            formats: Comma-separated export formats

        Returns:
            Export result dict
        """
        try:
            # Get GNN manager
            if hasattr(self.system, 'gnn_manager'):
                gnn_mgr = self.system.gnn_manager
            else:
                return {
                    "status": "error",
                    "message": "GNN manager not available. Train a model first."
                }

            # Parse formats
            format_list = [f.strip() for f in formats.split(',')]

            # Export models
            results = gnn_mgr.export_models(
                output_dir="exports/gnn_models",
                formats=format_list
            )

            # Format success message
            message_lines = ["✅ Models exported successfully!\n"]
            for model_name, model_results in results.items():
                message_lines.append(f"\n**{model_name}:**")
                for fmt, fmt_results in model_results.items():
                    if fmt_results.get('status') == 'success':
                        path = fmt_results.get('model_path', '')
                        size = fmt_results.get('file_size_mb', 0)
                        message_lines.append(f"  - {fmt.upper()}: {path} ({size:.2f} MB)")

            return {
                "status": "success",
                "message": "\n".join(message_lines),
                "results": results
            }

        except Exception as e:
            logger.error(f"Model export failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"❌ Export failed: {str(e)}"
            }

    def generate_gnn_report(self) -> Dict[str, Any]:
        """
        Generate performance report with visualizations

        Returns:
            Report generation result
        """
        try:
            # Get GNN manager
            if hasattr(self.system, 'gnn_manager'):
                gnn_mgr = self.system.gnn_manager
            else:
                return {
                    "status": "error",
                    "message": "GNN manager not available. Train a model first."
                }

            # Generate reports
            reports = gnn_mgr.generate_performance_report(
                output_dir="reports/gnn"
            )

            # Format success message
            message_lines = ["✅ Performance reports generated!\n"]
            for model_name, model_reports in reports.items():
                message_lines.append(f"\n**{model_name}:**")
                for report_type, report_path in model_reports.items():
                    message_lines.append(f"  - {report_type}: {report_path}")

            return {
                "status": "success",
                "message": "\n".join(message_lines),
                "reports": reports
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"❌ Report generation failed: {str(e)}"
            }

    def batch_predict_nodes(
        self,
        node_ids: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Batch prediction for multiple nodes

        Args:
            node_ids: Comma-separated node IDs
            top_k: Top predictions per node

        Returns:
            Batch prediction results
        """
        try:
            # Get GNN manager
            if hasattr(self.system, 'gnn_manager'):
                gnn_mgr = self.system.gnn_manager
            else:
                return {
                    "status": "error",
                    "message": "GNN manager not available"
                }

            # Create batch predictor
            predictor = gnn_mgr.create_batch_predictor(batch_size=32)

            # Parse node IDs
            node_id_list = [nid.strip() for nid in node_ids.split(',') if nid.strip()]

            if not node_id_list:
                return {
                    "status": "error",
                    "message": "No valid node IDs provided"
                }

            # Convert node IDs to indices
            # This would need to be implemented based on your node ID mapping
            # For now, assume node IDs are indices
            try:
                node_indices = [int(nid) for nid in node_id_list]
            except ValueError:
                return {
                    "status": "error",
                    "message": "Node IDs must be numeric indices"
                }

            # Get graph data
            if not gnn_mgr.graph_data:
                return {
                    "status": "error",
                    "message": "No graph data available"
                }

            # Batch predict
            predictions = predictor.predict_nodes_batch(
                gnn_mgr.graph_data.x,
                gnn_mgr.graph_data.edge_index,
                node_indices,
                top_k=top_k
            )

            return {
                "status": "success",
                "num_predictions": len(predictions),
                "predictions": predictions,
                "cache_stats": predictor.get_cache_stats()
            }

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"❌ Batch prediction failed: {str(e)}"
            }
