#!/usr/bin/env python3
"""
GNN Data Pipeline - GNN-first data processing for Research Compass
Converts documents and research data into graph representations optimized for GNN processing.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
import torch
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

from .document_processor import DocumentProcessor
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from ..ml.graph_converter import Neo4jToTorchGeometric


class GraphBuilder:
    """
    Builds heterogeneous graphs from extracted entities and relationships.
    Optimized for GNN processing with proper node and edge features.
    """
    
    def __init__(self):
        self.node_counter = 0
        self.edge_counter = 0
        self.node_id_map = {}
        self.node_types = {}
        self.node_features = {}
        
    def build_heterogeneous_graph(
        self,
        entities: Dict[str, List[Dict]],
        relationships: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Build heterogeneous graph from entities and relationships.
        
        Args:
            entities: Dictionary of entity lists by type
            relationships: Dictionary of relationship lists by type
            
        Returns:
            Graph data structure optimized for GNN processing
        """
        graph_data = {
            'nodes': {},
            'edges': {},
            'node_types': [],
            'edge_types': [],
            'features': {}
        }
        
        # Process nodes
        for node_type, entity_list in entities.items():
            node_ids = []
            node_features = []
            
            for entity in entity_list:
                node_id = self._get_or_create_node_id(entity['id'], node_type)
                node_ids.append(node_id)
                
                # Extract features for GNN
                features = self._extract_node_features(entity)
                node_features.append(features)
            
            graph_data['nodes'][node_type] = node_ids
            graph_data['features'][node_type] = np.array(node_features)
        
        # Process edges
        for edge_type, rel_list in relationships.items():
            edge_indices = []
            edge_features = []
            
            for rel in rel_list:
                source_id = self._get_node_id(rel['source'])
                target_id = self._get_node_id(rel['target'])
                
                if source_id is not None and target_id is not None:
                    edge_indices.append([source_id, target_id])
                    
                    # Extract edge features
                    features = self._extract_edge_features(rel)
                    edge_features.append(features)
            
            if edge_indices:
                graph_data['edges'][edge_type] = np.array(edge_indices)
                graph_data['edge_features'][edge_type] = np.array(edge_features)
        
        return graph_data
    
    def _get_or_create_node_id(self, entity_id: str, node_type: str) -> int:
        """Get or create node ID for entity."""
        if entity_id not in self.node_id_map:
            self.node_id_map[entity_id] = self.node_counter
            self.node_types[self.node_counter] = node_type
            self.node_counter += 1
        
        return self.node_id_map[entity_id]
    
    def _get_node_id(self, entity_id: str) -> Optional[int]:
        """Get node ID for entity."""
        return self.node_id_map.get(entity_id)
    
    def _extract_node_features(self, entity: Dict) -> np.ndarray:
        """Extract features from entity for GNN processing."""
        features = []
        
        # Text features
        if 'title' in entity:
            features.extend(self._text_to_features(entity['title']))
        if 'abstract' in entity:
            features.extend(self._text_to_features(entity['abstract'][:100]))  # Truncate
        
        # Metadata features
        if 'year' in entity:
            features.append(float(entity['year']) / 2025.0)  # Normalize year
        
        if 'venue' in entity:
            features.append(hash(entity['venue']) % 1000 / 1000.0)
        
        # Categorical features
        if 'type' in entity:
            features.append(hash(entity['type']) % 100 / 100.0)
        
        # Pad to fixed size
        while len(features) < 384:  # Standard embedding size
            features.append(0.0)
        
        return np.array(features[:384])
    
    def _extract_edge_features(self, relationship: Dict) -> np.ndarray:
        """Extract features from relationship for GNN processing."""
        features = []
        
        # Relationship type
        if 'type' in relationship:
            features.append(hash(relationship['type']) % 100 / 100.0)
        
        # Weight/strength
        if 'weight' in relationship:
            features.append(float(relationship['weight']))
        else:
            features.append(1.0)
        
        # Temporal features
        if 'timestamp' in relationship:
            features.append(float(relationship['timestamp']) / 2025.0)
        
        # Pad to fixed size
        while len(features) < 64:  # Edge feature size
            features.append(0.0)
        
        return np.array(features[:64])
    
    def _text_to_features(self, text: str) -> List[float]:
        """Convert text to numerical features."""
        if not text:
            return [0.0] * 100
        
        # Simple text features (in practice, use embeddings)
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 200.0)
        
        # Character-level features
        features.append(text.count('.') / 50.0)
        features.append(text.count(',') / 100.0)
        
        # Word-level features
        words = text.lower().split()
        features.append(len(set(words)) / max(len(words), 1))
        
        # Keyword features
        keywords = ['method', 'result', 'conclusion', 'analysis', 'study']
        for keyword in keywords:
            features.append(text.lower().count(keyword) / 10.0)
        
        # Pad to fixed size
        while len(features) < 100:
            features.append(0.0)
        
        return features[:100]


class GraphFeatureExtractor:
    """
    Extracts and enhances features for GNN processing.
    Optimizes node and edge features for better GNN performance.
    """
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_features(
        self,
        graph_data: Dict,
        documents: List[str]
    ) -> Dict:
        """
        Extract and enhance features for GNN processing.
        
        Args:
            graph_data: Graph structure
            documents: Original documents for context
            
        Returns:
            Enhanced feature dictionary
        """
        enhanced_features = {}
        
        # Process each node type
        for node_type, node_ids in graph_data['nodes'].items():
            base_features = graph_data['features'][node_type]
            
            # Enhance features with document context
            enhanced = self._enhance_node_features(
                base_features, node_type, documents
            )
            
            # Add structural features
            structural = self._add_structural_features(
                enhanced, graph_data, node_type
            )
            
            enhanced_features[node_type] = structural
        
        # Process edge features
        for edge_type in graph_data.get('edges', {}):
            edge_features = graph_data.get('edge_features', {}).get(edge_type)
            if edge_features is not None:
                enhanced_edge_features = self._enhance_edge_features(
                    edge_features, edge_type, graph_data
                )
                enhanced_features[f'edge_{edge_type}'] = enhanced_edge_features
        
        return enhanced_features
    
    def _enhance_node_features(
        self,
        base_features: np.ndarray,
        node_type: str,
        documents: List[str]
    ) -> np.ndarray:
        """Enhance node features with document context."""
        enhanced = base_features.copy()
        
        # Add document-level statistics
        if documents:
            doc_stats = self._compute_document_statistics(documents)
            for stat_value in doc_stats:
                enhanced = np.pad(enhanced, (0, 1), 'constant')[:384]
                enhanced[-1] = stat_value
        
        return enhanced
    
    def _add_structural_features(
        self,
        features: np.ndarray,
        graph_data: Dict,
        node_type: str
    ) -> np.ndarray:
        """Add structural graph features."""
        # Compute degree centrality
        if node_type in graph_data['nodes']:
            node_ids = graph_data['nodes'][node_type]
            degrees = self._compute_degrees(graph_data, node_ids)
            
            # Add degree information to features
            if len(degrees) == features.shape[0]:
                enhanced = np.zeros((features.shape[0], features.shape[1] + 1))
                enhanced[:, :-1] = features
                enhanced[:, -1] = degrees
                return enhanced
            else:
                # Handle mismatched dimensions
                try:
                    if len(features.shape) >= 2:
                        enhanced = np.zeros((features.shape[0], features.shape[1] + 1))
                        enhanced[:, :-1] = features
                    else:
                        # Handle 1D features
                        enhanced = np.zeros((features.shape[0], 1))
                        enhanced[:, 0] = features
                except (IndexError, AttributeError):
                    # Handle case where features is not properly shaped
                    enhanced = np.zeros((len(degrees), 1))
                # Pad or truncate degrees to match
                if len(degrees) > features.shape[0]:
                    degrees = degrees[:features.shape[0]]
                elif len(degrees) < features.shape[0]:
                    degrees = np.pad(degrees, (0, features.shape[0] - len(degrees)), 'constant')
                else:
                    degrees = degrees
                
                enhanced[:, -1] = degrees
                return enhanced
        
        return features
    
    def _enhance_edge_features(
        self,
        edge_features: np.ndarray,
        edge_type: str,
        graph_data: Dict
    ) -> np.ndarray:
        """Enhance edge features with structural information."""
        enhanced = edge_features.copy()
        
        # Add edge type encoding
        type_encoding = hash(edge_type) % 100 / 100.0
        type_col = np.full((enhanced.shape[0], 1), type_encoding)
        
        enhanced = np.hstack([enhanced, type_col])
        
        return enhanced
    
    def _compute_document_statistics(self, documents: List[str]) -> List[float]:
        """Compute statistics from documents."""
        if not documents:
            return [0.0] * 5
        
        stats = []
        
        # Average document length
        doc_lengths = []
        all_words = []
        
        for doc in documents:
            if isinstance(doc, Path):
                try:
                    # Try to load the document
                    text = self.document_processor.load_document(doc)
                    doc_lengths.append(len(text))
                    all_words.extend(text.lower().split())
                except:
                    # Fallback to path string length
                    doc_lengths.append(len(str(doc)))
                    all_words.extend(str(doc).lower().split())
            else:
                doc_lengths.append(len(doc))
                all_words.extend(doc.lower().split())
        
        if doc_lengths:
            avg_length = np.mean(doc_lengths)
            stats.append(avg_length / 10000.0)
        else:
            stats.append(0.0)
        
        # Vocabulary diversity
        if all_words:
            vocab_diversity = len(set(all_words)) / len(all_words)
            stats.append(vocab_diversity)
        else:
            stats.append(0.0)
        
        # Technical term density
        tech_terms = ['algorithm', 'method', 'analysis', 'framework', 'approach']
        tech_density = sum(all_words.count(term) for term in tech_terms) / len(all_words)
        stats.append(tech_density)
        
        # Citation density (simplified)
        citation_pattern = r'\[\d+\]|\([A-Z][a-z]+ et al\., \d{4}\)'
        citation_count = 0
        for doc in documents:
            if isinstance(doc, Path):
                try:
                    text = self.document_processor.load_document(doc)
                    citation_count += len(re.findall(citation_pattern, text))
                except:
                    text = str(doc)
                    citation_count += len(re.findall(citation_pattern, text))
            else:
                citation_count += len(re.findall(citation_pattern, doc))
        
        if all_words:
            citation_density = citation_count / len(all_words)
        else:
            citation_density = 0.0
        stats.append(citation_density)
        
        # Structural complexity
        sentence_lengths = []
        for doc in documents:
            if isinstance(doc, Path):
                try:
                    text = self.document_processor.load_document(doc)
                    sentence_lengths.extend([len(sentence.split()) for sentence in text.split('.')])
                except:
                    text = str(doc)
                    sentence_lengths.extend([len(sentence.split()) for sentence in text.split('.')])
            else:
                sentence_lengths.extend([len(sentence.split()) for sentence in doc.split('.')])
        
        if sentence_lengths:
            avg_sentence_length = np.mean(sentence_lengths)
            stats.append(avg_sentence_length / 50.0)
        else:
            stats.append(0.0)
        
        return stats
    
    def _compute_degrees(self, graph_data: Dict, node_ids: List[int]) -> np.ndarray:
        """Compute degree for each node."""
        degrees = np.zeros(len(node_ids))
        
        for edge_type, edges in graph_data.get('edges', {}).items():
            if edges is not None:
                for i, node_id in enumerate(node_ids):
                    # Count edges connected to this node
                    degree = np.sum(edges == node_id)
                    degrees[i] += degree
        
        return degrees


class GNNDataPipeline:
    """
    GNN-first data processing pipeline.
    Converts documents and research data into optimized graph representations.
    """
    
    def __init__(self, gnn_core):
        """Initialize pipeline with GNN core system."""
        self.gnn_core = gnn_core
        self.graph_builder = GraphBuilder()
        self.feature_extractor = GraphFeatureExtractor()
        self.document_processor = DocumentProcessor()
        
        # Initialize extractors
        try:
            self.entity_extractor = EntityExtractor()
        except:
            self.entity_extractor = None
            
        try:
            self.relationship_extractor = RelationshipExtractor()
        except:
            self.relationship_extractor = None
    
    def process_documents_to_graph(
        self,
        documents: List[Union[str, Path]]
    ) -> Tuple[Dict, Dict]:
        """
        Convert documents to graph representation optimized for GNN.
        
        Args:
            documents: List of document texts or paths
            
        Returns:
            Tuple of (graph_data, features)
        """
        logger.info(f"Processing {len(documents)} documents for GNN pipeline")
        
        # 1. Extract entities and relationships
        entities, relationships = self._extract_graph_components(documents)
        
        # 2. Build heterogeneous graph
        graph_data = self.graph_builder.build_heterogeneous_graph(
            entities, relationships
        )
        
        # 3. Extract and enhance features
        features = self.feature_extractor.extract_features(
            graph_data, documents
        )
        
        # 4. Convert to PyTorch Geometric format
        pyg_data = self._convert_to_pyg_format(graph_data, features)
        
        # 5. Optimize for GNN processing
        optimized_data = self._optimize_for_gnn(pyg_data)
        
        return optimized_data, features
    
    def _extract_graph_components(
        self,
        documents: List[Union[str, Path]]
    ) -> Tuple[Dict, Dict]:
        """Extract entities and relationships from documents."""
        entities = {
            'papers': [],
            'authors': [],
            'venues': [],
            'topics': [],
            'methods': [],
            'datasets': []
        }
        
        relationships = {
            'cites': [],
            'authored_by': [],
            'published_in': [],
            'discusses': [],
            'uses_method': [],
            'evaluated_on': []
        }
        
        for i, doc in enumerate(documents):
            # Process document text
            if isinstance(doc, Path):
                try:
                    text = self.document_processor.load_document(doc)
                except:
                    text = str(doc)
            else:
                text = doc
            
            # Extract entities
            if self.entity_extractor:
                doc_entities = self.entity_extractor.extract_entities(text)
                if isinstance(doc_entities, dict):
                    for entity_type, entity_list in doc_entities.items():
                        if entity_type in entities:
                            entities[entity_type].extend(entity_list)
                elif isinstance(doc_entities, list):
                    # Handle case where entity extractor returns list of Entity objects
                    for entity in doc_entities:
                        if hasattr(entity, 'label') and hasattr(entity, 'text'):
                            # Convert Entity object to dictionary
                            entity_dict = {
                                'id': f"{entity.label}_{entity.text}_{i}",
                                'text': entity.text,
                                'type': entity.label,
                                'start': entity.start,
                                'end': entity.end
                            }
                            
                            # Categorize by entity type
                            entity_type = entity.label.lower()
                            if entity_type in ['person', 'org']:
                                entities['authors'].append(entity_dict)
                            elif entity_type in ['gpe', 'loc']:
                                entities['venues'].append(entity_dict)
                            elif entity_type in ['topic', 'concept']:
                                entities['topics'].append(entity_dict)
                            else:
                                entities['papers'].append(entity_dict)
                        else:
                            # Handle case where it's already a dictionary
                            entities['papers'].append(entity)
            else:
                # Fallback entity extraction
                self._fallback_entity_extraction(text, entities, i)
            
            # Extract relationships
            if self.relationship_extractor:
                doc_relationships = self.relationship_extractor.extract_relationships(text)
                for rel_type, rel_list in doc_relationships.items():
                    if rel_type in relationships:
                        relationships[rel_type].extend(rel_list)
            else:
                # Fallback relationship extraction
                self._fallback_relationship_extraction(text, relationships, i)
        
        return entities, relationships
    
    def _fallback_entity_extraction(self, text: str, entities: Dict, doc_idx: int):
        """Fallback entity extraction when specialized extractor unavailable."""
        # Simple pattern-based extraction
        lines = text.split('\n')
        
        # Extract title (first non-empty line)
        title = None
        for line in lines:
            if line.strip():
                title = line.strip()
                break
        
        if title:
            entities['papers'].append({
                'id': f'paper_{doc_idx}',
                'title': title,
                'type': 'paper'
            })
        
        # Extract authors (simplified pattern)
        author_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)'
        authors = re.findall(author_pattern, text[:1000])  # First 1000 chars
        for author in authors[:5]:  # Limit to 5 authors
            entities['authors'].append({
                'id': f'author_{author}_{doc_idx}',
                'name': author,
                'type': 'author'
            })
        
        # Extract topics (simplified)
        topic_keywords = ['machine learning', 'deep learning', 'neural network', 'algorithm', 'method']
        for keyword in topic_keywords:
            if keyword.lower() in text.lower():
                entities['topics'].append({
                    'id': f'topic_{keyword}_{doc_idx}',
                    'name': keyword,
                    'type': 'topic'
                })
    
    def _fallback_relationship_extraction(self, text: str, relationships: Dict, doc_idx: int):
        """Fallback relationship extraction when specialized extractor unavailable."""
        # Simple citation extraction
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, text)
        
        for citation in citations[:10]:  # Limit to 10 citations
            relationships['cites'].append({
                'source': f'paper_{doc_idx}',
                'target': f'paper_{citation}',
                'type': 'citation'
            })
        
        # Authorship relationships
        relationships['authored_by'].append({
            'source': f'paper_{doc_idx}',
            'target': f'author_unknown_{doc_idx}',
            'type': 'authorship'
        })
    
    def _convert_to_pyg_format(self, graph_data: Dict, features: Dict) -> 'Data':
        """Convert graph data to PyTorch Geometric format."""
        try:
            from torch_geometric.data import Data
        except ImportError:
            logger.error("torch_geometric not installed")
            return None
        
        # Combine all nodes
        all_node_features = []
        node_type_mapping = {}
        
        for node_type, node_ids in graph_data['nodes'].items():
            if node_type in features:
                node_features = features[node_type]
                all_node_features.append(node_features)
                
                # Track node type mapping
                for i, node_id in enumerate(node_ids):
                    node_type_mapping[node_id] = node_type
        
        # Concatenate all node features
        if all_node_features:
            x = torch.tensor(np.vstack(all_node_features), dtype=torch.float)
        else:
            x = torch.zeros((1, 384), dtype=torch.float)
        
        # Combine all edges
        all_edges = []
        for edge_type, edges in graph_data.get('edges', {}).items():
            if edges is not None:
                all_edges.extend(edges)
        
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Store metadata
        data.node_type_mapping = node_type_mapping
        data.graph_data = graph_data
        
        return data
    
    def _optimize_for_gnn(self, data: 'Data') -> 'Data':
        """Optimize graph data for GNN processing."""
        if data is None:
            return None
        
        # Ensure graph is connected
        if data.edge_index.size(1) == 0:
            # Add self-loops
            data.edge_index = torch.stack([
                torch.arange(data.num_nodes),
                torch.arange(data.num_nodes)
            ], dim=0)
        
        # Normalize features
        if data.x is not None:
            data.x = F.normalize(data.x, p=2, dim=1)
        
        # Add edge attributes if missing
        if not hasattr(data, 'edge_attr'):
            data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        
        return data
    
    def create_temporal_graphs(
        self,
        documents: List[Union[str, Path]],
        time_windows: List[int]
    ) -> List['Data']:
        """
        Create temporal graphs for different time windows.
        
        Args:
            documents: Documents with timestamps
            time_windows: List of years for temporal snapshots
            
        Returns:
            List of PyG Data objects for each time window
        """
        temporal_graphs = []
        
        for window in time_windows:
            # Filter documents for this time window
            window_docs = self._filter_documents_by_time(documents, window)
            
            # Process documents for this window
            graph_data, features = self.process_documents_to_graph(window_docs)
            
            # Add temporal information
            if graph_data is not None:
                graph_data.time_window = window
                temporal_graphs.append(graph_data)
        
        return temporal_graphs
    
    def _filter_documents_by_time(
        self,
        documents: List[Union[str, Path]],
        year: int
    ) -> List[Union[str, Path]]:
        """Filter documents by publication year."""
        # Simplified filtering - in practice, use document metadata
        window_size = len(documents) // 5  # Assume 5 time windows
        start_idx = min(year % 5 * window_size, len(documents))
        end_idx = min(start_idx + window_size, len(documents))
        
        return documents[start_idx:end_idx]
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline processing."""
        return {
            'graph_builder_nodes': self.graph_builder.node_counter,
            'graph_builder_edges': self.graph_builder.edge_counter,
            'entity_extractor_available': self.entity_extractor is not None,
            'relationship_extractor_available': self.relationship_extractor is not None,
            'feature_cache_size': len(self.feature_extractor.feature_cache)
        }


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Data Pipeline Test")
    print("=" * 80)
    
    # Mock GNN core for testing
    class MockGNNCore:
        pass
    
    # Initialize pipeline
    pipeline = GNNDataPipeline(MockGNNCore())
    
    # Test document processing
    test_documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
    ]
    
    # Process documents
    graph_data, features = pipeline.process_documents_to_graph(test_documents)
    
    if graph_data is not None:
        print(f"\nGraph Processing Results:")
        print(f"  Nodes: {graph_data.num_nodes}")
        print(f"  Edges: {graph_data.num_edges}")
        print(f"  Features: {graph_data.x.shape}")
        print(f"  Node types: {len(set(graph_data.node_type_mapping.values()))}")
    
    # Get pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ GNN Data Pipeline test complete")