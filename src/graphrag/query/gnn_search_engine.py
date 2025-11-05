#!/usr/bin/env python3
"""
GNN Search Engine - Graph-based search and retrieval system
Uses graph neural networks for intelligent document search and retrieval.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class GraphQueryEncoder(nn.Module):
    """
    Encodes search queries using graph neural networks.
    
    Transforms text queries into graph-aware embeddings
    for better semantic search performance.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Query Encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout probability
        """
        super(GraphQueryEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Query processing layers
        self.query_encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Graph attention for query-document interaction
        try:
            from torch_geometric.nn import GATConv
            self.graph_attention = GATConv(
                hidden_dim, hidden_dim, heads=4, dropout=dropout
            )
        except ImportError:
            # Fallback to linear layer
            self.graph_attention = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query_tokens: torch.Tensor,
        graph_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode query with graph context.
        
        Args:
            query_tokens: Tokenized query [batch_size, seq_len]
            graph_embeddings: Document graph embeddings [num_nodes, hidden_dim]
            edge_index: Graph edge indices
            
        Returns:
            Query embedding with graph context
        """
        # Word embeddings
        word_emb = self.word_embeddings(query_tokens)
        word_emb = self.dropout(word_emb)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.query_encoder(word_emb)
        query_emb = hidden[-1]  # Use last hidden state
        
        # Graph attention
        if hasattr(self.graph_attention, '__class__') and 'GAT' in str(self.graph_attention.__class__):
            # PyG GAT
            graph_context = self.graph_attention(graph_embeddings, edge_index)
            # Aggregate graph context
            graph_context = torch.mean(graph_context, dim=0)
        else:
            # Linear fallback
            graph_context = torch.mean(graph_embeddings, dim=0)
            graph_context = self.graph_attention(graph_context)
        
        # Combine query and graph context
        combined_emb = query_emb + 0.3 * graph_context
        
        # Final projection
        final_emb = self.query_projection(combined_emb)
        
        return final_emb


class GraphDocumentRetriever(nn.Module):
    """
    Retrieves documents using graph neural networks.
    
    Performs graph-aware document retrieval with
    relevance scoring and ranking.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Document Retriever.
        
        Args:
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super(GraphDocumentRetriever, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Document encoding layers
        self.doc_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Graph neural network for document relationships
        try:
            from torch_geometric.nn import GCNConv
            self.gnn_layers = nn.ModuleList([
                GCNConv(embedding_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
        except ImportError:
            # Fallback to linear layers
            self.gnn_layers = nn.ModuleList([
                nn.Linear(embedding_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
        
        # Relevance scoring network
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Graph-aware ranking
        self.ranking_network = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # +1 for initial score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        document_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve and rank documents for query.
        
        Args:
            query_embedding: Encoded query
            document_embeddings: Document embeddings
            edge_index: Document relationship edges
            document_ids: Document IDs
            
        Returns:
            Retrieval results with scores
        """
        # Encode documents
        encoded_docs = self.doc_encoder(document_embeddings)
        
        # Apply GNN layers
        doc_features = encoded_docs
        for gnn_layer in self.gnn_layers:
            if hasattr(gnn_layer, '__class__') and 'GCN' in str(gnn_layer.__class__):
                # PyG GCN
                doc_features = gnn_layer(doc_features, edge_index)
            else:
                # Linear fallback
                doc_features = gnn_layer(doc_features)
            doc_features = F.relu(doc_features)
        
        # Compute relevance scores
        relevance_scores = []
        for i in range(document_embeddings.size(0)):
            doc_emb = doc_features[i]
            
            # Combine query and document embeddings
            combined_emb = torch.cat([query_embedding, doc_emb], dim=0)
            relevance_score = self.relevance_scorer(combined_emb)
            relevance_scores.append(relevance_score)
        
        relevance_scores = torch.stack(relevance_scores)
        
        # Graph-aware ranking
        final_scores = []
        for i, score in enumerate(relevance_scores):
            # Combine with graph features
            graph_feature = doc_features[i]
            ranking_input = torch.cat([graph_feature, score.unsqueeze(0)], dim=0)
            final_score = self.ranking_network(ranking_input)
            final_scores.append(final_score)
        
        final_scores = torch.stack(final_scores)
        
        return {
            'document_ids': document_ids,
            'relevance_scores': relevance_scores,
            'final_scores': final_scores,
            'ranked_indices': torch.argsort(final_scores, descending=True)
        }


class GNNSearchEngine:
    """
    GNN-powered search engine for research documents.
    
    Provides intelligent search capabilities using graph neural networks
    for better semantic understanding and relevance ranking.
    """
    
    def __init__(self, gnn_core):
        """
        Initialize GNN Search Engine.
        
        Args:
            gnn_core: GNN core system for graph processing
        """
        self.gnn_core = gnn_core
        self.query_encoder = None
        self.document_retriever = None
        self.search_cache = {}
        self.vocabulary = {}
        self.word_to_idx = {}
        
    def initialize_search_models(self):
        """Initialize search-specific GNN models."""
        if self.gnn_core.current_graph_data is not None:
            embedding_dim = self.gnn_core.current_graph_data.x.shape[1]
            
            # Initialize query encoder
            self.query_encoder = GraphQueryEncoder(
                vocab_size=10000,
                embedding_dim=embedding_dim,
                hidden_dim=256
            )
            
            # Initialize document retriever
            self.document_retriever = GraphDocumentRetriever(
                embedding_dim=embedding_dim,
                hidden_dim=256
            )
            
            # Build vocabulary from documents
            self._build_vocabulary()
            
            logger.info("GNN search models initialized")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = 'semantic',
        use_graph_context: bool = True
    ) -> List[Dict]:
        """
        Perform GNN-powered search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            search_type: Type of search ('semantic', 'keyword', 'hybrid')
            use_graph_context: Whether to use graph context
            
        Returns:
            List of search results with scores and explanations
        """
        if not self.query_encoder or not self.document_retriever:
            self.initialize_search_models()
        
        # Check cache first
        cache_key = f"{query}_{top_k}_{search_type}_{use_graph_context}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Process query
        processed_query = self._preprocess_query(query)
        
        # Encode query
        query_embedding = self._encode_query(processed_query, use_graph_context)
        
        # Retrieve documents
        retrieval_results = self._retrieve_documents(
            query_embedding, top_k * 2  # Retrieve more for re-ranking
        )
        
        # Re-rank based on search type
        if search_type == 'semantic':
            ranked_results = self._semantic_ranking(
                query_embedding, retrieval_results
            )
        elif search_type == 'keyword':
            ranked_results = self._keyword_ranking(
                processed_query, retrieval_results
            )
        else:  # hybrid
            ranked_results = self._hybrid_ranking(
                query, processed_query, query_embedding, retrieval_results
            )
        
        # Generate explanations
        explained_results = self._generate_search_explanations(
            query, ranked_results[:top_k], search_type
        )
        
        # Cache results
        self.search_cache[cache_key] = explained_results
        
        return explained_results
    
    def _build_vocabulary(self):
        """Build vocabulary from document corpus."""
        # Simplified vocabulary building
        # In practice, extract from actual documents
        
        common_words = [
            'machine', 'learning', 'neural', 'network', 'deep', 'algorithm',
            'model', 'data', 'training', 'prediction', 'classification',
            'regression', 'clustering', 'optimization', 'gradient', 'descent',
            'backpropagation', 'convolution', 'recurrent', 'attention',
            'transformer', 'bert', 'gpt', 'research', 'paper', 'method',
            'approach', 'technique', 'framework', 'architecture', 'design',
            'implementation', 'evaluation', 'performance', 'accuracy', 'precision',
            'recall', 'f1', 'score', 'metric', 'benchmark', 'dataset',
            'feature', 'extraction', 'selection', 'engineering', 'preprocessing'
        ]
        
        self.vocabulary = {word: i for i, word in enumerate(common_words)}
        self.word_to_idx = self.vocabulary
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess search query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Tokenize
        tokens = query.split()
        
        # Remove stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def _encode_query(
        self,
        processed_query: str,
        use_graph_context: bool
    ) -> torch.Tensor:
        """Encode query using GNN."""
        # Tokenize query
        tokens = processed_query.split()
        
        # Convert to indices
        token_indices = []
        for token in tokens:
            if token in self.word_to_idx:
                token_indices.append(self.word_to_idx[token])
            else:
                token_indices.append(0)  # Unknown token
        
        # Pad to fixed length
        max_len = 20
        if len(token_indices) < max_len:
            token_indices.extend([0] * (max_len - len(token_indices)))
        else:
            token_indices = token_indices[:max_len]
        
        query_tokens = torch.tensor([token_indices], dtype=torch.long)
        
        # Get graph context
        if use_graph_context and self.gnn_core.current_graph_data is not None:
            graph_embeddings = self.gnn_core.current_graph_data.x
            edge_index = self.gnn_core.current_graph_data.edge_index
        else:
            graph_embeddings = torch.randn(100, 256)  # Mock graph
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Encode query
        with torch.no_grad():
            query_embedding = self.query_encoder(
                query_tokens, graph_embeddings, edge_index
            )
        
        return query_embedding.squeeze(0)
    
    def _retrieve_documents(
        self,
        query_embedding: torch.Tensor,
        num_candidates: int
    ) -> Dict:
        """Retrieve candidate documents."""
        if self.gnn_core.current_graph_data is None:
            return {'document_ids': [], 'embeddings': torch.empty(0, 128)}
        
        # Get all document embeddings
        document_embeddings = self.gnn_core.current_graph_data.x
        edge_index = self.gnn_core.current_graph_data.edge_index
        document_ids = list(range(document_embeddings.size(0)))
        
        # Use document retriever
        with torch.no_grad():
            retrieval_results = self.document_retriever(
                query_embedding.unsqueeze(0),
                document_embeddings,
                edge_index,
                document_ids
            )
        
        # Get top candidates
        top_indices = retrieval_results['ranked_indices'][:num_candidates]
        
        return {
            'document_ids': [retrieval_results['document_ids'][i] for i in top_indices],
            'embeddings': document_embeddings[top_indices],
            'scores': retrieval_results['final_scores'][top_indices],
            'relevance_scores': retrieval_results['relevance_scores'][top_indices]
        }
    
    def _semantic_ranking(
        self,
        query_embedding: torch.Tensor,
        retrieval_results: Dict
    ) -> List[Dict]:
        """Rank results using semantic similarity."""
        ranked_results = []
        
        doc_embeddings = retrieval_results['embeddings']
        doc_ids = retrieval_results['document_ids']
        
        # Compute semantic similarity
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_emb.unsqueeze(0)
            ).item()
            
            ranked_results.append({
                'document_id': doc_ids[i],
                'semantic_score': similarity,
                'relevance_score': retrieval_results['relevance_scores'][i].item(),
                'final_score': 0.7 * similarity + 0.3 * retrieval_results['relevance_scores'][i].item()
            })
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_results
    
    def _keyword_ranking(
        self,
        processed_query: str,
        retrieval_results: Dict
    ) -> List[Dict]:
        """Rank results using keyword matching."""
        ranked_results = []
        query_terms = set(processed_query.split())
        
        doc_ids = retrieval_results['document_ids']
        
        for i, doc_id in enumerate(doc_ids):
            # Get document content (mock)
            doc_content = self._get_document_content(doc_id)
            doc_terms = set(doc_content.lower().split())
            
            # Compute keyword overlap
            overlap = len(query_terms.intersection(doc_terms))
            keyword_score = overlap / max(len(query_terms), 1)
            
            ranked_results.append({
                'document_id': doc_id,
                'keyword_score': keyword_score,
                'relevance_score': retrieval_results['relevance_scores'][i].item(),
                'final_score': 0.8 * keyword_score + 0.2 * retrieval_results['relevance_scores'][i].item()
            })
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_results
    
    def _hybrid_ranking(
        self,
        original_query: str,
        processed_query: str,
        query_embedding: torch.Tensor,
        retrieval_results: Dict
    ) -> List[Dict]:
        """Rank results using hybrid approach."""
        # Get semantic and keyword rankings
        semantic_results = self._semantic_ranking(query_embedding, retrieval_results)
        keyword_results = self._keyword_ranking(processed_query, retrieval_results)
        
        # Combine rankings
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result['document_id']
            combined_results[doc_id] = {
                'document_id': doc_id,
                'semantic_score': result['semantic_score'],
                'keyword_score': 0.0,
                'relevance_score': result['relevance_score']
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result['document_id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['keyword_score']
            else:
                combined_results[doc_id] = {
                    'document_id': doc_id,
                    'semantic_score': 0.0,
                    'keyword_score': result['keyword_score'],
                    'relevance_score': result['relevance_score']
                }
        
        # Compute final hybrid scores
        for result in combined_results.values():
            result['final_score'] = (
                0.4 * result['semantic_score'] +
                0.4 * result['keyword_score'] +
                0.2 * result['relevance_score']
            )
        
        # Sort and return
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results
    
    def _get_document_content(self, doc_id: int) -> str:
        """Get document content for keyword matching."""
        # Mock document content
        # In practice, retrieve from database or index
        
        contents = [
            "Machine learning algorithms for data analysis and prediction",
            "Deep neural networks with attention mechanisms for NLP tasks",
            "Graph neural networks for social network analysis and recommendation",
            "Convolutional neural networks for image classification and detection",
            "Recurrent neural networks for sequence modeling and time series",
            "Transformer architectures for natural language understanding",
            "Optimization methods for training deep neural networks",
            "Ensemble methods for improving model performance and robustness"
        ]
        
        return contents[doc_id % len(contents)]
    
    def _generate_search_explanations(
        self,
        query: str,
        ranked_results: List[Dict],
        search_type: str
    ) -> List[Dict]:
        """Generate explanations for search results."""
        explained_results = []
        
        for result in ranked_results:
            doc_id = result['document_id']
            doc_content = self._get_document_content(doc_id)
            
            explanation = {
                'document_id': doc_id,
                'title': f"Research Paper {doc_id}",
                'content': doc_content,
                'score': result['final_score'],
                'explanation': self._create_search_explanation(
                    query, result, doc_content, search_type
                ),
                'search_type': search_type,
                'relevance_factors': self._extract_relevance_factors(result, doc_content)
            }
            
            explained_results.append(explanation)
        
        return explained_results
    
    def _create_search_explanation(
        self,
        query: str,
        result: Dict,
        doc_content: str,
        search_type: str
    ) -> str:
        """Create explanation for search result."""
        if search_type == 'semantic':
            return f"Found based on semantic similarity (score: {result['final_score']:.3f})"
        elif search_type == 'keyword':
            return f"Found based on keyword matching (score: {result['final_score']:.3f})"
        else:  # hybrid
            semantic_score = result.get('semantic_score', 0)
            keyword_score = result.get('keyword_score', 0)
            return (f"Found using hybrid search - semantic: {semantic_score:.2f}, "
                   f"keyword: {keyword_score:.2f}, total: {result['final_score']:.3f}")
    
    def _extract_relevance_factors(self, result: Dict, doc_content: str) -> Dict:
        """Extract factors contributing to relevance."""
        factors = {}
        
        # Semantic similarity
        if 'semantic_score' in result:
            factors['semantic_similarity'] = result['semantic_score']
        
        # Keyword matching
        if 'keyword_score' in result:
            factors['keyword_match'] = result['keyword_score']
        
        # Graph relevance
        if 'relevance_score' in result:
            factors['graph_relevance'] = result['relevance_score']
        
        # Content length (longer documents might be more comprehensive)
        factors['content_length'] = len(doc_content) / 1000.0
        
        return factors
    
    def graph_traversal_search(
        self,
        start_node: int,
        max_depth: int = 3,
        query_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform graph traversal-based search.
        
        Args:
            start_node: Starting node in graph
            max_depth: Maximum traversal depth
            query_filter: Optional query to filter results
            
        Returns:
            List of nodes found through traversal
        """
        if self.gnn_core.current_graph_data is None:
            return []
        
        # Perform graph traversal
        visited = set()
        queue = [(start_node, 0)]  # (node, depth)
        traversal_results = []
        
        while queue:
            node, depth = queue.pop(0)
            
            if node in visited or depth > max_depth:
                continue
            
            visited.add(node)
            
            # Get node information
            node_info = self._get_node_info(node)
            
            # Apply query filter if provided
            if query_filter:
                if self._matches_query_filter(node_info, query_filter):
                    traversal_results.append({
                        'node_id': node,
                        'depth': depth,
                        'info': node_info
                    })
            else:
                traversal_results.append({
                    'node_id': node,
                    'depth': depth,
                    'info': node_info
                })
            
            # Add neighbors to queue
            neighbors = self._get_node_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return traversal_results
    
    def _get_node_info(self, node_id: int) -> Dict:
        """Get information about a node."""
        # Mock node information
        return {
            'title': f"Node {node_id}",
            'type': 'paper',
            'content': f"Content for node {node_id}",
            'metadata': {'year': 2020 + node_id % 5}
        }
    
    def _matches_query_filter(self, node_info: Dict, query_filter: str) -> bool:
        """Check if node matches query filter."""
        # Simple text matching
        content = f"{node_info.get('title', '')} {node_info.get('content', '')}"
        return query_filter.lower() in content.lower()
    
    def _get_node_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node."""
        if self.gnn_core.current_graph_data is None:
            return []
        
        edge_index = self.gnn_core.current_graph_data.edge_index
        neighbors = set()
        
        # Find outgoing edges
        outgoing = edge_index[1][edge_index[0] == node_id].tolist()
        neighbors.update(outgoing)
        
        # Find incoming edges
        incoming = edge_index[0][edge_index[1] == node_id].tolist()
        neighbors.update(incoming)
        
        return list(neighbors)
    
    def get_search_stats(self) -> Dict:
        """Get statistics about search performance."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'search_cache_size': len(self.search_cache),
            'models_initialized': self.query_encoder is not None,
            'graph_nodes': self.gnn_core.current_graph_data.num_nodes if self.gnn_core.current_graph_data else 0
        }


# Main execution for testing
if __name__ == "__main__":
    print("=" * 80)
    print("GNN Search Engine Test")
    print("=" * 80)
    
    # Mock GNN core for testing
    class MockGNNCore:
        def __init__(self):
            # Create mock graph data
            self.current_graph_data = type('GraphData', (), {
                'x': torch.randn(20, 128),
                'edge_index': torch.tensor([[i, i+1] for i in range(19)], dtype=torch.long).t(),
                'num_nodes': 20
            })()
    
    # Initialize search engine
    search_engine = GNNSearchEngine(MockGNNCore())
    
    # Test semantic search
    print("\n1. Testing Semantic Search...")
    results = search_engine.search(
        query="machine learning neural networks",
        top_k=5,
        search_type='semantic'
    )
    
    print(f"  Found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"    {i}. Paper {result['document_id']} (score: {result['score']:.3f})")
        print(f"       Explanation: {result['explanation']}")
    
    # Test keyword search
    print("\n2. Testing Keyword Search...")
    results = search_engine.search(
        query="deep learning attention",
        top_k=5,
        search_type='keyword'
    )
    
    print(f"  Found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"    {i}. Paper {result['document_id']} (score: {result['score']:.3f})")
    
    # Test hybrid search
    print("\n3. Testing Hybrid Search...")
    results = search_engine.search(
        query="graph neural networks",
        top_k=5,
        search_type='hybrid'
    )
    
    print(f"  Found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"    {i}. Paper {result['document_id']} (score: {result['score']:.3f})")
        print(f"       Factors: {list(result['relevance_factors'].keys())}")
    
    # Test graph traversal search
    print("\n4. Testing Graph Traversal Search...")
    traversal_results = search_engine.graph_traversal_search(
        start_node=0,
        max_depth=2,
        query_filter="neural"
    )
    
    print(f"  Traversal found {len(traversal_results)} nodes")
    for result in traversal_results[:3]:
        print(f"    Node {result['node_id']} at depth {result['depth']}")
    
    # Get stats
    stats = search_engine.get_search_stats()
    print(f"\nSearch Engine Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ GNN Search Engine test complete")