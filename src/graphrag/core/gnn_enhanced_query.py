#!/usr/bin/env python3
"""
GNN-Enhanced Query Engine - Integrate GNN predictions into query answering.

This module combines traditional retrieval with GNN reasoning to provide
richer, more contextual answers with transparent reasoning traces.
"""

from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GNNEnhancedQueryEngine:
    """
    Integrate GNN predictions into the main query answering pipeline.
    
    Combines vector search, graph traversal, and GNN predictions to provide
    comprehensive answers with multi-source evidence and reasoning traces.
    
    Example:
        >>> engine = GNNEnhancedQueryEngine(vector_search, graph_manager, gnn_manager, llm_manager)
        >>> result = engine.answer_with_gnn_reasoning("What are the latest advances in transformers?")
        >>> print(result['answer'])
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(self, vector_search, graph_manager, gnn_manager, llm_manager=None):
        """
        Initialize the enhanced query engine.
        
        Args:
            vector_search: VectorSearch instance for semantic retrieval
            graph_manager: GraphManager for graph traversal
            gnn_manager: GNNManager for GNN predictions
            llm_manager: Optional LLM for answer generation
        """
        self.vector_search = vector_search
        self.graph_manager = graph_manager
        self.gnn_manager = gnn_manager
        self.llm_manager = llm_manager
    
    def answer_with_gnn_reasoning(self, question: str, use_gnn: bool = True) -> Dict:
        """
        Answer questions using both retrieval and GNN reasoning.
        
        Workflow:
        1. Vector search for relevant chunks
        2. GNN link prediction to find related papers
        3. GNN node classification for context enrichment
        4. Combine evidence from all sources
        5. Generate answer with reasoning trace
        
        Args:
            question: User's research question
            use_gnn: Whether to use GNN predictions (default True)
            
        Returns:
            {
                'answer': str,
                'confidence': float,
                'sources': [
                    {
                        'type': 'retrieval|gnn_prediction|graph_traversal',
                        'content': str,
                        'confidence': float,
                        'reasoning': str
                    },
                    ...
                ],
                'gnn_reasoning_trace': [
                    {'step': str, 'prediction': str, 'explanation': str},
                    ...
                ],
                'visualization': str  # HTML of reasoning path
            }
            
        Example:
            >>> result = engine.answer_with_gnn_reasoning(
            ...     "What papers build on BERT?",
            ...     use_gnn=True
            ... )
            >>> for source in result['sources']:
            ...     print(f"{source['type']}: {source['content'][:50]}...")
        """
        result = {
            'answer': '',
            'confidence': 0.0,
            'sources': [],
            'gnn_reasoning_trace': [],
            'visualization': ''
        }
        
        try:
            # Step 1: Vector search for relevant content
            logger.info(f"Processing query: {question}")
            
            if self.vector_search and self.vector_search.index:
                try:
                    vector_results = self.vector_search.search(question, top_k=5)
                    
                    for text, score, metadata in vector_results:
                        result['sources'].append({
                            'type': 'retrieval',
                            'content': text,
                            'confidence': 1.0 / (1.0 + score),  # Convert distance to similarity
                            'reasoning': 'Retrieved via semantic similarity',
                            'metadata': metadata
                        })
                    
                    logger.info(f"Found {len(vector_results)} relevant chunks via vector search")
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            
            # Step 2: Extract entities from question for graph expansion
            entities = self._extract_entities(question)
            logger.info(f"Extracted entities: {entities}")
            
            # Step 3: Graph traversal for connected papers
            for entity in entities[:3]:  # Limit to avoid overload
                try:
                    neighbors = self.graph_manager.query_neighbors(entity, max_depth=2)
                    
                    for neighbor in neighbors[:5]:
                        result['sources'].append({
                            'type': 'graph_traversal',
                            'content': f"Related entity: {neighbor['name']}",
                            'confidence': 0.7 / (neighbor['distance'] + 1),
                            'reasoning': f"Connected via graph (distance: {neighbor['distance']})",
                            'metadata': {'distance': neighbor['distance'], 'type': neighbor['type']}
                        })
                    
                    logger.info(f"Found {len(neighbors)} neighbors for {entity}")
                except Exception as e:
                    logger.warning(f"Graph traversal failed for {entity}: {e}")
            
            # Step 4: GNN predictions (if enabled and available)
            if use_gnn and self.gnn_manager:
                try:
                    gnn_trace = []
                    
                    # Node classification for relevant papers
                    for entity in entities[:2]:
                        try:
                            topics = self.gnn_manager.predict_paper_topics(entity, top_k=3)
                            
                            if topics:
                                gnn_trace.append({
                                    'step': 'node_classification',
                                    'prediction': f"Topics for {entity}: {[t[0] for t in topics]}",
                                    'explanation': f"GNN classified with confidence {topics[0][1]:.2%}"
                                })
                                
                                result['sources'].append({
                                    'type': 'gnn_prediction',
                                    'content': f"GNN predicts topics: {', '.join(t[0] for t in topics)}",
                                    'confidence': topics[0][1] if topics else 0.5,
                                    'reasoning': 'GNN node classification',
                                    'metadata': {'topics': topics}
                                })
                        except Exception as e:
                            logger.warning(f"Node classification failed for {entity}: {e}")
                    
                    # Link prediction for related papers
                    for entity in entities[:2]:
                        try:
                            predictions = self.gnn_manager.predict_paper_citations(entity, top_k=5)
                            
                            if predictions:
                                gnn_trace.append({
                                    'step': 'link_prediction',
                                    'prediction': f"Predicted {len(predictions)} related papers for {entity}",
                                    'explanation': f"Top prediction score: {predictions[0][1]:.3f}"
                                })
                                
                                for paper_id, score in predictions[:3]:
                                    result['sources'].append({
                                        'type': 'gnn_prediction',
                                        'content': f"GNN predicts relation to: {paper_id}",
                                        'confidence': score,
                                        'reasoning': 'GNN link prediction',
                                        'metadata': {'predicted_paper': paper_id, 'score': score}
                                    })
                        except Exception as e:
                            logger.warning(f"Link prediction failed for {entity}: {e}")
                    
                    result['gnn_reasoning_trace'] = gnn_trace
                    logger.info(f"GNN reasoning completed with {len(gnn_trace)} steps")
                
                except Exception as e:
                    logger.error(f"GNN reasoning failed: {e}")
            
            # Step 5: Synthesize answer
            result['answer'] = self._synthesize_answer(question, result['sources'])
            result['confidence'] = self._calculate_confidence(result['sources'])
            
            # Step 6: Create visualization
            result['visualization'] = self._create_reasoning_viz(
                question, 
                result['sources'], 
                result['gnn_reasoning_trace']
            )
        
        except Exception as e:
            logger.error(f"Error in GNN-enhanced query: {e}")
            result['answer'] = f"Error processing query: {str(e)}"
        
        return result
    
    def find_implicit_connections(self, entity1: str, entity2: str) -> Dict:
        """
        Use GNN to find non-obvious connections between entities.
        
        Combines shortest path finding with GNN link prediction to discover
        both explicit and implicit connections.
        
        Args:
            entity1: First entity ID
            entity2: Second entity ID
            
        Returns:
            {
                'direct_paths': [paths],
                'gnn_predicted_links': [
                    {
                        'intermediate_node': str,
                        'confidence': float,
                        'explanation': str
                    },
                    ...
                ],
                'reasoning_narrative': str
            }
            
        Example:
            >>> connections = engine.find_implicit_connections("paper1", "paper2")
            >>> print(connections['reasoning_narrative'])
        """
        result = {
            'direct_paths': [],
            'gnn_predicted_links': [],
            'reasoning_narrative': ''
        }
        
        try:
            # Find direct paths via graph
            try:
                neighbors1 = self.graph_manager.query_neighbors(entity1, max_depth=3)
                neighbors2 = self.graph_manager.query_neighbors(entity2, max_depth=3)
                
                # Find common neighbors (potential paths)
                common = []
                neighbors1_ids = {n['name'] for n in neighbors1}
                neighbors2_ids = {n['name'] for n in neighbors2}
                common_ids = neighbors1_ids & neighbors2_ids
                
                result['direct_paths'] = [
                    f"{entity1} -> {common_id} -> {entity2}"
                    for common_id in list(common_ids)[:5]
                ]
            except Exception as e:
                logger.warning(f"Graph path finding failed: {e}")
            
            # Use GNN to find implicit connections
            if self.gnn_manager:
                try:
                    # Get predictions from entity1
                    predictions1 = self.gnn_manager.predict_paper_citations(entity1, top_k=10)
                    # Get predictions from entity2
                    predictions2 = self.gnn_manager.predict_paper_citations(entity2, top_k=10)
                    
                    # Find common predicted papers (potential bridges)
                    preds1_ids = {p[0] for p in predictions1}
                    preds2_ids = {p[0] for p in predictions2}
                    bridges = preds1_ids & preds2_ids
                    
                    for bridge_id in list(bridges)[:5]:
                        # Get confidence scores
                        score1 = next((s for p, s in predictions1 if p == bridge_id), 0.0)
                        score2 = next((s for p, s in predictions2 if p == bridge_id), 0.0)
                        avg_confidence = (score1 + score2) / 2
                        
                        result['gnn_predicted_links'].append({
                            'intermediate_node': bridge_id,
                            'confidence': avg_confidence,
                            'explanation': f"GNN predicts both papers relate to {bridge_id}"
                        })
                
                except Exception as e:
                    logger.warning(f"GNN implicit connection finding failed: {e}")
            
            # Build narrative
            narrative_parts = []
            
            if result['direct_paths']:
                narrative_parts.append(
                    f"Found {len(result['direct_paths'])} direct paths through the citation network."
                )
            
            if result['gnn_predicted_links']:
                narrative_parts.append(
                    f"GNN identified {len(result['gnn_predicted_links'])} potential implicit connections "
                    f"through related work."
                )
            
            if narrative_parts:
                result['reasoning_narrative'] = " ".join(narrative_parts)
            else:
                result['reasoning_narrative'] = (
                    f"No strong connections found between {entity1} and {entity2}. "
                    "They may be in different research areas."
                )
        
        except Exception as e:
            logger.error(f"Error finding implicit connections: {e}")
            result['reasoning_narrative'] = f"Error: {str(e)}"
        
        return result
    
    # Helper methods
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract potential entity names from question."""
        # Simple implementation - could use NER
        entities = []
        
        # Try to find entities in the graph
        words = question.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = " ".join(words[i:j])
                if len(phrase) > 3:
                    # Search for this phrase in the graph
                    try:
                        matches = self.graph_manager.search_entities(phrase, limit=1)
                        if matches:
                            entities.append(matches[0]['name'])
                    except:
                        pass
        
        return list(set(entities))[:5]  # Deduplicate and limit
    
    def _synthesize_answer(self, question: str, sources: List[Dict]) -> str:
        """Synthesize answer from multiple sources."""
        if not sources:
            return "No relevant information found."
        
        # If LLM is available, use it for synthesis
        if self.llm_manager:
            try:
                context = "\n\n".join([
                    f"[{s['type']}] {s['content']}"
                    for s in sorted(sources, key=lambda x: x['confidence'], reverse=True)[:10]
                ])
                
                prompt = f"""Based on the following information, answer this question: {question}

Context:
{context}

Provide a concise, accurate answer based on the evidence above."""
                
                answer = self.llm_manager.generate(prompt, max_tokens=300)
                return answer
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
        
        # Fallback: simple concatenation
        top_sources = sorted(sources, key=lambda x: x['confidence'], reverse=True)[:5]
        answer_parts = [
            f"Based on {len(sources)} sources:\n"
        ]
        
        for i, source in enumerate(top_sources, 1):
            answer_parts.append(f"{i}. [{source['type']}] {source['content'][:200]}...")
        
        return "\n".join(answer_parts)
    
    def _calculate_confidence(self, sources: List[Dict]) -> float:
        """Calculate overall confidence in answer."""
        if not sources:
            return 0.0
        
        # Weighted average with diversity penalty
        confidences = [s['confidence'] for s in sources]
        types = set(s['type'] for s in sources)
        
        avg_confidence = sum(confidences) / len(confidences)
        diversity_bonus = len(types) / 3.0  # Bonus for multiple source types
        
        return min(1.0, avg_confidence * (1 + diversity_bonus * 0.2))
    
    def _create_reasoning_viz(
        self, 
        question: str, 
        sources: List[Dict], 
        gnn_trace: List[Dict]
    ) -> str:
        """Create HTML visualization of reasoning process."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .question {{ font-size: 18px; font-weight: bold; margin-bottom: 20px; }}
                .reasoning-step {{ 
                    margin: 10px 0; 
                    padding: 15px; 
                    border-left: 4px solid #4CAF50;
                    background: #f9f9f9;
                }}
                .source {{
                    margin: 10px 0;
                    padding: 10px;
                    background: #e8f5e9;
                    border-radius: 5px;
                }}
                .source-type {{
                    font-weight: bold;
                    color: #2e7d32;
                }}
                .confidence {{
                    float: right;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="question">Question: {question}</div>
        """
        
        if gnn_trace:
            html += "<h3>GNN Reasoning Trace</h3>"
            for step in gnn_trace:
                html += f"""
                <div class="reasoning-step">
                    <strong>{step['step']}</strong><br>
                    {step['prediction']}<br>
                    <em>{step['explanation']}</em>
                </div>
                """
        
        if sources:
            html += "<h3>Evidence Sources</h3>"
            for source in sources[:10]:
                html += f"""
                <div class="source">
                    <span class="source-type">{source['type']}</span>
                    <span class="confidence">Confidence: {source['confidence']:.2%}</span><br>
                    {source['content'][:200]}...
                </div>
                """
        
        html += "</body></html>"
        return html


if __name__ == "__main__":
    import os
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.graphrag.core.graph_manager import GraphManager
    from src.graphrag.core.vector_search import VectorSearch
    from src.graphrag.ml.gnn_manager import GNNManager
    
    print("=" * 80)
    print("GNN-Enhanced Query Engine Test")
    print("=" * 80)
    
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Initialize components
    graph_manager = GraphManager(neo4j_uri, neo4j_user, neo4j_password)
    vector_search = VectorSearch()
    gnn_manager = GNNManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Initialize GNN
        gnn_manager.initialize_models()
        
        # Create engine
        engine = GNNEnhancedQueryEngine(
            vector_search,
            graph_manager,
            gnn_manager,
            llm_manager=None
        )
        
        # Test query
        question = "What are the latest advances in neural networks?"
        print(f"\nQuery: {question}")
        
        result = engine.answer_with_gnn_reasoning(question)
        
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Sources: {len(result['sources'])}")
        print(f"GNN reasoning steps: {len(result['gnn_reasoning_trace'])}")
        
        # Save visualization
        viz_path = Path("output/visualizations/query_reasoning.html")
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        with open(viz_path, 'w') as f:
            f.write(result['visualization'])
        print(f"\nVisualization saved to: {viz_path}")
        
        print("\n✓ GNN-Enhanced Query Engine test complete")
    
    except Exception as e:
        logger.exception(f"Test failed: {e}")
    
    finally:
        graph_manager.close()
        gnn_manager.close()
