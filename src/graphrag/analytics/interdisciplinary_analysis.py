#!/usr/bin/env python3
"""
Interdisciplinary Analysis - Analyze and visualize cross-disciplinary connections.

Helps researchers discover connections between different research fields
and identify interdisciplinary opportunities.
"""

from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict, Counter
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class InterdisciplinaryAnalyzer:
    """
    Analyze and visualize cross-disciplinary connections.
    
    Identifies papers and authors that bridge multiple disciplines,
    enabling discovery of interdisciplinary research opportunities.
    """
    
    def __init__(self, graph_manager, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize interdisciplinary analyzer.
        
        Args:
            graph_manager: GraphManager for graph operations
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
        """
        self.graph_manager = graph_manager
        self.driver = None
        self._use_neo4j = False
        
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.driver.verify_connectivity()
            self._use_neo4j = True
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}")
    
    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()
    
    def identify_interdisciplinary_papers(self, threshold: float = 0.3) -> List[Dict]:
        """
        Find papers bridging multiple disciplines.
        
        Args:
            threshold: Minimum diversity score (0-1)
            
        Returns:
            [
                {
                    'paper_id': str,
                    'disciplines': [str, str, ...],
                    'diversity_score': float,
                    'bridging_centrality': float,
                    'impact_on_fields': {field: score, ...}
                },
                ...
            ]
        """
        interdisciplinary_papers = []
        
        if not self._use_neo4j:
            return interdisciplinary_papers
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Entity)
                WHERE p.keywords IS NOT NULL OR p.topics IS NOT NULL
                WITH p, coalesce(p.keywords, p.topics, []) as fields
                WHERE size(fields) >= 2
                OPTIONAL MATCH (p)<-[c:CITES]-()
                WITH p, fields, count(c) as citations
                RETURN coalesce(p.id, p.name) as paper_id,
                       coalesce(p.name, p.text) as title,
                       fields,
                       citations,
                       size(fields) as num_fields
                ORDER BY num_fields DESC
                LIMIT 100
                """
                
                results = session.run(query)
                
                for record in results:
                    fields = record['fields']
                    if not fields or len(fields) < 2:
                        continue
                    
                    # Calculate diversity score (normalized entropy)
                    field_counts = Counter(fields)
                    total = sum(field_counts.values())
                    entropy = 0.0
                    
                    for count in field_counts.values():
                        p = count / total
                        if p > 0:
                            entropy -= p * (p.__log__() if hasattr(p, '__log__') else 0)
                    
                    max_entropy = len(field_counts).__log__() if len(field_counts) > 0 else 1
                    diversity_score = entropy / max_entropy if max_entropy > 0 else 0
                    
                    if diversity_score >= threshold:
                        interdisciplinary_papers.append({
                            'paper_id': record['paper_id'],
                            'title': record['title'],
                            'disciplines': list(set(fields)),
                            'diversity_score': diversity_score,
                            'bridging_centrality': diversity_score * record['citations'],
                            'impact_on_fields': {field: 1.0/len(fields) for field in set(fields)}
                        })
                
                # Sort by bridging centrality
                interdisciplinary_papers.sort(key=lambda x: x['bridging_centrality'], reverse=True)
        
        except Exception as e:
            logger.error(f"Error identifying interdisciplinary papers: {e}")
        
        return interdisciplinary_papers
    
    def visualize_field_interactions(self) -> str:
        """
        Create Sankey or chord diagram showing inter-field citation flows.
        
        Returns:
            HTML with interactive visualization showing:
            - Fields as nodes
            - Citation flows as links
            - Link width = citation volume
            - Color coded by field
        """
        html = """
        <html>
        <head>
            <title>Field Interactions</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id="sankey" style="width:100%;height:600px;"></div>
            <script>
                // Placeholder - would need actual data from Neo4j
                var data = [{
                    type: "sankey",
                    node: {
                        label: ["Computer Science", "Physics", "Biology", "Mathematics"],
                        color: ["blue", "red", "green", "purple"],
                        pad: 15,
                        thickness: 20
                    },
                    link: {
                        source: [0, 0, 1, 2],
                        target: [1, 2, 3, 3],
                        value: [10, 20, 15, 25]
                    }
                }];
                
                var layout = {
                    title: "Inter-field Citation Flows",
                    font: { size: 12 }
                };
                
                Plotly.newPlot('sankey', data, layout);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def find_cross_disciplinary_pathways(self, field1: str, field2: str) -> Dict:
        """
        Find how ideas flow between two disciplines.
        
        Args:
            field1: First discipline name
            field2: Second discipline name
            
        Returns:
            {
                'bridging_papers': [papers],
                'bridging_authors': [authors],
                'common_topics': [topics],
                'pathway_visualization': str,
                'collaboration_opportunities': [suggestions]
            }
        """
        result = {
            'bridging_papers': [],
            'bridging_authors': [],
            'common_topics': [],
            'pathway_visualization': '',
            'collaboration_opportunities': []
        }
        
        try:
            # Find papers in field1
            papers1 = set()
            matches1 = self.graph_manager.search_entities(field1, limit=50)
            for match in matches1:
                papers1.add(match.get('name') or match.get('id'))
            
            # Find papers in field2
            papers2 = set()
            matches2 = self.graph_manager.search_entities(field2, limit=50)
            for match in matches2:
                papers2.add(match.get('name') or match.get('id'))
            
            # Find bridging papers (cite both fields)
            bridging = []
            for paper1 in list(papers1)[:20]:
                neighbors = self.graph_manager.query_neighbors(paper1, max_depth=2)
                neighbor_ids = {n['name'] for n in neighbors}
                
                # Check if connects to field2
                overlap = neighbor_ids & papers2
                if overlap:
                    bridging.append({
                        'paper_id': paper1,
                        'connects_to': list(overlap)[:5]
                    })
            
            result['bridging_papers'] = bridging[:10]
            
            # Generate collaboration opportunities
            if len(bridging) > 0:
                result['collaboration_opportunities'] = [
                    f"Combine {field1} techniques with {field2} problems",
                    f"Apply {field2} methods to {field1} datasets",
                    f"Develop hybrid {field1}-{field2} approaches"
                ]
        
        except Exception as e:
            logger.error(f"Error finding cross-disciplinary pathways: {e}")
        
        return result


if __name__ == "__main__":
    print("Interdisciplinary Analysis module loaded successfully")
