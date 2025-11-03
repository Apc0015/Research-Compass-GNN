#!/usr/bin/env python3
"""
Advanced Citation Metrics - Compute sophisticated metrics on citation networks.

Provides deep insights into citation patterns including:
- Citation velocity and acceleration
- Sleeping beauty detection
- Disruption index
- Citation cascades
- Citation patterns and context
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class AdvancedCitationMetrics:
    """
    Compute advanced metrics on citation networks.
    
    Provides sophisticated citation analysis beyond simple counts,
    revealing temporal patterns, impact, and influence dynamics.
    
    Example:
        >>> metrics = AdvancedCitationMetrics(graph_manager, neo4j_uri, neo4j_user, neo4j_password)
        >>> disruption = metrics.calculate_disruption_index("paper123")
        >>> print(f"Disruption index: {disruption:.3f}")
    """
    
    def __init__(self, graph_manager, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize citation metrics analyzer.
        
        Args:
            graph_manager: GraphManager for graph operations
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
        """
        self.graph_manager = graph_manager
        self.uri = neo4j_uri
        self.driver = None
        self._use_neo4j = False
        
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.driver.verify_connectivity()
            self._use_neo4j = True
        except Exception as e:
            logger.warning(f"Neo4j unavailable for citation metrics: {e}")
    
    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()
    
    def compute_citation_velocity(self, paper_id: str) -> Dict:
        """
        Compute citation velocity for a paper.
        
        Returns:
            {
                'citations_per_year': [(year, count), ...],
                'velocity': float,
                'acceleration': float,
                'half_life': float,
                'prediction_next_year': int
            }
        """
        # This is implemented in temporal_analytics.py
        # Redirect to that implementation or duplicate here
        return {
            'citations_per_year': [],
            'velocity': 0.0,
            'acceleration': 0.0,
            'half_life': None,
            'prediction_next_year': 0
        }
    
    def analyze_citation_patterns(self, paper_id: str) -> Dict:
        """
        Analyze patterns in how a paper is cited.
        
        Args:
            paper_id: Paper node ID
            
        Returns:
            {
                'self_citation_rate': float,
                'citation_concentration': float,  # How concentrated among few papers
                'citation_diversity': float,  # Across how many fields
                'citing_paper_quality': {
                    'avg_citations': float,
                    'top_venues': [venues]
                },
                'citation_context': [
                    {
                        'citing_paper': str,
                        'context_type': 'background|method|comparison|extension',
                        'sentiment': 'positive|neutral|critical'
                    },
                    ...
                ]
            }
        """
        result = {
            'self_citation_rate': 0.0,
            'citation_concentration': 0.0,
            'citation_diversity': 0.0,
            'citing_paper_quality': {
                'avg_citations': 0.0,
                'top_venues': []
            },
            'citation_context': []
        }
        
        if not self._use_neo4j:
            return result
        
        try:
            with self.driver.session() as session:
                # Get all citing papers
                query = """
                MATCH (p:Entity {id: $paper_id})<-[c:CITES]-(citing:Entity)
                OPTIONAL MATCH (citing)<-[cc:CITES]-()
                RETURN coalesce(citing.id, citing.name) as citing_id,
                       coalesce(citing.authors, []) as authors,
                       count(cc) as citing_citations,
                       coalesce(citing.venue, 'Unknown') as venue
                """
                
                citing_papers = list(session.run(query, paper_id=paper_id))
                
                if not citing_papers:
                    return result
                
                # Get paper authors for self-citation detection
                author_query = """
                MATCH (p:Entity {id: $paper_id})
                RETURN coalesce(p.authors, []) as authors
                """
                paper_authors = set(session.run(author_query, paper_id=paper_id).single().get('authors', []))
                
                # Analyze patterns
                self_citations = 0
                all_citations = len(citing_papers)
                citation_counts = []
                venues = defaultdict(int)
                
                for record in citing_papers:
                    # Self-citation check
                    citing_authors = set(record.get('authors', []))
                    if paper_authors & citing_authors:
                        self_citations += 1
                    
                    # Quality metrics
                    citation_counts.append(record.get('citing_citations', 0))
                    venues[record.get('venue', 'Unknown')] += 1
                
                result['self_citation_rate'] = self_citations / all_citations if all_citations > 0 else 0
                
                # Citation concentration (Gini coefficient approximation)
                if citation_counts:
                    sorted_counts = sorted(citation_counts)
                    n = len(sorted_counts)
                    cumsum = np.cumsum(sorted_counts)
                    result['citation_concentration'] = (2 * sum((i+1) * count for i, count in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n if sum(sorted_counts) > 0 else 0
                
                # Citation diversity (number of unique venues)
                result['citation_diversity'] = len(venues) / all_citations if all_citations > 0 else 0
                
                # Citing paper quality
                result['citing_paper_quality']['avg_citations'] = np.mean(citation_counts) if citation_counts else 0
                result['citing_paper_quality']['top_venues'] = sorted(venues.items(), key=lambda x: x[1], reverse=True)[:5]
        
        except Exception as e:
            logger.error(f"Error analyzing citation patterns: {e}")
        
        return result
    
    def compute_sleeping_beauty_score(self, paper_id: str) -> Dict:
        """
        Identify "sleeping beauty" papers (late bloomers).
        
        A sleeping beauty is a paper that receives little attention initially
        but experiences a sudden surge in citations years later.
        
        Args:
            paper_id: Paper node ID
            
        Returns:
            {
                'sleeping_beauty_score': float,
                'awakening_year': int,
                'dormancy_period': int,
                'awakening_trigger': str,  # What caused recognition
                'citation_timeline': [(year, count), ...]
            }
        """
        result = {
            'sleeping_beauty_score': 0.0,
            'awakening_year': None,
            'dormancy_period': 0,
            'awakening_trigger': '',
            'citation_timeline': []
        }
        
        if not self._use_neo4j:
            return result
        
        try:
            with self.driver.session() as session:
                # Get citation timeline
                query = """
                MATCH (p:Entity {id: $paper_id})<-[c:CITES]-(citing:Entity)
                WITH citing, coalesce(citing.year, citing.publication_year, 2020) as year
                WITH year, count(citing) as citation_count
                ORDER BY year
                RETURN year, citation_count
                """
                
                timeline = [(r['year'], r['citation_count']) for r in session.run(query, paper_id=paper_id)]
                result['citation_timeline'] = timeline
                
                if len(timeline) < 3:
                    return result
                
                # Detect awakening (significant increase in citations)
                max_increase = 0
                awakening_idx = 0
                
                for i in range(1, len(timeline)):
                    prev_avg = np.mean([c for _, c in timeline[:i]])
                    current = timeline[i][1]
                    increase = current - prev_avg
                    
                    if increase > max_increase:
                        max_increase = increase
                        awakening_idx = i
                
                # Calculate sleeping beauty score
                # Higher score = longer dormancy + bigger awakening
                if awakening_idx > 0:
                    dormancy_period = awakening_idx
                    awakening_magnitude = max_increase / (np.mean([c for _, c in timeline[:awakening_idx]]) + 1)
                    
                    result['sleeping_beauty_score'] = dormancy_period * np.log1p(awakening_magnitude)
                    result['awakening_year'] = timeline[awakening_idx][0]
                    result['dormancy_period'] = dormancy_period
                    result['awakening_trigger'] = f"Citation surge of {max_increase:.0f} papers in {timeline[awakening_idx][0]}"
        
        except Exception as e:
            logger.error(f"Error computing sleeping beauty score: {e}")
        
        return result
    
    def calculate_disruption_index(self, paper_id: str) -> float:
        """
        Calculate how disruptive a paper is (Funk & Owen-Smith metric).
        
        Formula: (i - j) / (i + j + k)
        i = citations only citing focal paper
        j = citations citing both focal + its references
        k = citations only citing references
        
        A disruptive paper (index close to 1) creates a new research direction.
        A consolidating paper (index close to -1) builds on existing work.
        
        Args:
            paper_id: Paper node ID
            
        Returns:
            Disruption index (-1 to 1)
        """
        if not self._use_neo4j:
            return 0.0
        
        try:
            with self.driver.session() as session:
                query = """
                // Get papers that cite the focal paper
                MATCH (focal:Entity {id: $paper_id})<-[:CITES]-(citing:Entity)
                
                // Get references of the focal paper
                OPTIONAL MATCH (focal)-[:CITES]->(ref:Entity)
                WITH focal, citing, collect(DISTINCT ref) as refs
                
                // For each citing paper, check if it cites any references
                UNWIND refs as ref
                OPTIONAL MATCH (citing)-[:CITES]->(ref)
                WITH citing, count(ref) > 0 as cites_ref
                
                // Categorize citing papers
                WITH 
                    sum(CASE WHEN NOT cites_ref THEN 1 ELSE 0 END) as i,
                    sum(CASE WHEN cites_ref THEN 1 ELSE 0 END) as j
                
                // Count papers citing references but not focal (k)
                MATCH (focal:Entity {id: $paper_id})-[:CITES]->(ref:Entity)<-[:CITES]-(other:Entity)
                WHERE NOT (other)-[:CITES]->(focal)
                WITH i, j, count(DISTINCT other) as k
                
                RETURN i, j, k
                """
                
                result = session.run(query, paper_id=paper_id).single()
                
                if result:
                    i = result.get('i', 0)
                    j = result.get('j', 0)
                    k = result.get('k', 0)
                    
                    denominator = i + j + k
                    if denominator > 0:
                        disruption_index = (i - j) / denominator
                        return float(disruption_index)
        
        except Exception as e:
            logger.error(f"Error calculating disruption index: {e}")
        
        return 0.0
    
    def analyze_citation_cascades(self, paper_id: str) -> Dict:
        """
        Analyze how citations cascade through the network.
        
        Tracks how a paper's influence spreads through successive generations
        of citations (papers citing papers that cite the focal paper).
        
        Args:
            paper_id: Paper node ID
            
        Returns:
            {
                'cascade_depth': int,
                'cascade_width': int,
                'cascade_visualization': str,
                'influential_cascade_nodes': [papers],
                'cascade_velocity': float
            }
        """
        result = {
            'cascade_depth': 0,
            'cascade_width': 0,
            'cascade_visualization': '',
            'influential_cascade_nodes': [],
            'cascade_velocity': 0.0
        }
        
        if not self._use_neo4j:
            return result
        
        try:
            with self.driver.session() as session:
                # Find citation cascade (papers at increasing depths)
                cascade_levels = []
                
                for depth in range(1, 6):  # Up to 5 levels
                    query = f"""
                    MATCH path = (p:Entity {{id: $paper_id}})<-[:CITES*{depth}]-(citing:Entity)
                    RETURN DISTINCT coalesce(citing.id, citing.name) as paper,
                           coalesce(citing.year, citing.publication_year, 2020) as year
                    LIMIT 100
                    """
                    
                    level_papers = list(session.run(query, paper_id=paper_id))
                    
                    if not level_papers:
                        break
                    
                    cascade_levels.append({
                        'depth': depth,
                        'papers': [r['paper'] for r in level_papers],
                        'count': len(level_papers),
                        'avg_year': np.mean([r['year'] for r in level_papers])
                    })
                
                if cascade_levels:
                    result['cascade_depth'] = len(cascade_levels)
                    result['cascade_width'] = max(level['count'] for level in cascade_levels)
                    
                    # Cascade velocity (how fast it spreads over time)
                    if len(cascade_levels) > 1:
                        time_span = cascade_levels[-1]['avg_year'] - cascade_levels[0]['avg_year']
                        result['cascade_velocity'] = result['cascade_depth'] / (time_span + 1)
                    
                    # Find most influential nodes in cascade
                    influential = sorted(cascade_levels[0]['papers'][:10])
                    result['influential_cascade_nodes'] = influential
                    
                    # Create visualization
                    result['cascade_visualization'] = self._create_cascade_viz(cascade_levels)
        
        except Exception as e:
            logger.error(f"Error analyzing citation cascades: {e}")
        
        return result
    
    def _create_cascade_viz(self, cascade_levels: List[Dict]) -> str:
        """Create HTML visualization of citation cascade."""
        html = """
        <html>
        <head>
            <style>
                .cascade { padding: 20px; }
                .level { margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; }
                .level-title { font-weight: bold; color: #2e7d32; }
                .papers { margin-top: 10px; }
                .paper { display: inline-block; padding: 5px 10px; margin: 3px; 
                        background: #e8f5e9; border-radius: 3px; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="cascade">
                <h2>Citation Cascade Analysis</h2>
        """
        
        for level in cascade_levels:
            html += f"""
                <div class="level">
                    <div class="level-title">
                        Generation {level['depth']} - {level['count']} papers
                        (avg year: {level['avg_year']:.0f})
                    </div>
                    <div class="papers">
            """
            
            for paper in level['papers'][:20]:  # Show first 20
                html += f'<span class="paper">{paper[:15]}</span>'
            
            if level['count'] > 20:
                html += f'<span class="paper">... and {level["count"] - 20} more</span>'
            
            html += "</div></div>"
        
        html += "</div></body></html>"
        return html


if __name__ == "__main__":
    print("Advanced Citation Metrics module loaded successfully")
