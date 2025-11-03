#!/usr/bin/env python3
"""
Temporal Graph Analytics - Analyze how research topics and citations evolve over time.

This module provides comprehensive temporal analysis capabilities:
- Topic evolution tracking
- Temporal centrality metrics
- Emerging topics detection
- Citation velocity analysis
- H-index timeline computation
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import numpy as np
from collections import defaultdict
from neo4j import GraphDatabase
import networkx as nx

logger = logging.getLogger(__name__)


class TemporalGraphAnalytics:
    """
    Analyze temporal dynamics of the academic knowledge graph.
    
    This class provides tools for understanding how research evolves over time,
    including topic trends, researcher impact trajectories, and citation patterns.
    
    Example:
        >>> analytics = TemporalGraphAnalytics(graph_manager, neo4j_uri, neo4j_user, neo4j_password)
        >>> evolution = analytics.analyze_topic_evolution("machine learning", "yearly")
        >>> print(evolution['maturity_stage'])
        'growing'
    """
    
    def __init__(self, graph_manager, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize with graph manager and Neo4j connection.
        
        Args:
            graph_manager: GraphManager instance for graph operations
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
        """
        self.graph_manager = graph_manager
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.password = neo4j_password
        self.driver = None
        self._use_neo4j = False
        
        # Try to connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.driver.verify_connectivity()
            self._use_neo4j = True
            logger.info("TemporalAnalytics connected to Neo4j")
        except Exception as e:
            logger.warning(f"Neo4j unavailable for temporal analytics: {e}")
    
    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()
    
    def analyze_topic_evolution(self, topic: str, time_window: str = "yearly") -> Dict:
        """
        Analyze how a topic has evolved over time.
        
        Tracks paper count, citation count, and influential papers across time periods.
        Determines topic maturity stage based on growth patterns.
        
        Args:
            topic: Topic name to analyze (e.g., "neural networks")
            time_window: 'monthly', 'quarterly', 'yearly'
            
        Returns:
            {
                'timeline': [(year, paper_count, citation_count), ...],
                'trending_keywords': {year: [keywords], ...},
                'influential_papers_by_period': {year: [papers], ...},
                'growth_rate': float,
                'maturity_stage': 'emerging|growing|mature|declining'
            }
            
        Example:
            >>> evolution = analytics.analyze_topic_evolution("deep learning", "yearly")
            >>> print(f"Growth rate: {evolution['growth_rate']:.2%}")
            Growth rate: 45.2%
        """
        result = {
            'timeline': [],
            'trending_keywords': {},
            'influential_papers_by_period': {},
            'growth_rate': 0.0,
            'maturity_stage': 'emerging'
        }
        
        if not self._use_neo4j:
            logger.warning("Neo4j not available, returning empty result")
            return result
        
        try:
            with self.driver.session() as session:
                # Query papers related to topic over time
                query = """
                MATCH (p:Entity)
                WHERE toLower(coalesce(p.name, p.text, '')) CONTAINS toLower($topic)
                   OR toLower(coalesce(p.description, '')) CONTAINS toLower($topic)
                WITH p, coalesce(p.year, p.publication_year, 2020) as year
                OPTIONAL MATCH (p)<-[c:CITES]-()
                WITH year, count(DISTINCT p) as paper_count, count(c) as citation_count
                ORDER BY year
                RETURN year, paper_count, citation_count
                """
                
                timeline_result = session.run(query, topic=topic)
                timeline_data = [(r['year'], r['paper_count'], r['citation_count']) 
                               for r in timeline_result]
                result['timeline'] = timeline_data
                
                # Calculate growth rate
                if len(timeline_data) >= 2:
                    early_count = sum(p[1] for p in timeline_data[:len(timeline_data)//2])
                    late_count = sum(p[1] for p in timeline_data[len(timeline_data)//2:])
                    if early_count > 0:
                        result['growth_rate'] = (late_count - early_count) / early_count
                
                # Determine maturity stage
                result['maturity_stage'] = self._classify_maturity(result['growth_rate'], timeline_data)
                
                # Get influential papers by period
                for year_data in timeline_data[:5]:  # Last 5 periods
                    year = year_data[0]
                    papers_query = """
                    MATCH (p:Entity)
                    WHERE (toLower(coalesce(p.name, p.text, '')) CONTAINS toLower($topic)
                       OR toLower(coalesce(p.description, '')) CONTAINS toLower($topic))
                       AND coalesce(p.year, p.publication_year, 2020) = $year
                    OPTIONAL MATCH (p)<-[c:CITES]-()
                    WITH p, count(c) as citations
                    ORDER BY citations DESC
                    LIMIT 5
                    RETURN coalesce(p.id, p.name, p.text) as paper_id, 
                           coalesce(p.name, p.text) as title,
                           citations
                    """
                    papers_result = session.run(papers_query, topic=topic, year=year)
                    result['influential_papers_by_period'][year] = [
                        {'id': r['paper_id'], 'title': r['title'], 'citations': r['citations']}
                        for r in papers_result
                    ]
                
                # Extract trending keywords (simplified)
                result['trending_keywords'] = self._extract_trending_keywords(session, topic, timeline_data)
        
        except Exception as e:
            logger.error(f"Error analyzing topic evolution: {e}")
        
        return result
    
    def calculate_temporal_centrality(self, entity_id: str, time_periods: List[int]) -> Dict:
        """
        Calculate centrality metrics across multiple time periods.
        
        Computes PageRank and betweenness centrality for each time period,
        revealing how an entity's importance changes over time.
        
        Args:
            entity_id: Paper or Author ID
            time_periods: List of years [2020, 2021, 2022, ...]
            
        Returns:
            {
                'pagerank_timeline': [(year, score), ...],
                'betweenness_timeline': [(year, score), ...],
                'influence_trend': 'rising|stable|declining',
                'peak_influence_year': int,
                'trajectory_chart_data': Dict
            }
            
        Example:
            >>> centrality = analytics.calculate_temporal_centrality("paper123", [2020, 2021, 2022])
            >>> print(centrality['influence_trend'])
            'rising'
        """
        result = {
            'pagerank_timeline': [],
            'betweenness_timeline': [],
            'influence_trend': 'stable',
            'peak_influence_year': None,
            'trajectory_chart_data': {}
        }
        
        if not self._use_neo4j:
            return result
        
        try:
            pagerank_scores = []
            betweenness_scores = []
            
            for year in sorted(time_periods):
                # Build temporal subgraph for this year
                G = self._build_temporal_subgraph(year)
                
                if G.number_of_nodes() == 0:
                    continue
                
                # Calculate PageRank
                pr = nx.pagerank(G, alpha=0.85)
                pagerank_score = pr.get(entity_id, 0.0)
                pagerank_scores.append((year, pagerank_score))
                
                # Calculate Betweenness Centrality (on sample for performance)
                if G.number_of_nodes() > 1000:
                    # Sample for large graphs
                    bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
                else:
                    bc = nx.betweenness_centrality(G)
                
                betweenness_score = bc.get(entity_id, 0.0)
                betweenness_scores.append((year, betweenness_score))
            
            result['pagerank_timeline'] = pagerank_scores
            result['betweenness_timeline'] = betweenness_scores
            
            # Determine trend
            if len(pagerank_scores) >= 3:
                scores = [s for _, s in pagerank_scores]
                result['influence_trend'] = self._determine_trend(scores)
                result['peak_influence_year'] = pagerank_scores[np.argmax(scores)][0] if scores else None
            
            # Chart data
            result['trajectory_chart_data'] = {
                'years': [y for y, _ in pagerank_scores],
                'pagerank': [s for _, s in pagerank_scores],
                'betweenness': [s for _, s in betweenness_scores]
            }
        
        except Exception as e:
            logger.error(f"Error calculating temporal centrality: {e}")
        
        return result
    
    def detect_emerging_topics(self, min_year: int, acceleration_threshold: float = 2.0) -> List[Dict]:
        """
        Identify topics with accelerating growth.
        
        Detects research areas that show significant acceleration in publication
        volume, indicating emerging or rapidly growing fields.
        
        Args:
            min_year: Start year for analysis
            acceleration_threshold: Minimum growth acceleration factor (default 2.0 = 100% increase)
            
        Returns:
            [
                {
                    'topic': str,
                    'acceleration': float,
                    'recent_papers': int,
                    'key_contributors': [authors],
                    'breakthrough_papers': [papers]
                },
                ...
            ]
            
        Example:
            >>> emerging = analytics.detect_emerging_topics(2020, acceleration_threshold=2.5)
            >>> for topic in emerging[:3]:
            ...     print(f"{topic['topic']}: {topic['acceleration']:.1f}x growth")
        """
        emerging_topics = []
        
        if not self._use_neo4j:
            return emerging_topics
        
        try:
            with self.driver.session() as session:
                # Find topics with keywords/descriptions
                query = """
                MATCH (p:Entity)
                WHERE coalesce(p.year, p.publication_year, 2020) >= $min_year
                  AND (p.keywords IS NOT NULL OR p.topics IS NOT NULL)
                WITH p, 
                     coalesce(p.year, p.publication_year, 2020) as year,
                     coalesce(p.keywords, p.topics, []) as keywords
                UNWIND keywords as keyword
                WITH keyword, year, count(p) as paper_count
                WHERE paper_count > 2
                WITH keyword, collect({year: year, count: paper_count}) as timeline
                WHERE size(timeline) >= 2
                RETURN keyword, timeline
                LIMIT 100
                """
                
                result = session.run(query, min_year=min_year)
                
                for record in result:
                    topic = record['topic'] if 'topic' in record else record.get('keyword', 'Unknown')
                    timeline = record['timeline']
                    
                    # Calculate acceleration
                    acceleration = self._calculate_acceleration(timeline)
                    
                    if acceleration >= acceleration_threshold:
                        # Get key contributors
                        contributors_query = """
                        MATCH (a:Author)-[:AUTHORED_BY]-(p:Entity)
                        WHERE (p.keywords CONTAINS $topic OR p.topics CONTAINS $topic)
                          AND coalesce(p.year, p.publication_year, 2020) >= $min_year
                        WITH a, count(p) as paper_count
                        ORDER BY paper_count DESC
                        LIMIT 5
                        RETURN coalesce(a.name, a.id) as author
                        """
                        contributors_result = session.run(contributors_query, topic=topic, min_year=min_year)
                        contributors = [r['author'] for r in contributors_result]
                        
                        emerging_topics.append({
                            'topic': topic,
                            'acceleration': acceleration,
                            'recent_papers': timeline[-1]['count'] if timeline else 0,
                            'key_contributors': contributors,
                            'breakthrough_papers': []  # Could be enhanced
                        })
                
                # Sort by acceleration
                emerging_topics.sort(key=lambda x: x['acceleration'], reverse=True)
        
        except Exception as e:
            logger.error(f"Error detecting emerging topics: {e}")
        
        return emerging_topics
    
    def analyze_citation_velocity(self, paper_id: str) -> Dict:
        """
        Analyze citation accumulation rate over time.
        
        Computes how quickly a paper accumulates citations, its acceleration,
        and predicts future citations based on current trends.
        
        Args:
            paper_id: Paper node ID
            
        Returns:
            {
                'citations_per_year': [(year, count), ...],
                'velocity': float,  # citations per year
                'acceleration': float,  # change in velocity
                'half_life': float,  # years to 50% of total citations
                'prediction_next_year': int
            }
            
        Example:
            >>> velocity = analytics.analyze_citation_velocity("paper123")
            >>> print(f"Velocity: {velocity['velocity']:.1f} citations/year")
            >>> print(f"Predicted next year: {velocity['prediction_next_year']} citations")
        """
        result = {
            'citations_per_year': [],
            'velocity': 0.0,
            'acceleration': 0.0,
            'half_life': None,
            'prediction_next_year': 0
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
                
                timeline_result = session.run(query, paper_id=paper_id)
                citations_timeline = [(r['year'], r['citation_count']) for r in timeline_result]
                result['citations_per_year'] = citations_timeline
                
                if len(citations_timeline) >= 2:
                    # Calculate velocity (average citations per year)
                    total_citations = sum(c for _, c in citations_timeline)
                    years_span = citations_timeline[-1][0] - citations_timeline[0][0] + 1
                    result['velocity'] = total_citations / years_span if years_span > 0 else 0
                    
                    # Calculate acceleration (change in velocity)
                    mid_point = len(citations_timeline) // 2
                    early_velocity = sum(c for _, c in citations_timeline[:mid_point]) / mid_point if mid_point > 0 else 0
                    late_velocity = sum(c for _, c in citations_timeline[mid_point:]) / (len(citations_timeline) - mid_point)
                    result['acceleration'] = late_velocity - early_velocity
                    
                    # Calculate half-life
                    cumulative = 0
                    half_total = total_citations / 2
                    for i, (year, count) in enumerate(citations_timeline):
                        cumulative += count
                        if cumulative >= half_total:
                            result['half_life'] = i
                            break
                    
                    # Simple linear prediction
                    if result['velocity'] > 0:
                        result['prediction_next_year'] = int(result['velocity'] + result['acceleration'])
        
        except Exception as e:
            logger.error(f"Error analyzing citation velocity: {e}")
        
        return result
    
    def compute_h_index_timeline(self, author_id: str) -> Dict:
        """
        Calculate h-index evolution for an author.
        
        Tracks how an author's h-index changes over time, providing insights
        into research productivity and impact trajectory.
        
        Args:
            author_id: Author node ID
            
        Returns:
            {
                'h_index_by_year': [(year, h_index), ...],
                'current_h_index': int,
                'growth_rate': float,
                'projected_h_index_5y': int
            }
            
        Example:
            >>> h_index_data = analytics.compute_h_index_timeline("author123")
            >>> print(f"Current h-index: {h_index_data['current_h_index']}")
            >>> print(f"5-year projection: {h_index_data['projected_h_index_5y']}")
        """
        result = {
            'h_index_by_year': [],
            'current_h_index': 0,
            'growth_rate': 0.0,
            'projected_h_index_5y': 0
        }
        
        if not self._use_neo4j:
            return result
        
        try:
            with self.driver.session() as session:
                # Get all papers by author with their citation counts over time
                query = """
                MATCH (a:Author {id: $author_id})-[:AUTHORED_BY]-(p:Entity)
                WITH p, coalesce(p.year, p.publication_year, 2020) as pub_year
                OPTIONAL MATCH (p)<-[c:CITES]-(citing:Entity)
                WITH p, pub_year, citing, coalesce(citing.year, citing.publication_year, 2020) as cite_year
                WHERE cite_year >= pub_year
                WITH p, pub_year, cite_year, count(citing) as citations_in_year
                RETURN coalesce(p.id, p.name) as paper_id, pub_year, cite_year, citations_in_year
                ORDER BY cite_year
                """
                
                papers_result = session.run(query, author_id=author_id)
                
                # Build citation counts by year
                citation_data = defaultdict(lambda: defaultdict(int))
                all_years = set()
                
                for record in papers_result:
                    paper_id = record['paper_id']
                    cite_year = record['cite_year']
                    citations = record['citations_in_year']
                    citation_data[cite_year][paper_id] = citations
                    all_years.add(cite_year)
                
                # Calculate h-index for each year
                h_indices = []
                for year in sorted(all_years):
                    # Cumulative citations up to this year
                    cumulative_citations = defaultdict(int)
                    for y in all_years:
                        if y <= year:
                            for paper, cites in citation_data[y].items():
                                cumulative_citations[paper] += cites
                    
                    # Calculate h-index
                    citation_counts = sorted(cumulative_citations.values(), reverse=True)
                    h_index = self._calculate_h_index(citation_counts)
                    h_indices.append((year, h_index))
                
                result['h_index_by_year'] = h_indices
                result['current_h_index'] = h_indices[-1][1] if h_indices else 0
                
                # Calculate growth rate
                if len(h_indices) >= 2:
                    years_span = h_indices[-1][0] - h_indices[0][0]
                    h_change = h_indices[-1][1] - h_indices[0][1]
                    result['growth_rate'] = h_change / years_span if years_span > 0 else 0
                    
                    # Project 5 years
                    result['projected_h_index_5y'] = int(result['current_h_index'] + 5 * result['growth_rate'])
        
        except Exception as e:
            logger.error(f"Error computing h-index timeline: {e}")
        
        return result
    
    # Helper methods
    
    def _classify_maturity(self, growth_rate: float, timeline_data: List[Tuple]) -> str:
        """Classify topic maturity based on growth patterns."""
        if growth_rate > 1.0:
            return 'emerging'
        elif growth_rate > 0.2:
            return 'growing'
        elif growth_rate > -0.1:
            return 'mature'
        else:
            return 'declining'
    
    def _extract_trending_keywords(self, session, topic: str, timeline_data: List[Tuple]) -> Dict:
        """Extract trending keywords related to topic."""
        # Simplified implementation - could use TF-IDF or other methods
        return {}
    
    def _build_temporal_subgraph(self, year: int) -> nx.DiGraph:
        """Build citation graph for a specific year."""
        G = nx.DiGraph()
        
        if not self._use_neo4j:
            return G
        
        try:
            with self.driver.session() as session:
                # Get papers published up to this year
                query = """
                MATCH (p:Entity)
                WHERE coalesce(p.year, p.publication_year, 2020) <= $year
                WITH p
                MATCH (p)-[r:CITES]->(cited:Entity)
                RETURN coalesce(p.id, p.name) as source, 
                       coalesce(cited.id, cited.name) as target
                LIMIT 5000
                """
                
                result = session.run(query, year=year)
                for record in result:
                    G.add_edge(record['source'], record['target'])
        
        except Exception as e:
            logger.error(f"Error building temporal subgraph: {e}")
        
        return G
    
    def _determine_trend(self, scores: List[float]) -> str:
        """Determine if trend is rising, stable, or declining."""
        if len(scores) < 2:
            return 'stable'
        
        # Simple linear regression slope
        x = np.arange(len(scores))
        y = np.array(scores)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'rising'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_acceleration(self, timeline: List[Dict]) -> float:
        """Calculate growth acceleration from timeline data."""
        if len(timeline) < 2:
            return 0.0
        
        # Sort by year
        sorted_timeline = sorted(timeline, key=lambda x: x['year'])
        
        # Compare first half vs second half
        mid = len(sorted_timeline) // 2
        early_avg = np.mean([t['count'] for t in sorted_timeline[:mid]])
        late_avg = np.mean([t['count'] for t in sorted_timeline[mid:]])
        
        if early_avg == 0:
            return float('inf') if late_avg > 0 else 0.0
        
        return late_avg / early_avg
    
    def _calculate_h_index(self, citation_counts: List[int]) -> int:
        """Calculate h-index from sorted citation counts."""
        h = 0
        for i, citations in enumerate(citation_counts, 1):
            if citations >= i:
                h = i
            else:
                break
        return h


if __name__ == "__main__":
    import os
    from ..core.graph_manager import GraphManager
    
    # Test configuration
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    print("=" * 80)
    print("Temporal Graph Analytics Test")
    print("=" * 80)
    
    graph_manager = GraphManager(neo4j_uri, neo4j_user, neo4j_password)
    analytics = TemporalGraphAnalytics(graph_manager, neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Test topic evolution
        print("\n1. Topic Evolution Analysis")
        evolution = analytics.analyze_topic_evolution("machine learning", "yearly")
        print(f"   Timeline entries: {len(evolution['timeline'])}")
        print(f"   Growth rate: {evolution['growth_rate']:.2%}")
        print(f"   Maturity stage: {evolution['maturity_stage']}")
        
        # Test emerging topics
        print("\n2. Emerging Topics Detection")
        emerging = analytics.detect_emerging_topics(2020, acceleration_threshold=1.5)
        print(f"   Found {len(emerging)} emerging topics")
        for topic in emerging[:3]:
            print(f"   - {topic['topic']}: {topic['acceleration']:.2f}x acceleration")
        
        print("\n✓ Temporal analytics test complete")
    
    finally:
        analytics.close()
        graph_manager.close()
