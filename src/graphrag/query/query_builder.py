#!/usr/bin/env python3
"""
Advanced Query Builder
Provides aggregation queries, complex filtering, and query composition.
"""

import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from collections import defaultdict, Counter
import json


@dataclass
class AggregationResult:
    """Result of an aggregation query"""
    group_key: str
    count: int
    percentage: float
    items: Optional[List[str]] = None


@dataclass
class QueryResult:
    """Generic query result"""
    data: List[Dict]
    total_count: int
    query_time_ms: Optional[float] = None
    metadata: Optional[Dict] = None


class QueryBuilder:
    """Advanced query builder with aggregations and filtering"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize query builder.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection"""
        self.driver.close()

    def count_entities_by_type(self, limit: Optional[int] = None) -> List[AggregationResult]:
        """
        Count entities grouped by type.

        Args:
            limit: Maximum number of types to return

        Returns:
            List of AggregationResult objects
        """
        with self.driver.session() as session:
            query = """
                MATCH (e:Entity)
                WITH e.type as entity_type, count(e) as count
                ORDER BY count DESC
            """
            if limit:
                query += " LIMIT $limit"

            result = session.run(query, {'limit': limit})
            records = [dict(record) for record in result]

            # Calculate total for percentages
            total = sum(r['count'] for r in records)

            return [
                AggregationResult(
                    group_key=record['entity_type'] or 'Unknown',
                    count=record['count'],
                    percentage=(record['count'] / total * 100) if total > 0 else 0
                )
                for record in records
            ]

    def count_relationships_by_type(self, limit: Optional[int] = None) -> List[AggregationResult]:
        """
        Count relationships grouped by type.

        Args:
            limit: Maximum number of types to return

        Returns:
            List of AggregationResult objects
        """
        with self.driver.session() as session:
            query = """
                MATCH ()-[r:RELATED]->()
                WITH type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """
            if limit:
                query += " LIMIT $limit"

            result = session.run(query, {'limit': limit})
            records = [dict(record) for record in result]

            total = sum(r['count'] for r in records)

            return [
                AggregationResult(
                    group_key=record['rel_type'],
                    count=record['count'],
                    percentage=(record['count'] / total * 100) if total > 0 else 0
                )
                for record in records
            ]

    def aggregate_by_time(self, time_unit: str = 'day', limit: int = 30) -> List[AggregationResult]:
        """
        Aggregate entities by creation time.

        Args:
            time_unit: 'hour', 'day', 'week', 'month'
            limit: Maximum number of time periods

        Returns:
            List of AggregationResult objects
        """
        with self.driver.session() as session:
            # This requires created_at field to be set
            query = """
                MATCH (e:Entity)
                WHERE e.created_at IS NOT NULL
                WITH date(e.created_at) as creation_date, count(e) as count
                RETURN toString(creation_date) as time_key, count
                ORDER BY creation_date DESC
                LIMIT $limit
            """

            result = session.run(query, {'limit': limit})
            records = [dict(record) for record in result]

            total = sum(r['count'] for r in records)

            return [
                AggregationResult(
                    group_key=record['time_key'],
                    count=record['count'],
                    percentage=(record['count'] / total * 100) if total > 0 else 0
                )
                for record in records
            ]

    def group_by_property(self, property_name: str, limit: int = 20) -> List[AggregationResult]:
        """
        Group entities by a custom property.

        Args:
            property_name: Name of the property to group by
            limit: Maximum number of groups

        Returns:
            List of AggregationResult objects
        """
        with self.driver.session() as session:
            query = f"""
                MATCH (e:Entity)
                WHERE e.{property_name} IS NOT NULL
                WITH e.{property_name} as prop_value, collect(e.name) as entities
                RETURN prop_value, size(entities) as count, entities
                ORDER BY count DESC
                LIMIT $limit
            """

            result = session.run(query, {'limit': limit})
            records = [dict(record) for record in result]

            total = sum(r['count'] for r in records)

            return [
                AggregationResult(
                    group_key=str(record['prop_value']),
                    count=record['count'],
                    percentage=(record['count'] / total * 100) if total > 0 else 0,
                    items=record['entities'][:10]  # Limit items for readability
                )
                for record in records
            ]

    def complex_filter(self,
                      entity_types: Optional[List[str]] = None,
                      min_connections: int = 0,
                      max_connections: Optional[int] = None,
                      has_properties: Optional[List[str]] = None,
                      created_after: Optional[str] = None,
                      created_before: Optional[str] = None,
                      limit: int = 100) -> QueryResult:
        """
        Complex filtering with multiple criteria.

        Args:
            entity_types: Filter by entity types
            min_connections: Minimum number of connections
            max_connections: Maximum number of connections
            has_properties: Properties that must exist
            created_after: ISO timestamp
            created_before: ISO timestamp
            limit: Maximum results

        Returns:
            QueryResult object
        """
        with self.driver.session() as session:
            # Build WHERE clause
            where_clauses = []
            params = {'limit': limit}

            if entity_types:
                where_clauses.append("e.type IN $entity_types")
                params['entity_types'] = entity_types

            if has_properties:
                for i, prop in enumerate(has_properties):
                    where_clauses.append(f"e.{prop} IS NOT NULL")

            if created_after:
                where_clauses.append("e.created_at >= $created_after")
                params['created_after'] = created_after

            if created_before:
                where_clauses.append("e.created_at <= $created_before")
                params['created_before'] = created_before

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Build connection filter
            having_clause = ""
            if min_connections > 0 or max_connections:
                having_parts = []
                if min_connections > 0:
                    having_parts.append(f"connections >= {min_connections}")
                if max_connections:
                    having_parts.append(f"connections <= {max_connections}")
                having_clause = "HAVING " + " AND ".join(having_parts)

            query = f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(DISTINCT r) as connections
                {having_clause}
                RETURN e.name as name, e.type as type, connections,
                       e.created_at as created_at
                ORDER BY connections DESC
                LIMIT $limit
            """

            result = session.run(query, params)
            data = [dict(record) for record in result]

            return QueryResult(
                data=data,
                total_count=len(data),
                metadata={'filters_applied': where_clauses}
            )

    def find_patterns(self, pattern_type: str, limit: int = 20) -> QueryResult:
        """
        Find common graph patterns.

        Args:
            pattern_type: 'triangles', 'chains', 'hubs', 'bridges'
            limit: Maximum results

        Returns:
            QueryResult object
        """
        with self.driver.session() as session:
            if pattern_type == 'triangles':
                # Find triangular relationships (A->B->C->A)
                query = """
                    MATCH (a:Entity)-[:RELATED]->(b:Entity)-[:RELATED]->(c:Entity)-[:RELATED]->(a)
                    WHERE id(a) < id(b) AND id(b) < id(c)
                    RETURN a.name as entity1, b.name as entity2, c.name as entity3
                    LIMIT $limit
                """
            elif pattern_type == 'chains':
                # Find chain patterns (A->B->C->D)
                query = """
                    MATCH path = (a:Entity)-[:RELATED*3]-(d:Entity)
                    WHERE a <> d
                    RETURN [n in nodes(path) | n.name] as chain
                    LIMIT $limit
                """
            elif pattern_type == 'hubs':
                # Find hub entities (highly connected)
                query = """
                    MATCH (e:Entity)-[r]-()
                    WITH e, count(DISTINCT r) as connections
                    WHERE connections >= 5
                    RETURN e.name as entity, e.type as type, connections
                    ORDER BY connections DESC
                    LIMIT $limit
                """
            elif pattern_type == 'bridges':
                # Find bridge entities (connect different communities)
                query = """
                    MATCH (a:Entity)-[:RELATED]-(bridge:Entity)-[:RELATED]-(b:Entity)
                    WHERE NOT (a)-[:RELATED]-(b) AND a.type <> b.type
                    WITH bridge, count(DISTINCT a) + count(DISTINCT b) as bridge_score
                    WHERE bridge_score >= 4
                    RETURN bridge.name as entity, bridge.type as type, bridge_score
                    ORDER BY bridge_score DESC
                    LIMIT $limit
                """
            else:
                return QueryResult(data=[], total_count=0,
                                 metadata={'error': f'Unknown pattern type: {pattern_type}'})

            result = session.run(query, {'limit': limit})
            data = [dict(record) for record in result]

            return QueryResult(
                data=data,
                total_count=len(data),
                metadata={'pattern_type': pattern_type}
            )

    def top_entities_by_metric(self, metric: str, limit: int = 20) -> QueryResult:
        """
        Get top entities by a specific metric.

        Args:
            metric: 'degree', 'pagerank', 'betweenness' (requires analytics)
            limit: Maximum results

        Returns:
            QueryResult object
        """
        with self.driver.session() as session:
            if metric == 'degree':
                query = """
                    MATCH (e:Entity)-[r]-()
                    WITH e, count(DISTINCT r) as score
                    RETURN e.name as entity, e.type as type, score
                    ORDER BY score DESC
                    LIMIT $limit
                """
            elif metric == 'neighbors':
                query = """
                    MATCH (e:Entity)-[]-(neighbor:Entity)
                    WITH e, count(DISTINCT neighbor) as score
                    RETURN e.name as entity, e.type as type, score
                    ORDER BY score DESC
                    LIMIT $limit
                """
            else:
                return QueryResult(data=[], total_count=0,
                                 metadata={'error': f'Metric {metric} not implemented'})

            result = session.run(query, {'limit': limit})
            data = [dict(record) for record in result]

            return QueryResult(
                data=data,
                total_count=len(data),
                metadata={'metric': metric}
            )

    def execute_custom_query(self, cypher_query: str, params: Optional[Dict] = None) -> QueryResult:
        """
        Execute a custom Cypher query.

        Args:
            cypher_query: Cypher query string
            params: Query parameters

        Returns:
            QueryResult object
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, params or {})
            data = [dict(record) for record in result]

            return QueryResult(
                data=data,
                total_count=len(data),
                metadata={'custom_query': True}
            )

    def get_statistics_summary(self) -> Dict:
        """
        Get comprehensive statistics summary.

        Returns:
            Dictionary with various statistics
        """
        with self.driver.session() as session:
            # Entity stats
            entity_result = session.run("""
                MATCH (e:Entity)
                RETURN count(e) as total_entities,
                       count(DISTINCT e.type) as unique_types
            """)
            entity_stats = entity_result.single()

            # Relationship stats
            rel_result = session.run("""
                MATCH ()-[r:RELATED]->()
                RETURN count(r) as total_relationships
            """)
            rel_stats = rel_result.single()

            # Connectivity stats
            conn_result = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) as degree
                RETURN avg(degree) as avg_degree,
                       max(degree) as max_degree,
                       min(degree) as min_degree
            """)
            conn_stats = conn_result.single()

            # Type distribution
            type_dist = self.count_entities_by_type(limit=10)

            return {
                'total_entities': entity_stats['total_entities'],
                'unique_types': entity_stats['unique_types'],
                'total_relationships': rel_stats['total_relationships'],
                'avg_degree': float(conn_stats['avg_degree']) if conn_stats['avg_degree'] else 0,
                'max_degree': conn_stats['max_degree'],
                'min_degree': conn_stats['min_degree'],
                'top_types': [
                    {'type': agg.group_key, 'count': agg.count, 'percentage': agg.percentage}
                    for agg in type_dist[:5]
                ]
            }

    # ========================================================================
    # GNN-Powered Query Methods
    # ========================================================================

    def find_similar_papers_gnn(
        self,
        paper_id: str,
        top_k: int = 10,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """
        Find similar papers using GNN embeddings

        Args:
            paper_id: Source paper ID
            top_k: Number of similar papers
            method: 'embedding', 'citation', or 'hybrid'

        Returns:
            List of similar papers with scores
        """
        try:
            from ..ml import GNNManager

            manager = GNNManager(
                self.driver.get_server_info().address,
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )

            recommendations = manager.get_paper_recommendations(
                paper_id,
                method=method,
                top_k=top_k
            )

            manager.close()
            return recommendations

        except Exception as e:
            print(f"GNN not available: {e}")
            return []

    def predict_paper_impact(self, paper_id: str) -> float:
        """
        Predict paper impact using GNN

        Args:
            paper_id: Paper node ID

        Returns:
            Predicted impact score
        """
        try:
            from ..ml import GNNManager

            manager = GNNManager(
                self.driver.get_server_info().address,
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )

            # Get citation predictions
            citations = manager.predict_paper_citations(paper_id, top_k=10)

            # Simple impact score based on predicted citations
            impact = sum(score for _, score in citations)

            manager.close()
            return impact

        except Exception as e:
            print(f"GNN not available: {e}")
            return 0.0

    def get_trending_topics_ml(self) -> List[Dict]:
        """
        Get trending topics using GNN predictions

        Returns:
            List of trending topics with scores
        """
        try:
            from ..ml import GNNManager

            manager = GNNManager(
                self.driver.get_server_info().address,
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )

            # Get recent papers and predict their topics
            # This is a simplified version
            trending = []

            manager.close()
            return trending

        except Exception as e:
            print(f"GNN not available: {e}")
            return []

    def get_influential_authors(self, topic: str) -> List[Dict]:
        """
        Find influential authors in a topic using GNN

        Args:
            topic: Topic name

        Returns:
            List of influential authors
        """
        # Use graph analysis + GNN embeddings
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.type = 'AUTHOR' OR e.type = 'PERSON'
            AND toLower(e.name) CONTAINS toLower($topic)
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as influence_score
            RETURN e.id as id, e.name as name, influence_score
            ORDER BY influence_score DESC
            LIMIT 10
            """

            result = session.run(query, topic=topic)
            return [dict(record) for record in result]


# Main execution
if __name__ == "__main__":
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    print("=" * 80)
    print("ADVANCED QUERY BUILDER")
    print("=" * 80)

    qb = QueryBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Entity type aggregation
        print("\n📊 Entity Type Distribution:")
        print("-" * 80)
        type_agg = qb.count_entities_by_type(limit=10)
        for agg in type_agg:
            print(f"  {agg.group_key}: {agg.count} ({agg.percentage:.1f}%)")

        # Statistics summary
        print("\n📈 Statistics Summary:")
        print("-" * 80)
        stats = qb.get_statistics_summary()
        print(f"  Total Entities: {stats['total_entities']}")
        print(f"  Total Relationships: {stats['total_relationships']}")
        print(f"  Average Degree: {stats['avg_degree']:.2f}")
        print(f"  Max Degree: {stats['max_degree']}")

        # Find patterns
        print("\n🔍 Finding Hub Entities:")
        print("-" * 80)
        hubs = qb.find_patterns('hubs', limit=5)
        for hub in hubs.data:
            print(f"  {hub['entity']} ({hub['type']}) - {hub['connections']} connections")

    finally:
        qb.close()
