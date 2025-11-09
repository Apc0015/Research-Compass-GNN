#!/usr/bin/env python3
"""
Temporal Query System
Time-based queries, trending analysis, and temporal pattern detection.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from neo4j import GraphDatabase
import json


@dataclass
class TemporalResult:
    """Result of a temporal query"""
    timestamp: str
    entities: List[Dict]
    relationships: List[Dict]
    entity_count: int
    relationship_count: int


@dataclass
class TrendingEntity:
    """Entity with trending information"""
    entity_name: str
    entity_type: str
    current_connections: int
    previous_connections: int
    growth_rate: float
    trend: str  # 'rising', 'falling', 'stable'


@dataclass
class TemporalPattern:
    """Temporal pattern detected in the graph"""
    pattern_type: str
    entities_involved: List[str]
    first_occurrence: str
    last_occurrence: str
    frequency: int


class TemporalQuerySystem:
    """Advanced temporal queries and trend analysis"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize temporal query system.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection"""
        self.driver.close()

    def query_by_date_range(self,
                           start_date: str,
                           end_date: str,
                           entity_types: Optional[List[str]] = None,
                           limit: int = 100) -> TemporalResult:
        """
        Query entities created within a date range.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            entity_types: Filter by entity types
            limit: Maximum results

        Returns:
            TemporalResult object
        """
        with self.driver.session() as session:
            # Build query
            where_clauses = [
                "e.created_at >= $start_date",
                "e.created_at <= $end_date"
            ]
            params = {
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit
            }

            if entity_types:
                where_clauses.append("e.type IN $entity_types")
                params['entity_types'] = entity_types

            where_clause = " AND ".join(where_clauses)

            # Get entities
            entity_query = f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN e.name as name, e.type as type, e.created_at as created_at
                ORDER BY e.created_at DESC
                LIMIT $limit
            """
            result = session.run(entity_query, params)
            entities = [dict(record) for record in result]

            # Get relationships in this timeframe
            rel_query = f"""
                MATCH (e:Entity)-[r:RELATED]->(t:Entity)
                WHERE {where_clause}
                  AND r.created_at >= $start_date
                  AND r.created_at <= $end_date
                RETURN e.name as source, t.name as target,
                       type(r) as rel_type, r.created_at as created_at
                LIMIT $limit
            """
            result = session.run(rel_query, params)
            relationships = [dict(record) for record in result]

            return TemporalResult(
                timestamp=end_date,
                entities=entities,
                relationships=relationships,
                entity_count=len(entities),
                relationship_count=len(relationships)
            )

    def get_changes_between(self,
                           start_time: str,
                           end_time: str) -> Dict:
        """
        Get all changes between two timestamps.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Dictionary with added/removed entities and relationships
        """
        with self.driver.session() as session:
            # Entities added
            added_query = """
                MATCH (e:Entity)
                WHERE e.created_at >= $start_time
                  AND e.created_at <= $end_time
                RETURN e.name as name, e.type as type, e.created_at as created_at
                ORDER BY e.created_at
            """
            result = session.run(added_query, {
                'start_time': start_time,
                'end_time': end_time
            })
            added_entities = [dict(record) for record in result]

            # Entities deleted (if using soft delete with deleted_at)
            deleted_query = """
                MATCH (e:Entity)
                WHERE e.deleted_at >= $start_time
                  AND e.deleted_at <= $end_time
                RETURN e.name as name, e.type as type, e.deleted_at as deleted_at
                ORDER BY e.deleted_at
            """
            result = session.run(deleted_query, {
                'start_time': start_time,
                'end_time': end_time
            })
            deleted_entities = [dict(record) for record in result]

            # New relationships
            new_rels_query = """
                MATCH (s:Entity)-[r:RELATED]->(t:Entity)
                WHERE r.created_at >= $start_time
                  AND r.created_at <= $end_time
                RETURN s.name as source, t.name as target,
                       type(r) as rel_type, r.created_at as created_at
                ORDER BY r.created_at
            """
            result = session.run(new_rels_query, {
                'start_time': start_time,
                'end_time': end_time
            })
            new_relationships = [dict(record) for record in result]

            return {
                'time_range': {'start': start_time, 'end': end_time},
                'added_entities': added_entities,
                'deleted_entities': deleted_entities,
                'new_relationships': new_relationships,
                'summary': {
                    'entities_added': len(added_entities),
                    'entities_deleted': len(deleted_entities),
                    'relationships_added': len(new_relationships)
                }
            }

    def timeline_query(self,
                      entity_name: str,
                      include_neighbors: bool = True) -> List[Dict]:
        """
        Get timeline of an entity's evolution.

        Args:
            entity_name: Name of the entity
            include_neighbors: Include neighbor changes

        Returns:
            List of timeline events
        """
        with self.driver.session() as session:
            timeline = []

            # Entity creation
            creation_query = """
                MATCH (e:Entity)
                WHERE e.name = $entity_name
                RETURN e.created_at as timestamp, 'CREATED' as event_type,
                       e.name as entity, e.type as entity_type
            """
            result = session.run(creation_query, {'entity_name': entity_name})
            timeline.extend([dict(record) for record in result])

            # Relationships added
            if include_neighbors:
                rel_query = """
                    MATCH (e:Entity {name: $entity_name})-[r:RELATED]-(other:Entity)
                    WHERE r.created_at IS NOT NULL
                    RETURN r.created_at as timestamp, 'RELATIONSHIP_ADDED' as event_type,
                           other.name as related_entity, type(r) as rel_type
                    ORDER BY r.created_at
                """
                result = session.run(rel_query, {'entity_name': entity_name})
                timeline.extend([dict(record) for record in result])

            # Sort by timestamp
            timeline.sort(key=lambda x: x.get('timestamp', ''))

            return timeline

    def trending_entities(self,
                         time_window_hours: int = 24,
                         min_growth_rate: float = 0.1,
                         limit: int = 20) -> List[TrendingEntity]:
        """
        Find trending entities (rapidly gaining connections).

        Args:
            time_window_hours: Time window to analyze
            min_growth_rate: Minimum growth rate to consider trending
            limit: Maximum results

        Returns:
            List of TrendingEntity objects
        """
        with self.driver.session() as session:
            # Calculate cutoff time
            cutoff_time = (
                datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            ).isoformat()

            query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r_all:RELATED]-()
                WITH e, count(DISTINCT r_all) as current_connections
                OPTIONAL MATCH (e)-[r_recent:RELATED]-()
                WHERE r_recent.created_at >= $cutoff_time
                WITH e, current_connections, count(DISTINCT r_recent) as recent_connections
                WHERE recent_connections > 0
                WITH e, current_connections,
                     current_connections - recent_connections as previous_connections,
                     recent_connections
                WHERE previous_connections > 0
                WITH e, current_connections, previous_connections,
                     toFloat(recent_connections) / toFloat(previous_connections) as growth_rate
                WHERE growth_rate >= $min_growth_rate
                RETURN e.name as entity_name, e.type as entity_type,
                       current_connections, previous_connections, growth_rate
                ORDER BY growth_rate DESC
                LIMIT $limit
            """

            result = session.run(query, {
                'cutoff_time': cutoff_time,
                'min_growth_rate': min_growth_rate,
                'limit': limit
            })

            trending = []
            for record in result:
                growth = record['growth_rate']
                trend = 'rising' if growth > 0.2 else 'stable'

                trending.append(TrendingEntity(
                    entity_name=record['entity_name'],
                    entity_type=record['entity_type'],
                    current_connections=record['current_connections'],
                    previous_connections=record['previous_connections'],
                    growth_rate=growth,
                    trend=trend
                ))

            return trending

    def temporal_patterns(self,
                         pattern_type: str = 'recurring',
                         min_frequency: int = 2,
                         limit: int = 10) -> List[TemporalPattern]:
        """
        Detect temporal patterns in the graph.

        Args:
            pattern_type: Type of pattern ('recurring', 'seasonal')
            min_frequency: Minimum occurrences
            limit: Maximum results

        Returns:
            List of TemporalPattern objects
        """
        with self.driver.session() as session:
            if pattern_type == 'recurring':
                # Find entity pairs that connect multiple times
                query = """
                    MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
                    WHERE r.created_at IS NOT NULL
                    WITH e1, e2, collect(r.created_at) as timestamps
                    WHERE size(timestamps) >= $min_frequency
                    RETURN e1.name as entity1, e2.name as entity2,
                           timestamps[0] as first_occurrence,
                           timestamps[-1] as last_occurrence,
                           size(timestamps) as frequency
                    ORDER BY frequency DESC
                    LIMIT $limit
                """

                result = session.run(query, {
                    'min_frequency': min_frequency,
                    'limit': limit
                })

                patterns = []
                for record in result:
                    patterns.append(TemporalPattern(
                        pattern_type='recurring_connection',
                        entities_involved=[record['entity1'], record['entity2']],
                        first_occurrence=record['first_occurrence'],
                        last_occurrence=record['last_occurrence'],
                        frequency=record['frequency']
                    ))

                return patterns

            return []

    def get_graph_evolution_stats(self, time_buckets: int = 10) -> List[Dict]:
        """
        Get graph evolution statistics over time.

        Args:
            time_buckets: Number of time periods to analyze

        Returns:
            List of statistics per time period
        """
        with self.driver.session() as session:
            # Get min and max timestamps
            range_query = """
                MATCH (e:Entity)
                WHERE e.created_at IS NOT NULL
                RETURN min(e.created_at) as min_time, max(e.created_at) as max_time
            """
            result = session.run(range_query)
            record = result.single()

            if not record or not record['min_time']:
                return []

            min_time = record['min_time']
            max_time = record['max_time']

            # For simplicity, group by date
            stats_query = """
                MATCH (e:Entity)
                WHERE e.created_at IS NOT NULL
                WITH date(e.created_at) as creation_date
                WITH creation_date, count(*) as count
                RETURN toString(creation_date) as date, count
                ORDER BY creation_date
            """

            result = session.run(stats_query)
            stats = [dict(record) for record in result]

            # Calculate cumulative counts
            cumulative = 0
            for stat in stats:
                cumulative += stat['count']
                stat['cumulative_count'] = cumulative

            return stats

    def compare_time_periods(self,
                            period1_start: str,
                            period1_end: str,
                            period2_start: str,
                            period2_end: str) -> Dict:
        """
        Compare two time periods.

        Args:
            period1_start: Period 1 start time
            period1_end: Period 1 end time
            period2_start: Period 2 start time
            period2_end: Period 2 end time

        Returns:
            Comparison statistics
        """
        period1 = self.query_by_date_range(period1_start, period1_end)
        period2 = self.query_by_date_range(period2_start, period2_end)

        return {
            'period1': {
                'start': period1_start,
                'end': period1_end,
                'entity_count': period1.entity_count,
                'relationship_count': period1.relationship_count
            },
            'period2': {
                'start': period2_start,
                'end': period2_end,
                'entity_count': period2.entity_count,
                'relationship_count': period2.relationship_count
            },
            'comparison': {
                'entity_change': period2.entity_count - period1.entity_count,
                'entity_change_pct': (
                    ((period2.entity_count - period1.entity_count) / period1.entity_count * 100)
                    if period1.entity_count > 0 else 0
                ),
                'relationship_change': period2.relationship_count - period1.relationship_count,
                'relationship_change_pct': (
                    ((period2.relationship_count - period1.relationship_count) /
                     period1.relationship_count * 100)
                    if period1.relationship_count > 0 else 0
                )
            }
        }

    def get_activity_timeline(self, granularity: str = 'day', limit: int = 30) -> List[Dict]:
        """
        Get activity timeline with specified granularity.

        Args:
            granularity: 'hour', 'day', 'week', 'month'
            limit: Number of time periods

        Returns:
            List of activity statistics
        """
        with self.driver.session() as session:
            query = """
                MATCH (e:Entity)
                WHERE e.created_at IS NOT NULL
                WITH date(e.created_at) as activity_date
                RETURN toString(activity_date) as date,
                       count(*) as entity_count
                ORDER BY activity_date DESC
                LIMIT $limit
            """

            result = session.run(query, {'limit': limit})
            return [dict(record) for record in result]


# Main execution
if __name__ == "__main__":
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable must be set")

    print("=" * 80)
    print("TEMPORAL QUERY SYSTEM")
    print("=" * 80)

    tq = TemporalQuerySystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Get activity timeline
        print("\n📅 Activity Timeline (Last 10 Days):")
        print("-" * 80)
        timeline = tq.get_activity_timeline(granularity='day', limit=10)
        for item in timeline:
            print(f"  {item['date']}: {item['entity_count']} entities added")

        # Graph evolution stats
        print("\n📈 Graph Evolution Statistics:")
        print("-" * 80)
        evolution = tq.get_graph_evolution_stats(time_buckets=10)
        for stat in evolution[-5:]:  # Last 5 periods
            print(f"  {stat['date']}: +{stat['count']} entities (Total: {stat['cumulative_count']})")

        # Trending entities (if recent data exists)
        print("\n🔥 Trending Entities (Last 24 Hours):")
        print("-" * 80)
        trending = tq.trending_entities(time_window_hours=24, min_growth_rate=0.1, limit=5)
        if trending:
            for entity in trending:
                print(f"  {entity.entity_name} ({entity.entity_type})")
                print(f"    Growth: {entity.growth_rate:.2%} ({entity.trend})")
                print(f"    Connections: {entity.previous_connections} → {entity.current_connections}")
        else:
            print("  No trending entities found")

    finally:
        tq.close()
