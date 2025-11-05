#!/usr/bin/env python3
"""
Advanced Query System
Multi-hop reasoning, graph-based summarization, and intelligent querying
"""

# Standard library imports
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Third-party imports
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

class AdvancedQuerySystem:
    """Advanced querying with graph reasoning"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def close(self):
        """Close database connection"""
        self.driver.close()

    def find_connection(self, entity1: str, entity2: str, max_hops: int = 5) -> Dict:
        """Find how two entities are connected"""

        with self.driver.session() as session:
            # Find shortest path
            query = f"""
                MATCH (e1:Entity), (e2:Entity)
                WHERE toLower(e1.text) CONTAINS toLower($entity1)
                  AND toLower(e2.text) CONTAINS toLower($entity2)
                WITH e1, e2
                MATCH path = shortestPath((e1)-[*..{max_hops}]-(e2))
                RETURN [node in nodes(path) | node.text] as path,
                       [rel in relationships(path) | type(rel)] as rel_types,
                       length(path) as hops
                LIMIT 1
            """

            result = session.run(query, entity1=entity1, entity2=entity2)
            record = result.single()

            if not record:
                return {
                    'connected': False,
                    'message': f"No connection found between '{entity1}' and '{entity2}' within {max_hops} hops"
                }

            path = record['path']
            rel_types = record['rel_types']
            hops = record['hops']

            # Build explanation
            explanation = []
            for i in range(len(path) - 1):
                explanation.append(f"{path[i]} -> {path[i+1]}")

            return {
                'connected': True,
                'entity1': entity1,
                'entity2': entity2,
                'path': path,
                'relationships': rel_types,
                'hops': hops,
                'explanation': " -> ".join(path)
            }

    def get_entity_context(self, entity_text: str, depth: int = 2) -> Dict:
        """Get rich context about an entity from the graph"""

        with self.driver.session() as session:
            # Find the entity
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.text) = toLower($text)
                RETURN e.id as id, e.text as text, e.type as type
                LIMIT 1
            """, text=entity_text)

            entity = result.single()
            if not entity:
                return {'found': False, 'message': f"Entity '{entity_text}' not found"}

            entity_id = entity['id']

            # Get direct connections
            result = session.run(f"""
                MATCH (e:Entity {{id: $entity_id}})-[r]-(connected:Entity)
                RETURN connected.text as text,
                       connected.type as type,
                       type(r) as relationship,
                       CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
                LIMIT 50
            """, entity_id=entity_id)

            connections = [dict(record) for record in result]

            # Get neighbors by type
            by_type = {}
            for conn in connections:
                conn_type = conn['type']
                if conn_type not in by_type:
                    by_type[conn_type] = []
                by_type[conn_type].append(conn['text'])

            # Get entity importance (degree centrality)
            degree = len(connections)

            return {
                'found': True,
                'entity': entity['text'],
                'type': entity['type'],
                'connections': connections,
                'connection_count': len(connections),
                'connections_by_type': by_type,
                'importance_score': min(degree / 10.0, 1.0)  # Normalized score
            }

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search across all entities"""

        # Get all entities
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.id as id, e.text as text, e.type as type
                LIMIT 1000
            """)

            entities = [dict(record) for record in result]

        if not entities:
            return []

        # Create embeddings
        query_embedding = self.embedding_model.encode([query])[0]
        entity_texts = [e['text'] for e in entities]
        entity_embeddings = self.embedding_model.encode(entity_texts)

        # Calculate cosine similarity
        similarities = []
        for i, emb in enumerate(entity_embeddings):
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append((entities[i], similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k with context
        results = []
        for entity, score in similarities[:top_k]:
            context = self.get_entity_context(entity['text'])
            results.append({
                'entity': entity['text'],
                'type': entity['type'],
                'similarity': float(score),
                'connections': context.get('connection_count', 0),
                'related_entities': list(context.get('connections_by_type', {}).keys())
            })

        return results

    def multi_hop_query(self, start_entity: str, target_type: str, max_hops: int = 3) -> List[Dict]:
        """Find entities of a specific type within N hops of a start entity"""

        with self.driver.session() as session:
            query = f"""
                MATCH (start:Entity)
                WHERE toLower(start.text) CONTAINS toLower($start_entity)
                WITH start
                MATCH path = (start)-[*1..{max_hops}]-(target:Entity)
                WHERE target.type = $target_type
                WITH DISTINCT target, length(path) as distance
                ORDER BY distance ASC
                LIMIT 20
                RETURN target.text as entity, target.type as type, distance
            """

            result = session.run(query, start_entity=start_entity, target_type=target_type)
            return [dict(record) for record in result]

    def graph_based_summary(self, topic: Optional[str] = None, top_k: int = 10) -> Dict:
        """Generate summary of the knowledge graph or a specific topic"""

        with self.driver.session() as session:
            # Get entity type distribution
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """)
            type_distribution = [dict(record) for record in result]

            # Get most connected entities
            result = session.run("""
                MATCH (e:Entity)-[r]-()
                WITH e, count(r) as connections
                ORDER BY connections DESC
                LIMIT $top_k
                RETURN e.text as entity, e.type as type, connections
            """, top_k=top_k)
            hub_entities = [dict(record) for record in result]

            # If topic specified, filter
            if topic:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.text) CONTAINS toLower($topic)
                       OR toLower(e.type) CONTAINS toLower($topic)
                    WITH e
                    MATCH (e)-[r]-(connected:Entity)
                    RETURN e.text as entity, e.type as type,
                           collect(DISTINCT connected.type) as connected_types,
                           count(r) as connections
                    ORDER BY connections DESC
                    LIMIT $top_k
                """, topic=topic, top_k=top_k)
                topic_entities = [dict(record) for record in result]

                return {
                    'topic': topic,
                    'entities_found': len(topic_entities),
                    'key_entities': topic_entities,
                    'entity_types': type_distribution
                }

            # Overall summary
            total_entities = sum(t['count'] for t in type_distribution)
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            total_relationships = result.single()['count']

            return {
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'entity_type_distribution': type_distribution,
                'hub_entities': hub_entities,
                'most_common_type': type_distribution[0] if type_distribution else None
            }

    def find_similar_entities(self, entity_text: str, top_k: int = 10) -> List[Dict]:
        """Find entities similar to a given entity based on connections"""

        with self.driver.session() as session:
            # Find entities with similar connection patterns
            query = """
                MATCH (e:Entity)
                WHERE toLower(e.text) = toLower($entity_text)
                WITH e
                MATCH (e)-[r]-(neighbor:Entity)
                WITH e, collect(DISTINCT neighbor.type) as my_types, count(r) as my_connections
                MATCH (other:Entity)-[r2]-(neighbor2:Entity)
                WHERE other <> e
                WITH other, collect(DISTINCT neighbor2.type) as other_types,
                     count(r2) as other_connections, my_types
                WITH other, other_types, other_connections,
                     [type IN my_types WHERE type IN other_types] as common_types
                WHERE size(common_types) > 0
                RETURN other.text as entity, other.type as type,
                       size(common_types) as similarity_score,
                       other_connections as connections
                ORDER BY similarity_score DESC, connections DESC
                LIMIT $top_k
            """

            result = session.run(query, entity_text=entity_text, top_k=top_k)
            return [dict(record) for record in result]

    def export_query_examples(self, output_file: str = "query_examples.txt"):
        """Generate example queries users can try"""

        output_path = Path("output") / output_file
        output_path.parent.mkdir(exist_ok=True)

        examples = """
========================================
ADVANCED QUERY EXAMPLES
========================================

1. FIND CONNECTIONS
-------------------
Find how two entities are related:
  >>> query_system.find_connection("AI", "healthcare")
  >>> query_system.find_connection("Tesla", "Elon Musk")

2. GET ENTITY CONTEXT
---------------------
Get detailed information about an entity:
  >>> query_system.get_entity_context("OpenAI")
  >>> query_system.get_entity_context("Python")

3. SEMANTIC SEARCH
------------------
Find entities semantically similar to a query:
  >>> query_system.semantic_search("machine learning algorithms")
  >>> query_system.semantic_search("renewable energy")

4. MULTI-HOP QUERIES
--------------------
Find entities of a type within N hops:
  >>> query_system.multi_hop_query("AI", "Person", max_hops=3)
  >>> query_system.multi_hop_query("Tesla", "Organization", max_hops=2)

5. GRAPH SUMMARIES
------------------
Get summary of the entire graph:
  >>> query_system.graph_based_summary()

Get summary of a specific topic:
  >>> query_system.graph_based_summary(topic="climate")

6. FIND SIMILAR ENTITIES
-------------------------
Find entities with similar connection patterns:
  >>> query_system.find_similar_entities("Google")
  >>> query_system.find_similar_entities("JavaScript")

========================================
EXAMPLE USAGE IN PYTHON
========================================

from advanced_query import AdvancedQuerySystem

# Initialize
query_system = AdvancedQuerySystem()

try:
    # Example 1: Find connections
    result = query_system.find_connection("AI", "healthcare")
    if result['connected']:
        print(f"Connection: {result['explanation']}")

    # Example 2: Get entity context
    context = query_system.get_entity_context("OpenAI")
    if context['found']:
        print(f"Connections: {context['connection_count']}")
        print(f"Connected types: {list(context['connections_by_type'].keys())}")

    # Example 3: Semantic search
    results = query_system.semantic_search("artificial intelligence")
    for r in results[:5]:
        print(f"  {r['entity']} ({r['type']}) - Score: {r['similarity']:.3f}")

finally:
    query_system.close()
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(examples)

        print(f"✓ Query examples saved to: {output_path}")
        return output_path

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED QUERY SYSTEM")
    print("=" * 80)

    query_system = AdvancedQuerySystem()

    try:
        # Example 1: Graph summary
        print("\n📊 GRAPH SUMMARY")
        print("-" * 80)
        summary = query_system.graph_based_summary()
        print(f"Total Entities: {summary['total_entities']}")
        print(f"Total Relationships: {summary['total_relationships']}")
        print("\nTop Entity Types:")
        for et in summary['entity_type_distribution'][:5]:
            print(f"  {et['type']}: {et['count']}")
        print("\nHub Entities:")
        for hub in summary['hub_entities'][:5]:
            print(f"  {hub['entity']} ({hub['type']}) - {hub['connections']} connections")

        # Example 2: Find connection
        print("\n\n🔍 FINDING CONNECTIONS")
        print("-" * 80)
        # Try to find a connection between two entities
        with query_system.driver.session() as session:
            # Get two random entities
            result = session.run("""
                MATCH (e:Entity)
                WITH e ORDER BY rand() LIMIT 2
                RETURN collect(e.text) as entities
            """)
            entities = result.single()['entities']

            if len(entities) >= 2:
                print(f"Finding connection between '{entities[0]}' and '{entities[1]}'...")
                connection = query_system.find_connection(entities[0], entities[1])
                if connection['connected']:
                    print(f"✓ Connected in {connection['hops']} hops")
                    print(f"  Path: {connection['explanation']}")
                else:
                    print(f"✗ {connection['message']}")

        # Example 3: Semantic search
        print("\n\n🔎 SEMANTIC SEARCH")
        print("-" * 80)
        query = "artificial intelligence"
        print(f"Searching for: '{query}'")
        results = query_system.semantic_search(query, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['entity']} ({r['type']}) - Similarity: {r['similarity']:.3f}")

        # Example 4: Entity context
        print("\n\n📋 ENTITY CONTEXT")
        print("-" * 80)
        # Get context for a hub entity
        if summary['hub_entities']:
            hub = summary['hub_entities'][0]
            entity_name = hub['entity']
            print(f"Getting context for '{entity_name}'...")
            context = query_system.get_entity_context(entity_name)
            if context['found']:
                print(f"  Type: {context['type']}")
                print(f"  Connections: {context['connection_count']}")
                print(f"  Connected to: {', '.join(list(context['connections_by_type'].keys())[:5])}")

        # Export examples
        print("\n\n📝 EXPORTING QUERY EXAMPLES")
        print("-" * 80)
        query_system.export_query_examples()

    finally:
        query_system.close()
