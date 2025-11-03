"""
Graph Manager Module
Manages Neo4j graph database operations for entities and relationships.
"""

from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
import logging
import networkx as nx
from pathlib import Path
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphManager:
    """Manages graph database operations for GraphRAG."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the graph manager.

        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        # Store connection details for testing
        self.uri = uri
        self.user = user
        self.password = password

        # Attempt to connect to Neo4j; if unavailable, fall back to an in-memory NetworkX graph
        self.driver = None
        self._use_neo4j = False
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self._use_neo4j = True
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.warning(f"Neo4j unavailable, falling back to in-memory graph: {e}")
            # In-memory fallback
            self._graph = nx.DiGraph()
            # Simple counters to simulate node/relationship IDs
            self._node_props = {}

    def close(self):
        """Close the database connection."""
        if getattr(self, 'driver', None):
            try:
                self.driver.close()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_entity(self, entity_name: str, entity_type: str, properties: Dict = None):
        """
        Create an entity node in the graph.

        Args:
            entity_name: Name of the entity
            entity_type: Type/label of the entity
            properties: Additional properties for the entity
        """
        properties = dict(properties or {})

        # Ensure canonical fields exist so other modules/tests can rely on them
        if 'id' not in properties:
            properties['id'] = uuid.uuid4().hex
        # Keep name/text consistent with older code that expects name/text fields
        properties.setdefault('name', entity_name)
        properties.setdefault('text', entity_name)
        properties.setdefault('type', entity_type)
        properties.setdefault('created_at', datetime.utcnow().isoformat())

        if self._use_neo4j:
            with self.driver.session() as session:
                # MERGE by id to avoid accidental duplicates, but always set name/text/type properties
                session.run(
                    f"""
                    MERGE (e:{entity_type} {{id: $id}})
                    SET e += $properties
                    """,
                    id=properties['id'],
                    properties=properties
                )
        else:
            # Use generated id as key in the in-memory graph
            key = properties['id']
            node_props = dict(properties)
            # store a label compatible with older code
            node_props.setdefault('label', entity_type)
            self._graph.add_node(key, **node_props)
            self._node_props[key] = node_props

    def create_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict = None
    ):
        """
        Create a relationship between two entities.
        Uses case-insensitive matching to find nodes.

        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship
            properties: Additional properties for the relationship
        """
        properties = properties or {}
        if self._use_neo4j:
            with self.driver.session() as session:
                # Try case-insensitive match to be more flexible
                result = session.run(
                    """
                    // Find source and target nodes (case-insensitive by name)
                    MATCH (s) WHERE toLower(coalesce(s.name, s.text, s.id)) = toLower($source)
                    MATCH (t) WHERE toLower(coalesce(t.name, t.text, t.id)) = toLower($target)
                    WITH s, t
                    MERGE (s)-[r:RELATED {type: $relation_type}]->(t)
                    SET r += $properties
                    RETURN count(r) as created
                    """,
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    properties=properties
                )

                # Check if relationship was actually created
                record = result.single()
                if not record or record['created'] == 0:
                    raise ValueError(f"Could not find nodes: '{source}' or '{target}'")
        else:
            # Case-insensitive match for node keys
            src_key = next((n for n in self._graph.nodes if n.lower() == source.lower()), None)
            tgt_key = next((n for n in self._graph.nodes if n.lower() == target.lower()), None)
            if not src_key or not tgt_key:
                raise ValueError(f"Could not find nodes: '{source}' or '{target}'")

            # store edges with type and label 'RELATED' for compatibility
            self._graph.add_edge(src_key, tgt_key, type=relation_type, label='RELATED', **properties)

    def query_neighbors(self, entity_name: str, max_depth: int = 1) -> List[Dict]:
        """
        Query neighboring entities up to a given depth.

        Args:
            entity_name: Name of the entity to start from
            max_depth: Maximum depth to traverse

        Returns:
            List of neighboring entities with their relationships
        """
        if self._use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    """
                    // Match the start node by id, name or text (case-insensitive)
                    MATCH (e)
                    WHERE toLower(coalesce(e.id, e.name, e.text)) = toLower($name)
                    WITH e
                    MATCH path = (e)-[*1..$max_depth]-(neighbor)
                    RETURN DISTINCT coalesce(neighbor.name, neighbor.text, neighbor.id) as name,
                           labels(neighbor)[0] as type,
                           length(path) as distance
                    ORDER BY distance
                    """,
                    name=entity_name,
                    max_depth=max_depth
                )
                return [dict(record) for record in result]
        else:
            # BFS up to max_depth
            if entity_name not in self._graph:
                # try case-insensitive match
                entity_name = next((n for n in self._graph.nodes if n.lower() == entity_name.lower()), entity_name)

            nodes = []
            for target, dist in nx.single_source_shortest_path_length(self._graph.to_undirected(), entity_name, cutoff=max_depth).items():
                if target == entity_name:
                    continue
                nodes.append({'name': target, 'type': self._graph.nodes[target].get('label', ''), 'distance': dist})
            return nodes

    def get_entity_context(self, entity_name: str, max_depth: int = 2) -> str:
        """
        Get textual context about an entity from the graph.

        Args:
            entity_name: Name of the entity
            max_depth: Maximum depth to traverse

        Returns:
            Formatted text description of the entity and its context
        """
        neighbors = self.query_neighbors(entity_name, max_depth)

        context_lines = [f"Entity: {entity_name}"]
        if neighbors:
            context_lines.append("Related entities:")
            for neighbor in neighbors:
                context_lines.append(
                    f"  - {neighbor['name']} ({neighbor['type']}) "
                    f"at distance {neighbor['distance']}"
                )
        else:
            context_lines.append("No related entities found.")

        return "\n".join(context_lines)

    def clear_graph(self):
        """Delete all nodes and relationships from the graph."""
        if self._use_neo4j:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph cleared")
        else:
            self._graph.clear()
            self._node_props.clear()
            logger.info("In-memory graph cleared")

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary with node and relationship counts
        """
        if self._use_neo4j:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

                return {
                    "node_count": node_count,
                    "relationship_count": rel_count
                }
        else:
            return {
                "node_count": self._graph.number_of_nodes(),
                "relationship_count": self._graph.number_of_edges()
            }

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test Neo4j database connection.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self._use_neo4j:
            try:
                with self.driver.session() as session:
                    # Get database info
                    result = session.run("CALL dbms.components() YIELD name, versions, edition")
                    record = result.single()

                    if record:
                        version = record.get("versions", ["Unknown"])[0]
                        edition = record.get("edition", "Unknown")

                        # Get stats
                        stats = self.get_graph_stats()

                        return True, (
                            f"✓ Connected to Neo4j {edition} {version}\n"
                            f"  Nodes: {stats['node_count']}, "
                            f"Relationships: {stats['relationship_count']}"
                        )
                    else:
                        return True, "✓ Connected to Neo4j (version unknown)"

            except Exception as e:
                return False, f"✗ Neo4j connection failed: {str(e)}"
        else:
            stats = self.get_graph_stats()
            return False, (
                f"⚠ Using in-memory graph (Neo4j not connected)\n"
                f"  Nodes: {stats['node_count']}, "
                f"Relationships: {stats['relationship_count']}"
            )

    def health_check(self) -> Tuple[bool, str]:
        """
        Lightweight health check wrapper for external callers.

        Returns:
            Tuple[bool, str] where bool indicates healthy status and str is a human message.
        """
        return self.test_connection()

    def reconnect(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Reconnect to Neo4j with new credentials.

        Args:
            uri: New Neo4j URI (optional, uses existing if not provided)
            user: New username (optional, uses existing if not provided)
            password: New password (optional, uses existing if not provided)
        """
        # Update credentials if provided
        if uri:
            self.uri = uri
        if user:
            self.user = user
        if password:
            self.password = password

        # Close existing connection
        self.close()

        # Attempt new connection
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._use_neo4j = True
            logger.info("Successfully reconnected to Neo4j database")
            return True, "✓ Successfully connected to Neo4j"
        except Exception as e:
            logger.error(f"Failed to reconnect to Neo4j: {e}")
            self._use_neo4j = False
            # Initialize in-memory fallback if not already
            if not hasattr(self, '_graph'):
                self._graph = nx.DiGraph()
                self._node_props = {}
            return False, f"✗ Connection failed: {str(e)}"

    def export_subgraph(
        self,
        entity_name: str,
        max_depth: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Export a subgraph around an entity.

        Args:
            entity_name: Central entity name
            max_depth: Maximum depth to include

        Returns:
            Tuple of (nodes, edges) as dictionaries
        """
        if self._use_neo4j:
            with self.driver.session() as session:
                # Get nodes around the entity (match by id/name/text)
                nodes_result = session.run(
                    """
                    MATCH (e)
                    WHERE toLower(coalesce(e.id, e.name, e.text)) = toLower($name)
                    WITH e
                    MATCH path = (e)-[*0..$max_depth]-(n)
                    RETURN DISTINCT coalesce(n.id, n.name, n.text) as id, labels(n)[0] as label, coalesce(n.name, n.text, n.id) as display
                    """,
                    name=entity_name,
                    max_depth=max_depth
                )
                nodes = [dict(record) for record in nodes_result]

                # Get edges
                edges_result = session.run(
                    """
                    MATCH (e)
                    WHERE toLower(coalesce(e.id, e.name, e.text)) = toLower($name)
                    WITH e
                    MATCH path = (e)-[*0..$max_depth]-(n)
                    MATCH (n1)-[r]->(n2)
                    WHERE toLower(coalesce(n1.id, n1.name, n1.text)) = toLower($name)
                       OR toLower(coalesce(n2.id, n2.name, n2.text)) = toLower($name)
                       OR ( (n1)-[*0..$max_depth]-(e) )
                       OR ( (n2)-[*0..$max_depth]-(e) )
                    RETURN DISTINCT coalesce(n1.id, n1.name, n1.text) as source, coalesce(n2.id, n2.name, n2.text) as target,
                           type(r) as type, r.type as label
                    """,
                    name=entity_name,
                    max_depth=max_depth
                )
                edges = [dict(record) for record in edges_result]

                return nodes, edges
        else:
            # Collect nodes within max_depth using BFS
            if entity_name not in self._graph:
                entity_name = next((n for n in self._graph.nodes if n.lower() == entity_name.lower()), entity_name)

            sub_nodes = list(nx.single_source_shortest_path_length(self._graph.to_undirected(), entity_name, cutoff=max_depth).keys())
            nodes = [{'name': n, 'label': self._graph.nodes[n].get('label', '')} for n in sub_nodes]

            edges = []
            for u, v, data in self._graph.edges(data=True):
                if u in sub_nodes and v in sub_nodes:
                    edges.append({'source': u, 'target': v, 'type': data.get('type', ''), 'label': data.get('type', '')})

            return nodes, edges

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for entities by name.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        if self._use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE toLower(coalesce(n.name, n.text, n.id)) CONTAINS toLower($query)
                    RETURN coalesce(n.id, n.name, n.text) as id, coalesce(n.name, n.text, n.id) as name, labels(n)[0] as type
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit
                )
                return [dict(record) for record in result]
        else:
            matches = [
                {'name': n, 'type': self._graph.nodes[n].get('label', '')}
                for n in self._graph.nodes
                if query.lower() in n.lower()
            ]
            return matches[:limit]
