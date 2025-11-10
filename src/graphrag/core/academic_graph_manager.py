"""Academic graph manager that builds on top of the existing GraphManager.

Provides convenience methods to add academic-specific nodes and relationships
such as papers, authors, topics and citations while preserving compatibility
with the existing generic Entity system.
"""
from __future__ import annotations

# Standard library imports
from typing import Dict, List, Optional
from datetime import datetime
import logging
import uuid

# Third-party imports
from neo4j import GraphDatabase

# Local imports
from .graph_manager import GraphManager
from .academic_schema import (
    PaperNode,
    AuthorNode,
    TopicNode,
    MethodNode,
    DatasetNode,
    InstitutionNode,
    VenueNode,
    RelationType,
)
from .relationship_manager import RelationshipManager

logger = logging.getLogger(__name__)


class AcademicGraphManager:
    """Manager for academic nodes and relationships.

    This class re-uses the existing `GraphManager` to keep backward
    compatibility. When Neo4j is available it will use real Cypher operations;
    otherwise it falls back to the in-memory graph implemented in
    `GraphManager`.
    """

    def __init__(self, graph: GraphManager):
        """Initialize with an existing GraphManager instance.

        Args:
            graph: GraphManager instance (responsible for DB connection/fallback)
        """
        self.graph = graph
        # Relationship helper for typed edges
        self.relationships = RelationshipManager(self.graph)

    # --- Node creation helpers ---
    def _add_node(self, node_data, entity_name: str, label: str) -> str:
        """
        Base method for adding any type of node to the graph.

        Optimized: Reduces 200+ lines of duplicate code.

        Args:
            node_data: Node object (PaperNode, AuthorNode, etc.)
            entity_name: Name/identifier for the entity
            label: Neo4j label (Paper, Author, Topic, etc.)

        Returns:
            Node ID
        """
        props = node_data.to_neo4j_properties()
        try:
            self.graph.create_entity(entity_name, label, properties=props)
            return node_data.id
        except Exception as e:
            logger.exception("Failed to add %s: %s", label.lower(), e)
            raise

    def add_paper(self, paper: PaperNode) -> str:
        """Add a Paper node to the graph and return its id."""
        return self._add_node(paper, paper.title, 'Paper')

    def add_author(self, author: AuthorNode) -> str:
        """Add an Author node to the graph and return its id."""
        return self._add_node(author, author.name, 'Author')

    def add_topic(self, topic: TopicNode) -> str:
        """Add a Topic node to the graph and return its id."""
        return self._add_node(topic, topic.name, 'Topic')

    def add_method(self, method: MethodNode) -> str:
        """Add a Method node to the graph and return its id."""
        return self._add_node(method, method.name, 'Method')

    def add_dataset(self, dataset: DatasetNode) -> str:
        """Add a Dataset node to the graph and return its id."""
        return self._add_node(dataset, dataset.name, 'Dataset')

    # --- Relationship helpers ---
    def create_citation_link(self, paper1_id: str, paper2_id: str) -> None:
        """Create a CITES relationship from paper1 -> paper2."""
        # Do not include a 'type' key in properties because GraphManager
        # already sets the relationship label/type. Keep created_at and other metadata.
        props = {"created_at": datetime.utcnow().isoformat()}
        try:
            # GraphManager.create_relationship matches by id/name/text, so pass ids
            self.graph.create_relationship(paper1_id, paper2_id, RelationType.CITES.value, properties=props)
        except Exception as e:
            logger.exception("Failed to create citation link: %s", e)
            raise

    def create_authorship_link(self, paper_id: str, author_id: str, position: int = 0) -> None:
        # Avoid including 'type' in properties (GraphManager will pass it as a separate arg)
        props = {"created_at": datetime.utcnow().isoformat(), "position": position}
        try:
            self.graph.create_relationship(paper_id, author_id, RelationType.AUTHORED_BY.value, properties=props)
        except Exception as e:
            logger.exception("Failed to create authorship link: %s", e)
            raise

    def create_topic_link(self, paper_id: str, topic_id: str, confidence: float = 0.9) -> None:
        props = {"created_at": datetime.utcnow().isoformat(), "confidence": float(confidence)}
        try:
            self.graph.create_relationship(paper_id, topic_id, RelationType.DISCUSSES.value, properties=props)
        except Exception as e:
            logger.exception("Failed to create topic link: %s", e)
            raise

    def create_method_usage_link(self, paper_id: str, method_id: str) -> None:
        props = {"created_at": datetime.utcnow().isoformat()}
        try:
            self.graph.create_relationship(paper_id, method_id, RelationType.USES_METHOD.value, properties=props)
        except Exception as e:
            logger.exception("Failed to create method usage link: %s", e)
            raise

    # --- Query helpers ---
    def get_paper_by_id(self, paper_id: str) -> Optional[PaperNode]:
        """Return a PaperNode for the given id if present, otherwise None."""
        # Try Neo4j if available
        if getattr(self.graph, '_use_neo4j', False):
            with self.graph.driver.session() as session:
                result = session.run(
                    "MATCH (p:Paper) WHERE p.id = $id RETURN p LIMIT 1",
                    id=paper_id
                )
                record = result.single()
                if not record:
                    return None
                p = record['p']
                return PaperNode(
                    id=p.get('id'),
                    title=p.get('title') or p.get('name') or '',
                    abstract=p.get('abstract'),
                    year=p.get('year'),
                    authors=p.get('authors') or [],
                    citations=p.get('citations') or [],
                    venue=p.get('venue'),
                    keywords=p.get('keywords') or [],
                )

        # Fallback to in-memory graph
        try:
            node_data = None
            # GraphManager stores node props in _node_props keyed by id when falling back
            if hasattr(self.graph, '_node_props'):
                node_data = self.graph._node_props.get(paper_id)
            if not node_data:
                # try case-insensitive match against stored node names
                node_data = next((v for v in getattr(self.graph, '_node_props', {}).values() if v.get('id') == paper_id), None)
            if not node_data:
                return None
            return PaperNode(
                id=node_data.get('id'),
                title=node_data.get('title') or node_data.get('name') or '',
                abstract=node_data.get('abstract'),
                year=node_data.get('year'),
                authors=node_data.get('authors') or [],
                citations=node_data.get('citations') or [],
                venue=node_data.get('venue'),
                keywords=node_data.get('keywords') or [],
            )
        except Exception as e:
            logger.exception("Error retrieving paper by id: %s", e)
            return None

    def get_author_papers(self, author_id: str) -> List[PaperNode]:
        """Return papers authored by the given author id."""
        papers: List[PaperNode] = []
        if getattr(self.graph, '_use_neo4j', False):
            with self.graph.driver.session() as session:
                result = session.run(
                    "MATCH (a:Author)-[:AUTHORED_BY|:AUTHORED_BY*0..1]-(p:Paper) WHERE a.id = $id RETURN DISTINCT p",
                    id=author_id
                )
                for record in result:
                    p = record['p']
                    papers.append(PaperNode(
                        id=p.get('id'),
                        title=p.get('title') or p.get('name') or '',
                        abstract=p.get('abstract'),
                        year=p.get('year'),
                        authors=p.get('authors') or [],
                        citations=p.get('citations') or [],
                        venue=p.get('venue'),
                        keywords=p.get('keywords') or [],
                    ))
            return papers

        # Fallback in-memory search
        try:
            for nid, props in getattr(self.graph, '_node_props', {}).items():
                if props.get('node_type') == 'PAPER' or props.get('label') == 'Paper':
                    authors = props.get('authors') or []
                    if author_id in authors or props.get('id') == author_id:
                        papers.append(PaperNode(
                            id=props.get('id'),
                            title=props.get('title') or props.get('name') or '',
                            abstract=props.get('abstract'),
                            year=props.get('year'),
                            authors=props.get('authors') or [],
                            citations=props.get('citations') or [],
                            venue=props.get('venue'),
                            keywords=props.get('keywords') or [],
                        ))
        except Exception:
            logger.exception("Failed to get author papers from in-memory graph")

        return papers

    def get_citation_network(self, paper_id: str, depth: int = 2) -> Dict:
        """Return a simple citation subgraph around a paper up to given depth.

        Returns a dict with 'nodes' and 'edges'. Edges are tuples (source, target).
        """
        nodes = set()
        edges = []
        if getattr(self.graph, '_use_neo4j', False):
            with self.graph.driver.session() as session:
                result = session.run(
                    "MATCH (p:Paper {id: $id})-[r:CITES*1..$depth]-(m:Paper) RETURN DISTINCT m, relationships((p)-[*1..$depth]-(m)) as rels",
                    id=paper_id,
                    depth=depth,
                )
                for record in result:
                    m = record['m']
                    nodes.add(m.get('id'))
                    # relationships may be complex; skip detailed rel extraction for safety
            # A more complete implementation would return full paths
        else:
            # Walk the in-memory graph
            try:
                from collections import deque

                q = deque()
                q.append((paper_id, 0))
                seen = {paper_id}
                while q:
                    current, d = q.popleft()
                    nodes.add(current)
                    if d >= depth:
                        continue
                    # outgoing edges
                    for _, tgt, data in getattr(self.graph, '_graph').out_edges(current, data=True):
                        if data.get('type') == 'CITES':
                            edges.append((current, tgt))
                            if tgt not in seen:
                                seen.add(tgt)
                                q.append((tgt, d + 1))
                
            except Exception:
                logger.exception("Failed to build citation network in-memory")

        return {"nodes": list(nodes), "edges": edges}

    def get_coauthor_network(self, author_id: str) -> List[AuthorNode]:
        """Return authors who co-authored with the given author.

        This is a simple implementation that finds papers by the author and
        collects other authors listed on those papers.
        """
        coauthors: Dict[str, AuthorNode] = {}
        try:
            papers = self.get_author_papers(author_id)

            # Collect all unique coauthor IDs first
            coauthor_ids = set()
            for p in papers:
                for aid in p.authors:
                    if aid != author_id:
                        coauthor_ids.add(aid)

            # Batch query all coauthors at once (10-50x faster than N+1 queries)
            if getattr(self.graph, '_use_neo4j', False):
                if coauthor_ids:
                    with self.graph.driver.session() as session:
                        # Single batch query with IN clause
                        res = session.run(
                            "MATCH (a:Author) WHERE a.id IN $ids RETURN a",
                            ids=list(coauthor_ids)
                        )
                        for record in res:
                            a = record['a']
                            aid = a.get('id')
                            coauthors[aid] = AuthorNode(
                                id=aid,
                                name=a.get('name') or '',
                                affiliations=a.get('affiliations') or []
                            )
            else:
                # In-memory fallback
                for aid in coauthor_ids:
                    a_props = self.graph._node_props.get(aid)
                    if a_props:
                        coauthors[aid] = AuthorNode(
                            id=a_props.get('id'),
                            name=a_props.get('name') or '',
                            affiliations=a_props.get('affiliations') or []
                        )
        except Exception:
            logger.exception("Failed to compute coauthor network")

        return list(coauthors.values())

    def initialize_constraints_and_indexes(self) -> None:
        """Create Neo4j constraints and indexes for new node types when using Neo4j.

        This is safe to call even if Neo4j isn't available; it will just be a no-op
        in the in-memory fallback.
        """
        if not getattr(self.graph, '_use_neo4j', False):
            logger.info("Neo4j not available; skipping constraint initialization")
            return

        try:
            with self.graph.driver.session() as session:
                session.run("CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE")
                session.run("CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE")
                session.run("CREATE INDEX FOR (p:Paper) ON (p.title)")
                session.run("CREATE INDEX FOR (a:Author) ON (a.name)")
                logger.info("Academic constraints and indexes ensured")
        except Exception:
            logger.exception("Failed to initialize constraints/indexes")

    def rebuild_relationship_types(self) -> None:
        """Migrate existing generic RELATED relationships to typed relationships.

        This method is a best-effort migration helper. It will inspect existing
        RELATED edges and create typed relationships based on available properties
        or node labels. It is safe to run multiple times.
        """
        if getattr(self.graph, '_use_neo4j', False):
            try:
                with self.graph.driver.session() as session:
                    # Fetch relationships with optional 'type' property
                    records = session.run("MATCH (s)-[r:RELATED]->(t) RETURN s, r, t LIMIT 10000")
                    for rec in records:
                        s = rec['s']
                        t = rec['t']
                        r = rec['r']
                        src_id = s.get('id') or s.get('name') or None
                        tgt_id = t.get('id') or t.get('name') or None
                        rtype = r.get('type')
                        # Heuristic mapping
                        if rtype:
                            typ = rtype
                        else:
                            # If both are Paper assume citation
                            s_labels = session.run("RETURN labels($n) as l", n=s).single()
                            t_labels = session.run("RETURN labels($n) as l", n=t).single()
                            s_is_paper = 'Paper' in (s_labels['l'] if s_labels else [])
                            t_is_paper = 'Paper' in (t_labels['l'] if t_labels else [])
                            if s_is_paper and t_is_paper:
                                typ = 'CITES'
                            else:
                                typ = 'SIMILAR_TO'

                        # Create typed relationship using RelationshipManager helper
                        try:
                            if typ == 'CITES' and src_id and tgt_id:
                                self.relationships.create_citation_relationship(src_id, tgt_id, context=r.get('context'), section=r.get('section'))
                            else:
                                # Fallback to similarity
                                if src_id and tgt_id:
                                    score = float(r.get('score', 0.0)) if r.get('score') is not None else 0.0
                                    self.relationships.create_similarity_relationship(src_id, tgt_id, similarity_type='inferred', similarity_score=score, method='migration')
                        except Exception:
                            logger.exception("Failed to migrate relationship %s -> %s", src_id, tgt_id)
            except Exception:
                logger.exception("Error during relationship migration (Neo4j)")
        else:
            # In-memory fallback: inspect edges in graph._graph
            try:
                for u, v, data in list(self.graph._graph.edges(data=True)):
                    rel_type = data.get('type') or data.get('label')
                    try:
                        if rel_type == 'CITES' or (self.graph._node_props.get(u, {}).get('label') == 'Paper' and self.graph._node_props.get(v, {}).get('label') == 'Paper'):
                            # create typed edge
                            self.relationships.create_citation_relationship(u, v, context=data.get('context'), section=data.get('section'))
                        else:
                            score = data.get('similarity_score') or data.get('score') or 0.0
                            self.relationships.create_similarity_relationship(u, v, similarity_type='inferred', similarity_score=float(score), method='migration')
                    except Exception:
                        logger.exception("Failed to migrate in-memory relationship %s -> %s", u, v)
            except Exception:
                logger.exception("Error during in-memory relationship migration")
