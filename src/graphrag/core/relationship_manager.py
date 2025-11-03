"""Typed relationship manager for academic graphs.

Provides high-level helpers to create typed, property-rich relationships while
keeping backward compatibility with the existing generic RELATED edges.
"""
from __future__ import annotations

from typing import Dict, Optional
from datetime import datetime
import logging

from .graph_manager import GraphManager

logger = logging.getLogger(__name__)


class RelationshipManager:
    """Manage specialized relationships between academic nodes.

    This class uses the project's GraphManager for low-level operations and
    ensures properties/labels are applied consistently across Neo4j and the
    in-memory fallback.
    """

    def __init__(self, graph: GraphManager):
        self.graph = graph

    # --- Creation helpers ---
    def create_citation_relationship(self, citing_paper_id: str, cited_paper_id: str, context: Optional[str] = None, section: Optional[str] = None) -> None:
        props = {
            'context': context,
            'section': section,
            'created_at': datetime.utcnow().isoformat()
        }
        # remove None values
        props = {k: v for k, v in props.items() if v is not None}
        try:
            # Use GraphManager.create_relationship which keeps backward compatibility
            self.graph.create_relationship(citing_paper_id, cited_paper_id, 'CITES', properties=props)
        except Exception:
            logger.exception("Failed to create CITES relationship: %s -> %s", citing_paper_id, cited_paper_id)
            raise

    def create_authorship_relationship(self, paper_id: str, author_id: str, author_position: int = 0, contribution: Optional[str] = None) -> None:
        props = {
            'position': int(author_position),
            'contribution': contribution,
            'created_at': datetime.utcnow().isoformat()
        }
        props = {k: v for k, v in props.items() if v is not None}
        try:
            self.graph.create_relationship(paper_id, author_id, 'AUTHORED_BY', properties=props)
        except Exception:
            logger.exception("Failed to create AUTHORED_BY relationship: %s -> %s", paper_id, author_id)
            raise

    def create_topic_relationship(self, paper_id: str, topic_id: str, confidence: float, evidence: Optional[str] = None) -> None:
        props = {
            'confidence': float(confidence),
            'evidence': evidence,
            'created_at': datetime.utcnow().isoformat()
        }
        props = {k: v for k, v in props.items() if v is not None}
        try:
            self.graph.create_relationship(paper_id, topic_id, 'DISCUSSES', properties=props)
        except Exception:
            logger.exception("Failed to create DISCUSSES relationship: %s -> %s", paper_id, topic_id)
            raise

    def create_method_usage_relationship(self, paper_id: str, method_id: str, usage_type: str, performance: Optional[Dict] = None) -> None:
        props = {
            'usage_type': usage_type,
            'performance': performance,
            'created_at': datetime.utcnow().isoformat()
        }
        props = {k: v for k, v in props.items() if v is not None}
        try:
            self.graph.create_relationship(paper_id, method_id, 'USES_METHOD', properties=props)
        except Exception:
            logger.exception("Failed to create USES_METHOD relationship: %s -> %s", paper_id, method_id)
            raise

    def create_dataset_evaluation_relationship(self, paper_id: str, dataset_id: str, metrics: Dict) -> None:
        props = {
            'metrics': metrics,
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            self.graph.create_relationship(paper_id, dataset_id, 'EVALUATED_ON', properties=props)
        except Exception:
            logger.exception("Failed to create EVALUATED_ON relationship: %s -> %s", paper_id, dataset_id)
            raise

    def create_similarity_relationship(self, entity1_id: str, entity2_id: str, similarity_type: str, similarity_score: float, method: str) -> None:
        props = {
            'similarity_type': similarity_type,
            'similarity_score': float(similarity_score),
            'method': method,
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            # Create bidirectional similarity edges for compatibility
            self.graph.create_relationship(entity1_id, entity2_id, 'SIMILAR_TO', properties=props)
            self.graph.create_relationship(entity2_id, entity1_id, 'SIMILAR_TO', properties=props)
        except Exception:
            logger.exception("Failed to create SIMILAR_TO relationship: %s <-> %s", entity1_id, entity2_id)
            raise

    def create_influence_relationship(self, influenced_paper_id: str, influencing_paper_id: str, influence_type: str, influence_score: float) -> None:
        props = {
            'influence_type': influence_type,
            'influence_score': float(influence_score),
            'created_at': datetime.utcnow().isoformat()
        }
        try:
            self.graph.create_relationship(influenced_paper_id, influencing_paper_id, 'INFLUENCED_BY', properties=props)
        except Exception:
            logger.exception("Failed to create INFLUENCED_BY relationship: %s -> %s", influenced_paper_id, influencing_paper_id)
            raise

    def create_affiliation_relationship(self, author_id: str, institution_id: str, start_year: Optional[int] = None, end_year: Optional[int] = None, role: Optional[str] = None) -> None:
        props = {'start_year': start_year, 'end_year': end_year, 'role': role, 'created_at': datetime.utcnow().isoformat()}
        props = {k: v for k, v in props.items() if v is not None}
        try:
            self.graph.create_relationship(author_id, institution_id, 'AFFILIATED_WITH', properties=props)
        except Exception:
            logger.exception("Failed to create AFFILIATED_WITH relationship: %s -> %s", author_id, institution_id)
            raise

    def create_publication_relationship(self, paper_id: str, venue_id: str, year: int, acceptance_type: Optional[str] = None) -> None:
        props = {'year': int(year), 'type': acceptance_type, 'created_at': datetime.utcnow().isoformat()}
        props = {k: v for k, v in props.items() if v is not None}
        try:
            self.graph.create_relationship(paper_id, venue_id, 'PUBLISHED_IN', properties=props)
        except Exception:
            logger.exception("Failed to create PUBLISHED_IN relationship: %s -> %s", paper_id, venue_id)
            raise

    def enrich_relationship(self, rel_id: str, additional_properties: Dict) -> None:
        """Add/merge properties into an existing relationship identified by internal id or unique key.

        For Neo4j this expects rel_id to be the relationship identity or a unique key. In the
        in-memory fallback this will try to find a relationship by matching common fields.
        """
        if getattr(self.graph, '_use_neo4j', False):
            try:
                with self.graph.driver.session() as session:
                    # Attempt to match by internal id if numeric
                    if isinstance(rel_id, int) or (isinstance(rel_id, str) and rel_id.isdigit()):
                        session.run("MATCH ()-[r]->() WHERE id(r) = $rid SET r += $props", rid=int(rel_id), props=additional_properties)
                    else:
                        # As a best-effort merge, try rel id as a uuid stored in rel.id
                        session.run("MATCH ()-[r]->() WHERE coalesce(r.id, '') = $rid SET r += $props", rid=str(rel_id), props=additional_properties)
            except Exception:
                logger.exception("Failed to enrich relationship %s", rel_id)
                raise
        else:
            # Best-effort: find any edge carrying an 'id' or matching rel_id as tuple
            try:
                for u, v, data in self.graph._graph.edges(data=True):
                    if data.get('id') == rel_id or str((u, v)) == str(rel_id):
                        # merge properties
                        data.update(additional_properties)
                        return
                # If not found, log and raise
                raise KeyError(f"Relationship {rel_id} not found in in-memory graph")
            except Exception:
                logger.exception("Failed to enrich in-memory relationship %s", rel_id)
                raise
