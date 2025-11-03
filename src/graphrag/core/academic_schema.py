"""Academic schema dataclasses and enums for specialized academic node types.

This module defines concrete dataclasses for academic entities (Paper, Author,
Topic, Method, Dataset, Institution, Venue) and enums for node and relation
types. Each dataclass provides a helper to convert itself into a dictionary of
properties suitable for storing in Neo4j.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class NodeType(Enum):
    PAPER = "PAPER"
    AUTHOR = "AUTHOR"
    TOPIC = "TOPIC"
    METHOD = "METHOD"
    DATASET = "DATASET"
    INSTITUTION = "INSTITUTION"
    VENUE = "VENUE"


class RelationType(Enum):
    CITES = "CITES"
    AUTHORED_BY = "AUTHORED_BY"
    DISCUSSES = "DISCUSSES"
    USES_METHOD = "USES_METHOD"
    EVALUATED_ON = "EVALUATED_ON"
    SIMILAR_TO = "SIMILAR_TO"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    PUBLISHED_IN = "PUBLISHED_IN"
    INFLUENCED_BY = "INFLUENCED_BY"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


@dataclass
class PaperNode:
    """Represents an academic paper."""

    id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    authors: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.PAPER, init=False)

    def to_neo4j_properties(self) -> Dict:
        """Convert to a dict suitable for Neo4j node properties.

        Dates are converted to ISO strings and lists are kept as lists so Neo4j
        drivers can serialize them.
        """
        props = {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract or "",
            "year": self.year,
            "authors": list(self.authors),
            "citations": list(self.citations),
            "venue": self.venue,
            "keywords": list(self.keywords),
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }
        return props


@dataclass
class AuthorNode:
    id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    orcid: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.AUTHOR, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "affiliations": list(self.affiliations),
            "orcid": self.orcid,
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }


@dataclass
class TopicNode:
    id: str
    name: str
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.TOPIC, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "keywords": list(self.keywords),
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }


@dataclass
class MethodNode:
    id: str
    name: str
    description: Optional[str] = None
    introduced_in: Optional[str] = None  # e.g., paper id or year
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.METHOD, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "introduced_in": self.introduced_in,
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }


@dataclass
class DatasetNode:
    id: str
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.DATASET, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }


@dataclass
class InstitutionNode:
    id: str
    name: str
    country: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.INSTITUTION, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }


@dataclass
class VenueNode:
    id: str
    title: str
    venue_type: Optional[str] = None  # conference/journal
    issn: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    node_type: NodeType = field(default=NodeType.VENUE, init=False)

    def to_neo4j_properties(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "venue_type": self.venue_type,
            "issn": self.issn,
            "created_at": _iso(self.created_at),
            "node_type": self.node_type.value,
        }
