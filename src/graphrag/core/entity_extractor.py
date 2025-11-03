"""
Entity Extraction Module
Extracts entities and relationships from text using NLP.
"""

from typing import List, Dict, Tuple, Set
import spacy
import spacy.cli
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str


class EntityExtractor:
    """Extracts entities and relationships using spaCy NLP."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the entity extractor.

        Args:
            model_name: spaCy model to use for NLP
        """
        # Try primary model; fall back to 'en_core_web_sm' and attempt auto-download
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            fallback = "en_core_web_sm"
            try:
                self.nlp = spacy.load(fallback)
            except OSError:
                # Try to download the small model automatically
                try:
                    # Use the already-imported spacy module to download the fallback model
                    spacy.cli.download(fallback)
                    self.nlp = spacy.load(fallback)
                except Exception:
                    raise ValueError(
                        f"spaCy model '{model_name}' not found and automatic download of '{fallback}' failed. "
                        f"Install manually with: python -m spacy download {model_name} or python -m spacy download {fallback}"
                    )

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text to process

        Returns:
            List of Entity objects
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))

        return entities

    def extract_relationships(self, text: str) -> List[Relationship]:
        """
        Extract relationships between entities using dependency parsing.

        Args:
            text: Input text to process

        Returns:
            List of Relationship objects
        """
        doc = self.nlp(text)
        relationships = []

        # Create entity map for quick lookup
        entity_map = {ent.start: ent for ent in doc.ents}

        # Extract relationships using dependency parsing
        for token in doc:
            if token.dep_ in ("nsubj", "dobj", "pobj", "attr"):
                # Find subject and object
                subject = None
                obj = None
                verb = token.head

                # Get subject
                for child in verb.children:
                    if child.dep_ == "nsubj" and child.i in entity_map:
                        subject = entity_map[child.i].text

                # Get object
                if token.i in entity_map:
                    obj = entity_map[token.i].text

                if subject and obj and subject != obj:
                    relationships.append(Relationship(
                        source=subject,
                        target=obj,
                        relation_type=verb.lemma_
                    ))

        return relationships

    def extract_entities_and_relationships(
        self, text: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract both entities and relationships from text.

        Args:
            text: Input text to process

        Returns:
            Tuple of (entities, relationships)
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text)
        return entities, relationships

    def get_entity_types(self, entities: List[Entity]) -> Set[str]:
        """
        Get unique entity types from a list of entities.

        Args:
            entities: List of Entity objects

        Returns:
            Set of unique entity type labels
        """
        return {entity.label for entity in entities}
