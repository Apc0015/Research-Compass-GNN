"""
Citation Type Classifier
Heuristic-based classification of citation relationships into semantic types
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from enum import Enum


class CitationType(Enum):
    """Types of citations in academic papers"""
    EXTENDS = 0        # Building upon / extending previous work
    METHODOLOGY = 1    # Using methods from cited paper
    BACKGROUND = 2     # General background / related work
    COMPARISON = 3     # Comparing approaches


class CitationTypeClassifier:
    """
    Classify citation types based on paper metadata and graph structure

    Heuristic rules:
    - EXTENDS: Citing paper is in same topic, cites recent papers (< 3 year gap)
    - METHODOLOGY: Cross-topic citation with high centrality cited paper
    - BACKGROUND: Old citations (> 5 year gap) or highly cited papers
    - COMPARISON: Multiple citations to same topic cluster

    Args:
        homogeneous_data: PyTorch Geometric Data object
        temporal_info: Optional dict with publication years {node_id: year}
        citation_counts: Optional citation count per paper
        seed: Random seed for reproducibility

    Example:
        >>> classifier = CitationTypeClassifier(data)
        >>> edge_types = classifier.classify_citations()
        >>> # Returns: [0, 1, 2, 3, ...] for each edge
    """

    def __init__(
        self,
        homogeneous_data,
        temporal_info: Dict[int, int] = None,
        citation_counts: np.ndarray = None,
        seed: int = 42
    ):
        self.data = homogeneous_data
        self.edge_index = homogeneous_data.edge_index
        self.num_edges = self.edge_index.shape[1]
        self.temporal_info = temporal_info or self._generate_synthetic_years()
        self.citation_counts = citation_counts or self._compute_citation_counts()
        self.seed = seed

        np.random.seed(seed)

    def _generate_synthetic_years(self) -> Dict[int, int]:
        """Generate synthetic publication years maintaining temporal constraints"""
        years = {}
        current_year = 2024

        # Papers are ordered temporally (earlier papers cited by later ones)
        for i in range(self.data.num_nodes):
            # Span 10 years, older papers first
            year = current_year - int(10 * (1 - i / self.data.num_nodes))
            years[i] = year

        return years

    def _compute_citation_counts(self) -> np.ndarray:
        """Compute in-degree (citation count) for each paper"""
        counts = np.zeros(self.data.num_nodes, dtype=int)
        for dst in self.edge_index[1]:
            counts[dst.item()] += 1
        return counts

    def classify_citations(self) -> torch.Tensor:
        """
        Classify each citation edge into a type

        Returns:
            edge_types: [num_edges] tensor with citation types (0-3)
        """
        edge_types = []

        # Get topic labels if available
        has_labels = hasattr(self.data, 'y')

        for i in range(self.num_edges):
            src = self.edge_index[0, i].item()  # Citing paper
            dst = self.edge_index[1, i].item()  # Cited paper

            # Compute features for classification
            year_gap = self.temporal_info[src] - self.temporal_info[dst]
            dst_citations = self.citation_counts[dst]
            same_topic = False

            if has_labels:
                same_topic = (self.data.y[src] == self.data.y[dst]).item()

            # Classify based on heuristics
            edge_type = self._apply_rules(
                year_gap, dst_citations, same_topic
            )
            edge_types.append(edge_type)

        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)

        # Validation: Ensure all edge types are in valid range [0-3]
        if edge_types_tensor.numel() > 0:
            min_type = edge_types_tensor.min().item()
            max_type = edge_types_tensor.max().item()
            assert min_type >= 0 and max_type <= 3, \
                f"Invalid edge types: range [{min_type}, {max_type}], expected [0, 3]"
            print(f"✅ Edge type validation passed: {self.num_edges} edges classified into types [0-3]")

        return edge_types_tensor

    def _apply_rules(
        self,
        year_gap: int,
        dst_citations: int,
        same_topic: bool
    ) -> int:
        """
        Apply heuristic rules to determine citation type

        Rules (in priority order):
        1. EXTENDS: Same topic + recent (< 3 years) + moderate citations
        2. METHODOLOGY: Cross-topic + highly cited (> 75th percentile)
        3. BACKGROUND: Old (> 5 years) OR very highly cited (> 90th percentile)
        4. COMPARISON: Otherwise (default)
        """
        high_citation_threshold = np.percentile(self.citation_counts, 75)
        very_high_citation_threshold = np.percentile(self.citation_counts, 90)

        # Rule 1: EXTENDS
        if same_topic and year_gap <= 3 and dst_citations < high_citation_threshold:
            return CitationType.EXTENDS.value

        # Rule 2: METHODOLOGY
        if not same_topic and dst_citations > high_citation_threshold:
            return CitationType.METHODOLOGY.value

        # Rule 3: BACKGROUND
        if year_gap > 5 or dst_citations > very_high_citation_threshold:
            return CitationType.BACKGROUND.value

        # Rule 4: COMPARISON (default)
        return CitationType.COMPARISON.value

    def get_type_distribution(self, edge_types: torch.Tensor) -> Dict[str, int]:
        """Get distribution of citation types"""
        distribution = {}
        for cit_type in CitationType:
            count = (edge_types == cit_type.value).sum().item()
            distribution[cit_type.name] = count
        return distribution

    def create_typed_edge_index(
        self,
        edge_types: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create separate edge indices for each citation type

        Args:
            edge_types: [num_edges] tensor of types

        Returns:
            Dictionary {citation_type_name: edge_index}
        """
        typed_edges = {}

        for cit_type in CitationType:
            mask = (edge_types == cit_type.value)
            typed_edges[cit_type.name.lower()] = self.edge_index[:, mask]

        return typed_edges

    def print_statistics(self, edge_types: torch.Tensor):
        """Print citation type statistics"""
        print("\n" + "=" * 70)
        print("CITATION TYPE DISTRIBUTION")
        print("=" * 70)

        dist = self.get_type_distribution(edge_types)
        total = edge_types.shape[0]

        print(f"\nTotal Citations: {total}")
        print("\nBreakdown:")
        for cit_type, count in dist.items():
            pct = 100 * count / total
            print(f"  • {cit_type:12s}: {count:5d} ({pct:5.1f}%)")

        print("\n" + "=" * 70)


def classify_citation_types(data, **kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function to classify citations

    Args:
        data: PyTorch Geometric Data object
        **kwargs: Additional arguments for CitationTypeClassifier

    Returns:
        edge_types: Tensor of types for each edge
        typed_edge_dict: Dictionary of edge indices per type

    Example:
        >>> edge_types, typed_edges = classify_citation_types(data)
        >>> print(f"Total edge types: {len(typed_edges)}")
    """
    classifier = CitationTypeClassifier(data, **kwargs)
    edge_types = classifier.classify_citations()

    # Print distribution
    classifier.print_statistics(edge_types)

    typed_edge_dict = classifier.create_typed_edge_index(edge_types)

    return edge_types, typed_edge_dict
