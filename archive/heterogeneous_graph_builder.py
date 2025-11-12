"""
Heterogeneous Graph Builder
Converts homogeneous citation network to heterogeneous graph with multiple node and edge types
"""

import torch
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeterogeneousGraphBuilder:
    """
    Build heterogeneous graphs from citation networks

    Node Types:
    - paper: Research papers
    - author: Paper authors
    - venue: Publication venues (conferences/journals)
    - topic: Research topics/categories

    Edge Types:
    - (paper, cites, paper): Citation relationships
    - (paper, written_by, author): Authorship
    - (paper, published_in, venue): Publication venue
    - (paper, belongs_to, topic): Topic classification
    - (author, writes, paper): Reverse authorship
    - (venue, publishes, paper): Reverse publication
    - (topic, contains, paper): Reverse topic

    Args:
        homogeneous_data: PyTorch Geometric Data object
        num_authors_per_paper: Average authors per paper (default: 2-3)
        num_venues: Number of unique venues (default: 10-20)
        author_collaboration_prob: Probability authors collaborate (default: 0.3)

    Example:
        >>> from torch_geometric.datasets import Planetoid
        >>> data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        >>> builder = HeterogeneousGraphBuilder(data)
        >>> hetero_data = builder.build()
    """

    def __init__(
        self,
        homogeneous_data,
        num_authors_per_paper: Tuple[int, int] = (2, 4),
        num_venues: int = 15,
        author_collaboration_prob: float = 0.3,
        venue_topic_correlation: float = 0.7,
        seed: int = 42
    ):
        self.data = homogeneous_data
        self.num_papers = homogeneous_data.num_nodes
        self.num_authors_per_paper = num_authors_per_paper
        self.num_venues = num_venues
        self.author_collab_prob = author_collaboration_prob
        self.venue_topic_corr = venue_topic_correlation
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def build(self) -> HeteroData:
        """
        Build heterogeneous graph from homogeneous citation network

        Returns:
            HeteroData object with multiple node and edge types
        """
        hetero_data = HeteroData()

        # 1. Add paper nodes (from original graph)
        hetero_data['paper'].x = self.data.x
        hetero_data['paper'].y = self.data.y

        # Check if masks exist before copying, otherwise create default 60/20/20 split
        if hasattr(self.data, 'train_mask') and self.data.train_mask is not None:
            hetero_data['paper'].train_mask = self.data.train_mask
            hetero_data['paper'].val_mask = self.data.val_mask
            hetero_data['paper'].test_mask = self.data.test_mask
            print(f"✅ Using existing masks: train={self.data.train_mask.sum()}, "
                  f"val={self.data.val_mask.sum()}, test={self.data.test_mask.sum()}")
        else:
            # Create default 60/20/20 split
            num_nodes = self.num_papers
            perm = torch.randperm(num_nodes)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[perm[:train_size]] = True
            val_mask[perm[train_size:train_size+val_size]] = True
            test_mask[perm[train_size+val_size:]] = True

            hetero_data['paper'].train_mask = train_mask
            hetero_data['paper'].val_mask = val_mask
            hetero_data['paper'].test_mask = test_mask
            print(f"✅ Created default masks (60/20/20): train={train_mask.sum()}, "
                  f"val={val_mask.sum()}, test={test_mask.sum()}")

        # Validate mask shapes match node count
        assert hetero_data['paper'].train_mask.shape[0] == self.num_papers, \
            f"train_mask size mismatch: {hetero_data['paper'].train_mask.shape[0]} != {self.num_papers}"
        assert hetero_data['paper'].val_mask.shape[0] == self.num_papers, \
            f"val_mask size mismatch: {hetero_data['paper'].val_mask.shape[0]} != {self.num_papers}"
        assert hetero_data['paper'].test_mask.shape[0] == self.num_papers, \
            f"test_mask size mismatch: {hetero_data['paper'].test_mask.shape[0]} != {self.num_papers}"

        # 2. Generate author nodes and edges
        author_data = self._generate_authors()
        hetero_data['author'].x = author_data['features']
        hetero_data['paper', 'written_by', 'author'].edge_index = author_data['paper_to_author']
        hetero_data['author', 'writes', 'paper'].edge_index = author_data['author_to_paper']

        # 3. Generate venue nodes and edges
        venue_data = self._generate_venues()
        hetero_data['venue'].x = venue_data['features']
        hetero_data['paper', 'published_in', 'venue'].edge_index = venue_data['paper_to_venue']
        hetero_data['venue', 'publishes', 'paper'].edge_index = venue_data['venue_to_paper']

        # 4. Generate topic nodes and edges
        topic_data = self._generate_topics()
        hetero_data['topic'].x = topic_data['features']
        hetero_data['paper', 'belongs_to', 'topic'].edge_index = topic_data['paper_to_topic']
        hetero_data['topic', 'contains', 'paper'].edge_index = topic_data['topic_to_paper']

        # 5. Add citation edges (from original graph)
        hetero_data['paper', 'cites', 'paper'].edge_index = self.data.edge_index

        # Store metadata
        hetero_data.num_papers = self.num_papers
        hetero_data.num_authors = author_data['num_authors']
        hetero_data.num_venues = self.num_venues
        hetero_data.num_topics = len(torch.unique(self.data.y))

        # Validate before returning
        if not self.validate(hetero_data):
            raise ValueError("Heterogeneous graph validation failed")

        return hetero_data

    def validate(self, hetero_data: HeteroData) -> bool:
        """
        Validate heterogeneous graph structure

        Checks:
        - Node features exist for all node types
        - Node counts are positive
        - Edge indices are within bounds
        - Edge index shapes are correct

        Args:
            hetero_data: HeteroData object to validate

        Returns:
            True if validation passes, False otherwise
        """
        errors = []

        # Check node counts and features
        for node_type in hetero_data.node_types:
            if not hasattr(hetero_data[node_type], 'x') or hetero_data[node_type].x is None:
                errors.append(f"Missing features for node type: {node_type}")
                continue

            num_nodes = hetero_data[node_type].x.shape[0]
            if num_nodes == 0:
                errors.append(f"Zero nodes for type: {node_type}")

        # Check edge indices
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index

            # Validate edge index shape
            if edge_index.shape[0] != 2:
                errors.append(f"Invalid edge_index shape for {edge_type}: {edge_index.shape}")
                continue

            # Validate indices are within bounds
            if edge_index.shape[1] > 0:
                src_max = edge_index[0].max().item()
                dst_max = edge_index[1].max().item()

                src_num_nodes = hetero_data[src_type].x.shape[0]
                dst_num_nodes = hetero_data[dst_type].x.shape[0]

                if src_max >= src_num_nodes:
                    errors.append(f"Invalid source index in {edge_type}: "
                                 f"{src_max} >= {src_num_nodes}")
                if dst_max >= dst_num_nodes:
                    errors.append(f"Invalid destination index in {edge_type}: "
                                 f"{dst_max} >= {dst_num_nodes}")

        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✅ Heterogeneous graph validation passed")
        return True

    def _generate_authors(self) -> Dict:
        """Generate synthetic author nodes and authorship edges"""
        # Estimate total authors (papers * avg_authors * uniqueness_factor)
        avg_authors = np.mean(self.num_authors_per_paper)
        num_unique_authors = int(self.num_papers * avg_authors * 0.6)  # 60% unique

        # Generate author features (smaller than paper features)
        author_dim = min(128, self.data.x.shape[1] // 2)
        author_features = torch.randn(num_unique_authors, author_dim)

        # Generate authorship edges
        paper_to_author = []
        author_to_paper = []

        author_id = 0
        for paper_id in range(self.num_papers):
            # Number of authors for this paper
            num_authors = np.random.randint(*self.num_authors_per_paper)

            # Assign authors (with collaboration probability)
            for _ in range(num_authors):
                if np.random.random() < self.author_collab_prob and author_id > 0:
                    # Collaborate with existing author
                    selected_author = np.random.randint(0, min(author_id, num_unique_authors))
                else:
                    # New author
                    selected_author = author_id % num_unique_authors
                    author_id += 1

                paper_to_author.append([paper_id, selected_author])
                author_to_paper.append([selected_author, paper_id])

        paper_to_author = torch.tensor(paper_to_author, dtype=torch.long).t()
        author_to_paper = torch.tensor(author_to_paper, dtype=torch.long).t()

        # Validation: Ensure all edge indices are within valid range
        if paper_to_author.shape[1] > 0:
            # Validate paper indices
            max_paper_idx = paper_to_author[0].max().item()
            assert max_paper_idx < self.num_papers, \
                f"Invalid paper index in authorship: {max_paper_idx} >= {self.num_papers}"

            # Validate author indices
            max_author_idx = paper_to_author[1].max().item()
            assert max_author_idx < num_unique_authors, \
                f"Invalid author index: {max_author_idx} >= {num_unique_authors}"

            print(f"✅ Author edge validation passed: {paper_to_author.shape[1]} edges, "
                  f"papers [0-{max_paper_idx}], authors [0-{max_author_idx}]")

        return {
            'features': author_features,
            'paper_to_author': paper_to_author,
            'author_to_paper': author_to_paper,
            'num_authors': num_unique_authors
        }

    def _generate_venues(self) -> Dict:
        """Generate synthetic venue nodes and publication edges"""
        # Generate venue features
        venue_dim = 64
        venue_features = torch.randn(self.num_venues, venue_dim)

        # Assign papers to venues (correlated with topics)
        paper_to_venue = []
        venue_to_paper = []

        # If we have topic labels, use topic-venue correlation
        if hasattr(self.data, 'y'):
            num_topics = len(torch.unique(self.data.y))
            # Create topic-to-venue mapping (some venues specialize in topics)
            topic_to_preferred_venues = {}
            for topic in range(num_topics):
                # Each topic has 2-3 preferred venues
                preferred = np.random.choice(
                    self.num_venues,
                    size=min(3, self.num_venues),
                    replace=False
                )
                topic_to_preferred_venues[topic] = preferred

            # Assign papers to venues based on topic
            for paper_id in range(self.num_papers):
                topic = self.data.y[paper_id].item()

                if np.random.random() < self.venue_topic_corr:
                    # Choose preferred venue for this topic
                    venue_id = np.random.choice(topic_to_preferred_venues[topic])
                else:
                    # Random venue
                    venue_id = np.random.randint(0, self.num_venues)

                paper_to_venue.append([paper_id, venue_id])
                venue_to_paper.append([venue_id, paper_id])
        else:
            # Random assignment if no topics
            for paper_id in range(self.num_papers):
                venue_id = np.random.randint(0, self.num_venues)
                paper_to_venue.append([paper_id, venue_id])
                venue_to_paper.append([venue_id, paper_id])

        paper_to_venue = torch.tensor(paper_to_venue, dtype=torch.long).t()
        venue_to_paper = torch.tensor(venue_to_paper, dtype=torch.long).t()

        # Validation: Ensure all edge indices are within valid range
        if paper_to_venue.shape[1] > 0:
            # Validate paper indices
            max_paper_idx = paper_to_venue[0].max().item()
            assert max_paper_idx < self.num_papers, \
                f"Invalid paper index in venue: {max_paper_idx} >= {self.num_papers}"

            # Validate venue indices
            max_venue_idx = paper_to_venue[1].max().item()
            assert max_venue_idx < self.num_venues, \
                f"Invalid venue index: {max_venue_idx} >= {self.num_venues}"

            print(f"✅ Venue edge validation passed: {paper_to_venue.shape[1]} edges, "
                  f"papers [0-{max_paper_idx}], venues [0-{max_venue_idx}]")

        return {
            'features': venue_features,
            'paper_to_venue': paper_to_venue,
            'venue_to_paper': venue_to_paper
        }

    def _generate_topics(self) -> Dict:
        """Generate topic nodes from paper labels"""
        if not hasattr(self.data, 'y'):
            raise ValueError("Data must have labels (y) to generate topics")

        num_topics = len(torch.unique(self.data.y))

        # Generate topic features (small, abstract representations)
        topic_dim = 32
        topic_features = torch.randn(num_topics, topic_dim)

        # Create paper-topic edges from labels
        paper_to_topic = []
        topic_to_paper = []

        for paper_id in range(self.num_papers):
            topic_id = self.data.y[paper_id].item()
            paper_to_topic.append([paper_id, topic_id])
            topic_to_paper.append([topic_id, paper_id])

        paper_to_topic = torch.tensor(paper_to_topic, dtype=torch.long).t()
        topic_to_paper = torch.tensor(topic_to_paper, dtype=torch.long).t()

        # Validation: Ensure all edge indices are within valid range
        if paper_to_topic.shape[1] > 0:
            # Validate paper indices
            max_paper_idx = paper_to_topic[0].max().item()
            assert max_paper_idx < self.num_papers, \
                f"Invalid paper index in topic: {max_paper_idx} >= {self.num_papers}"

            # Validate topic indices
            max_topic_idx = paper_to_topic[1].max().item()
            assert max_topic_idx < num_topics, \
                f"Invalid topic index: {max_topic_idx} >= {num_topics}"

            print(f"✅ Topic edge validation passed: {paper_to_topic.shape[1]} edges, "
                  f"papers [0-{max_paper_idx}], topics [0-{max_topic_idx}]")

        return {
            'features': topic_features,
            'paper_to_topic': paper_to_topic,
            'topic_to_paper': topic_to_paper
        }

    def print_statistics(self, hetero_data: HeteroData):
        """Print statistics about the heterogeneous graph"""
        print("\n" + "=" * 70)
        print("HETEROGENEOUS GRAPH STATISTICS")
        print("=" * 70)

        print("\n📊 Node Types:")
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].x.shape[0]
            feat_dim = hetero_data[node_type].x.shape[1]
            print(f"  • {node_type:10s}: {num_nodes:5d} nodes, {feat_dim:4d} features")

        print("\n🔗 Edge Types:")
        for edge_type in hetero_data.edge_types:
            src, rel, dst = edge_type
            num_edges = hetero_data[edge_type].edge_index.shape[1]
            print(f"  • ({src:10s}, {rel:12s}, {dst:10s}): {num_edges:6d} edges")

        print("\n" + "=" * 70)


def convert_to_heterogeneous(data, **kwargs) -> HeteroData:
    """
    Convenience function to convert homogeneous graph to heterogeneous

    Args:
        data: PyTorch Geometric Data object
        **kwargs: Additional arguments for HeterogeneousGraphBuilder

    Returns:
        HeteroData object

    Example:
        >>> data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        >>> hetero_data = convert_to_heterogeneous(data, num_venues=20)
    """
    builder = HeterogeneousGraphBuilder(data, **kwargs)
    hetero_data = builder.build()
    builder.print_statistics(hetero_data)
    return hetero_data
