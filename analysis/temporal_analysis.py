"""
Temporal Analysis for Citation Networks

Analyzes how research evolves over time:
- Citation patterns over time
- Topic evolution
- Emerging research areas
- Paper impact trajectories
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path


class TemporalAnalyzer:
    """
    Analyzer for temporal patterns in citation networks

    Example:
        >>> analyzer = TemporalAnalyzer()
        >>> analyzer.add_temporal_data(data, years, paper_titles)
        >>> trends = analyzer.analyze_topic_evolution()
        >>> analyzer.plot_citation_growth()
    """

    def __init__(self):
        self.data = None
        self.years = None
        self.paper_titles = None
        self.figures = {}

    def add_temporal_data(
        self,
        data,
        years: Optional[List[int]] = None,
        paper_titles: Optional[List[str]] = None
    ):
        """
        Add temporal information to the analyzer

        Args:
            data: PyG Data object
            years: Publication year for each paper (or None for synthetic)
            paper_titles: Optional paper titles
        """
        self.data = data
        self.paper_titles = paper_titles

        # If no years provided, generate synthetic years
        if years is None:
            # Assume papers are ordered chronologically
            base_year = 2010
            num_years = 10
            self.years = np.linspace(base_year, base_year + num_years, data.num_nodes, dtype=int)
        else:
            self.years = np.array(years)

    def analyze_citation_velocity(self, node_idx: int) -> Dict:
        """
        Compute citation velocity for a specific paper

        Citation velocity = citations per year since publication

        Args:
            node_idx: Index of the paper

        Returns:
            Dictionary with velocity statistics
        """
        edge_index = self.data.edge_index.cpu().numpy()

        # Count incoming citations
        incoming_citations = (edge_index[1] == node_idx).sum()

        # Years since publication
        paper_year = self.years[node_idx]
        current_year = self.years.max()
        years_since_pub = max(1, current_year - paper_year)

        velocity = incoming_citations / years_since_pub

        return {
            'node_idx': node_idx,
            'total_citations': int(incoming_citations),
            'publication_year': int(paper_year),
            'years_since_publication': int(years_since_pub),
            'citation_velocity': float(velocity)
        }

    def identify_emerging_topics(
        self,
        lookback_years: int = 3,
        min_papers: int = 5
    ) -> List[Dict]:
        """
        Identify emerging research topics (topics with accelerating growth)

        Args:
            lookback_years: How many recent years to consider
            min_papers: Minimum papers to be considered a topic

        Returns:
            List of emerging topics with acceleration metrics
        """
        current_year = self.years.max()
        recent_years = current_year - lookback_years

        # Count papers per topic over time
        topics = self.data.y.cpu().numpy()
        num_topics = topics.max() + 1

        topic_growth = []

        for topic in range(num_topics):
            topic_mask = topics == topic

            # Papers in recent period
            recent_mask = topic_mask & (self.years >= recent_years)
            num_recent = recent_mask.sum()

            # Papers in earlier period
            earlier_mask = topic_mask & (self.years < recent_years)
            num_earlier = max(1, earlier_mask.sum())

            if num_recent >= min_papers:
                # Growth rate
                growth_rate = (num_recent - num_earlier) / num_earlier

                # Acceleration (second derivative)
                mid_year = recent_years + lookback_years // 2
                mid_mask = topic_mask & (self.years >= mid_year) & (self.years < current_year)
                early_recent_mask = topic_mask & (self.years >= recent_years) & (self.years < mid_year)

                num_mid = mid_mask.sum()
                num_early_recent = max(1, early_recent_mask.sum())

                acceleration = (num_mid / max(1, (current_year - mid_year))) - \
                              (num_early_recent / max(1, (mid_year - recent_years)))

                topic_growth.append({
                    'topic': int(topic),
                    'total_papers': int(topic_mask.sum()),
                    'recent_papers': int(num_recent),
                    'growth_rate': float(growth_rate),
                    'acceleration': float(acceleration),
                    'is_emerging': acceleration > 0.5
                })

        # Sort by acceleration
        topic_growth.sort(key=lambda x: x['acceleration'], reverse=True)

        return topic_growth

    def analyze_topic_evolution(self) -> Dict:
        """
        Analyze how topics evolve over time

        Returns:
            Dictionary with evolution statistics
        """
        topics = self.data.y.cpu().numpy()
        num_topics = topics.max() + 1

        # Compute topic distribution per year
        unique_years = np.unique(self.years)
        topic_by_year = {year: np.zeros(num_topics) for year in unique_years}

        for year in unique_years:
            year_mask = self.years == year
            year_topics = topics[year_mask]

            for topic in range(num_topics):
                topic_by_year[year][topic] = (year_topics == topic).sum()

        # Compute trends
        evolution = {
            'years': unique_years.tolist(),
            'topic_counts': {
                f'topic_{i}': [float(topic_by_year[year][i]) for year in unique_years]
                for i in range(num_topics)
            }
        }

        # Peak year for each topic
        for topic in range(num_topics):
            counts = [topic_by_year[year][topic] for year in unique_years]
            peak_idx = np.argmax(counts)
            evolution[f'topic_{topic}_peak_year'] = int(unique_years[peak_idx])

        return evolution

    def plot_citation_growth(self, title: str = 'Citation Network Growth Over Time') -> plt.Figure:
        """
        Plot how the citation network grows over time

        Args:
            title: Plot title

        Returns:
            Matplotlib figure
        """
        unique_years = np.unique(self.years)
        edge_index = self.data.edge_index.cpu().numpy()

        papers_by_year = []
        citations_by_year = []

        for year in unique_years:
            # Cumulative papers up to this year
            papers_by_year.append((self.years <= year).sum())

            # Citations between papers published up to this year
            papers_mask = self.years <= year
            paper_ids = np.where(papers_mask)[0]
            citations_mask = np.isin(edge_index[0], paper_ids) & np.isin(edge_index[1], paper_ids)
            citations_by_year.append(citations_mask.sum())

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(unique_years, papers_by_year, 'o-', label='Papers', linewidth=2, markersize=8)
        ax.plot(unique_years, citations_by_year, 's-', label='Citations', linewidth=2, markersize=8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add growth rate annotation
        total_papers = len(papers_by_year)
        if total_papers > 1:
            years_span = unique_years[-1] - unique_years[0]
            paper_growth_rate = (papers_by_year[-1] - papers_by_year[0]) / years_span if years_span > 0 else 0
            citation_growth_rate = (citations_by_year[-1] - citations_by_year[0]) / years_span if years_span > 0 else 0

            text = f"Avg Growth:\nPapers: {paper_growth_rate:.1f}/year\nCitations: {citation_growth_rate:.1f}/year"
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self.figures['citation_growth'] = fig
        return fig

    def plot_topic_evolution(self, title: str = 'Research Topic Evolution') -> plt.Figure:
        """
        Plot how research topics evolve over time

        Args:
            title: Plot title

        Returns:
            Matplotlib figure
        """
        evolution = self.analyze_topic_evolution()
        unique_years = evolution['years']
        num_topics = len([k for k in evolution['topic_counts'].keys()])

        fig, ax = plt.subplots(figsize=(14, 7))

        colors = sns.color_palette('husl', num_topics)

        for i in range(num_topics):
            counts = evolution['topic_counts'][f'topic_{i}']
            ax.plot(unique_years, counts, 'o-',
                   label=f'Topic {i}',
                   linewidth=2,
                   markersize=6,
                   color=colors[i])

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures['topic_evolution'] = fig
        return fig

    def plot_topic_heatmap(self, title: str = 'Topic Distribution Over Time') -> plt.Figure:
        """
        Plot topic distribution heatmap

        Args:
            title: Plot title

        Returns:
            Matplotlib figure
        """
        evolution = self.analyze_topic_evolution()
        unique_years = evolution['years']
        num_topics = len([k for k in evolution['topic_counts'].keys()])

        # Create matrix
        matrix = np.zeros((num_topics, len(unique_years)))
        for i in range(num_topics):
            matrix[i, :] = evolution['topic_counts'][f'topic_{i}']

        # Normalize by year (show proportion)
        matrix_normalized = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            matrix_normalized,
            xticklabels=unique_years,
            yticklabels=[f'Topic {i}' for i in range(num_topics)],
            cmap='YlOrRd',
            cbar_kws={'label': 'Proportion of Papers'},
            ax=ax
        )

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Topic', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        self.figures['topic_heatmap'] = fig
        return fig

    def plot_citation_velocity_distribution(
        self,
        top_k: int = 20,
        title: str = 'Top Papers by Citation Velocity'
    ) -> plt.Figure:
        """
        Plot distribution of citation velocities

        Args:
            top_k: Number of top papers to show
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Compute velocities for all papers
        velocities = []
        for node_idx in range(self.data.num_nodes):
            vel = self.analyze_citation_velocity(node_idx)
            velocities.append(vel)

        # Sort by velocity
        velocities.sort(key=lambda x: x['citation_velocity'], reverse=True)
        top_papers = velocities[:top_k]

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(top_k)
        velocities_list = [p['citation_velocity'] for p in top_papers]

        # Create labels
        labels = []
        for p in top_papers:
            if self.paper_titles and p['node_idx'] < len(self.paper_titles):
                label = self.paper_titles[p['node_idx']][:40]
            else:
                label = f"Paper {p['node_idx']}"
            label += f"\n({p['publication_year']}, {p['total_citations']} cites)"
            labels.append(label)

        bars = ax.barh(y_pos, velocities_list, color='coral', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Citations per Year', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add values
        for i, (bar, vel) in enumerate(zip(bars, velocities_list)):
            ax.text(vel + max(velocities_list)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{vel:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        self.figures['citation_velocity'] = fig
        return fig

    def save_all(self, output_dir: str):
        """Save all generated figures"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in self.figures.items():
            filepath = output_path / f"{name}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved {filepath}")

    def close_all(self):
        """Close all figures"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()


def generate_temporal_report(
    data,
    years: Optional[List[int]] = None,
    paper_titles: Optional[List[str]] = None,
    output_dir: str = 'results/temporal'
) -> str:
    """
    Generate comprehensive temporal analysis report

    Args:
        data: PyG Data object
        years: Publication years (optional)
        paper_titles: Paper titles (optional)
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    analyzer = TemporalAnalyzer()
    analyzer.add_temporal_data(data, years, paper_titles)

    # Generate analysis
    evolution = analyzer.analyze_topic_evolution()
    emerging = analyzer.identify_emerging_topics()

    # Generate plots
    analyzer.plot_citation_growth()
    analyzer.plot_topic_evolution()
    analyzer.plot_topic_heatmap()
    analyzer.plot_citation_velocity_distribution()

    # Save plots
    analyzer.save_all(output_dir)

    # Generate markdown report
    output_path = Path(output_dir)
    report_path = output_path / 'temporal_analysis_report.md'

    with open(report_path, 'w') as f:
        f.write("# Temporal Analysis Report\n\n")

        f.write("## Citation Network Growth\n\n")
        f.write(f"- **Total Papers:** {data.num_nodes}\n")
        f.write(f"- **Total Citations:** {data.num_edges}\n")
        f.write(f"- **Year Range:** {analyzer.years.min()} - {analyzer.years.max()}\n\n")

        f.write("## Topic Evolution\n\n")
        for i, year in enumerate(evolution['years']):
            f.write(f"### Year {year}\n")
            for topic_key, counts in evolution['topic_counts'].items():
                topic_num = topic_key.split('_')[1]
                f.write(f"- {topic_key}: {int(counts[i])} papers\n")
            f.write("\n")

        f.write("## Emerging Topics\n\n")
        f.write("Topics with highest growth acceleration:\n\n")
        for topic_info in emerging[:5]:
            f.write(f"### Topic {topic_info['topic']}\n")
            f.write(f"- Total Papers: {topic_info['total_papers']}\n")
            f.write(f"- Recent Papers: {topic_info['recent_papers']}\n")
            f.write(f"- Growth Rate: {topic_info['growth_rate']:.2f}\n")
            f.write(f"- Acceleration: {topic_info['acceleration']:.2f}\n")
            f.write(f"- Status: {'🔥 Emerging' if topic_info['is_emerging'] else 'Stable'}\n\n")

    print(f"Generated temporal analysis report: {report_path}")
    return str(report_path)
