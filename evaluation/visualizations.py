"""
Visualization utilities for evaluation metrics

Includes:
- Confusion matrices
- ROC curves
- Training/validation curves
- Per-class performance charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class MetricsVisualizer:
    """
    Visualizer for evaluation metrics

    Example:
        >>> viz = MetricsVisualizer()
        >>> viz.plot_confusion_matrix(cm, class_names)
        >>> viz.plot_roc_curves(roc_data)
        >>> viz.save_all('results/')
    """

    def __init__(self):
        self.figures = {}

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = 'Confusion Matrix'
    ) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize (show percentages)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        self.figures['confusion_matrix'] = fig
        return fig

    def plot_roc_curves(
        self,
        roc_data: Dict,
        title: str = 'ROC Curves per Class'
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple classes

        Args:
            roc_data: Dictionary with class_name -> {fpr, tpr, auc}
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve for each class
        for class_name, data in roc_data.items():
            ax.plot(
                data['fpr'],
                data['tpr'],
                label=f"{class_name} (AUC = {data['auc']:.3f})",
                linewidth=2
            )

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures['roc_curves'] = fig
        return fig

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_metrics: List[float],
        metric_name: str = 'Accuracy',
        title: str = 'Training Progress'
    ) -> plt.Figure:
        """
        Plot training and validation curves

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            val_metrics: List of validation metrics (e.g., accuracy)
            metric_name: Name of the metric
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(train_losses) + 1)

        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot metric
        ax2.plot(epochs, val_metrics, 'g-', label=f'Validation {metric_name}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(f'Validation {metric_name}', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures['training_curves'] = fig
        return fig

    def plot_per_class_performance(
        self,
        per_class_metrics: Dict,
        metric_names: List[str] = ['precision', 'recall', 'f1'],
        title: str = 'Per-Class Performance'
    ) -> plt.Figure:
        """
        Plot per-class performance metrics

        Args:
            per_class_metrics: Dict of class_name -> {precision, recall, f1}
            metric_names: List of metrics to plot
            title: Plot title

        Returns:
            Matplotlib figure
        """
        class_names = list(per_class_metrics.keys())
        num_classes = len(class_names)
        num_metrics = len(metric_names)

        # Prepare data
        data = {metric: [] for metric in metric_names}
        for class_name in class_names:
            for metric in metric_names:
                data[metric].append(per_class_metrics[class_name][metric])

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(num_classes)
        width = 0.8 / num_metrics

        for i, metric in enumerate(metric_names):
            offset = (i - num_metrics / 2) * width + width / 2
            ax.bar(
                x + offset,
                data[metric],
                width,
                label=metric.capitalize(),
                alpha=0.8
            )

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        self.figures['per_class_performance'] = fig
        return fig

    def plot_learning_rate_schedule(
        self,
        lr_history: List[float],
        title: str = 'Learning Rate Schedule'
    ) -> plt.Figure:
        """
        Plot learning rate over epochs

        Args:
            lr_history: List of learning rates per epoch
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, 'b-', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visibility

        plt.tight_layout()
        self.figures['lr_schedule'] = fig
        return fig

    def plot_multi_task_losses(
        self,
        link_losses: List[float],
        node_losses: List[float],
        total_losses: List[float],
        title: str = 'Multi-Task Learning Losses'
    ) -> plt.Figure:
        """
        Plot losses for multi-task learning

        Args:
            link_losses: Link prediction losses
            node_losses: Node classification losses
            total_losses: Combined losses
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = range(1, len(total_losses) + 1)

        ax.plot(epochs, link_losses, 'b-', label='Link Prediction Loss', linewidth=2)
        ax.plot(epochs, node_losses, 'r-', label='Node Classification Loss', linewidth=2)
        ax.plot(epochs, total_losses, 'g-', label='Total Loss', linewidth=2, linestyle='--')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures['multi_task_losses'] = fig
        return fig

    def plot_model_comparison(
        self,
        model_names: List[str],
        accuracies: List[float],
        training_times: List[float],
        memory_usages: List[float],
        title: str = 'Model Comparison'
    ) -> plt.Figure:
        """
        Plot comparison of multiple models

        Args:
            model_names: List of model names
            accuracies: List of test accuracies
            training_times: List of training times (seconds)
            memory_usages: List of memory usages (MB)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.arange(len(model_names))

        # Accuracy
        axes[0].bar(x, accuracies, color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Test Accuracy', fontsize=11)
        axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].grid(True, axis='y', alpha=0.3)

        # Training time
        axes[1].bar(x, training_times, color='coral', alpha=0.8)
        axes[1].set_ylabel('Training Time (s)', fontsize=11)
        axes[1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(True, axis='y', alpha=0.3)

        # Memory usage
        axes[2].bar(x, memory_usages, color='seagreen', alpha=0.8)
        axes[2].set_ylabel('Memory Usage (MB)', fontsize=11)
        axes[2].set_title('Memory Usage Comparison', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[2].grid(True, axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures['model_comparison'] = fig
        return fig

    def save_all(self, output_dir: str):
        """
        Save all generated figures

        Args:
            output_dir: Directory to save figures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in self.figures.items():
            filepath = output_path / f"{name}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved {filepath}")

    def close_all(self):
        """Close all figures to free memory"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
