"""
Comprehensive Evaluation Report Generator

Generates professional evaluation reports with:
- Performance metrics tables
- Visualizations
- Statistical comparisons
- Markdown and HTML output
"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
from .metrics import NodeClassificationMetrics, LinkPredictionMetrics, StatisticalTests
from .visualizations import MetricsVisualizer


class EvaluationReportGenerator:
    """
    Generate comprehensive evaluation reports

    Example:
        >>> generator = EvaluationReportGenerator()
        >>> generator.add_model_results('GCN', y_true, y_pred, y_prob, training_time, memory_usage)
        >>> generator.generate_report('results/')
    """

    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Topic {i}" for i in range(num_classes)]

        self.models = {}
        self.visualizer = MetricsVisualizer()

    def add_model_results(
        self,
        model_name: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None,
        training_time: float = 0.0,
        memory_usage: float = 0.0,
        num_parameters: int = 0,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        val_metrics: Optional[List[float]] = None,
        lr_history: Optional[List[float]] = None,
        additional_info: Optional[Dict] = None
    ):
        """
        Add results for a model

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            training_time: Training time in seconds
            memory_usage: Memory usage in MB
            num_parameters: Number of model parameters
            train_losses: Training losses per epoch (optional)
            val_losses: Validation losses per epoch (optional)
            val_metrics: Validation metrics per epoch (optional)
            lr_history: Learning rate history (optional)
            additional_info: Additional information dictionary (optional)
        """
        # Compute metrics
        metrics_computer = NodeClassificationMetrics(self.num_classes, self.class_names)
        metrics = metrics_computer.compute(y_true, y_pred, y_prob)

        # Store results
        self.models[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'training_time': training_time,
            'memory_usage': memory_usage,
            'num_parameters': num_parameters,
            'train_losses': train_losses or [],
            'val_losses': val_losses or [],
            'val_metrics': val_metrics or [],
            'lr_history': lr_history or [],
            'additional_info': additional_info or {}
        }

    def generate_markdown_report(self) -> str:
        """
        Generate markdown report

        Returns:
            Markdown formatted string
        """
        report = []

        # Header
        report.append("# GNN Model Evaluation Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Number of Models:** {len(self.models)}\n")
        report.append("---\n")

        # Summary table
        report.append("## Overall Performance Summary\n")
        report.append("| Model | Accuracy | F1 (Macro) | Precision | Recall | Training Time (s) | Memory (MB) | Parameters |")
        report.append("|-------|----------|------------|-----------|--------|-------------------|-------------|------------|")

        for model_name, data in self.models.items():
            metrics = data['metrics']
            report.append(
                f"| {model_name} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['f1_macro']:.4f} | "
                f"{metrics['precision_macro']:.4f} | "
                f"{metrics['recall_macro']:.4f} | "
                f"{data['training_time']:.2f} | "
                f"{data['memory_usage']:.2f} | "
                f"{data['num_parameters']:,} |"
            )

        report.append("\n")

        # Per-model detailed results
        for model_name, data in self.models.items():
            metrics = data['metrics']

            report.append(f"## {model_name} - Detailed Results\n")

            # Overall metrics
            report.append("### Overall Metrics\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Accuracy | {metrics['accuracy']:.4f} |")
            report.append(f"| Precision (Macro) | {metrics['precision_macro']:.4f} |")
            report.append(f"| Precision (Weighted) | {metrics['precision_weighted']:.4f} |")
            report.append(f"| Recall (Macro) | {metrics['recall_macro']:.4f} |")
            report.append(f"| Recall (Weighted) | {metrics['recall_weighted']:.4f} |")
            report.append(f"| F1 (Macro) | {metrics['f1_macro']:.4f} |")
            report.append(f"| F1 (Weighted) | {metrics['f1_weighted']:.4f} |")

            if 'roc_auc_macro' in metrics and metrics['roc_auc_macro'] is not None:
                report.append(f"| ROC AUC (Macro) | {metrics['roc_auc_macro']:.4f} |")
                report.append(f"| ROC AUC (Weighted) | {metrics['roc_auc_weighted']:.4f} |")

            report.append("\n")

            # Per-class metrics
            report.append("### Per-Class Performance\n")
            report.append("| Class | Precision | Recall | F1-Score |")
            report.append("|-------|-----------|--------|----------|")

            for class_name, class_metrics in metrics['per_class'].items():
                report.append(
                    f"| {class_name} | "
                    f"{class_metrics['precision']:.4f} | "
                    f"{class_metrics['recall']:.4f} | "
                    f"{class_metrics['f1']:.4f} |"
                )

            report.append("\n")

            # Training info
            report.append("### Training Information\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Training Time | {data['training_time']:.2f} seconds |")
            report.append(f"| Memory Usage | {data['memory_usage']:.2f} MB |")
            report.append(f"| Model Parameters | {data['num_parameters']:,} |")

            if data['train_losses']:
                report.append(f"| Epochs Trained | {len(data['train_losses'])} |")
                report.append(f"| Final Train Loss | {data['train_losses'][-1]:.4f} |")
                report.append(f"| Final Val Loss | {data['val_losses'][-1]:.4f} |")

            if data['lr_history']:
                report.append(f"| Initial LR | {data['lr_history'][0]:.6f} |")
                report.append(f"| Final LR | {data['lr_history'][-1]:.6f} |")

            report.append("\n")

            # Additional info
            if data['additional_info']:
                report.append("### Additional Information\n")
                for key, value in data['additional_info'].items():
                    report.append(f"- **{key}:** {value}")
                report.append("\n")

            report.append("---\n")

        # Model comparison
        if len(self.models) > 1:
            report.append("## Model Comparison\n")

            report.append("### Statistical Comparison\n")

            model_names = list(self.models.keys())
            accuracies = [self.models[m]['metrics']['accuracy'] for m in model_names]

            best_model = model_names[accuracies.index(max(accuracies))]
            worst_model = model_names[accuracies.index(min(accuracies))]

            report.append(f"- **Best Model:** {best_model} (Accuracy: {max(accuracies):.4f})")
            report.append(f"- **Worst Model:** {worst_model} (Accuracy: {min(accuracies):.4f})")
            report.append(f"- **Accuracy Range:** {max(accuracies) - min(accuracies):.4f}")
            report.append("\n")

        return "\n".join(report)

    def generate_visualizations(self, output_dir: str):
        """
        Generate all visualizations

        Args:
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, data in self.models.items():
            metrics = data['metrics']

            # Confusion matrix
            self.visualizer.plot_confusion_matrix(
                metrics['confusion_matrix'],
                self.class_names,
                normalize=True,
                title=f'{model_name} - Confusion Matrix (Normalized)'
            )

            # ROC curves
            if 'roc_curves' in metrics:
                self.visualizer.plot_roc_curves(
                    metrics['roc_curves'],
                    title=f'{model_name} - ROC Curves'
                )

            # Training curves
            if data['train_losses']:
                self.visualizer.plot_training_curves(
                    data['train_losses'],
                    data['val_losses'],
                    data['val_metrics'],
                    metric_name='Accuracy',
                    title=f'{model_name} - Training Progress'
                )

            # Per-class performance
            self.visualizer.plot_per_class_performance(
                metrics['per_class'],
                metric_names=['precision', 'recall', 'f1'],
                title=f'{model_name} - Per-Class Performance'
            )

            # Learning rate schedule
            if data['lr_history']:
                self.visualizer.plot_learning_rate_schedule(
                    data['lr_history'],
                    title=f'{model_name} - Learning Rate Schedule'
                )

        # Model comparison
        if len(self.models) > 1:
            model_names = list(self.models.keys())
            accuracies = [self.models[m]['metrics']['accuracy'] for m in model_names]
            training_times = [self.models[m]['training_time'] for m in model_names]
            memory_usages = [self.models[m]['memory_usage'] for m in model_names]

            self.visualizer.plot_model_comparison(
                model_names,
                accuracies,
                training_times,
                memory_usages,
                title='Model Comparison'
            )

        # Save all figures
        self.visualizer.save_all(output_dir)

    def save_results_json(self, filepath: str):
        """
        Save results as JSON

        Args:
            filepath: Path to save JSON file
        """
        results = {}

        for model_name, data in self.models.items():
            # Convert tensors to lists for JSON serialization
            results[model_name] = {
                'accuracy': float(data['metrics']['accuracy']),
                'f1_macro': float(data['metrics']['f1_macro']),
                'precision_macro': float(data['metrics']['precision_macro']),
                'recall_macro': float(data['metrics']['recall_macro']),
                'training_time': float(data['training_time']),
                'memory_usage': float(data['memory_usage']),
                'num_parameters': int(data['num_parameters']),
                'per_class': {
                    class_name: {
                        k: float(v) for k, v in class_metrics.items()
                    }
                    for class_name, class_metrics in data['metrics']['per_class'].items()
                }
            }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def generate_report(self, output_dir: str):
        """
        Generate complete report with visualizations

        Args:
            output_dir: Directory to save report and visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        markdown = self.generate_markdown_report()
        with open(output_path / 'evaluation_report.md', 'w') as f:
            f.write(markdown)
        print(f"Saved evaluation report to {output_path / 'evaluation_report.md'}")

        # Generate visualizations
        viz_dir = output_path / 'visualizations'
        self.generate_visualizations(str(viz_dir))
        print(f"Saved visualizations to {viz_dir}")

        # Save JSON results
        self.save_results_json(str(output_path / 'results.json'))
        print(f"Saved results JSON to {output_path / 'results.json'}")

        print("\n✅ Report generation complete!")
