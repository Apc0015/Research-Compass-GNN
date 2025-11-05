#!/usr/bin/env python3
"""
GNN Visualization - Performance metrics and training visualization

Provides utilities for:
- Training history plotting
- Confusion matrices
- ROC curves
- Model comparison charts
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> str:
    """
    Plot training history (loss and metrics over epochs)

    Args:
        history: Training history dict with keys like 'train_loss', 'val_loss', etc.
        save_path: Path to save plot (PNG/SVG)
        title: Plot title

    Returns:
        HTML string with embedded plot or path to saved image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Install with: pip install matplotlib")
        return "Error: matplotlib not available"

    # Create figure with subplots
    metrics = list(history.keys())
    num_metrics = len(metrics)

    if num_metrics == 0:
        return "No metrics to plot"

    # Determine layout (2 columns)
    num_rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot each metric
    for idx, (metric_name, values) in enumerate(history.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', markersize=3, linewidth=2, label=metric_name)

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric_name.replace("_", " ").title()} over Epochs', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add min/max annotations for loss metrics
        if 'loss' in metric_name.lower():
            min_val = min(values)
            min_epoch = values.index(min_val) + 1
            ax.annotate(
                f'Best: {min_val:.4f}',
                xy=(min_epoch, min_val),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
        elif 'acc' in metric_name.lower() or 'auc' in metric_name.lower():
            max_val = max(values)
            max_epoch = values.index(max_val) + 1
            ax.annotate(
                f'Best: {max_val:.4f}',
                xy=(max_epoch, max_val),
                xytext=(10, -10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    # Hide empty subplots
    for idx in range(num_metrics, num_rows * 2):
        row = idx // 2
        col = idx % 2
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")
        return save_path
    else:
        # Return as base64 for embedding
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" />'


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> str:
    """
    Plot confusion matrix for classification results

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save plot
        normalize: Normalize confusion matrix

    Returns:
        HTML string or path to saved image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        logger.warning(f"Required library not installed: {e}")
        return "Error: matplotlib or sklearn not available"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix' + (' (Normalized)' if normalize else '')
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()

    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" />'


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
) -> str:
    """
    Plot ROC curve for binary classification or link prediction

    Args:
        y_true: True labels (0/1)
        y_scores: Predicted scores/probabilities
        save_path: Path to save plot
        title: Plot title

    Returns:
        HTML string or path to saved image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError as e:
        logger.warning(f"Required library not installed: {e}")
        return "Error: matplotlib or sklearn not available"

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" />'


def compare_model_performance(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> str:
    """
    Compare performance of multiple models

    Args:
        results: Dict of {model_name: {metric_name: value}}
        save_path: Path to save plot

    Returns:
        HTML string or path to saved image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed")
        return "Error: matplotlib not available"

    # Extract metrics and models
    models = list(results.keys())
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    metrics = sorted(list(all_metrics))

    # Prepare data
    num_metrics = len(metrics)
    num_models = len(models)

    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [results[model].get(metric, 0) for model in models]
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))

        bars = ax.bar(range(num_models), values, color=colors, alpha=0.7)
        ax.set_xticks(range(num_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()

    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" />'


def generate_performance_report(
    model_name: str,
    task: str,
    history: Dict[str, List[float]],
    final_metrics: Dict[str, float],
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Generate comprehensive performance report with all visualizations

    Args:
        model_name: Name of the model
        task: Task type (node_classification, link_prediction, etc.)
        history: Training history
        final_metrics: Final evaluation metrics
        output_dir: Directory to save report files

    Returns:
        Dict with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_files = {}

    # 1. Training history plot
    history_plot = output_path / f"{model_name}_history.png"
    plot_training_history(history, save_path=str(history_plot), title=f"{model_name} Training History")
    report_files['history_plot'] = str(history_plot)

    # 2. Generate markdown report
    report_md = output_path / f"{model_name}_report.md"

    with open(report_md, 'w') as f:
        f.write(f"# {model_name} Performance Report\n\n")
        f.write(f"**Task:** {task}\n\n")
        f.write(f"**Generated:** {np.datetime64('now')}\n\n")

        f.write("## Training Configuration\n\n")
        f.write(f"- Total Epochs: {len(history.get('train_loss', []))}\n")
        f.write(f"- Final Training Loss: {history.get('train_loss', [0])[-1]:.4f}\n")
        if 'val_loss' in history:
            f.write(f"- Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write("\n")

        f.write("## Final Metrics\n\n")
        for metric, value in sorted(final_metrics.items()):
            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
        f.write("\n")

        f.write("## Training History\n\n")
        f.write(f"![Training History]({history_plot.name})\n\n")

        f.write("## Epoch-by-Epoch Results\n\n")
        f.write("| Epoch | Train Loss | Val Loss | Val Acc/AUC |\n")
        f.write("|-------|-----------|----------|-------------|\n")

        epochs = len(history.get('train_loss', []))
        for i in range(min(10, epochs)):  # First 10 epochs
            train_loss = history.get('train_loss', [0])[i]
            val_loss = history.get('val_loss', [0])[i] if 'val_loss' in history else 0
            val_metric = history.get('val_acc', history.get('val_auc', [0]))[i] if 'val_acc' in history or 'val_auc' in history else 0
            f.write(f"| {i+1} | {train_loss:.4f} | {val_loss:.4f} | {val_metric:.4f} |\n")

        if epochs > 10:
            f.write("| ... | ... | ... | ... |\n")
            # Last epoch
            train_loss = history.get('train_loss', [0])[-1]
            val_loss = history.get('val_loss', [0])[-1] if 'val_loss' in history else 0
            val_metric = history.get('val_acc', history.get('val_auc', [0]))[-1] if 'val_acc' in history or 'val_auc' in history else 0
            f.write(f"| {epochs} | {train_loss:.4f} | {val_loss:.4f} | {val_metric:.4f} |\n")

    report_files['report'] = str(report_md)

    logger.info(f"Performance report generated in {output_dir}")
    return report_files


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    dummy_history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26],
        'val_loss': [0.85, 0.65, 0.55, 0.45, 0.40, 0.38, 0.36, 0.35, 0.34, 0.33],
        'train_acc': [0.6, 0.7, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.85],
        'val_acc': [0.58, 0.68, 0.72, 0.75, 0.77, 0.78, 0.79, 0.80, 0.80, 0.81]
    }

    print("Generating test plots...")
    plot_training_history(dummy_history, save_path="test_history.png")
    print("✓ Training history plot saved")

    # Test confusion matrix
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])
    plot_confusion_matrix(y_true, y_pred, class_names=['A', 'B', 'C'], save_path="test_cm.png")
    print("✓ Confusion matrix saved")

    # Test ROC curve
    y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.85])
    plot_roc_curve(y_true_binary, y_scores, save_path="test_roc.png")
    print("✓ ROC curve saved")

    print("\nAll test plots generated successfully!")
