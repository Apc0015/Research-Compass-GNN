"""
Comprehensive Evaluation Metrics for GNN Models

Includes:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves and AUC
- Per-class performance
- Statistical significance tests
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class NodeClassificationMetrics:
    """
    Comprehensive metrics for node classification tasks

    Example:
        >>> metrics = NodeClassificationMetrics(num_classes=5)
        >>> results = metrics.compute(y_true, y_pred, y_prob)
        >>> print(metrics.get_report())
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_prob: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute all metrics

        Args:
            y_true: True labels [num_samples]
            y_pred: Predicted labels [num_samples]
            y_prob: Prediction probabilities [num_samples, num_classes] (optional)

        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()

        results = {}

        # Basic accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (macro and weighted)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        results['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # ROC AUC (if probabilities provided)
        if y_prob is not None:
            try:
                # Multiclass ROC AUC
                results['roc_auc_macro'] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class='ovr',
                    average='macro'
                )
                results['roc_auc_weighted'] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class='ovr',
                    average='weighted'
                )

                # Per-class ROC curves
                results['roc_curves'] = {}
                for i, class_name in enumerate(self.class_names):
                    y_true_binary = (y_true == i).astype(int)
                    if len(np.unique(y_true_binary)) > 1:  # Need both classes
                        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        results['roc_curves'][class_name] = {
                            'fpr': fpr,
                            'tpr': tpr,
                            'thresholds': thresholds,
                            'auc': roc_auc
                        }
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC: {e}")
                results['roc_auc_macro'] = None

        return results

    def get_report(self, y_true, y_pred) -> str:
        """
        Get sklearn classification report as string

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )


class LinkPredictionMetrics:
    """
    Metrics for link prediction tasks

    Example:
        >>> metrics = LinkPredictionMetrics()
        >>> results = metrics.compute(y_true, y_pred, y_score)
    """

    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_score: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute link prediction metrics

        Args:
            y_true: True labels (0 or 1) [num_edges]
            y_pred: Predicted labels (0 or 1) [num_edges]
            y_score: Prediction scores [num_edges] (optional)

        Returns:
            Dictionary with metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.cpu().numpy()

        results = {}

        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = results['confusion_matrix'].ravel()
        results['true_positive'] = tp
        results['false_positive'] = fp
        results['true_negative'] = tn
        results['false_negative'] = fn

        # Specificity
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ROC AUC (if scores provided)
        if y_score is not None:
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                results['roc_auc'] = auc(fpr, tpr)
                results['fpr'] = fpr
                results['tpr'] = tpr
                results['thresholds'] = thresholds
            except Exception as e:
                print(f"Warning: Could not compute ROC curve: {e}")
                results['roc_auc'] = None

        return results


class StatisticalTests:
    """
    Statistical significance tests for model comparison

    Example:
        >>> tests = StatisticalTests()
        >>> p_value = tests.paired_t_test(model1_accs, model2_accs)
    """

    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Paired t-test for comparing two models

        Args:
            scores1: Scores from model 1 (e.g., accuracy per fold)
            scores2: Scores from model 2
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Mean difference and confidence interval
        diff = scores1 - scores2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        n = len(diff)
        se = std_diff / np.sqrt(n)

        # 95% confidence interval
        ci = stats.t.interval(1 - alpha, n - 1, loc=mean_diff, scale=se)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_difference': mean_diff,
            'confidence_interval': ci,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'model1_std': np.std(scores1),
            'model2_std': np.std(scores2)
        }

    @staticmethod
    def compute_confidence_interval(
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute mean and confidence interval

        Args:
            scores: List of scores
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        scores = np.array(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        se = std / np.sqrt(n)

        alpha = 1 - confidence
        ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)

        return mean, ci[0], ci[1]


def compute_class_distribution(y: torch.Tensor, num_classes: int) -> Dict:
    """
    Compute class distribution statistics

    Args:
        y: Labels tensor
        num_classes: Number of classes

    Returns:
        Dictionary with distribution info
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    distribution = {}
    for cls in range(num_classes):
        count = counts[unique == cls][0] if cls in unique else 0
        distribution[f'class_{cls}'] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }

    # Class balance metrics
    percentages = [distribution[f'class_{i}']['percentage'] for i in range(num_classes)]
    distribution['imbalance_ratio'] = max(percentages) / min(percentages) if min(percentages) > 0 else float('inf')
    distribution['entropy'] = stats.entropy([c / total for c in counts])

    return distribution
