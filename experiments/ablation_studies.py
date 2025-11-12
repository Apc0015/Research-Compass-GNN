"""
Ablation Study Framework

Systematic studies to understand which components contribute to performance:
- Feature ablation (graph vs features)
- Architecture ablation (layers, dimensions, heads)
- Training ablation (LR scheduler, dropout, weight decay)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import time

from models import GCNModel, GATModel
from training.trainer import GCNTrainer
from data.dataset_utils import create_synthetic_citation_network


class AblationStudy:
    """
    Framework for running systematic ablation studies

    Example:
        >>> study = AblationStudy(data)
        >>> results = study.run_architecture_ablation()
        >>> study.save_results('results/ablation')
    """

    def __init__(self, data, epochs: int = 100, num_runs: int = 3):
        """
        Args:
            data: PyG Data object
            epochs: Training epochs per experiment
            num_runs: Number of runs per configuration (for statistical robustness)
        """
        self.data = data
        self.epochs = epochs
        self.num_runs = num_runs
        self.results = {}

    def run_single_experiment(
        self,
        model,
        lr: float = 0.01,
        weight_decay: float = 5e-4
    ) -> Dict:
        """
        Run a single training experiment

        Returns:
            Dictionary with results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        trainer = GCNTrainer(model, optimizer, device)
        data_device = self.data.to(device)

        start_time = time.time()

        # Training loop
        for epoch in range(self.epochs):
            trainer.train_epoch(data_device, data_device.train_mask)
            val_metrics = trainer.validate(data_device, data_device.val_mask)
            trainer.step_scheduler(val_metrics['accuracy'])
            trainer.save_best_model(val_metrics['accuracy'], epoch)

        training_time = time.time() - start_time

        # Evaluate
        trainer.load_best_model()
        model.eval()

        with torch.no_grad():
            out = model(data_device.x, data_device.edge_index)
            pred = out.argmax(dim=1)

            test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

        return {
            'test_acc': test_acc,
            'training_time': training_time,
            'num_parameters': model.count_parameters()
        }

    def feature_ablation(self) -> Dict:
        """
        Ablate features to test graph structure importance

        Experiments:
        1. Full model (features + graph)
        2. Graph only (random features)
        3. Features only (no graph edges)
        """
        print("\n" + "=" * 70)
        print("🔬 Feature Ablation Study")
        print("=" * 70)

        results = {}
        num_classes = self.data.y.max().item() + 1
        input_dim = self.data.x.shape[1]

        # 1. Full model (baseline)
        print("\n1️⃣  Full Model (Features + Graph)")
        full_model_scores = []

        for run in range(self.num_runs):
            model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
            result = self.run_single_experiment(model)
            full_model_scores.append(result['test_acc'])
            print(f"   Run {run+1}: {result['test_acc']:.4f}")

        results['full_model'] = {
            'mean': float(np.mean(full_model_scores)),
            'std': float(np.std(full_model_scores)),
            'scores': full_model_scores
        }

        # 2. Graph only (random features)
        print("\n2️⃣  Graph Only (Random Features)")
        graph_only_scores = []

        # Replace features with random
        data_random = self.data.clone()
        data_random.x = torch.randn_like(data_random.x)

        for run in range(self.num_runs):
            model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
            original_data = self.data
            self.data = data_random
            result = self.run_single_experiment(model)
            self.data = original_data
            graph_only_scores.append(result['test_acc'])
            print(f"   Run {run+1}: {result['test_acc']:.4f}")

        results['graph_only'] = {
            'mean': float(np.mean(graph_only_scores)),
            'std': float(np.std(graph_only_scores)),
            'scores': graph_only_scores
        }

        # 3. Features only (remove edges)
        print("\n3️⃣  Features Only (No Graph)")
        from baselines.traditional_ml import train_mlp_baseline

        features_only_scores = []

        for run in range(self.num_runs):
            result = train_mlp_baseline(self.data, epochs=self.epochs)
            features_only_scores.append(result['test_acc'])
            print(f"   Run {run+1}: {result['test_acc']:.4f}")

        results['features_only'] = {
            'mean': float(np.mean(features_only_scores)),
            'std': float(np.std(features_only_scores)),
            'scores': features_only_scores
        }

        # Summary
        print("\n" + "=" * 70)
        print("📊 Feature Ablation Results:")
        print(f"   Full Model:      {results['full_model']['mean']:.4f} ± {results['full_model']['std']:.4f}")
        print(f"   Graph Only:      {results['graph_only']['mean']:.4f} ± {results['graph_only']['std']:.4f}")
        print(f"   Features Only:   {results['features_only']['mean']:.4f} ± {results['features_only']['std']:.4f}")

        graph_contribution = results['full_model']['mean'] - results['features_only']['mean']
        feature_contribution = results['full_model']['mean'] - results['graph_only']['mean']

        print(f"\n   Graph contribution:   +{graph_contribution:.4f}")
        print(f"   Feature contribution: +{feature_contribution:.4f}")
        print("=" * 70)

        self.results['feature_ablation'] = results
        return results

    def architecture_ablation(self) -> Dict:
        """
        Ablate architecture parameters

        Experiments:
        1. Number of layers (1, 2, 3, 4)
        2. Hidden dimensions (64, 128, 256, 512)
        3. Dropout rates (0.0, 0.3, 0.5, 0.7)
        """
        print("\n" + "=" * 70)
        print("🔬 Architecture Ablation Study")
        print("=" * 70)

        results = {}
        num_classes = self.data.y.max().item() + 1
        input_dim = self.data.x.shape[1]

        # 1. Number of layers
        print("\n1️⃣  Varying Number of Layers")
        layer_configs = [1, 2, 3, 4]
        results['num_layers'] = {}

        for num_layers in layer_configs:
            print(f"\n   Testing {num_layers} layers:")
            scores = []

            for run in range(self.num_runs):
                model = GCNModel(
                    input_dim=input_dim,
                    hidden_dim=128,
                    output_dim=num_classes,
                    num_layers=num_layers
                )
                result = self.run_single_experiment(model)
                scores.append(result['test_acc'])
                print(f"      Run {run+1}: {result['test_acc']:.4f}")

            results['num_layers'][num_layers] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': scores
            }

        # 2. Hidden dimensions
        print("\n2️⃣  Varying Hidden Dimensions")
        hidden_configs = [64, 128, 256, 512]
        results['hidden_dim'] = {}

        for hidden_dim in hidden_configs:
            print(f"\n   Testing hidden_dim={hidden_dim}:")
            scores = []

            for run in range(self.num_runs):
                model = GCNModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=num_classes,
                    num_layers=3
                )
                result = self.run_single_experiment(model)
                scores.append(result['test_acc'])
                print(f"      Run {run+1}: {result['test_acc']:.4f}")

            results['hidden_dim'][hidden_dim] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': scores
            }

        # 3. Dropout rates
        print("\n3️⃣  Varying Dropout Rates")
        dropout_configs = [0.0, 0.3, 0.5, 0.7]
        results['dropout'] = {}

        for dropout in dropout_configs:
            print(f"\n   Testing dropout={dropout}:")
            scores = []

            for run in range(self.num_runs):
                model = GCNModel(
                    input_dim=input_dim,
                    hidden_dim=128,
                    output_dim=num_classes,
                    num_layers=3,
                    dropout=dropout
                )
                result = self.run_single_experiment(model)
                scores.append(result['test_acc'])
                print(f"      Run {run+1}: {result['test_acc']:.4f}")

            results['dropout'][dropout] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': scores
            }

        # Summary
        print("\n" + "=" * 70)
        print("📊 Architecture Ablation Results:")
        print("\n   Num Layers:")
        for num_layers, data in results['num_layers'].items():
            print(f"      {num_layers} layers: {data['mean']:.4f} ± {data['std']:.4f}")

        print("\n   Hidden Dimensions:")
        for hidden_dim, data in results['hidden_dim'].items():
            print(f"      {hidden_dim}d: {data['mean']:.4f} ± {data['std']:.4f}")

        print("\n   Dropout:")
        for dropout, data in results['dropout'].items():
            print(f"      {dropout}: {data['mean']:.4f} ± {data['std']:.4f}")
        print("=" * 70)

        self.results['architecture_ablation'] = results
        return results

    def training_ablation(self) -> Dict:
        """
        Ablate training hyperparameters

        Experiments:
        1. With/without LR scheduler
        2. Different learning rates
        3. Different weight decay values
        """
        print("\n" + "=" * 70)
        print("🔬 Training Ablation Study")
        print("=" * 70)

        results = {}
        num_classes = self.data.y.max().item() + 1
        input_dim = self.data.x.shape[1]

        # 1. Learning rate scheduler
        print("\n1️⃣  With/Without LR Scheduler")

        # Without scheduler
        print("\n   Without LR Scheduler:")
        no_scheduler_scores = []

        for run in range(self.num_runs):
            model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            # No scheduler
            from training.trainer import BaseTrainer
            trainer = BaseTrainer(model, optimizer, device, scheduler_config=None)

            # Manually disable scheduler
            trainer.scheduler = None

            data_device = self.data.to(device)

            for epoch in range(self.epochs):
                def loss_fn(m, d):
                    out = m(d.x, d.edge_index)
                    return torch.nn.functional.cross_entropy(out[d.train_mask], d.y[d.train_mask])

                trainer.train_epoch(data_device, loss_fn)

            model.eval()
            with torch.no_grad():
                out = model(data_device.x, data_device.edge_index)
                pred = out.argmax(dim=1)
                test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

            no_scheduler_scores.append(test_acc)
            print(f"      Run {run+1}: {test_acc:.4f}")

        # With scheduler (from feature ablation)
        print("\n   With LR Scheduler:")
        with_scheduler_scores = []

        for run in range(self.num_runs):
            model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
            result = self.run_single_experiment(model)
            with_scheduler_scores.append(result['test_acc'])
            print(f"      Run {run+1}: {result['test_acc']:.4f}")

        results['lr_scheduler'] = {
            'without': {
                'mean': float(np.mean(no_scheduler_scores)),
                'std': float(np.std(no_scheduler_scores))
            },
            'with': {
                'mean': float(np.mean(with_scheduler_scores)),
                'std': float(np.std(with_scheduler_scores))
            }
        }

        # Summary
        print("\n" + "=" * 70)
        print("📊 Training Ablation Results:")
        print(f"   Without LR Scheduler: {results['lr_scheduler']['without']['mean']:.4f} ± {results['lr_scheduler']['without']['std']:.4f}")
        print(f"   With LR Scheduler:    {results['lr_scheduler']['with']['mean']:.4f} ± {results['lr_scheduler']['with']['std']:.4f}")

        improvement = results['lr_scheduler']['with']['mean'] - results['lr_scheduler']['without']['mean']
        print(f"   Improvement: +{improvement:.4f}")
        print("=" * 70)

        self.results['training_ablation'] = results
        return results

    def save_results(self, output_dir: str):
        """Save all ablation study results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(output_path / 'ablation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Saved ablation results to {output_path / 'ablation_results.json'}")

        # Generate markdown report
        with open(output_path / 'ablation_report.md', 'w') as f:
            f.write("# Ablation Study Report\n\n")

            if 'feature_ablation' in self.results:
                f.write("## Feature Ablation\n\n")
                fa = self.results['feature_ablation']
                f.write("| Configuration | Accuracy | Std |\n")
                f.write("|---------------|----------|-----|\n")
                for config, data in fa.items():
                    f.write(f"| {config} | {data['mean']:.4f} | {data['std']:.4f} |\n")
                f.write("\n")

            if 'architecture_ablation' in self.results:
                f.write("## Architecture Ablation\n\n")
                aa = self.results['architecture_ablation']

                f.write("### Number of Layers\n\n")
                f.write("| Layers | Accuracy | Std |\n")
                f.write("|--------|----------|-----|\n")
                for layers, data in aa['num_layers'].items():
                    f.write(f"| {layers} | {data['mean']:.4f} | {data['std']:.4f} |\n")
                f.write("\n")

                f.write("### Hidden Dimensions\n\n")
                f.write("| Hidden Dim | Accuracy | Std |\n")
                f.write("|------------|----------|-----|\n")
                for dim, data in aa['hidden_dim'].items():
                    f.write(f"| {dim} | {data['mean']:.4f} | {data['std']:.4f} |\n")
                f.write("\n")

            if 'training_ablation' in self.results:
                f.write("## Training Ablation\n\n")
                ta = self.results['training_ablation']
                f.write("| Configuration | Accuracy | Std |\n")
                f.write("|---------------|----------|-----|\n")
                f.write(f"| Without LR Scheduler | {ta['lr_scheduler']['without']['mean']:.4f} | {ta['lr_scheduler']['without']['std']:.4f} |\n")
                f.write(f"| With LR Scheduler | {ta['lr_scheduler']['with']['mean']:.4f} | {ta['lr_scheduler']['with']['std']:.4f} |\n")

        print(f"📄 Saved ablation report to {output_path / 'ablation_report.md'}")


def run_comprehensive_ablation(data, output_dir: str = 'results/ablation'):
    """
    Run comprehensive ablation studies

    Args:
        data: PyG Data object
        output_dir: Output directory

    Returns:
        Dictionary with all results
    """
    study = AblationStudy(data, epochs=100, num_runs=3)

    # Run all ablations
    study.feature_ablation()
    study.architecture_ablation()
    study.training_ablation()

    # Save results
    study.save_results(output_dir)

    return study.results
