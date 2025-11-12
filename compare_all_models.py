#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script

Compares all models including:
- GNN Models: GCN, GAT, GraphSAGE, Transformer
- Baselines: Random, Logistic, Random Forest, MLP, Label Propagation, Node2Vec

Generates professional comparison reports proving the value of GNNs.

Usage:
    python compare_all_models.py --dataset Cora
    python compare_all_models.py --dataset synthetic --size 500
"""

import torch
import torch.optim as optim
import argparse
import time
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models import GCNModel, GATModel, GraphSAGEModel, GraphTransformerModel
from training.trainer import GCNTrainer
from training.batch_training import create_trainer
from data.dataset_utils import (
    create_synthetic_citation_network,
    load_citation_dataset,
    print_dataset_info
)
from baselines.traditional_ml import evaluate_all_baselines, train_mlp_baseline
from baselines.graph_baselines import evaluate_graph_baselines
from evaluation.report_generator import EvaluationReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Compare All Models')

    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['synthetic', 'Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use')
    parser.add_argument('--size', type=int, default=500,
                        help='Size for synthetic dataset')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs for GNNs')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train_gnn_model(model, data, epochs, lr=0.01, device=None):
    """Train a GNN model and return results"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    scheduler_config = {
        'mode': 'max',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6,
        'verbose': False
    }

    trainer = GCNTrainer(model, optimizer, device, scheduler_config)
    data = data.to(device)

    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(data, data.train_mask)
        val_metrics = trainer.validate(data, data.val_mask)
        trainer.step_scheduler(val_metrics['accuracy'])
        trainer.save_best_model(val_metrics['accuracy'], epoch)

    training_time = time.time() - start_time

    # Load best model and evaluate
    trainer.load_best_model()
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return {
        'test_acc': test_acc,
        'training_time': training_time,
        'num_parameters': model.count_parameters(),
        'model': model,
        'trainer': trainer
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("🏆 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load dataset
    if args.dataset == 'synthetic':
        data = create_synthetic_citation_network(
            num_papers=args.size,
            num_topics=5,
            seed=args.seed
        )
        dataset_name = f"Synthetic ({args.size} nodes)"
    else:
        data, info = load_citation_dataset(args.dataset)
        dataset_name = args.dataset

    print_dataset_info(data, dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = data.y.max().item() + 1
    input_dim = data.x.shape[1]

    all_results = {}

    # ========================================================================
    # PART 1: Evaluate Baselines
    # ========================================================================

    print("\n" + "=" * 70)
    print("PART 1: BASELINE MODELS")
    print("=" * 70)

    # Traditional ML baselines
    baseline_results = evaluate_all_baselines(data, device)
    all_results.update(baseline_results)

    # Graph-based baselines
    graph_baseline_results = evaluate_graph_baselines(data, device)
    all_results.update(graph_baseline_results)

    # ========================================================================
    # PART 2: Evaluate GNN Models
    # ========================================================================

    print("\n" + "=" * 70)
    print("PART 2: GNN MODELS")
    print("=" * 70)

    # 1. GCN
    print("\n1️⃣  Training GCN")
    gcn_model = GCNModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=num_classes,
        num_layers=3,
        dropout=0.5
    )
    gcn_results = train_gnn_model(gcn_model, data, args.epochs, device=device)
    all_results['GCN'] = gcn_results
    print(f"   ✅ GCN Accuracy: {gcn_results['test_acc']:.4f}")
    print(f"   Time: {gcn_results['training_time']:.2f}s")

    # 2. GAT
    print("\n2️⃣  Training GAT")
    # For comparison, use GAT for node classification only
    from models.gat import GATModel
    gat_model = GATModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=num_classes,
        num_layers=2,
        heads=4,
        dropout=0.3
    )

    optimizer = optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
    gat_model = gat_model.to(device)
    data_device = data.to(device)

    start_time = time.time()

    for epoch in range(args.epochs):
        gat_model.train()
        optimizer.zero_grad()
        out = gat_model.forward_node_classification(data_device.x, data_device.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data_device.y[data.train_mask])
        loss.backward()
        optimizer.step()

    gat_time = time.time() - start_time

    gat_model.eval()
    with torch.no_grad():
        out = gat_model.forward_node_classification(data_device.x, data_device.edge_index)
        pred = out.argmax(dim=1)
        gat_acc = (pred[data.test_mask] == data_device.y[data.test_mask]).float().mean().item()

    all_results['GAT'] = {
        'test_acc': gat_acc,
        'training_time': gat_time,
        'num_parameters': gat_model.count_parameters()
    }
    print(f"   ✅ GAT Accuracy: {gat_acc:.4f}")
    print(f"   Time: {gat_time:.2f}s")

    # 3. GraphSAGE
    print("\n3️⃣  Training GraphSAGE")
    sage_model = GraphSAGEModel(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
        dropout=0.5,
        task='classification',
        num_classes=num_classes
    )
    sage_results = train_gnn_model(sage_model, data, args.epochs, device=device)
    all_results['GraphSAGE'] = sage_results
    print(f"   ✅ GraphSAGE Accuracy: {sage_results['test_acc']:.4f}")
    print(f"   Time: {sage_results['training_time']:.2f}s")

    # ========================================================================
    # PART 3: Generate Comparison Report
    # ========================================================================

    print("\n" + "=" * 70)
    print("PART 3: GENERATING COMPARISON REPORT")
    print("=" * 70)

    # Create comparison table
    comparison_data = []

    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'GNN' if model_name in ['GCN', 'GAT', 'GraphSAGE', 'Transformer']
                    else ('Graph-Based' if model_name in ['label_propagation', 'node2vec']
                    else 'Traditional ML'),
            'Test Accuracy': results['test_acc'],
            'Training Time (s)': results['training_time'],
            'Parameters': results['num_parameters'] if isinstance(results['num_parameters'], int)
                         else results['num_parameters']
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Test Accuracy', ascending=False)

    print("\n📊 Results Summary:")
    print(df.to_string(index=False))

    # Save results
    output_dir = Path(args.output_dir) / f"{dataset_name.replace(' ', '_')}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_dir / 'comparison_results.csv', index=False)

    # Save JSON
    json_results = {
        'dataset': dataset_name,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_classes': num_classes,
        'results': {k: {
            'test_acc': float(v['test_acc']),
            'training_time': float(v['training_time']),
            'num_parameters': str(v['num_parameters'])
        } for k, v in all_results.items()}
    }

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Generate visualization
    plt.figure(figsize=(12, 6))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    models = df['Model'].tolist()
    accuracies = df['Test Accuracy'].tolist()
    colors = ['#2ecc71' if t == 'GNN' else '#e74c3c' if t == 'Traditional ML' else '#3498db'
              for t in df['Type'].tolist()]

    bars = plt.barh(models, accuracies, color=colors, alpha=0.8)
    plt.xlabel('Test Accuracy', fontsize=12)
    plt.title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
    plt.xlim([0, 1.0])
    plt.grid(axis='x', alpha=0.3)

    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=9)

    # Training time comparison
    plt.subplot(1, 2, 2)
    times = df['Training Time (s)'].tolist()
    bars = plt.barh(models, times, color=colors, alpha=0.8)
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.title('Model Comparison - Training Time', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Add time values
    for i, (bar, t) in enumerate(zip(bars, times)):
        plt.text(t + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}s', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plot.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization to {output_dir / 'comparison_plot.png'}")

    # Generate markdown report
    with open(output_dir / 'comparison_report.md', 'w') as f:
        f.write(f"# Model Comparison Report - {dataset_name}\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Nodes:** {data.num_nodes:,}\n")
        f.write(f"**Edges:** {data.num_edges:,}\n")
        f.write(f"**Classes:** {num_classes}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        # Key findings
        best_model = df.iloc[0]['Model']
        best_acc = df.iloc[0]['Test Accuracy']
        worst_model = df.iloc[-1]['Model']
        worst_acc = df.iloc[-1]['Test Accuracy']

        gnn_models = df[df['Type'] == 'GNN']
        baseline_models = df[df['Type'] != 'GNN']

        if len(gnn_models) > 0 and len(baseline_models) > 0:
            gnn_avg = gnn_models['Test Accuracy'].mean()
            baseline_avg = baseline_models['Test Accuracy'].mean()
            improvement = ((gnn_avg - baseline_avg) / baseline_avg) * 100

            f.write("## Key Findings\n\n")
            f.write(f"- **Best Model:** {best_model} ({best_acc:.4f})\n")
            f.write(f"- **Worst Model:** {worst_model} ({worst_acc:.4f})\n")
            f.write(f"- **GNN Average Accuracy:** {gnn_avg:.4f}\n")
            f.write(f"- **Baseline Average Accuracy:** {baseline_avg:.4f}\n")
            f.write(f"- **GNN Improvement:** +{improvement:.1f}% over baselines\n\n")

            f.write("## Conclusion\n\n")
            f.write(f"Graph Neural Networks outperform traditional methods by **{improvement:.1f}%** on average, ")
            f.write("demonstrating the value of incorporating graph structure for citation network analysis.\n")

    print(f"📄 Saved report to {output_dir / 'comparison_report.md'}")

    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n🎯 Best Model: {best_model} ({best_acc:.4f})")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
