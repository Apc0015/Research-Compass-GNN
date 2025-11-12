#!/usr/bin/env python3
"""
Enhanced GNN Training Script - Research Compass

Demonstrates Phase 1 enhancements:
1. Learning Rate Scheduling
2. Multi-Task Learning for GAT
3. Comprehensive Evaluation Metrics
4. Mini-Batch Training

Usage:
    python train_enhanced.py --model GCN --dataset synthetic --epochs 100
    python train_enhanced.py --model GAT --multitask --dataset Cora --epochs 50
    python train_enhanced.py --model GraphSAGE --dataset synthetic --size 5000 --minibatch
"""

import torch
import torch.optim as optim
import argparse
from pathlib import Path
import time
import json

# Import our modules
from models import GCNModel, GATModel, GraphSAGEModel, GraphTransformerModel
from training.trainer import GCNTrainer, MultiTaskGATTrainer
from training.batch_training import create_trainer
from evaluation.metrics import NodeClassificationMetrics, LinkPredictionMetrics
from evaluation.report_generator import EvaluationReportGenerator
from data.dataset_utils import (
    create_synthetic_citation_network,
    load_citation_dataset,
    print_dataset_info,
    create_link_prediction_split
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced GNN Training')

    # Model selection
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSAGE', 'Transformer'],
                        help='Model architecture')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use')
    parser.add_argument('--size', type=int, default=200,
                        help='Number of papers for synthetic dataset')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension')

    # Features
    parser.add_argument('--multitask', action='store_true',
                        help='Use multi-task learning for GAT')
    parser.add_argument('--minibatch', action='store_true',
                        help='Force mini-batch training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for mini-batch training')

    # Scheduler parameters
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='Factor for learning rate reduction')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train_gcn(data, args):
    """Train GCN with learning rate scheduling"""
    print("\n" + "=" * 70)
    print("🧪 Training GCN (Graph Convolutional Network)")
    print("=" * 70)

    # Create model
    model = GCNModel(
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=data.y.max().item() + 1,
        num_layers=3,
        dropout=0.5
    )

    print(f"\n📊 Model: {model}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler configuration
    scheduler_config = {
        'mode': 'max',  # Maximize validation accuracy
        'factor': args.scheduler_factor,
        'patience': args.scheduler_patience,
        'min_lr': args.min_lr,
        'verbose': True
    }

    # Create trainer
    if args.minibatch or data.num_nodes >= 1000:
        from training.batch_training import MiniBatchTrainer
        trainer = MiniBatchTrainer(
            model, data, optimizer,
            batch_size=args.batch_size,
            num_neighbors=[10, 5]
        )
        print(f"✅ Using Mini-Batch Training (batch_size={args.batch_size})")
    else:
        trainer = GCNTrainer(model, optimizer, scheduler_config=scheduler_config)
        print("✅ Using Full-Batch Training")

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(data, data.train_mask)

        # Validate
        val_metrics = trainer.validate(data, data.val_mask)

        # Step scheduler
        if hasattr(trainer, 'step_scheduler'):
            trainer.step_scheduler(val_metrics['accuracy'])

            # Save best model
            is_best = trainer.save_best_model(val_metrics['accuracy'], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            lr = train_metrics.get('lr', args.lr)
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {lr:.6f}"
            )

    total_time = time.time() - start_time

    # Load best model
    if hasattr(trainer, 'load_best_model'):
        trainer.load_best_model()
        print(f"\n✅ Loaded best model from epoch {trainer.best_epoch}")

    # Test evaluation
    print("\n📊 Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(trainer.device), data.edge_index.to(trainer.device))
        pred = out.argmax(dim=1)
        prob = torch.softmax(out, dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]
        y_prob = prob[data.test_mask]

    test_acc = (y_pred == y_true).float().mean().item()
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    return {
        'model': model,
        'trainer': trainer,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'training_time': total_time,
        'test_accuracy': test_acc
    }


def train_gat_multitask(data, args):
    """Train GAT with multi-task learning"""
    print("\n" + "=" * 70)
    print("🧪 Training GAT with Multi-Task Learning")
    print("   Task 1: Link Prediction (70%)")
    print("   Task 2: Node Classification (30%)")
    print("=" * 70)

    # Create link prediction splits
    from data.dataset_utils import create_link_prediction_split
    train_edges, val_pos_edges, test_pos_edges, neg_edges = create_link_prediction_split(
        data, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    # Split negative edges
    num_val = val_pos_edges.shape[1]
    val_neg_edges = neg_edges[:, :num_val]
    test_neg_edges = neg_edges[:, num_val:]

    # Create model
    model = GATModel(
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=data.y.max().item() + 1,
        num_layers=2,
        heads=4,
        dropout=0.3
    )

    print(f"\n📊 Model: {model}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler configuration
    scheduler_config = {
        'mode': 'max',
        'factor': args.scheduler_factor,
        'patience': args.scheduler_patience,
        'min_lr': args.min_lr,
        'verbose': True
    }

    # Create trainer
    trainer = MultiTaskGATTrainer(
        model, optimizer,
        scheduler_config=scheduler_config,
        link_weight=0.7,
        node_weight=0.3
    )

    print("✅ Multi-Task Trainer initialized")

    # Move data to device
    data = data.to(trainer.device)
    train_edges = train_edges.to(trainer.device)
    val_pos_edges = val_pos_edges.to(trainer.device)
    val_neg_edges = val_neg_edges.to(trainer.device)

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Prepare training edges
        pos_train = train_edges
        num_pos = pos_train.shape[1]

        # Sample negative edges
        neg_train = []
        edge_set = set(map(tuple, train_edges.t().tolist()))
        while len(neg_train) < num_pos:
            src = torch.randint(0, data.num_nodes, (1,)).item()
            dst = torch.randint(0, data.num_nodes, (1,)).item()
            if src != dst and (src, dst) not in edge_set:
                neg_train.append([src, dst])
        neg_train = torch.tensor(neg_train, dtype=torch.long).t().to(trainer.device)

        # Train
        train_metrics = trainer.train_epoch(
            data, pos_train, neg_train, data.train_mask
        )

        # Validate
        val_pos_neg = torch.cat([val_pos_edges, val_neg_edges], dim=1)
        val_metrics = trainer.validate(
            data, val_pos_neg[:, :num_val], val_neg_edges, data.val_mask
        )

        # Step scheduler
        trainer.step_scheduler(val_metrics['link_acc'])
        is_best = trainer.save_best_model(val_metrics['link_acc'], epoch)

        # Print progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Total Loss: {train_metrics['total_loss']:.4f} | "
                f"Link Loss: {train_metrics['link_loss']:.4f} | "
                f"Node Loss: {train_metrics['node_loss']:.4f} | "
                f"Link Acc: {val_metrics['link_acc']:.4f} | "
                f"Node Acc: {val_metrics['node_acc']:.4f}"
            )

    total_time = time.time() - start_time

    # Load best model
    trainer.load_best_model()
    print(f"\n✅ Loaded best model from epoch {trainer.best_epoch}")

    # Test evaluation
    print("\n📊 Evaluating on test set...")
    model.eval()

    with torch.no_grad():
        # Node classification
        out = model.forward_node_classification(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        prob = torch.softmax(out, dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]
        y_prob = prob[data.test_mask]

        # Link prediction
        test_edges = torch.cat([test_pos_edges, test_neg_edges], dim=1)
        link_pred = model(data.x, data.edge_index, test_edges)
        link_labels = torch.cat([
            torch.ones(test_pos_edges.shape[1]),
            torch.zeros(test_neg_edges.shape[1])
        ]).to(trainer.device)

        link_acc = ((torch.sigmoid(link_pred) > 0.5) == link_labels).float().mean().item()

    test_acc = (y_pred == y_true).float().mean().item()
    print(f"✅ Test Node Classification Accuracy: {test_acc:.4f}")
    print(f"✅ Test Link Prediction Accuracy: {link_acc:.4f}")

    return {
        'model': model,
        'trainer': trainer,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'training_time': total_time,
        'test_accuracy': test_acc,
        'link_accuracy': link_acc
    }


def main():
    """Main training function"""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("🚀 Enhanced GNN Training - Research Compass")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load dataset
    if args.dataset == 'synthetic':
        data = create_synthetic_citation_network(
            num_papers=args.size,
            num_topics=5,
            seed=args.seed
        )
    else:
        data, info = load_citation_dataset(args.dataset)

    # Print dataset info
    print_dataset_info(data, args.dataset)

    # Train model
    if args.model == 'GCN':
        results = train_gcn(data, args)
    elif args.model == 'GAT' and args.multitask:
        results = train_gat_multitask(data, args)
    else:
        print(f"❌ Configuration not yet implemented: {args.model}")
        return

    # Generate evaluation report
    print("\n📊 Generating comprehensive evaluation report...")

    report_gen = EvaluationReportGenerator(
        num_classes=data.y.max().item() + 1,
        class_names=[f"Topic {i}" for i in range(data.y.max().item() + 1)]
    )

    # Get memory usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    # Add results
    report_gen.add_model_results(
        model_name=args.model,
        y_true=results['y_true'],
        y_pred=results['y_pred'],
        y_prob=results['y_prob'],
        training_time=results['training_time'],
        memory_usage=memory_mb,
        num_parameters=results['model'].count_parameters(),
        train_losses=results['trainer'].train_losses if hasattr(results['trainer'], 'train_losses') else [],
        val_losses=results['trainer'].val_losses if hasattr(results['trainer'], 'val_losses') else [],
        val_metrics=results['trainer'].val_metrics if hasattr(results['trainer'], 'val_metrics') else [],
        lr_history=results['trainer'].lr_history if hasattr(results['trainer'], 'lr_history') else [],
        additional_info={
            'dataset': args.dataset,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size if args.minibatch else 'Full-batch',
            'multi_task': args.multitask
        }
    )

    # Generate report
    output_dir = Path(args.output_dir) / f"{args.model}_{args.dataset}_{int(time.time())}"
    report_gen.generate_report(str(output_dir))

    # Save model
    if args.save_model:
        model_path = output_dir / f"{args.model}_best.pt"
        torch.save(results['model'].state_dict(), model_path)
        print(f"💾 Saved model to {model_path}")

    print("\n✅ Training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == '__main__':
    main()
