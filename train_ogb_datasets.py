#!/usr/bin/env python3
"""
Training script for Open Graph Benchmark (OGB) datasets
Supports ogbn-arxiv and ogbn-mag datasets for Research Compass GNN

Usage:
    python train_ogb_datasets.py --dataset ogbn-arxiv --model GCN --epochs 100
    python train_ogb_datasets.py --dataset ogbn-mag --model GAT --epochs 50
"""

import sys
from pathlib import Path
# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import argparse
from pathlib import Path
import time
import json

# Import our modules
from models import GCNModel, GATModel
from training.trainer import GCNTrainer
from training.batch_training import create_trainer
from evaluation.metrics import NodeClassificationMetrics
from evaluation.report_generator import EvaluationReportGenerator
from data.dataset_utils import (
    print_dataset_info,
    create_link_prediction_split,
    move_to_device
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GNN on OGB Datasets')
    
    # Model selection
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT'],
                        help='Model architecture')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv', 'ogbn-mag'],
                        help='OGB dataset to use')
    
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

def load_ogb_dataset(args):
    """Load OGB dataset and convert to PyG format"""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        print("❌ Error: ogb package not installed.")
        print("Please install with: pip install ogb")
        return None, None
    
    print(f"🔬 Loading OGB dataset: {args.dataset}")
    
    # Download and load dataset
    dataset = PygNodePropPredDataset(name=args.dataset, root='data/ogb')
    
    # Get the graph data
    data = dataset[0]
    
    # Convert to proper format for node classification
    # OGB datasets are typically for node property prediction
    # We need to adapt them for our node classification task
    
    # Create train/val/test masks (OGB provides splits)
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']] = True
    data.test_mask[split_idx['test']] = True
    
    # Ensure labels are in correct format
    if hasattr(data, 'y'):
        data.y = data.y.view(-1)
    
    # Dataset info
    info = {
        'name': args.dataset,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes,
        'num_train': data.train_mask.sum().item(),
        'num_val': data.val_mask.sum().item(),
        'num_test': data.test_mask.sum().item(),
        'is_directed': True,
        'has_node_features': True,
        'has_edge_features': False,
        'dataset_type': 'OGB'
    }
    
    print(f"✅ Loaded {args.dataset}")
    print(f"   Nodes: {info['num_nodes']:,}, Edges: {info['num_edges']:,}")
    print(f"   Features: {info['num_features']:,}, Classes: {info['num_classes']}")
    print(f"   Train: {info['num_train']:,}, Val: {info['num_val']:,}, Test: {info['num_test']:,}")
    
    # Dataset-specific information
    if args.dataset == 'ogbn-arxiv':
        print(f"   📄 arXiv papers: Computer science papers from arXiv")
        print(f"   🏷️  Task: Paper subject classification ({dataset.num_classes} categories)")
        print(f"   ⚠️  Note: Large dataset - may require significant memory")
    elif args.dataset == 'ogbn-mag':
        print(f"   📄 Microsoft Academic Graph: Large-scale heterogeneous network")
        print(f"   🏷️  Task: Conference venue classification ({dataset.num_classes} categories)")
        print(f"   ⚠️  Note: Very large dataset - requires significant memory and time")
    
    return data, info

def train_gcn(data, args, dataset_info):
    """Train GCN on OGB dataset"""
    print("\n" + "=" * 70)
    print("🧪 Training GCN (Graph Convolutional Network)")
    print("=" * 70)
    
    # Create model
    model = GCNModel(
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset_info['num_classes'],
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
    if args.minibatch or data.num_nodes >= 10000:
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

def train_gat(data, args, dataset_info):
    """Train GAT on OGB dataset"""
    print("\n" + "=" * 70)
    print("🧪 Training GAT (Graph Attention Network)")
    print("=" * 70)
    
    # Create model
    model = GATModel(
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset_info['num_classes'],
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
        'mode': 'max',  # Maximize validation accuracy
        'factor': args.scheduler_factor,
        'patience': args.scheduler_patience,
        'min_lr': args.min_lr,
        'verbose': True
    }
    
    # Create trainer
    if args.minibatch or data.num_nodes >= 10000:
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

def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Fix torch loading issue for OGB datasets
    import torch.serialization
    import torch_geometric.data.data
    torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
    
    print("\n" + "=" * 70)
    print("🚀 Training GNN on OGB Datasets - Research Compass")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Load dataset
    data, dataset_info = load_ogb_dataset(args)
    if data is None:
        print("❌ Failed to load dataset")
        return
    
    # Print dataset info
    print_dataset_info(data, args.dataset)
    
    # Train model
    if args.model == 'GCN':
        results = train_gcn(data, args, dataset_info)
    elif args.model == 'GAT':
        results = train_gat(data, args, dataset_info)
    else:
        print(f"❌ Model {args.model} not yet implemented")
        return
    
    # Generate evaluation report
    print("\n📊 Generating comprehensive evaluation report...")
    
    report_gen = EvaluationReportGenerator(
        num_classes=dataset_info['num_classes'],
        class_names=[f"Class {i}" for i in range(dataset_info['num_classes'])]
    )
    
    # Get memory usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Fix PyTorch loading issue for OGB datasets
    import torch.serialization
    torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
    
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
            'dataset_type': 'OGB'
        }
    )
    
    # Generate report
    output_dir = Path(args.output_dir) / f"{args.model}_{args.dataset}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
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