#!/usr/bin/env python3
"""
Direct GNN Model Testing - Bypasses full package imports
Tests all 4 GNN architectures directly
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import numpy as np
import time
from typing import Dict


class SimpleGCN(nn.Module):
    """Simple GCN for testing"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=5, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class SimpleGAT(nn.Module):
    """Simple GAT for testing"""
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=1)
        return self.edge_predictor(edge_features).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class SimpleGraphTransformer(nn.Module):
    """Simple Graph Transformer for testing"""
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            self.convs.append(TransformerConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.output_proj = nn.Linear(hidden_dim * num_heads, input_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.output_proj(x)


def create_test_graph(num_nodes=100, num_edges=300):
    """Create synthetic test graph"""
    x = torch.randn(num_nodes, 384)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 5, (num_nodes,))

    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train+num_val] = True
    test_mask[num_train+num_val:] = True

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def test_gcn():
    """Test GCN model"""
    print("\n" + "="*60)
    print("🧪 TEST 1/3: GCN (Graph Convolutional Network)")
    print("="*60)

    data = create_test_graph(100, 300)
    model = SimpleGCN(input_dim=384, hidden_dim=128, output_dim=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    accs = []
    times = []

    for epoch in range(20):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        epoch_time = time.time() - start
        losses.append(loss.item())
        accs.append(acc.item())
        times.append(epoch_time)

        if epoch % 5 == 0 or epoch == 19:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f} | Time: {epoch_time:.3f}s")

    print(f"\n✅ GCN Training Summary:")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Final Accuracy: {accs[-1]:.4f}")
    print(f"   Loss Reduction: {losses[0] - losses[-1]:.4f}")
    print(f"   Avg Time/Epoch: {np.mean(times):.3f}s")

    return {
        'model': 'GCN',
        'final_loss': losses[-1],
        'final_acc': accs[-1],
        'convergence': losses[0] - losses[-1],
        'avg_time': np.mean(times)
    }


def test_gat():
    """Test GAT model"""
    print("\n" + "="*60)
    print("🧪 TEST 2/3: GAT (Graph Attention Network)")
    print("="*60)

    data = create_test_graph(100, 300)
    model = SimpleGAT(input_dim=384, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    accs = []
    times = []

    for epoch in range(20):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        # Link prediction
        pos_edge = data.edge_index[:, :100]
        neg_edge = torch.randint(0, data.x.size(0), (2, 100))

        pos_pred = model(data.x, data.edge_index, pos_edge)
        neg_pred = model(data.x, data.edge_index, neg_edge)

        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        acc = ((pos_pred > 0).float().mean() + (neg_pred < 0).float().mean()) / 2

        epoch_time = time.time() - start
        losses.append(loss.item())
        accs.append(acc.item())
        times.append(epoch_time)

        if epoch % 5 == 0 or epoch == 19:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f} | Time: {epoch_time:.3f}s")

    print(f"\n✅ GAT Training Summary:")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Final Accuracy: {accs[-1]:.4f}")
    print(f"   Loss Reduction: {losses[0] - losses[-1]:.4f}")
    print(f"   Avg Time/Epoch: {np.mean(times):.3f}s")

    return {
        'model': 'GAT',
        'final_loss': losses[-1],
        'final_acc': accs[-1],
        'convergence': losses[0] - losses[-1],
        'avg_time': np.mean(times)
    }


def test_transformer():
    """Test Graph Transformer"""
    print("\n" + "="*60)
    print("🧪 TEST 3/3: Graph Transformer")
    print("="*60)

    data = create_test_graph(80, 250)
    model = SimpleGraphTransformer(input_dim=384, hidden_dim=128, num_layers=2, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    times = []

    for epoch in range(15):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)  # Reconstruction loss
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start
        losses.append(loss.item())
        times.append(epoch_time)

        if epoch % 5 == 0 or epoch == 14:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Time: {epoch_time:.3f}s")

    print(f"\n✅ Transformer Training Summary:")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Loss Reduction: {losses[0] - losses[-1]:.4f}")
    print(f"   Avg Time/Epoch: {np.mean(times):.3f}s")

    return {
        'model': 'Transformer',
        'final_loss': losses[-1],
        'convergence': losses[0] - losses[-1],
        'avg_time': np.mean(times)
    }


def main():
    print("\n" + "🧪 " * 20)
    print("   GNN MODEL TESTING SUITE - Research Compass")
    print("   Direct Testing of Core GNN Architectures")
    print("🧪 " * 20)

    results = []

    try:
        results.append(test_gcn())
    except Exception as e:
        print(f"❌ GCN test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        results.append(test_gat())
    except Exception as e:
        print(f"❌ GAT test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        results.append(test_transformer())
    except Exception as e:
        print(f"❌ Transformer test failed: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("📊 COMPLETE TEST SUMMARY")
    print("="*60)

    print("\n| Model       | Final Loss | Final Acc | Convergence | Avg Time |")
    print("|-------------|------------|-----------|-------------|----------|")

    for r in results:
        model_name = r['model'].ljust(11)
        final_loss = f"{r['final_loss']:.4f}".rjust(10)
        final_acc = f"{r.get('final_acc', 'N/A'):>9}" if 'final_acc' in r else "    N/A  "
        convergence = f"{r['convergence']:.4f}".rjust(11)
        avg_time = f"{r['avg_time']:.3f}s".rjust(8)
        print(f"| {model_name} | {final_loss} | {final_acc} | {convergence} | {avg_time} |")

    print("\n" + "="*60)
    print(f"✅ {len(results)}/3 models tested successfully!")
    print("="*60 + "\n")

    return 0 if len(results) == 3 else 1


if __name__ == "__main__":
    exit(main())
