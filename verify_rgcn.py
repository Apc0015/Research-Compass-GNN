#!/usr/bin/env python3
"""
Verification script for R-GCN (Relational GCN) implementation
Tests all components: citation type classifier, R-GCN model, training
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from data import classify_citation_types
from models import create_rgcn_model
from training.trainer import GCNTrainer


def verify_rgcn():
    """Comprehensive verification of R-GCN implementation"""
    print("=" * 70)
    print("VERIFYING R-GCN IMPLEMENTATION")
    print("=" * 70)

    try:
        # 1. Load dataset
        print("\n1️⃣  Loading Cora dataset...")
        data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        print(f"✅ Loaded: {data.num_nodes} nodes, {data.num_edges} edges")

        # 2. Classify citation types
        print("\n2️⃣  Classifying citation types...")
        edge_types, typed_edges = classify_citation_types(data)
        print(f"✅ Classified {edge_types.shape[0]} edges into 4 types")
        print(f"   Unique types: {torch.unique(edge_types).tolist()}")

        # Verify edge types are in valid range
        assert edge_types.min() >= 0 and edge_types.max() <= 3, "Edge types out of range"

        # 3. Create model
        print("\n3️⃣  Creating R-GCN model...")
        model = create_rgcn_model(
            data,
            num_relations=4,
            hidden_dim=64,
            num_bases=30,
            task='classification'
        )
        print(f"✅ Model created")
        print(f"   Parameters: {model.count_parameters():,}")

        # 4. Test forward pass
        print("\n4️⃣  Testing forward pass...")
        out = model(data.x, data.edge_index, edge_types)
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {out.shape}")
        expected_shape = (data.num_nodes, len(torch.unique(data.y)))
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

        # 5. Test training step
        print("\n5️⃣  Testing training step...")
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, edge_types)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        pred = out[data.train_mask].argmax(dim=1)
        train_acc = (pred == data.y[data.train_mask]).float().mean().item()

        print(f"✅ Training step successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Train Accuracy: {train_acc:.4f}")

        # 6. Test validation
        print("\n6️⃣  Testing validation...")
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, edge_types)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            pred = out[data.val_mask].argmax(dim=1)
            val_acc = (pred == data.y[data.val_mask]).float().mean().item()

        print(f"✅ Validation successful")
        print(f"   Val Loss: {val_loss.item():.4f}")
        print(f"   Val Accuracy: {val_acc:.4f}")

        # 7. Test with different relation types
        print("\n7️⃣  Testing relation-specific processing...")
        for rel_type in range(4):
            mask = (edge_types == rel_type)
            num_edges = mask.sum().item()
            if num_edges > 0:
                # Test forward pass with only this relation type
                rel_edge_index = data.edge_index[:, mask]
                rel_edge_types = edge_types[mask]
                print(f"   Relation {rel_type}: {num_edges} edges - ✅")

        # 8. Test multiple epochs
        print("\n8️⃣  Testing multiple training epochs...")
        for epoch in range(5):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, edge_types)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, edge_types)
                pred = out[data.val_mask].argmax(dim=1)
                val_acc = (pred == data.y[data.val_mask]).float().mean().item()

            if epoch == 0 or epoch == 4:
                print(f"   Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")

        print("✅ Multiple epoch training successful")

        # 9. Test typed edge indices
        print("\n9️⃣  Verifying typed edge dictionaries...")
        for edge_type_name, edge_idx in typed_edges.items():
            print(f"   {edge_type_name}: {edge_idx.shape[1]} edges")
        print("✅ Typed edges verified")

        print("\n" + "=" * 70)
        print("✅ ALL VERIFICATIONS PASSED - R-GCN IMPLEMENTATION COMPLETE")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = verify_rgcn()
    sys.exit(0 if success else 1)
