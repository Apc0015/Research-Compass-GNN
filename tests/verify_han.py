#!/usr/bin/env python3
"""
Verification script for HAN (Heterogeneous Attention Network) implementation
Tests all components: graph builder, model, trainer
"""

import sys
from pathlib import Path
# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid

from data import convert_to_heterogeneous
from models import create_han_model
from training.trainer import HANTrainer


def verify_han():
    """Comprehensive verification of HAN implementation"""
    print("=" * 70)
    print("VERIFYING HAN IMPLEMENTATION")
    print("=" * 70)

    try:
        # 1. Load dataset
        print("\n1️⃣  Loading Cora dataset...")
        data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        print(f"✅ Loaded: {data.num_nodes} nodes, {data.num_edges} edges")

        # 2. Convert to heterogeneous
        print("\n2️⃣  Converting to heterogeneous graph...")
        hetero_data = convert_to_heterogeneous(
            data,
            num_venues=10,
            num_authors_per_paper=(2, 3),
            author_collaboration_prob=0.3
        )
        print(f"✅ Created heterogeneous graph")
        print(f"   Node types: {hetero_data.node_types}")
        print(f"   Edge types: {len(hetero_data.edge_types)}")

        # 3. Create model
        print("\n3️⃣  Creating HAN model...")
        model = create_han_model(
            hetero_data,
            hidden_dim=64,
            num_heads=4,
            task='classification',
            num_classes=7
        )
        print(f"✅ Model created")
        print(f"   Parameters: {model.count_parameters():,}")

        # 4. Test forward pass
        print("\n4️⃣  Testing forward pass...")
        out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        print(f"✅ Forward pass successful")
        print(f"   Output shape (paper): {out['paper'].shape}")
        expected_shape = (hetero_data['paper'].x.shape[0], 7)
        assert out['paper'].shape == expected_shape, f"Expected {expected_shape}, got {out['paper'].shape}"

        # 5. Test training step
        print("\n5️⃣  Testing training step...")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        trainer = HANTrainer(model, optimizer, target_node_type='paper')

        train_metrics = trainer.train_epoch(hetero_data)
        print(f"✅ Training step successful")
        print(f"   Loss: {train_metrics['loss']:.4f}")
        print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"   Time: {train_metrics['time']:.4f}s")

        # 6. Test validation
        print("\n6️⃣  Testing validation...")
        val_metrics = trainer.validate(hetero_data, return_attention=True)
        print(f"✅ Validation successful")
        print(f"   Loss: {val_metrics['loss']:.4f}")
        print(f"   Accuracy: {val_metrics['accuracy']:.4f}")

        # 7. Check attention weights
        print("\n7️⃣  Checking attention weights...")
        attention = trainer.get_attention_weights()
        if attention is not None:
            print(f"✅ Attention weights available")
            for node_type, attn in attention.items():
                if attn is not None:
                    print(f"   {node_type}: shape {attn.shape}")
        else:
            print("⚠️  No attention weights captured")

        # 8. Test multiple epochs
        print("\n8️⃣  Testing multiple training epochs...")
        for epoch in range(5):
            train_metrics = trainer.train_epoch(hetero_data)
            val_metrics = trainer.validate(hetero_data)

            if epoch == 0 or epoch == 4:
                print(f"   Epoch {epoch}: Loss={train_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        print("✅ Multiple epoch training successful")

        print("\n" + "=" * 70)
        print("✅ ALL VERIFICATIONS PASSED - HAN IMPLEMENTATION COMPLETE")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = verify_han()
    sys.exit(0 if success else 1)
