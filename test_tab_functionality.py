#!/usr/bin/env python3
"""
Test Real Data Training tab functionality end-to-end
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from launcher import process_pdfs, train_gnn_live, app_state
import torch

def test_real_data_training():
    """Test the complete workflow of Real Data Training tab"""

    print("=" * 60)
    print("Testing Real Data Training Tab Functionality")
    print("=" * 60)

    # Test 1: Check if sample PDFs exist
    print("\n1. Checking for sample PDFs...")
    pdf_dir = "test_pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.txt')]  # Using .txt as mock PDFs

    if not pdf_files:
        print("❌ No sample files found in test_pdfs/")
        return False

    print(f"✅ Found {len(pdf_files)} sample files: {pdf_files}")

    # Test 2: Create mock file objects
    print("\n2. Creating mock file objects...")
    class MockFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path

        def read(self):
            with open(self.path, 'rb') as f:
                return f.read()

    mock_files = []
    for pdf_file in pdf_files[:3]:  # Test with first 3 files
        file_path = os.path.join(pdf_dir, pdf_file)
        mock_files.append(MockFile(file_path))

    print(f"✅ Created {len(mock_files)} mock file objects")

    # Test 3: Test process_pdfs function
    print("\n3. Testing process_pdfs()...")
    try:
        # Clear previous state
        app_state.graph_data = None
        app_state.paper_list = []

        status, graph_stats, paper_choices, initial_graph = process_pdfs(
            files=mock_files,
            extract_citations=True,
            build_graph=True,
            extract_metadata=True
        )

        print("Process Status:")
        print(status)
        print("\nGraph Stats:")
        print(graph_stats if graph_stats else "No stats")

        if app_state.graph_data is None:
            print("❌ Graph data not created")
            return False

        print(f"\n✅ Graph created: {app_state.graph_data.num_nodes} nodes, {app_state.graph_data.num_edges} edges")
        print(f"✅ Initial visualization created: {initial_graph is not None}")

    except Exception as e:
        print(f"❌ Error in process_pdfs: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Test train_gnn_live function
    print("\n4. Testing train_gnn_live()...")
    try:
        # Run training for just 10 epochs as a test
        results = list(train_gnn_live(
            model_type="GCN",
            epochs=10,
            learning_rate=0.01,
            task_type="Node Classification"
        ))

        if not results:
            print("❌ No results from training")
            return False

        # Get final result
        final_status, final_graph, results_summary, training_plot = results[-1]

        print("Training Status (last update):")
        print(final_status[-500:] if len(final_status) > 500 else final_status)

        print(f"\n✅ Training completed with {len(results)} updates")
        print(f"✅ Final visualization created: {final_graph is not None}")
        print(f"✅ Results summary created: {results_summary is not None}")
        print(f"✅ Training plot created: {training_plot is not None}")

        if app_state.trained_model is None:
            print("❌ Model not saved to app_state")
            return False

        print(f"✅ Model saved: {app_state.model_type}")

    except Exception as e:
        print(f"❌ Error in train_gnn_live: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Verify model can make predictions
    print("\n5. Testing model predictions...")
    try:
        model = app_state.trained_model
        data = app_state.graph_data.to(app_state.device)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        print(f"✅ Predictions working: Test accuracy = {test_acc.item():.4f}")

    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Real Data Training tab is functional!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_real_data_training()
    sys.exit(0 if success else 1)
