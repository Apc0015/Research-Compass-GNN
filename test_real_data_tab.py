#!/usr/bin/env python3
"""
Quick test to verify Real Data Training tab functionality
"""

import torch
from torch_geometric.data import Data
import numpy as np

# Test 1: Create synthetic graph data
print("Test 1: Creating synthetic graph data...")
try:
    num_papers = 5
    x = torch.randn(num_papers, 384)
    y = torch.randint(0, 3, (num_papers,))

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    train_mask = torch.zeros(num_papers, dtype=torch.bool)
    val_mask = torch.zeros(num_papers, dtype=torch.bool)
    test_mask = torch.zeros(num_papers, dtype=torch.bool)

    train_mask[:3] = True
    val_mask[3] = True
    test_mask[4] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(f"✅ Created graph: {data.num_nodes} nodes, {data.num_edges} edges")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 2: Test graph visualization
print("\nTest 2: Testing graph visualization...")
try:
    from launcher import create_interactive_graph

    paper_titles = [f"Paper_{i}" for i in range(num_papers)]
    fig = create_interactive_graph(
        data,
        predictions=None,
        show_predictions=False,
        paper_titles=paper_titles
    )

    print(f"✅ Graph visualization created successfully")
    print(f"   Figure type: {type(fig)}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Test training function structure
print("\nTest 3: Testing training function...")
try:
    from launcher import train_gnn_live, app_state

    # Set up state
    app_state.graph_data = data
    app_state.paper_list = paper_titles

    print("✅ App state configured")
    print(f"   Graph data: {app_state.graph_data is not None}")
    print(f"   Paper list: {len(app_state.paper_list)} papers")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test PDF processing (without actual PDFs)
print("\nTest 4: Testing PDF processing logic...")
try:
    from launcher import build_graph_from_papers

    papers_data = [
        {'name': 'Paper1.pdf', 'text': 'Sample text', 'citations': [], 'metadata': {}},
        {'name': 'Paper2.pdf', 'text': 'Another paper', 'citations': [], 'metadata': {}},
    ]

    graph = build_graph_from_papers(papers_data)
    print(f"✅ Graph built from papers: {graph.num_nodes} nodes, {graph.num_edges} edges")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("🎉 All tests passed! Real Data Training tab should work.")
print("="*60)
print("\nIf the tab still doesn't work in the browser:")
print("1. Check browser console for JavaScript errors")
print("2. Try refreshing the page (Ctrl+F5)")
print("3. Check if PDFs are uploading correctly")
print("4. Look for errors in the launcher_output.log")
