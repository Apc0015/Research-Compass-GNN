# Archive

This directory contains advanced features and models that have been archived to simplify the main interface.

## Archived Models

The following GNN models have been moved here to focus on core functionality:

### Models (`models/`)
- **han.py** - Heterogeneous Attention Network (HAN)
  - Multi-relational graphs with hierarchical attention
  - 4 node types, 7 edge types
  
- **rgcn.py** - Relational Graph Convolutional Network (R-GCN)
  - Citation type-aware convolutions
  - 4 citation types: EXTENDS, METHODOLOGY, BACKGROUND, COMPARISON
  
- **graphsage.py** - GraphSAGE
  - Inductive learning with neighbor sampling
  
- **graph_transformer.py** - Graph Transformer
  - Full attention mechanism over graph structure

## Archived Data Processing

### Data Features
- **citation_type_classifier.py** - Classifies citation relationships into 4 types
- **heterogeneous_graph_builder.py** - Converts homogeneous graphs to heterogeneous representations

### Test Files
- **verify_han.py** - HAN model verification tests
- **verify_rgcn.py** - R-GCN model verification tests

## Why Archived?

These advanced features were archived to:
1. **Simplify the UI** - Focus on core GCN and GAT models
2. **Reduce complexity** - Remove dependencies on heterogeneous and relational graph features
3. **Improve maintainability** - Easier to understand and extend core functionality
4. **Better user experience** - Clearer interface for most use cases

## Using Archived Models

If you need these advanced features, they can still be used by:

1. **Importing from archive:**
   ```python
   import sys
   sys.path.insert(0, 'archive/models')
   from han import HANModel, create_han_model
   from rgcn import RGCNModel, create_rgcn_model
   ```

2. **Moving back to main directory:**
   ```bash
   cp archive/models/han.py models/
   # Update models/__init__.py to include imports
   ```

3. **Updating configuration:**
   - Add model choices back to `scripts/launcher.py`
   - Update model selection logic in training functions

## Archived Date

November 12, 2025

## Original Features

All archived features were fully functional and tested. They represent advanced GNN capabilities for:
- Multi-relational citation networks
- Citation type analysis
- Inductive learning scenarios
- Full graph attention mechanisms
