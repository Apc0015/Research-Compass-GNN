#!/usr/bin/env python3
"""
GNN Model Export - Export trained models for deployment

Supports:
- ONNX format (cross-platform)
- TorchScript format (PyTorch deployment)
- Model metadata and version tracking
"""

import logging
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def export_to_torchscript(
    model: torch.nn.Module,
    example_inputs: Tuple,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export model to TorchScript format for PyTorch deployment

    Args:
        model: Trained PyTorch model
        example_inputs: Example inputs for tracing (tuple of tensors)
        save_path: Path to save TorchScript model (.pt)
        metadata: Optional metadata to save alongside model

    Returns:
        Export result dict with status and paths
    """
    try:
        logger.info("Exporting model to TorchScript...")

        # Set model to eval mode
        model.eval()

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)

        # Save traced model
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.jit.save(traced_model, str(save_path))

        # Save metadata
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            metadata['export_format'] = 'torchscript'
            metadata['export_date'] = datetime.now().isoformat()

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")

        logger.info(f"✓ Model exported to TorchScript: {save_path}")

        return {
            "status": "success",
            "format": "torchscript",
            "model_path": str(save_path),
            "metadata_path": str(metadata_path) if metadata else None,
            "file_size_mb": save_path.stat().st_size / (1024 * 1024)
        }

    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        return {
            "status": "error",
            "format": "torchscript",
            "error": str(e)
        }


def export_to_onnx(
    model: torch.nn.Module,
    example_inputs: Tuple,
    save_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 14,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export model to ONNX format for cross-platform deployment

    Args:
        model: Trained PyTorch model
        example_inputs: Example inputs for tracing
        save_path: Path to save ONNX model (.onnx)
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes specification for variable-size inputs
        opset_version: ONNX opset version
        metadata: Optional metadata

    Returns:
        Export result dict
    """
    try:
        logger.info("Exporting model to ONNX...")

        # Set model to eval mode
        model.eval()

        # Default names if not provided
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # Save path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs,
                str(save_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True
            )

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")
        except ImportError:
            logger.warning("onnx package not available for verification. Install with: pip install onnx")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")

        # Save metadata
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            metadata['export_format'] = 'onnx'
            metadata['export_date'] = datetime.now().isoformat()
            metadata['opset_version'] = opset_version
            metadata['input_names'] = input_names
            metadata['output_names'] = output_names

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")

        logger.info(f"✓ Model exported to ONNX: {save_path}")

        return {
            "status": "success",
            "format": "onnx",
            "model_path": str(save_path),
            "metadata_path": str(metadata_path) if metadata else None,
            "file_size_mb": save_path.stat().st_size / (1024 * 1024),
            "opset_version": opset_version
        }

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return {
            "status": "error",
            "format": "onnx",
            "error": str(e)
        }


def export_gnn_model(
    model: torch.nn.Module,
    example_x: torch.Tensor,
    example_edge_index: torch.Tensor,
    output_dir: str,
    model_name: str,
    formats: list = ['torchscript', 'onnx'],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export GNN model to multiple formats

    Args:
        model: Trained GNN model
        example_x: Example node features tensor
        example_edge_index: Example edge index tensor
        output_dir: Output directory
        model_name: Name for the model files
        formats: List of formats to export to
        metadata: Model metadata (performance, config, etc.)

    Returns:
        Dict with export results for each format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Prepare example inputs
    example_inputs = (example_x, example_edge_index)

    # Add model architecture info to metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        'model_name': model_name,
        'model_type': model.__class__.__name__,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'input_shape_x': list(example_x.shape),
        'input_shape_edge_index': list(example_edge_index.shape)
    })

    # Export to each requested format
    if 'torchscript' in formats:
        torchscript_path = output_path / f"{model_name}.pt"
        results['torchscript'] = export_to_torchscript(
            model,
            example_inputs,
            str(torchscript_path),
            metadata=metadata.copy()
        )

    if 'onnx' in formats:
        onnx_path = output_path / f"{model_name}.onnx"

        # Dynamic axes for GNN (variable graph sizes)
        dynamic_axes = {
            'node_features': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'output': {0: 'num_nodes'}
        }

        results['onnx'] = export_to_onnx(
            model,
            example_inputs,
            str(onnx_path),
            input_names=['node_features', 'edge_index'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            metadata=metadata.copy()
        )

    return results


def load_torchscript_model(model_path: str) -> torch.nn.Module:
    """
    Load a TorchScript model for inference

    Args:
        model_path: Path to .pt file

    Returns:
        Loaded model
    """
    try:
        model = torch.jit.load(model_path)
        model.eval()
        logger.info(f"✓ TorchScript model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load TorchScript model: {e}")
        raise


def load_onnx_model(model_path: str):
    """
    Load an ONNX model for inference

    Args:
        model_path: Path to .onnx file

    Returns:
        ONNX runtime session
    """
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        logger.info(f"✓ ONNX model loaded from {model_path}")
        return session
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        raise
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise


def create_model_package(
    model_path: str,
    output_dir: str,
    include_files: Optional[list] = None
) -> str:
    """
    Create a deployment package with model and dependencies

    Args:
        model_path: Path to model file
        output_dir: Output directory for package
        include_files: Additional files to include

    Returns:
        Path to created package (.zip)
    """
    import zipfile
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    package_name = f"{model_path.stem}_package.zip"
    package_path = output_path / package_name

    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model file
        zipf.write(model_path, arcname=model_path.name)

        # Add metadata if exists
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            zipf.write(metadata_path, arcname=metadata_path.name)

        # Add additional files
        if include_files:
            for file_path in include_files:
                file_path = Path(file_path)
                if file_path.exists():
                    zipf.write(file_path, arcname=file_path.name)

        # Create README
        readme_content = f"""# Model Deployment Package

**Model:** {model_path.stem}
**Created:** {datetime.now().isoformat()}

## Contents

- `{model_path.name}`: Trained model
- `{metadata_path.name}`: Model metadata (if available)

## Usage

### TorchScript Model (.pt)

```python
import torch

# Load model
model = torch.jit.load('{model_path.name}')
model.eval()

# Inference
with torch.no_grad():
    output = model(node_features, edge_index)
```

### ONNX Model (.onnx)

```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession('{model_path.name}')

# Inference
outputs = session.run(
    None,
    {{'node_features': x_numpy, 'edge_index': edge_index_numpy}}
)
```

## Requirements

- PyTorch (for .pt models)
- onnxruntime (for .onnx models)

Install with:
```bash
pip install torch onnxruntime
```
"""

        zipf.writestr('README.md', readme_content)

    logger.info(f"✓ Model package created: {package_path}")
    return str(package_path)


# Example usage
if __name__ == "__main__":
    print("GNN Model Export Utilities")
    print("=" * 50)
    print("\nFeatures:")
    print("- Export to TorchScript (.pt)")
    print("- Export to ONNX (.onnx)")
    print("- Model packaging with metadata")
    print("- Cross-platform deployment support")
    print("\nUsage example in code - see function docstrings")
