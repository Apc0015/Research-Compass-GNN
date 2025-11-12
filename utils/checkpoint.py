"""
Model checkpointing utilities for Research Compass GNN

Provides automatic model checkpoint saving based on performance metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


class ModelCheckpoint:
    """
    Model checkpointing with automatic best model tracking

    Saves model checkpoints based on monitored metric (e.g., validation accuracy).
    Keeps only top-k best checkpoints to save disk space.

    Features:
    - Automatic best model detection
    - Top-k checkpoint retention
    - Optimizer state saving
    - Scheduler state saving
    - Training metadata saving

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     checkpoint_dir='./checkpoints',
        ...     monitor='val_acc',
        ...     mode='max',
        ...     keep_top_k=3
        ... )
        >>>
        >>> for epoch in range(100):
        ...     # Train model
        ...     metrics = {'val_acc': 0.85, 'val_loss': 0.3}
        ...     checkpoint.save(model, epoch, metrics, optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        monitor: str = 'val_acc',
        mode: str = 'max',
        keep_top_k: int = 3,
        save_optimizer: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_acc', 'val_loss', etc.)
            mode: 'max' to maximize metric, 'min' to minimize
            keep_top_k: Keep only top-k best checkpoints (-1 for all)
            save_optimizer: Whether to save optimizer state
            verbose: Print checkpoint save messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.keep_top_k = keep_top_k
        self.save_optimizer = save_optimizer
        self.verbose = verbose

        # Initialize best score
        self.best_score = float('-inf') if mode == 'max' else float('inf')

        # Track all saved checkpoints
        self.checkpoints = []  # List of (score, epoch, path) tuples

    def is_better(self, score: float) -> bool:
        """Check if current score is better than best score"""
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        **extra_state
    ) -> Optional[Path]:
        """
        Save model checkpoint if metric improved

        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            metrics: Dictionary of metrics (must contain monitored metric)
            optimizer: Optimizer to save (optional)
            scheduler: LR scheduler to save (optional)
            **extra_state: Additional state to save

        Returns:
            Path to saved checkpoint if saved, None otherwise

        Raises:
            KeyError: If monitored metric not in metrics dictionary
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"Monitored metric '{self.monitor}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )

        score = metrics[self.monitor]
        is_best = self.is_better(score)

        # Only save if this is a new best
        if not is_best:
            return None

        # Update best score
        self.best_score = score

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
            'monitor': self.monitor,
        }

        # Add optimizer state
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Add extra state
        checkpoint.update(extra_state)

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{epoch:03d}_{self.monitor}{score:.4f}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self.checkpoints.append((score, epoch, checkpoint_path))

        if self.verbose:
            print(f"✅ Saved checkpoint: {checkpoint_path.name}")
            print(f"   {self.monitor}: {score:.4f} (best: {self.best_score:.4f})")

        # Also save as 'best_model.pt' for convenience
        best_path = self.checkpoint_dir / 'best_model.pt'
        shutil.copy(checkpoint_path, best_path)

        # Remove old checkpoints if needed
        self._cleanup_checkpoints()

        return checkpoint_path

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only top-k"""
        if self.keep_top_k <= 0:
            return  # Keep all checkpoints

        # Sort checkpoints by score
        if self.mode == 'max':
            self.checkpoints.sort(reverse=True, key=lambda x: x[0])
        else:
            self.checkpoints.sort(key=lambda x: x[0])

        # Remove checkpoints beyond top-k
        if len(self.checkpoints) > self.keep_top_k:
            for score, epoch, path in self.checkpoints[self.keep_top_k:]:
                if path.exists() and path.name != 'best_model.pt':
                    path.unlink()
                    if self.verbose:
                        print(f"🗑️  Removed old checkpoint: {path.name}")

            # Keep only top-k in list
            self.checkpoints = self.checkpoints[:self.keep_top_k]

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load best model checkpoint

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load model to (optional)

        Returns:
            Dictionary with checkpoint metadata

        Raises:
            FileNotFoundError: If no checkpoint found
        """
        best_path = self.checkpoint_dir / 'best_model.pt'

        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_path}")

        # Load checkpoint
        checkpoint = torch.load(best_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            score = checkpoint.get('best_score', 'unknown')
            print(f"✅ Loaded best model from epoch {epoch}")
            print(f"   {self.monitor}: {score}")

        return checkpoint

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load specific checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load model to (optional)

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Loaded checkpoint from epoch {epoch}")

        return checkpoint

    def get_best_score(self) -> float:
        """Get current best score"""
        return self.best_score

    def get_num_checkpoints(self) -> int:
        """Get number of saved checkpoints"""
        return len(self.checkpoints)

    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint(dir='{self.checkpoint_dir}', "
            f"monitor='{self.monitor}', mode='{self.mode}', "
            f"best_score={self.best_score:.4f})"
        )
