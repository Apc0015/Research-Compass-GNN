"""
Logging utilities for Research Compass GNN

Provides centralized logging with file and console handlers
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file. If None, logs to console only
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_format: Custom log format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('my_module', 'logs/training.log')
        >>> logger.info('Training started')
        >>> logger.warning('Validation accuracy dropped')
        >>> logger.error('Training failed')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Using existing logger')
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training progress

    Tracks and logs training metrics, model performance, and events.

    Example:
        >>> logger = TrainingLogger('training.log')
        >>> logger.log_epoch(epoch=1, train_loss=0.5, train_acc=0.85, val_acc=0.82)
        >>> logger.log_best_model(epoch=10, val_acc=0.90)
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        name: str = 'training',
        level: int = logging.INFO
    ):
        """
        Initialize training logger

        Args:
            log_file: Path to log file
            name: Logger name
            level: Logging level
        """
        self.logger = setup_logger(name, log_file, level)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_acc: float,
        **kwargs
    ):
        """
        Log epoch metrics

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
            **kwargs: Additional metrics to log
        """
        msg = (
            f"Epoch {epoch:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Add additional metrics
        if kwargs:
            extra = " | ".join(f"{k}: {v:.4f}" for k, v in kwargs.items())
            msg += f" | {extra}"

        self.logger.info(msg)

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log when new best model is found"""
        self.logger.info(
            f"🏆 New best model at epoch {epoch}: "
            f"{metric_name}={metric_value:.4f}"
        )

    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping"""
        self.logger.info(
            f"⚠️  Early stopping triggered at epoch {epoch} "
            f"(patience: {patience})"
        )

    def log_training_start(self, model_name: str, num_params: int, device: str):
        """Log training start"""
        self.logger.info("=" * 70)
        self.logger.info(f"🚀 Training {model_name}")
        self.logger.info(f"   Parameters: {num_params:,}")
        self.logger.info(f"   Device: {device}")
        self.logger.info("=" * 70)

    def log_training_complete(self, total_time: float, best_val_acc: float):
        """Log training completion"""
        self.logger.info("=" * 70)
        self.logger.info("✅ Training Complete!")
        self.logger.info(f"   Total Time: {total_time:.2f}s")
        self.logger.info(f"   Best Val Acc: {best_val_acc:.4f}")
        self.logger.info("=" * 70)

    def log_error(self, error: Exception):
        """Log error with traceback"""
        self.logger.error(f"❌ Error: {str(error)}", exc_info=True)

    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
