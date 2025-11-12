"""
Settings and configuration management for Research Compass GNN

Provides centralized configuration loading from config.yaml
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Settings:
    """
    Configuration manager for Research Compass GNN

    Loads and manages all configuration from config.yaml.
    Provides convenient access to model, training, and path configurations.

    Example:
        >>> settings = Settings()
        >>> settings.load()
        >>> model_config = settings.get_model_config('han')
        >>> training_config = settings.get_training_config()
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings manager

        Args:
            config_path: Path to config.yaml. If None, searches in project root.
        """
        if config_path is None:
            # Search for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self.config = {}
        self._loaded = False

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            Dictionary containing all configuration

        Raises:
            FileNotFoundError: If config.yaml not found
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self._loaded = True
            print(f"✅ Configuration loaded from {self.config_path}")
            return self.config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def _ensure_loaded(self):
        """Ensure configuration is loaded"""
        if not self._loaded:
            self.load()

    def get_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        self._ensure_loaded()
        return self.config

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model

        Args:
            model_name: Model name ('gcn', 'gat', 'han', 'rgcn', etc.)

        Returns:
            Dictionary with model-specific configuration

        Example:
            >>> settings = Settings()
            >>> han_config = settings.get_model_config('han')
            >>> print(han_config['hidden_dim'])  # 128
        """
        self._ensure_loaded()
        model_name = model_name.lower()

        if 'models' not in self.config:
            raise KeyError("No 'models' section in configuration")

        if model_name not in self.config['models']:
            raise KeyError(f"Model '{model_name}' not found in configuration")

        return self.config['models'][model_name]

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration

        Returns:
            Dictionary with training parameters

        Example:
            >>> settings = Settings()
            >>> training = settings.get_training_config()
            >>> print(training['epochs'])  # 100
        """
        self._ensure_loaded()

        if 'training' not in self.config:
            raise KeyError("No 'training' section in configuration")

        return self.config['training']

    def get_paths_config(self) -> Dict[str, Path]:
        """
        Get paths configuration

        Returns:
            Dictionary with Path objects for all configured paths

        Example:
            >>> settings = Settings()
            >>> paths = settings.get_paths_config()
            >>> print(paths['models'])  # PosixPath('./checkpoints')
        """
        self._ensure_loaded()

        if 'paths' not in self.config:
            raise KeyError("No 'paths' section in configuration")

        # Convert string paths to Path objects
        paths = {}
        for key, value in self.config['paths'].items():
            paths[key] = Path(value)

        return paths

    def get_device(self) -> torch.device:
        """
        Get device based on configuration

        Returns:
            torch.device object

        Example:
            >>> settings = Settings()
            >>> device = settings.get_device()
            >>> print(device)  # cuda:0 or cpu
        """
        self._ensure_loaded()

        device_config = self.config.get('device', {})

        # Check if specific device is forced
        force_device = device_config.get('force_device')
        if force_device:
            return torch.device(force_device)

        # Auto-detect CUDA availability
        if device_config.get('auto_detect', True):
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default to CPU
        return torch.device('cpu')

    def create_directories(self):
        """
        Create all directories specified in paths configuration

        Example:
            >>> settings = Settings()
            >>> settings.create_directories()
            ✅ Created directory: ./data
            ✅ Created directory: ./checkpoints
            ...
        """
        paths = self.get_paths_config()

        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                print(f"✅ Created directory: {path}")

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        self._ensure_loaded()
        return self.config.get('logging', {})

    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpointing configuration"""
        self._ensure_loaded()
        return self.config.get('checkpointing', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        self._ensure_loaded()
        return self.config.get('evaluation', {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        self._ensure_loaded()
        return self.config.get('visualization', {})

    def get_reproducibility_config(self) -> Dict[str, Any]:
        """Get reproducibility configuration"""
        self._ensure_loaded()
        return self.config.get('reproducibility', {})

    def set_reproducibility(self):
        """
        Set random seeds for reproducibility

        Example:
            >>> settings = Settings()
            >>> settings.set_reproducibility()
            ✅ Reproducibility set with seed: 42
        """
        repro_config = self.get_reproducibility_config()

        seed = repro_config.get('seed', 42)
        deterministic = repro_config.get('deterministic', True)
        benchmark = repro_config.get('benchmark', False)

        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif benchmark:
            torch.backends.cudnn.benchmark = True

        print(f"✅ Reproducibility set with seed: {seed}, "
              f"deterministic: {deterministic}, benchmark: {benchmark}")

    def __repr__(self) -> str:
        return f"Settings(config_path='{self.config_path}', loaded={self._loaded})"


# Global settings instance
_global_settings = None


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load global configuration

    Args:
        config_path: Optional path to config.yaml

    Returns:
        Settings instance

    Example:
        >>> load_config()
        >>> config = get_config()
    """
    global _global_settings
    _global_settings = Settings(config_path)
    _global_settings.load()
    return _global_settings


def get_config() -> Settings:
    """
    Get global settings instance

    Returns:
        Settings instance

    Raises:
        RuntimeError: If load_config() hasn't been called yet
    """
    global _global_settings
    if _global_settings is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _global_settings


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Convenience function to get model config"""
    return get_config().get_model_config(model_name)


def get_training_config() -> Dict[str, Any]:
    """Convenience function to get training config"""
    return get_config().get_training_config()


def get_paths_config() -> Dict[str, Path]:
    """Convenience function to get paths config"""
    return get_config().get_paths_config()
