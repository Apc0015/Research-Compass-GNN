"""
Baseline Models for Comparison

Implements simple baselines to demonstrate the value of GNNs:
- Random Baseline
- MLP (features only, no graph)
- Logistic Regression
- Random Forest
- SVM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Dict, Optional
import time


class MLPBaseline(nn.Module):
    """
    Multi-Layer Perceptron baseline (no graph structure)

    Uses only node features, ignoring graph structure.
    This proves the value of incorporating graph information.

    Args:
        input_dim: Feature dimension
        hidden_dim: Hidden layer size
        output_dim: Number of classes
        num_layers: Number of hidden layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 5,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (no edge_index needed)

        Args:
            x: Node features [num_nodes, input_dim]

        Returns:
            Predictions [num_nodes, output_dim]
        """
        return self.mlp(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomBaseline:
    """
    Random prediction baseline

    Predicts random class labels. Represents worst-case performance.
    Expected accuracy: 1/num_classes (e.g., 20% for 5 classes)
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def predict(self, num_samples: int) -> np.ndarray:
        """Generate random predictions"""
        return np.random.randint(0, self.num_classes, num_samples)

    def evaluate(self, y_true: np.ndarray) -> float:
        """Evaluate random baseline"""
        y_pred = self.predict(len(y_true))
        return accuracy_score(y_true, y_pred)


class TraditionalMLBaseline:
    """
    Traditional ML baselines using scikit-learn

    Supports:
    - Logistic Regression
    - Random Forest
    - SVM

    These use only node features (no graph structure).
    """

    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
        Args:
            model_type: 'logistic', 'random_forest', or 'svm'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type

        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                multi_class='multinomial'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TraditionalMLBaseline':
        """Train the model"""
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy"""
        return self.model.score(X, y)


def train_mlp_baseline(
    data,
    epochs: int = 100,
    lr: float = 0.01,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Train MLP baseline and return results

    Args:
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Dictionary with results
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Create model
    model = MLPBaseline(
        input_dim=data.x.shape[1],
        hidden_dim=128,
        output_dim=data.y.max().item() + 1,
        num_layers=2,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        pred = out.argmax(dim=1)

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'num_parameters': model.count_parameters()
    }


def evaluate_all_baselines(data, device: Optional[torch.device] = None) -> Dict:
    """
    Evaluate all baseline models

    Args:
        data: PyTorch Geometric Data object
        device: Device for PyTorch models

    Returns:
        Dictionary with results for all baselines
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # Convert to numpy for sklearn
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train = X[data.train_mask.cpu().numpy()]
    y_train = y[data.train_mask.cpu().numpy()]
    X_test = X[data.test_mask.cpu().numpy()]
    y_test = y[data.test_mask.cpu().numpy()]

    num_classes = data.y.max().item() + 1

    print("\n" + "=" * 70)
    print("📊 Evaluating Baseline Models")
    print("=" * 70)

    # 1. Random Baseline
    print("\n1️⃣  Random Baseline")
    random_baseline = RandomBaseline(num_classes)
    random_acc = random_baseline.evaluate(y_test)
    results['random'] = {
        'test_acc': random_acc,
        'training_time': 0,
        'num_parameters': 0
    }
    print(f"   Accuracy: {random_acc:.4f}")

    # 2. Logistic Regression
    print("\n2️⃣  Logistic Regression")
    start = time.time()
    lr_model = TraditionalMLBaseline('logistic')
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - start
    lr_acc = lr_model.score(X_test, y_test)
    results['logistic'] = {
        'test_acc': lr_acc,
        'training_time': lr_time,
        'num_parameters': 'N/A'
    }
    print(f"   Accuracy: {lr_acc:.4f}")
    print(f"   Training time: {lr_time:.2f}s")

    # 3. Random Forest
    print("\n3️⃣  Random Forest")
    start = time.time()
    rf_model = TraditionalMLBaseline('random_forest', n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_acc = rf_model.score(X_test, y_test)
    results['random_forest'] = {
        'test_acc': rf_acc,
        'training_time': rf_time,
        'num_parameters': 'N/A'
    }
    print(f"   Accuracy: {rf_acc:.4f}")
    print(f"   Training time: {rf_time:.2f}s")

    # 4. MLP
    print("\n4️⃣  MLP (Neural Network, no graph)")
    mlp_results = train_mlp_baseline(data, epochs=100, device=device)
    results['mlp'] = mlp_results
    print(f"   Accuracy: {mlp_results['test_acc']:.4f}")
    print(f"   Training time: {mlp_results['training_time']:.2f}s")
    print(f"   Parameters: {mlp_results['num_parameters']:,}")

    print("\n" + "=" * 70)
    print("✅ Baseline Evaluation Complete")
    print("=" * 70)

    return results
