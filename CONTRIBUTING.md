# Contributing to Research Compass

Thank you for your interest in contributing to Research Compass! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors. Please:

- Be respectful and considerate in your communication
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git for version control
- Basic understanding of:
  - Graph databases (Neo4j)
  - Machine learning (PyTorch)
  - Natural language processing
  - Web development (Gradio)

### First Contribution

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Research-Compass.git
   cd Research-Compass
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/Apc0015/Research-Compass.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install PyTorch Geometric (optional)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install development dependencies
pip install pytest black flake8 mypy
```

### 2. Configuration

```bash
# Create environment file
cp .env.example .env

# Edit .env with your local settings
nano .env
```

### 3. Run Local Instance

```bash
# Start the application
python launcher.py

# For development with auto-reload
python launcher.py --dev
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes** - Fix issues in the codebase
2. **New Features** - Add new functionality
3. **Documentation** - Improve docs, examples, tutorials
4. **Tests** - Add or improve test coverage
5. **Code Quality** - Refactoring, performance improvements
6. **Examples** - Add usage examples or sample notebooks

### Contribution Workflow

1. **Check existing issues** - See if your idea is already being worked on
2. **Open an issue** - Discuss your proposed changes (for large changes)
3. **Get feedback** - Wait for maintainer response
4. **Implement changes** - Write code following our standards
5. **Add tests** - Ensure your changes are tested
6. **Submit PR** - Open a pull request for review

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Double quotes for strings
- **Naming:**
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Private methods: `_leading_underscore`

### Code Formatting

```bash
# Format code with black
black src/ --line-length 100

# Check linting with flake8
flake8 src/ --max-line-length 100

# Type checking with mypy
mypy src/
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional

def process_document(
    file_path: str,
    chunk_size: int = 500,
    metadata: Optional[Dict[str, str]] = None
) -> List[str]:
    """Process a document and return chunks.

    Args:
        file_path: Path to the document file
        chunk_size: Size of text chunks
        metadata: Optional metadata dictionary

    Returns:
        List of text chunks
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_gnn_model(
    graph_data: Data,
    model_type: str = "GAT",
    epochs: int = 50
) -> Dict[str, float]:
    """Train a Graph Neural Network model.

    This function trains a GNN model on the provided graph data
    using the specified architecture and hyperparameters.

    Args:
        graph_data: PyTorch Geometric Data object
        model_type: Type of GNN architecture (GAT, GCN, etc.)
        epochs: Number of training epochs

    Returns:
        Dictionary containing training metrics:
            - 'train_loss': Final training loss
            - 'val_loss': Final validation loss
            - 'accuracy': Model accuracy on test set

    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If training fails

    Example:
        >>> graph = load_graph_data()
        >>> metrics = train_gnn_model(graph, model_type="GAT", epochs=100)
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
    pass
```

### Error Handling

**Don't** use broad exception handling:

```python
# ❌ Bad
try:
    result = risky_operation()
except Exception:
    pass
```

**Do** use specific exceptions:

```python
# ✅ Good
from neo4j.exceptions import Neo4jError

try:
    result = graph.query(cypher_query)
except Neo4jError as e:
    logger.error(f"Neo4j query failed: {e}")
    raise
except ConnectionError as e:
    logger.warning(f"Connection lost, using fallback: {e}")
    result = fallback_operation()
```

### Logging

Use the logging module appropriately:

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("General informational messages")
logger.warning("Warning messages for recoverable issues")
logger.error("Error messages for failures")
logger.critical("Critical issues requiring immediate attention")
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core/test_graph_manager.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
from src.graphrag.core.graph_manager import GraphManager

class TestGraphManager:
    """Test suite for GraphManager class."""

    @pytest.fixture
    def graph_manager(self):
        """Fixture to create a GraphManager instance."""
        return GraphManager(use_neo4j=False)

    def test_add_node(self, graph_manager):
        """Test adding a node to the graph."""
        node_id = graph_manager.add_node(
            label="Paper",
            properties={"title": "Test Paper"}
        )
        assert node_id is not None
        assert graph_manager.node_exists(node_id)

    def test_add_relationship(self, graph_manager):
        """Test adding a relationship between nodes."""
        node1 = graph_manager.add_node("Paper", {"title": "Paper 1"})
        node2 = graph_manager.add_node("Paper", {"title": "Paper 2"})

        rel_id = graph_manager.add_relationship(
            node1, node2, "CITES"
        )
        assert rel_id is not None
```

### Test Coverage

- Aim for **80%+ code coverage**
- All new features must include tests
- Bug fixes should include regression tests
- Test both success and failure cases

---

## Pull Request Process

### Before Submitting

1. **Update from main:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Format code:**
   ```bash
   black src/ --line-length 100
   flake8 src/
   ```

4. **Update documentation** if needed

5. **Update CHANGELOG.md** with your changes

### PR Template

When opening a pull request, include:

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks** must pass (linting, tests)
2. **Code review** by at least one maintainer
3. **Address feedback** - Make requested changes
4. **Approval** - Get approval from maintainer
5. **Merge** - Maintainer will merge your PR

---

## Issue Reporting

### Bug Reports

When reporting bugs, include:

```markdown
**Describe the bug**
Clear description of what the bug is

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
 - OS: [e.g. Ubuntu 22.04]
 - Python version: [e.g. 3.11.5]
 - Research Compass version: [e.g. 1.0.0]

**Additional context**
Any other context about the problem
```

### Feature Requests

When requesting features, include:

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of what you want to happen

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other context or screenshots
```

---

## Project Structure

Understanding the codebase structure:

```
src/graphrag/
├── core/           # Core system components
├── analytics/      # Analytics and recommendations
├── ml/            # Machine learning and GNN
├── visualization/ # Graph visualization
├── ui/            # User interface
├── indexing/      # Document indexing
├── query/         # Query engines
├── utils/         # Utilities
└── evaluation/    # ML evaluation
```

### Module Responsibilities

- **core/**: Document processing, graph management, LLM integration
- **analytics/**: Citation analysis, recommendations, temporal trends
- **ml/**: GNN models, training, embeddings
- **visualization/**: Interactive graphs, attention maps
- **ui/**: Gradio web interface
- **indexing/**: Document indexing, chunking strategies
- **query/**: Search engines, query builders

---

## Development Best Practices

### 1. Branching Strategy

- `main` - Stable production code
- `develop` - Integration branch (if used)
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/documentation-update` - Documentation

### 2. Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)

footer (optional)
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(gnn): add graph transformer architecture

fix(core): resolve Neo4j connection timeout issue

docs(readme): update installation instructions
```

### 3. Code Review Checklist

When reviewing code, check:

- [ ] Code is readable and well-documented
- [ ] Tests are comprehensive
- [ ] Error handling is appropriate
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Backward compatibility maintained
- [ ] Documentation updated

---

## Getting Help

- **Documentation:** Check README.md first
- **Issues:** Search existing issues
- **Discussions:** Start a discussion for questions
- **Email:** Contact maintainers for sensitive issues

---

## Recognition

Contributors will be recognized in:

- Project README
- Release notes
- Contributor list

Thank you for contributing to Research Compass! 🎉

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
