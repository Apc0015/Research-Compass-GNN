# Changelog

All notable changes to Research Compass will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional project documentation structure
- MIT License
- Contributing guidelines
- This changelog

### Removed
- Redundant audit and review documents

## [1.0.0] - 2025-11-05

### Added
- Complete Graph Neural Network integration with PyTorch Geometric
- Interactive Graph & GNN Dashboard with model training UI
- Unified multi-provider LLM support (Ollama, OpenRouter, OpenAI, LM Studio)
- Intelligent caching system with 10-100x performance improvements
- Streaming AI responses with real-time display
- Temporal analysis for research trend tracking
- Advanced citation metrics (disruption index, sleeping beauty detection)
- Cross-disciplinary discovery engine
- Personalized recommendation system (hybrid + GNN-powered)
- Interactive graph visualization with Pyvis and Plotly
- Citation network explorer
- GNN attention visualization and explainability
- Automated setup script (`setup.sh`)
- Comprehensive configuration system (CLI > ENV > YAML > defaults)
- Health monitoring and system status checks
- Cache management interface
- Multi-format document processing (PDF, DOCX, TXT, Markdown)
- arXiv and web URL fetching
- Relationship extraction and inference
- Academic metadata extraction

### Changed
- Unified configuration architecture with dataclass-based validation
- Improved error handling in core modules (14 instances)
- Renamed `ml/gnn_explainer.py` to `ml/gnn_interpretation.py` for clarity
- Enhanced README with complete usage guide and troubleshooting

### Fixed
- Critical setup issues with missing `.env` file
- Duplicate `gnn_explainer.py` naming conflict
- Incorrect `sys.path` imports in test blocks
- Neo4j/NetworkX fallback mechanism
- Configuration loading and precedence
- LLM provider initialization

### Security
- Added input validation for file uploads
- Implemented secure credential management via environment variables

## [0.1.0] - 2025-10-01

### Added
- Initial project structure
- Basic graph management with Neo4j and NetworkX
- Document processing pipeline
- Entity extraction with spaCy
- Vector search with FAISS
- Basic LLM integration
- Gradio web interface
- Graph visualization

---

## Version History

- **1.0.0** - Production-ready release with GNN integration
- **0.1.0** - Initial alpha release

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

1. **Update Dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   python -m spacy download en_core_web_sm
   ```

2. **Update Configuration:**
   - Copy `.env.example` to `.env`
   - Update `LLM_PROVIDER` settings (old `USE_OLLAMA` deprecated)
   - Review new configuration options in `.env.example`

3. **Install PyTorch Geometric (optional):**
   ```bash
   # For CPU:
   pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
   ```

4. **Run Database Migrations:**
   - No schema changes required
   - Existing graphs remain compatible

5. **Test Your Setup:**
   ```bash
   python launcher.py
   ```

---

## Breaking Changes

### 1.0.0

- **Configuration:** `USE_OLLAMA` and `USE_OPENAI` deprecated, use `LLM_PROVIDER` instead
- **Imports:** `ml.gnn_explainer` renamed to `ml.gnn_interpretation`
- **Environment:** Python 3.11+ now required (3.13 not supported)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
