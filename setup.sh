#!/bin/bash
# Research Compass - Automated Setup Script
# This script helps set up the Research Compass environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

echo "================================================"
echo "Research Compass - Automated Setup"
echo "================================================"
echo ""

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 11 ]; then
    print_error "Python 3.11+ required. Found: $PYTHON_VERSION"
    print_info "Please install Python 3.11 or higher"
    exit 1
else
    print_success "Python $PYTHON_VERSION detected"
fi

# Check if .env exists
print_info "Checking environment configuration..."
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.example .env
    print_success ".env file created"
    print_info "Please edit .env file with your credentials before running the app"
else
    print_success ".env file exists"
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
if pip install -r requirements.txt; then
    print_success "Core dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Download spaCy model
print_info "Downloading spaCy language model..."
if python -m spacy download en_core_web_sm; then
    print_success "spaCy model downloaded"
else
    print_warning "Failed to download spaCy model (may already exist)"
fi

# Check for PyTorch Geometric
print_info "Checking PyTorch Geometric installation..."
if python -c "import torch_geometric" 2>/dev/null; then
    print_success "PyTorch Geometric already installed"
else
    print_warning "PyTorch Geometric not installed"
    print_info "For GNN features, install manually:"
    echo ""
    echo "  CPU version:"
    echo "    pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"
    echo ""
    echo "  GPU version (CUDA 11.8):"
    echo "    pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
    echo ""
fi

# Check for Ollama
print_info "Checking for Ollama (recommended LLM provider)..."
if command -v ollama &> /dev/null; then
    print_success "Ollama found"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama server is running"

        # Check for models
        MODELS=$(ollama list 2>/dev/null | tail -n +2)
        if [ -n "$MODELS" ]; then
            print_success "Ollama models installed:"
            echo "$MODELS" | awk '{print "  - " $1}'
        else
            print_warning "No Ollama models found"
            print_info "Install a model with: ollama pull llama3.2"
        fi
    else
        print_warning "Ollama is installed but not running"
        print_info "Start Ollama with: ollama serve"
    fi
else
    print_warning "Ollama not found"
    print_info "Install Ollama for local LLM support:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  ollama serve"
    echo "  ollama pull llama3.2"
fi

# Check for Neo4j
print_info "Checking for Neo4j database..."
if curl -s http://localhost:7474 &> /dev/null; then
    print_success "Neo4j detected at http://localhost:7474"
else
    print_warning "Neo4j not detected (optional - falls back to NetworkX)"
    print_info "To use Neo4j features:"
    echo ""
    echo "  Option 1: Neo4j Aura (Cloud - Free tier available)"
    echo "    - Visit: https://neo4j.com/cloud/aura/"
    echo "    - Create free account and database"
    echo "    - Update .env with connection details"
    echo ""
    echo "  Option 2: Local Neo4j Docker"
    echo "    docker run -d --name neo4j \\"
    echo "      -p 7474:7474 -p 7687:7687 \\"
    echo "      -e NEO4J_AUTH=neo4j/password \\"
    echo "      neo4j:latest"
    echo ""
fi

# Create required directories
print_info "Creating required directories..."
mkdir -p data/documents data/indices data/cache output/visualizations output/reports output/exports models/gnn
print_success "Directories created"

# Summary
echo ""
echo "================================================"
echo "Setup Summary"
echo "================================================"
echo ""

print_success "Core setup complete!"
echo ""
print_info "Next steps:"
echo "  1. Edit .env file with your credentials"
echo "  2. (Optional) Install PyTorch Geometric for GNN features"
echo "  3. (Optional) Set up Ollama for local LLM support"
echo "  4. (Optional) Configure Neo4j database"
echo "  5. Launch the application: python launcher.py"
echo ""
print_info "Access the application at: http://localhost:7860"
echo ""
echo "================================================"
