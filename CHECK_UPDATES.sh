#!/bin/bash

echo "🔍 CHECKING RESEARCH COMPASS UPDATES..."
echo ""
echo "📍 Current Location: $(pwd)"
echo "🌿 Current Branch: $(git branch --show-current)"
echo ""
echo "✅ NEW FILES CREATED:"
echo ""

files=(
    "GNN_WORKFLOW_GUIDE.md"
    "QUICK_REFERENCE_GNN.md"
    "src/graphrag/ui/graph_gnn_dashboard.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file (NOT FOUND)"
    fi
done

echo ""
echo "📊 UI CHANGES:"
if grep -q "Graph & GNN Dashboard" src/graphrag/ui/unified_launcher.py; then
    echo "  ✅ Tab 2 'Graph & GNN Dashboard' added to UI"
    echo "  ✅ Line: $(grep -n "Graph & GNN Dashboard" src/graphrag/ui/unified_launcher.py | head -1)"
else
    echo "  ❌ Tab 2 not found in UI"
fi

echo ""
echo "🌐 TO SEE CHANGES IN UI:"
echo "  1. Run: python launcher.py"
echo "  2. Open: http://localhost:7860"
echo "  3. Look for Tab 2: '🕸️ Graph & GNN Dashboard'"
echo ""
echo "📚 TO READ GUIDES:"
echo "  1. Quick Start: cat QUICK_REFERENCE_GNN.md | less"
echo "  2. Full Guide: cat GNN_WORKFLOW_GUIDE.md | less"
echo ""
echo "🔄 GITHUB STATUS:"
git log --oneline -3
echo ""
echo "Branch pushed to: https://github.com/Apc0015/Research-Compass/tree/$(git branch --show-current)"
echo ""
