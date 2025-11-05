#!/bin/bash

echo "🎨 Research Compass - Presentation Quick Start"
echo "=============================================="
echo ""

# Check if python-pptx is installed
if ! python3 -c "import pptx" 2>/dev/null; then
    echo "📦 Installing python-pptx..."
    pip install python-pptx
    echo "✅ Installed!"
    echo ""
fi

echo "🎯 Generating PowerPoint presentation..."
python3 generate_slides.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Presentation created:"
    ls -lh Research_Compass_GNN_Presentation.pptx 2>/dev/null || ls -lh *.pptx
    echo ""
    echo "📂 Location: $(pwd)"
    echo ""
    echo "💡 To open:"
    echo "   - Double-click the .pptx file"
    echo "   - Or: libreoffice Research_Compass_GNN_Presentation.pptx"
else
    echo "❌ Error generating presentation"
fi
