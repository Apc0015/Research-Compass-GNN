#!/usr/bin/env python3
"""
Research Compass - Simple Demo Script
Shows exactly how to use your GNN-powered research platform
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_research_compass():
    """Demonstrate how to use Research Compass step by step."""
    
    print("🧭 Research Compass - Step-by-Step Demo")
    print("=" * 50)
    
    # Step 1: Import and initialize
    print("\n📦 Step 1: Import and Initialize System")
    try:
        from src.graphrag.core.academic_rag_system import AcademicRAGSystem
        print("✅ Successfully imported AcademicRAGSystem")
        
        # Initialize the system
        system = AcademicRAGSystem()
        print("✅ System initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False
    
    # Step 2: Check system components
    print("\n🔍 Step 2: Check System Components")
    try:
        # Check if core components are available
        components = {
            'Document Processor': hasattr(system, 'doc_processor'),
            'Graph Manager': hasattr(system, 'graph'),
            'Academic Graph Manager': hasattr(system, 'academic'),
            'Recommendation Engine': hasattr(system, 'recommendation_engine'),
            'Impact Metrics': hasattr(system, 'impact_metrics'),
            'GNN Manager': hasattr(system, 'gnn_manager'),
        }
        
        for component, available in components.items():
            status = "✅" if available else "❌"
            print(f"  {status} {component}: {'Available' if available else 'Not Available'}")
        
    except Exception as e:
        print(f"❌ Error checking components: {e}")
    
    # Step 3: Process a simple document
    print("\n📄 Step 3: Process a Simple Document")
    try:
        # Use one of your existing PDF files
        pdf_path = "data/docs/NCT06145295_Prot_SAP_000.pdf"
        if os.path.exists(pdf_path):
            print(f"📁 Processing: {pdf_path}")
            
            # Process the document
            result = system.process_academic_paper(pdf_path)
            print("✅ Document processed successfully")
            print(f"   Created: {result.get('created', {})}")
        else:
            print(f"⚠️  File not found: {pdf_path}")
            print("   Using text document instead...")
            
            # Create a simple text document
            test_text = """
            Machine Learning in Clinical Trials
            
            Abstract:
            This paper discusses the application of machine learning techniques 
            in clinical trial design and analysis. We demonstrate how neural networks 
            can improve patient selection and outcome prediction.
            
            Authors: Dr. John Smith, Dr. Jane Doe
            Year: 2024
            Venue: Journal of Medical AI
            """
            
            # Process as text
            from src.graphrag.core.document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            
            # Create chunks
            chunks = doc_processor.chunk_text(test_text, "demo_paper")
            print(f"✅ Created {len(chunks)} text chunks")
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
    
    # Step 4: Test basic functionality
    print("\n🔬 Step 4: Test Basic Functionality")
    try:
        # Test query
        query = "What are the main applications of machine learning in clinical trials?"
        print(f"🔍 Testing query: {query}")
        
        # Simple query test
        response = system.query_academic(query)
        print("✅ Query processed successfully")
        print(f"   Sources found: {len(response.get('sources', []))}")
        print(f"   Recommendations: {len(response.get('recommendations', []))}")
        
    except Exception as e:
        print(f"❌ Error with query: {e}")
    
    # Step 5: Show how to launch the web interface
    print("\n🌐 Step 5: Launch Web Interface")
    print("To start the web interface, run:")
    print("   python launcher.py")
    print("")
    print("Then open your browser to:")
    print("   http://localhost:7860")
    print("")
    print("In the web interface, you can:")
    print("   📤 Upload PDF documents")
    print("   🔍 Ask questions about your research")
    print("   💡 Get personalized recommendations")
    print("   🕸️ Explore citation networks")
    print("   📊 Analyze research trends")
    
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("Your Research Compass is ready to use!")
    print("🚀 Run 'python launcher.py' to start the web interface")
    
    return True

if __name__ == "__main__":
    demo_research_compass()