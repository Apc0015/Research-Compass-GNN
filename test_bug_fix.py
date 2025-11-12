#!/usr/bin/env python3
"""
Test the bug fix for process_pdfs handling different upload types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from launcher import process_pdfs, app_state

def test_different_upload_types():
    """Test process_pdfs with different upload types"""

    print("=" * 70)
    print("Testing Bug Fix: process_pdfs with different upload types")
    print("=" * 70)

    # Create sample PDF content (simple text as PDF would be complex)
    sample_pdf_content = b"""
    Sample Research Paper
    This is a test paper about Graph Neural Networks.
    References: [Smith et al., 2020], [Jones, 2021]
    """

    # Test 1: bytes object (type="binary" in Gradio)
    print("\n1. Testing with bytes object (simulating type='binary')...")
    try:
        app_state.graph_data = None
        app_state.paper_list = []

        files = [sample_pdf_content, sample_pdf_content]
        status, graph_stats, _, _ = process_pdfs(
            files=files,
            extract_citations=True,
            build_graph=True,
            extract_metadata=True
        )

        print("✅ bytes object handled successfully!")
        print(f"   Files processed: {len(app_state.paper_list)}")
        print(f"   Filenames: {app_state.paper_list}")

    except AttributeError as e:
        if "'bytes' object has no attribute 'name'" in str(e):
            print(f"❌ FAILED: Original bug still present - {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: file-like object with .name attribute
    print("\n2. Testing with file-like object (has .name attribute)...")
    try:
        app_state.graph_data = None
        app_state.paper_list = []

        import io

        class MockFile:
            def __init__(self, content, name):
                self.content = content
                self.name = name
                self._pos = 0

            def read(self):
                return self.content

            def seek(self, pos):
                self._pos = pos

        files = [
            MockFile(sample_pdf_content, "paper1.pdf"),
            MockFile(sample_pdf_content, "paper2.pdf")
        ]

        status, graph_stats, _, _ = process_pdfs(
            files=files,
            extract_citations=True,
            build_graph=True,
            extract_metadata=True
        )

        print("✅ File-like object handled successfully!")
        print(f"   Files processed: {len(app_state.paper_list)}")
        print(f"   Filenames: {app_state.paper_list}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: dict with 'name' key (Gradio sometimes sends this)
    print("\n3. Testing with dict containing 'name' key...")
    try:
        app_state.graph_data = None
        app_state.paper_list = []

        files = [
            {'name': 'dict_paper1.pdf', 'data': sample_pdf_content},
            {'name': 'dict_paper2.pdf', 'data': sample_pdf_content}
        ]

        status, graph_stats, _, _ = process_pdfs(
            files=files,
            extract_citations=True,
            build_graph=True,
            extract_metadata=True
        )

        print("✅ Dict upload handled successfully!")
        print(f"   Files processed: {len(app_state.paper_list)}")
        print(f"   Filenames: {app_state.paper_list}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Mixed types
    print("\n4. Testing with mixed upload types...")
    try:
        app_state.graph_data = None
        app_state.paper_list = []

        files = [
            sample_pdf_content,  # bytes
            MockFile(sample_pdf_content, "named.pdf"),  # file-like with .name
            {'name': 'dict.pdf', 'data': sample_pdf_content}  # dict
        ]

        status, graph_stats, _, _ = process_pdfs(
            files=files,
            extract_citations=True,
            build_graph=True,
            extract_metadata=True
        )

        print("✅ Mixed types handled successfully!")
        print(f"   Files processed: {len(app_state.paper_list)}")
        print(f"   Filenames: {app_state.paper_list}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ ALL UPLOAD TYPE TESTS PASSED - Bug fix verified!")
    print("=" * 70)
    return True

def test_port_fallback():
    """Test the port fallback helper"""
    print("\n" + "=" * 70)
    print("Testing Port Fallback Helper")
    print("=" * 70)

    from launcher import find_available_port

    # Test finding available port
    port = find_available_port(start_port=7860, max_tries=10)
    print(f"\n✅ Port fallback helper works!")
    print(f"   Found available port: {port}")

    # Test with different starting port
    port2 = find_available_port(start_port=8000, max_tries=5)
    print(f"   Alternative port: {port2}")

    return True

if __name__ == "__main__":
    success1 = test_different_upload_types()
    success2 = test_port_fallback()

    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED - Bug fixes verified and working!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
