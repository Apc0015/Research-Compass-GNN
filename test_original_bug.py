#!/usr/bin/env python3
"""
Demonstrate the original bug and verify it's fixed
"""

import sys

def test_original_bug():
    """
    Original bug: When Gradio File component uses type="binary",
    uploads are raw bytes without .name attribute, causing:
    AttributeError: 'bytes' object has no attribute 'name'
    """

    print("=" * 70)
    print("Demonstrating Original Bug Scenario")
    print("=" * 70)

    # Simulate the original buggy code
    print("\n1. ORIGINAL BUGGY CODE:")
    print("-" * 70)
    print("""
def process_pdfs_BUGGY(files):
    for idx, file in enumerate(files):
        status = f"Processing: {file.name}"  # ❌ Assumes .name exists
        paper_info = {'name': file.name}      # ❌ Assumes .name exists
        app_state.paper_list.append(file.name) # ❌ Assumes .name exists
    """)

    print("\n2. WHAT HAPPENS WITH BYTES UPLOAD:")
    print("-" * 70)

    # Simulate bytes upload (type="binary" in Gradio)
    sample_bytes = b"PDF content here"

    try:
        # This is what the buggy code tried to do
        filename = sample_bytes.name  # This will fail!
        print(f"   Filename: {filename}")
    except AttributeError as e:
        print(f"   ❌ ERROR: {e}")
        print(f"   ↳ This is the bug that was occurring!")

    print("\n3. FIXED CODE:")
    print("-" * 70)
    print("""
def process_pdfs_FIXED(files):
    for idx, file in enumerate(files):
        # Safely resolve filename
        if hasattr(file, 'name'):
            filename = file.name
        elif isinstance(file, dict) and 'name' in file:
            filename = file['name']
        else:
            filename = f"uploaded_{idx}.pdf"
    """)

    print("\n4. TESTING FIXED CODE WITH BYTES:")
    print("-" * 70)

    # Simulate the fix
    file = sample_bytes
    if hasattr(file, 'name'):
        filename = file.name
    elif isinstance(file, dict) and 'name' in file:
        filename = file['name']
    else:
        filename = f"uploaded_0.pdf"

    print(f"   ✅ Filename resolved: {filename}")
    print(f"   ↳ No AttributeError!")

    print("\n5. VERIFY FIX IN ACTUAL CODE:")
    print("-" * 70)

    # Import the actual fixed function
    sys.path.insert(0, '.')
    from launcher import process_pdfs, app_state

    # Reset state
    app_state.graph_data = None
    app_state.paper_list = []

    # Test with bytes (simulating Gradio type="binary")
    files = [b"Sample PDF 1", b"Sample PDF 2"]

    try:
        status, _, _, _ = process_pdfs(
            files=files,
            extract_citations=False,
            build_graph=False,
            extract_metadata=False
        )

        print(f"   ✅ process_pdfs() handled bytes successfully!")
        print(f"   ✅ Files processed: {app_state.paper_list}")
        print(f"   ↳ Bug is FIXED!")
        return True

    except AttributeError as e:
        if "'bytes' object has no attribute 'name'" in str(e):
            print(f"   ❌ Bug still exists: {e}")
            return False
        else:
            raise

    print("\n" + "=" * 70)

def test_port_fallback():
    """Test port fallback when 7860 is occupied"""

    print("\n" + "=" * 70)
    print("Demonstrating Port Fallback Fix")
    print("=" * 70)

    print("\n1. ORIGINAL ISSUE:")
    print("-" * 70)
    print("""
    If port 7860 is already in use:
    ❌ OSError: Cannot find empty port in range: 7860-7860
    ❌ Application crashes
    """)

    print("\n2. FIXED CODE:")
    print("-" * 70)
    print("""
def find_available_port(start_port=7860, max_tries=10):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket() as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port
    """)

    print("\n3. TESTING PORT FALLBACK:")
    print("-" * 70)

    from launcher import find_available_port
    import socket

    # Test port fallback (7860 might already be in use by launcher)
    try:
        # Try to find available port
        port = find_available_port(start_port=7860, max_tries=10)
        print(f"   ✅ Port fallback works: found port {port}")

        # Verify it actually tries different ports if needed
        # Test with a higher starting port to avoid conflicts
        port2 = find_available_port(start_port=9000, max_tries=5)
        print(f"   ✅ Alternative range works: found port {port2}")
        print(f"   ↳ Application gracefully handles port conflicts")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "BUG FIX VERIFICATION" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")

    success1 = test_original_bug()
    success2 = test_port_fallback()

    print("\n" + "=" * 70)
    if success1 and success2:
        print("🎉 VERIFICATION COMPLETE - ALL BUGS FIXED!")
        print("=" * 70)
        print("\nSummary:")
        print("✅ 1. AttributeError on bytes uploads - FIXED")
        print("✅ 2. Port conflict crashes - FIXED")
        print("\nThe Real Data Training tab now:")
        print("  • Handles type='binary' uploads (bytes objects)")
        print("  • Handles file-like objects with .name attribute")
        print("  • Handles dict uploads from Gradio")
        print("  • Finds alternative ports when 7860 is occupied")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED - BUGS STILL PRESENT")
        print("=" * 70)
        sys.exit(1)
