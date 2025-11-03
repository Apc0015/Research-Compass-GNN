#!/usr/bin/env python3
"""
Research Compass - Unified Launcher

Launch the comprehensive Research Compass platform with all features.
Supports both production and development modes.

Usage:
    Production:  python launcher.py [--port PORT] [--share]
    Development: python launcher.py --dev [--port PORT] [--share]

Features:
    - 📤 Multiple file upload & web URL processing
    - 🔍 Research Assistant with GNN reasoning
    - 💬 Streaming responses with intelligent caching
    - 📊 Temporal analysis & trend detection
    - 💡 Personalized recommendations
    - 🕸️ Interactive citation explorer
    - 🔬 Discovery engine for cross-disciplinary research
    - 📈 Advanced citation metrics
    - ⚙️  Settings management with connection testing
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_dev_mode():
    """Setup auto-reload monitoring for development mode."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class CodeChangeHandler(FileSystemEventHandler):
            """Handle file system events and trigger reloads."""

            def __init__(self, restart_callback):
                self.restart_callback = restart_callback
                self.last_reload = 0
                self.reload_delay = 2  # seconds

            def on_modified(self, event):
                # Only reload for Python files
                if event.src_path.endswith('.py'):
                    current_time = time.time()
                    # Debounce: only reload if enough time has passed
                    if current_time - self.last_reload > self.reload_delay:
                        print(f"\n🔄 Detected change in: {event.src_path}")
                        print("♻️  Reloading application...\n")
                        self.last_reload = current_time
                        self.restart_callback()

        # Setup observer
        observer = Observer()

        def reload_app():
            """Reload application on file changes."""
            try:
                import importlib
                from src.graphrag.ui import unified_launcher
                from src.graphrag.core import academic_rag_system

                importlib.reload(unified_launcher)
                importlib.reload(academic_rag_system)

                print("✅ Modules reloaded successfully!\n")
            except Exception as e:
                print(f"❌ Error reloading: {e}\n")

        handler = CodeChangeHandler(reload_app)

        # Directories to monitor
        watch_dirs = [
            project_root / "src" / "graphrag",
            project_root / "config"
        ]

        for watch_dir in watch_dirs:
            if watch_dir.exists():
                observer.schedule(handler, str(watch_dir), recursive=True)
                print(f"   📁 Watching: {watch_dir}")

        observer.start()
        print("✅ File monitoring active!\n")

        return observer

    except ImportError:
        print("⚠️  watchdog not installed - auto-reload disabled")
        print("   Install with: pip install watchdog")
        print("   Running in standard mode...\n")
        return None


def print_banner(dev_mode=False, port=7860, share=False):
    """Print application banner."""
    mode_str = "Development Mode 🔧" if dev_mode else "Production Mode 🚀"

    print("=" * 70)
    print(f"🧭 Research Compass - {mode_str}")
    print("=" * 70)
    print("\n✨ Available Features:")
    print("  📤 Multiple file upload & web URL processing")
    print("  🔍 Research Assistant with GNN reasoning")
    print("  💬 Streaming responses")
    print("  💾 Intelligent caching")
    print("  📊 Temporal analysis")
    print("  💡 Personalized recommendations")
    print("  🕸️ Citation explorer")
    print("  🔬 Discovery engine")
    print("  📈 Advanced metrics")
    print("  ⚙️  Settings management")

    if dev_mode:
        print("\n🔄 Development Features:")
        print("  📁 Auto-reload on code changes")
        print("  ⚡ Hot-reload without restart")
        print("  💡 Tip: Edit & save - app reloads automatically!")

    print(f"\n🌐 Configuration:")
    print(f"   Port: {port}")
    print(f"   Share: {'Yes (Public URL)' if share else 'No (Local Only)'}")
    print("=" * 70 + "\n")


def initialize_system():
    """Initialize the AcademicRAGSystem."""
    try:
        from src.graphrag.core.academic_rag_system import AcademicRAGSystem
        print("⏳ Initializing AcademicRAGSystem...")
        system = AcademicRAGSystem()
        print("✅ System initialized successfully\n")
        return system
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize system: {e}")
        print("⏳ Launching with limited functionality...\n")
        return None


def main():
    """Launch the unified Research Compass UI."""
    parser = argparse.ArgumentParser(
        description='Launch Research Compass UI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Production:  python launcher.py
  Development: python launcher.py --dev
  Custom port: python launcher.py --port 8080
  Public URL:  python launcher.py --share
        """
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run on (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public Gradio link'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable development mode with auto-reload'
    )

    args = parser.parse_args()

    # Print banner
    print_banner(dev_mode=args.dev, port=args.port, share=args.share)

    # Setup dev mode if requested
    observer = None
    if args.dev:
        print("👀 Setting up file monitoring...")
        observer = setup_dev_mode()

    # Initialize system
    print("🚀 Starting server...\n")
    system = initialize_system()

    # Import and launch UI
    from src.graphrag.ui.unified_launcher import launch_unified_ui

    try:
        launch_unified_ui(system=system, port=args.port, share=args.share)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        if observer:
            observer.stop()
            observer.join()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if observer:
            observer.stop()
            observer.join()
        sys.exit(1)


if __name__ == '__main__':
    main()
