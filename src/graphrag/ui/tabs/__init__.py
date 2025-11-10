"""
UI Tabs Module for Research Compass.

This package contains modular UI tab components extracted from the monolithic
unified_launcher.py as part of Phase 4 UI refactoring.

Each tab is now a separate module for better:
- Maintainability (smaller, focused files)
- Testability (can test tabs independently)
- Readability (clear separation of concerns)
- Reusability (tabs can be imported and used elsewhere)

Modules:
- upload_tab: Document upload and URL processing UI
- (more tabs to be extracted)

Phase 4 Refactoring Goal:
Split the massive 2,451-line create_unified_ui() function into modular components.
"""

from .upload_tab import create_upload_tab

__all__ = [
    "create_upload_tab",
]
