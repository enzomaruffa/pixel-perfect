#!/usr/bin/env python3
"""
Pixel Perfect - Streamlit Web Interface

A visual pipeline builder for sophisticated image processing operations.
"""

import sys
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.components.layout import render_header, render_main_content, render_sidebar
from ui.components.session import initialize_session_state
from ui.design_system import apply_global_styles


def main():
    """Main Streamlit application entry point."""

    # Configure Streamlit page
    st.set_page_config(
        page_title="Pixel Perfect - Image Processing",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "# Pixel Perfect\nSophisticated image processing with visual pipeline builder",
        },
    )

    # Apply professional design system
    apply_global_styles()

    # Initialize session state
    initialize_session_state()

    # Render application layout
    render_header()

    # Main layout: sidebar + content
    with st.sidebar:
        render_sidebar()

    # Main content area
    render_main_content()


if __name__ == "__main__":
    main()
