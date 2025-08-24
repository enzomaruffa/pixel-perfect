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

    # Custom CSS for wider sidebar
    st.markdown(
        """
    <style>
        /* Make sidebar wider for better usability */
        .css-1d391kg {width: 400px !important;}
        .css-1cyp50f {min-width: 400px !important; max-width: 400px !important;}

        /* Adjust main content margin */
        .main .block-container {margin-left: 420px;}

        /* Responsive adjustments */
        @media (max-width: 1024px) {
            .css-1d391kg {width: 350px !important;}
            .css-1cyp50f {min-width: 350px !important; max-width: 350px !important;}
            .main .block-container {margin-left: 370px;}
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

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
