#!/usr/bin/env python3
"""
Pixel Perfect - Streamlit Cloud Deployment Entry Point

This is the main entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for a file named 'streamlit_app.py' in the root directory.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ui.streamlit_app import main  # noqa: E402

# Run the main app
main()
