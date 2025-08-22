#!/usr/bin/env python3
"""Main entry point for the pixel-perfect CLI application."""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli import cli

if __name__ == "__main__":
    cli()
