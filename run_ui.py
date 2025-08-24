#!/usr/bin/env python3
"""
Launch script for the Pixel Perfect Streamlit UI.

This script launches the Streamlit web interface for the pixel-perfect
image processing framework.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit UI."""

    # Change to src directory for proper imports
    src_dir = Path(__file__).parent / "src"

    # Launch Streamlit
    cmd = [
        "uv",
        "run",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.port=8501",
        "--server.address=localhost",
    ]

    print("🎨 Launching Pixel Perfect UI...")
    print(f"   Directory: {src_dir}")
    print(f"   Command: {' '.join(cmd)}")
    print("   App will be available at: http://localhost:8501")
    print()

    try:
        subprocess.run(cmd, cwd=src_dir, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Pixel Perfect UI")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching UI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
