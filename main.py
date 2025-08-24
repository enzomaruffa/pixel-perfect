#!/usr/bin/env python3
"""Backward compatibility wrapper - now points to transform CLI."""

import subprocess
import sys

if __name__ == "__main__":
    # Use the installed package entry point
    sys.exit(subprocess.call(["uv", "run", "pixel-transform"] + sys.argv[1:]))
