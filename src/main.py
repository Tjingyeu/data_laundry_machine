#!/usr/bin/env python3
"""
Data Laundry Machine - AI-powered data cleaning agent

Usage:
    python -m src.main clean <input_file> <output_file> [options]
    python -m src.main analyze <input_file>
    python -m src.main providers
    python -m src.main init
"""

import sys
import os

# 添加 src 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.commands import cli


if __name__ == "__main__":
    cli()
