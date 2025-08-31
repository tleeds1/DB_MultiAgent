#!/usr/bin/env python3
"""
Main entry point for the Multi-Database Intelligent Agent System
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.main_cli import main

if __name__ == "__main__":
    main()
