#!/usr/bin/env python3
"""Hook entry point. Reads the hook JSON from stdin, prints a decision (or nothing)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automode.hook import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
