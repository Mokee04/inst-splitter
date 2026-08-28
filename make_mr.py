#!/usr/bin/env python3
"""
make_mr.py - Single file MR / Instrumental & Vocals generator CLI
(Wrapper calling inst_splitter.cli.main_single)
"""

import sys
from inst_splitter.cli import main_single

if __name__ == "__main__":
    main_single()
