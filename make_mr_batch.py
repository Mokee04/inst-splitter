#!/usr/bin/env python3
"""
make_mr_batch.py - Batch MR / Instrumental & Vocals generator CLI
(Wrapper calling inst_splitter.cli.main_batch)
"""

import sys
from inst_splitter.cli import main_batch

if __name__ == "__main__":
    main_batch()
