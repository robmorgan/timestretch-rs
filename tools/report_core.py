#!/usr/bin/env python3
"""Generate the current timestretch report; historical reports are archived."""
import argparse
from pathlib import Path
from qualify import summarize
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('target/qualification'))
    a=p.parse_args();summarize(a.output);print(a.output/'index.html')
