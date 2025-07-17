"""Tests for the credit risk transition matrix library."""

import pytest
import sys
from pathlib import Path

# Add src to path so we can import our module
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
