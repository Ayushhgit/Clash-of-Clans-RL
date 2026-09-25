"""Makes `pytest` work from the repo root without setting PYTHONPATH."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
