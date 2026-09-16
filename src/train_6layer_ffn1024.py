"""Train the 6-layer FFN-1024 baseline with the same five seeds and test summary."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == '__main__':
    run_experiment(6, dim_feedforward=1024)
