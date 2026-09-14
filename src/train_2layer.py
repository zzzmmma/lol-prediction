"""Train the 2-layer baseline with seeds 42, 43, 44 and summarize test scores."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == '__main__':
    run_experiment(2)
