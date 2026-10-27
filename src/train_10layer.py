"""Train the 256D-ID, 10-layer model with five seeds and final split evaluations."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == '__main__':
    run_experiment(10, evaluate_training=True)
