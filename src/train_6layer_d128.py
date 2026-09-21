"""Train the 6-layer, 128-dimensional model with five seeds and training/test summaries."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == '__main__':
    run_experiment(6, dim_feedforward=512, evaluate_training=True, embedding_dim=128)
