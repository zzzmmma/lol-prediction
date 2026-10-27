"""Train five seeds with 16D token embeddings, reporting Training and Playoff scores."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


if __name__ == '__main__':
    run_experiment(6, evaluate_training=True, token_embedding_dim=16)
