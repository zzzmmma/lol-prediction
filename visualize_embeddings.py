"""Plot player and champion embeddings from the 6L / FFN512 / 256D model.

Run from any directory with: python visualize_embeddings.py
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "runs/layers_6_20260921_111124_289760/seed_137/last.pt"
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
COLORS = {
    "TOP": "#4e79a7",
    "JUNGLE": "#59a14f",
    "MID": "#f28e2b",
    "ADC": "#e15759",
    "SUPPORT": "#b07aa1",
}


def load_vocab(path):
    with path.open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    if not isinstance(vocab, dict) or "<UNK>" not in vocab:
        raise ValueError(f"Invalid vocabulary: {path}")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError(f"Vocabulary IDs must be consecutive and unique: {path}")
    return vocab


def load_embeddings(checkpoint_path, vocabs):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {"embedding_dim": 256, "num_layers": 6, "feedforward_dim": 512}
    config = checkpoint.get("model_config", {})
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected a 6L / FFN512 / 256D checkpoint: {checkpoint_path}")

    result = {}
    for kind, vocab in vocabs.items():
        if checkpoint.get(f"{kind}_vocab") != vocab:
            raise ValueError(f"{kind}_vocab.json does not match checkpoint IDs")
        weights = checkpoint["model_state_dict"][f"{kind}_embedding.weight"]
        if tuple(weights.shape) != (len(vocab), 256) or not bool(torch.isfinite(weights).all()):
            raise ValueError(f"Invalid {kind} embedding matrix")
        names = sorted((name for name in vocab if name != "<UNK>"), key=vocab.__getitem__)
        result[kind] = (names, weights[[vocab[name] for name in names]].float().numpy())
    return result


def load_roles(train_path, names_by_kind):
    counts = {kind: defaultdict(Counter) for kind in names_by_kind}
    columns = [(kind, f"{side}_{kind}_{role.lower()}", role)
               for kind in names_by_kind
               for side in ("blue", "red")
               for role in ROLES]
    with train_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = {column for _, column, _ in columns} - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Missing train.csv columns: {sorted(missing_columns)}")
        for row in reader:
            for kind, column, role in columns:
                name = (row[column] or "").strip()
                if name and name != "<UNK>":
                    counts[kind][name][role] += 1

    result = {}
    for kind, names in names_by_kind.items():
        missing_names = [name for name in names if not counts[kind][name]]
        if missing_names:
            raise ValueError(f"No training role for {kind}: {missing_names}")
        # ROLES order breaks equal-frequency ties deterministically.
        result[kind] = [max(ROLES, key=lambda role: counts[kind][name][role])
                        for name in names]
    return result


def save_plot(points, names, roles, title, axes, path, label_names):
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for role in ROLES:
        indices = [index for index, value in enumerate(roles) if value == role]
        ax.scatter(points[indices, 0], points[indices, 1], label=f"{role} ({len(indices)})",
                   color=COLORS[role], s=26, alpha=0.75, edgecolors="none")
    if label_names:
        for (x, y), name in zip(points, names):
            ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=6, alpha=0.8)
    ax.set(title=title, xlabel=axes[0], ylabel=axes[1])
    ax.grid(alpha=0.2)
    ax.legend(title="Most frequent training role", loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--player-vocab", type=Path, default=ROOT / "data/splits/player_vocab.json")
    parser.add_argument("--champion-vocab", type=Path, default=ROOT / "data/splits/champion_vocab.json")
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/train.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/embedding_visualization")
    parser.add_argument("--label-names", action="store_true", help="Annotate every point with its name")
    args = parser.parse_args()

    vocabs = {"player": load_vocab(args.player_vocab),
              "champion": load_vocab(args.champion_vocab)}
    embeddings = load_embeddings(args.checkpoint, vocabs)
    roles = load_roles(args.train, {kind: names for kind, (names, _) in embeddings.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for kind, (names, matrix) in embeddings.items():
        pca = PCA(n_components=2)
        pca_points = pca.fit_transform(matrix)
        ratios = pca.explained_variance_ratio_
        print(f"{kind.capitalize()} PCA explained variance ratio: "
              f"PC1={ratios[0]:.6f}, PC2={ratios[1]:.6f}, total={ratios.sum():.6f}")
        save_plot(pca_points, names, roles[kind], f"{kind.capitalize()} embeddings — PCA",
                  ("PC1", "PC2"), args.output_dir / f"{kind}_pca.png", args.label_names)

        tsne_points = TSNE(n_components=2, random_state=137, init="pca",
                           learning_rate="auto").fit_transform(matrix)
        save_plot(tsne_points, names, roles[kind], f"{kind.capitalize()} embeddings — t-SNE",
                  ("t-SNE 1", "t-SNE 2"), args.output_dir / f"{kind}_tsne.png", args.label_names)


if __name__ == "__main__":
    main()
