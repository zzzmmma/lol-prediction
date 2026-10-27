# ID embedding dimension / Transformer depth experiments

All models use d_model=256, nhead=8, FFN=512, dropout=0.1 and seeds
137, 482, 911, 2027, 7643. The 16D models use separate player/champion
Linear(16, 256) projections before adding the existing learnable position,
side, role and type embeddings. Training defaults, tasks, losses and class
weights reuse the baseline. Each seed evaluates the final model on the
complete Regular Season training set and Playoff test set.

Reuse the existing 256D/6L and 16D/6L runs. Run only the four new experiments
when ready (each command performs a full five-seed experiment):

```powershell
.venv/Scripts/python.exe src/train_8layer.py
.venv/Scripts/python.exe src/train_10layer.py
.venv/Scripts/python.exe src/train_8layer_emb16.py
.venv/Scripts/python.exe src/train_10layer_emb16.py
```

Results go to unique timestamped folders under `runs/layers_8_*`,
`runs/layers_10_*`, `runs/layers_8_emb16_*`, and `runs/layers_10_emb16_*`.
Every seed directory contains `last.pt`, `config.json`, `training_metrics.json`,
`test_metrics.json` and the baseline logs/CSV reports. The console prints all
task metrics per seed and their five-seed mean ± sample standard deviation
(ddof=1), for both Training and Playoff.

After the four experiments finish, pass all six experiment roots to:

```powershell
.venv/Scripts/python.exe src/compare_embedding_layers.py `
  --emb256-6l runs/layers_6_20260914_212350_572417 `
  --emb16-6l runs/layers_6_emb16_20261027_230106_319953 `
  --emb256-8l runs/<256D_8L_run> `
  --emb256-10l runs/<256D_10L_run> `
  --emb16-8l runs/<16D_8L_run> `
  --emb16-10l runs/<16D_10L_run>
```

The comparison script never trains. It reads saved reports; if an older run
has no final `training_metrics.json`, it evaluates that run's saved checkpoint
against its original training dataset. These supplementary reports are saved
only inside the new comparison directory. Online training-epoch scores are
not substituted for final model evaluation. Original experiment folders and
checkpoints are never written to.

The new `runs/embedding_layers_comparison_<timestamp>/` contains:

- `winner_summary.csv`: Winner Prediction Training/Playoff mean ± std as numeric columns.
- `all_tasks.csv`: all tasks and both splits, with mean/std columns.
- `per_seed.csv`: individual seed Accuracy/Macro F1 and source directories.
- `summary.json` and `summary.txt`: complete results, with Winner Prediction first.

Scores use the 0..1 scale. If a task has missing metrics in any seed, its
aggregate is N/A rather than a mean over fewer seeds.

The currently saved 256D/6L run used 438 Playoff games, while the 16D/6L run
used 446. The comparison script rejects different test dataset fingerprints
by default. To display these existing results together, append
`--allow-different-test`; the output then records a warning, dataset hashes
and game counts. Those Playoff scores cannot establish a controlled model
comparison. Training data, training settings and task definitions must still
match. No existing result is recomputed or overwritten to hide this difference.
