"""Dataset-level masked metrics with fixed class ordering and no sklearn dependency."""
import torch
from torch.nn import functional as F

from .dataset import IGNORE_INDEX, TASK_CLASSES
from .preprocess import TARGETS


def classification_metrics(matrix):
    total = sum(map(sum, matrix))
    classes = []
    for i in range(len(matrix)):
        tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        classes.append({'precision': precision if total else None, 'recall': recall if total else None,
                        'f1': f1 if total else None, 'support': support, 'predicted': predicted})
    return {'accuracy': sum(matrix[i][i] for i in range(len(matrix))) / total if total else None,
            'macro_f1': sum(c['f1'] for c in classes) / len(classes) if total else None,
            'valid_samples': total, 'per_class': classes}


class MetricAccumulator:
    def __init__(self):
        self.matrices = {t: torch.zeros((len(c), len(c)), dtype=torch.long) for t, c in TASK_CLASSES.items()}
        self.loss_sums = dict.fromkeys(TARGETS, 0.0)
        self.total_samples = 0

    @torch.no_grad()
    def update(self, logits, labels):
        self.total_samples += labels.shape[0]
        for i, task in enumerate(TARGETS):
            valid = labels[:, i] != IGNORE_INDEX
            if not bool(valid.any()):
                continue
            truth, scores = labels[valid, i], logits[task][valid]
            self.loss_sums[task] += F.cross_entropy(scores, truth, reduction='sum').item()
            n = len(TASK_CLASSES[task])
            counts = torch.bincount(truth * n + scores.argmax(dim=1), minlength=n * n)
            self.matrices[task] += counts.reshape(n, n).cpu()

    def compute(self):
        tasks = {}
        for task, matrix in self.matrices.items():
            result = classification_metrics(matrix.tolist())
            result['per_class'] = dict(zip(TASK_CLASSES[task], result['per_class']))
            result['ignored_samples'] = self.total_samples - result['valid_samples']
            result['loss'] = self.loss_sums[task] / result['valid_samples'] if result['valid_samples'] else None
            result['class_order'] = list(TASK_CLASSES[task])
            result['confusion_matrix'] = matrix.tolist()  # Rows=true, columns=predicted.
            tasks[task] = result
        losses = [t['loss'] for t in tasks.values() if t['loss'] is not None]
        return {'samples': self.total_samples, 'loss': sum(losses) / len(losses) if losses else None,
                'tasks': tasks}


def metric_rows(report, epoch, split):
    return [dict(epoch=epoch, split=split, task=task,
                 **{key: result[key] for key in ('loss', 'accuracy', 'macro_f1', 'valid_samples', 'ignored_samples')})
            for task, result in report['tasks'].items()]


def class_rows(report):
    return [dict(task=task, **{'class': name}, **metrics)
            for task, result in report['tasks'].items() for name, metrics in result['per_class'].items()]


def print_metrics(report, emit=print, per_class=True):
    def fmt(value):
        return '--' if value is None else f'{value:.4f}'
    for task, values in report['tasks'].items():
        emit(f'{task}: Accuracy={fmt(values["accuracy"])} Macro F1={fmt(values["macro_f1"])} '
             f'valid={values["valid_samples"]} ignored={values["ignored_samples"]}')
        if per_class:
            for name, c in values['per_class'].items():
                emit(f'  {name:<5} precision={fmt(c["precision"])} recall={fmt(c["recall"])} '
                     f'F1={fmt(c["f1"])} support={c["support"]}')
