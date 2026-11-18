"""Summarize the recorded metrics for the paper's matched experiments."""

import statistics

from .artifacts import read_json
from .config import ABLATIONS, BACKBONES, DATASETS, MAIN_METHODS, SEEDS, result_path

COLUMNS = ("p_c", "mcc", "f1", "auc", "auprc", "recall", "fpr")


def saved_metrics(path, split):
    metrics = read_json(path / f"{split}.metrics.json")["metrics"]
    return dict({k: metrics.get(k) for k in COLUMNS[:-1]},
                fpr=metrics.get("patched_fpr", metrics.get("normal_fpr")))


def summarize(root, family):
    rows = []
    experiments = [(b, d, MAIN_METHODS) for b in BACKBONES for d in DATASETS] if family == "main" else [
        ("unixcoder", "primevul_paired", (*MAIN_METHODS[1:], *ABLATIONS))
    ]
    for backbone, dataset, methods in experiments:
        for split in ("valid", "test"):
            for method in methods:
                values = [saved_metrics(result_path(root, backbone, dataset, seed, method), split) for seed in SEEDS]
                columns = {key: [value[key] for value in values] for key in COLUMNS}
                summary = {key: dict(mean=statistics.mean(group), std=statistics.stdev(group))
                           for key, group in columns.items() if all(n is not None for n in group)}
                rows.append(dict(backbone=backbone, dataset=dataset, split=split, method=method, metrics=summary))
    return dict(seeds=SEEDS, records=rows)
