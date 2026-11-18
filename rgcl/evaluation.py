"""Frozen single-function inference and the paper's fixed threshold metrics."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import MAX_EMPHASIS_FACTOR
from .artifacts import write_json, write_jsonl
from .data import collate, data_loader, make_feature, load_rows, encode_source
from .metrics import metrics, paired_metrics
from .model import checkpoint_metadata, forward_batch, load_checkpoint, load_tokenizer


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def prediction_record(row, line_numbers, output, index):
    p = float(output["probability"][index])
    record = dict(sample_key=row["sample_key"], pair_key=row["pair_key"], target=row["target"],
                  probability=p, prediction=int(p > .5))
    if "line_emphasis_factors" in output:
        n = int(line_numbers.ge(0).sum())
        record.update(logit=float(output["logits"][index]), line_numbers=line_numbers[:n].tolist(),
                      line_emphasis_factors=output["line_emphasis_factors"][index, :n].cpu().tolist(),
                      effective_token_mass=output["effective_token_mass"][index, :n].cpu().tolist())
    return record


@torch.no_grad()
def predict_rows(model, rows, loader, dataset, device):
    model.eval()
    predictions, loss = [], 0.
    for batch in loader:
        output = forward_batch(model, batch, device, with_labels=model.is_plm_classifier)
        if model.is_plm_classifier:
            loss += float(output["bce"])
        else:
            loss += float(F.binary_cross_entropy_with_logits(output["logits"], batch["labels"].float().to(device)))
        for i, numbers in enumerate(batch["line_numbers"]):
            row = rows[len(predictions)]
            if int(batch["labels"][i]) != row["target"]:
                raise ValueError("Prediction order differs from the evaluation membership")
            predictions.append(prediction_record(row, numbers, output, i))
    if len(predictions) != len(rows):
        raise ValueError("Incomplete evaluation membership")
    scores = [p["probability"] for p in predictions]
    if dataset == "primevul_paired":
        result = paired_metrics(rows, scores)
    else:
        result = metrics([r["target"] for r in rows], scores)
        result.update(comparison=">", normal_fpr=result["fp"] / (result["fp"] + result["tn"]), normal_kinds={})
        for kind in ("ordinary", "repaired"):
            values = [p for r, p in zip(rows, scores) if r["target"] == 0 and r.get("normal_kind") == kind]
            false_positives = sum(p > .5 for p in values)
            result["normal_kinds"][kind] = dict(functions=len(values), false_positives=false_positives,
                                                fpr=false_positives / len(values) if values else None)
    result["loss"] = loss / len(loader)
    return result, predictions


def evaluate(checkpoint, data_dir, prepared_dir, split, output_dir, *,
             device="auto", model_source=None):
    device = resolve_device(device)
    meta = checkpoint_metadata(checkpoint)
    tokenizer = load_tokenizer(meta["backbone"], model_source)
    rows = load_rows(data_dir, prepared_dir, meta["dataset"], split)
    model = load_checkpoint(checkpoint, device=device, model_source=model_source)
    loader = data_loader([make_feature(row, tokenizer) for row in rows])
    result, predictions = predict_rows(model, rows, loader, meta["dataset"], device)
    record = dict(selected_epoch=int(meta["epoch"]), metrics=result)
    write_json(Path(output_dir) / f"{split}.metrics.json", record)
    write_jsonl(Path(output_dir) / f"{split}.predictions.jsonl", predictions)
    return record


@torch.no_grad()
def predict(checkpoint, source, *, device="auto", model_source=None):
    device = resolve_device(device)
    meta = checkpoint_metadata(checkpoint)
    tokenizer = load_tokenizer(meta["backbone"], model_source)
    model = load_checkpoint(checkpoint, device=device, model_source=model_source)
    feature = dict(encode_source(source, tokenizer), labels=0)
    output = forward_batch(model, collate([feature]), device)
    probability = float(output["probability"][0])
    result = dict(probability=probability, prediction=int(probability > .5))
    if not model.is_plm_classifier:
        result.update(line_numbers=feature["line_numbers"], line_emphasis_factors=output["line_emphasis_factors"][0].cpu().tolist())
    return result


def regional_emphasis(annotation, prediction):
    """Measure real-region emphasis after the source-only forward pass."""
    numbers = prediction["line_numbers"]
    mass = prediction["effective_token_mass"]
    factors = prediction["line_emphasis_factors"]
    changed = set(annotation["lines_ground_truth"])
    selected = [int(n in changed) for n in numbers]
    if (annotation["sample_key"] != prediction["sample_key"]
        or not numbers or not len(numbers) == len(mass) == len(factors)
        or any(n <= 0 for n in mass) or any(not 1 <= r <= MAX_EMPHASIS_FACTOR for r in factors)):
        raise ValueError("Invalid regional coordinates, emphasis factors or edit labels")
    total_mass = sum(mass)
    region_mass = sum(n * y for n, y in zip(mass, selected))
    background_mass = total_mass - region_mass
    weighted_total = sum(n * r for n, r in zip(mass, factors))
    weighted_region = sum(n * r * y for n, r, y in zip(mass, factors, selected))
    weighted_background = sum(n * r * (1 - y) for n, r, y in zip(mass, factors, selected))
    eligible = region_mass > 0 and background_mass > 0
    ratio = (weighted_region / region_mass) / (weighted_background / background_mass) if eligible else None
    return dict(region_token_mass=region_mass, background_token_mass=background_mass,
                token_coverage=region_mass / total_mass, regional_pooling_weight=weighted_region / weighted_total,
                emphasis_ratio=ratio, eligible=eligible,
                region_margin=max(0., math.log(2) - math.log(ratio)) if eligible else None)
