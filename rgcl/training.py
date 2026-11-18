"""One training loop for all paper classifiers and repair-guidance ablations."""

from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from .artifacts import write_json, write_jsonl
from .config import (
    ADAM_EPSILON, BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM, MAX_LENGTH, MAX_EMPHASIS_FACTOR,
    REGION_METHODS, REGIONAL_LOSS_WEIGHT, SIDE_TARGETS, WARMUP_FRACTION, check_run, selected_by,
)
from .data import class_weight, data_loader, load_rows, make_feature
from .evaluation import predict_rows, resolve_device
from .model import build_model, forward_batch, load_tokenizer, pretrained_source, save_checkpoint
from .progress import TrainingProgress


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(run, model, optimizer, metadata, loop, *, scheduler=None):
    """Only called at optimizer boundaries; an interrupted write keeps last.pt intact."""
    run = Path(run)
    log = run / "best.training.jsonl"
    checkpoint = dict(
        config={
            k: v for k, v in metadata.items()
            if k not in ("environment", "checkpoint_interval")
        },
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        loop=loop,
        rng=rng_state(),
        log_position=log.stat().st_size if log.exists() else 0,
    )
    temporary = run / "last.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(run / "last.pt")


def load_training_state(run, model, optimizer, metadata, *, scheduler=None):
    run = Path(run)
    cp = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    expected = {
        k: v for k, v in metadata.items()
        if k not in ("environment", "checkpoint_interval")
    }
    if cp["config"] != expected:
        raise ValueError(f"{run}: resume configuration or data members differ")
    model.load_state_dict(cp["model"])
    optimizer.load_state_dict(cp["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(cp["scheduler"])
    # Work after the last committed optimizer step will be replayed, including
    # validation and candidate selection. Discard its incomplete/duplicate logs.
    log = run / "best.training.jsonl"
    position = cp["log_position"]
    if position and (not log.exists() or log.stat().st_size < position):
        raise ValueError(f"{log}: training history is shorter than the checkpoint")
    if log.exists():
        with log.open("r+b") as stream:
            stream.truncate(position)
    return cp["loop"], cp["rng"]


class AdamW(torch.optim.Optimizer):
    """The Transformers 4.19 AdamW rule, including epsilon before bias correction."""

    def __init__(self, params, lr=2e-5, eps=1e-8, weight_decay=0.0):
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, eps=eps, betas=(0.9, 0.999), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("sparse gradients are not supported")
                state = self.state[p]
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                    )
                state["step"] += 1
                b1, b2 = group["betas"]
                g = p.grad
                avg, sq = state["exp_avg"], state["exp_avg_sq"]
                avg.mul_(b1).add_(g, alpha=1 - b1)
                sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = sq.sqrt().add_(group["eps"])
                step_size = (
                    group["lr"]
                    * math.sqrt(1 - b2 ** state["step"])
                    / (1 - b1 ** state["step"])
                )
                p.addcdiv_(avg, denom, value=-step_size)
                # Original Transformers AdamW applies decoupled decay AFTER its
                # adaptive update. The decay never enters either moment.
                decay = group.get("weight_decay", 0.0)
                if decay:
                    p.add_(p, alpha=-group["lr"] * decay)
        return loss


def randomize_regions(features, seed):
    """Choose a fixed random region with the same conserved token mass."""
    rng = random.Random(seed)
    records = []
    for feature in features:
        original = feature["line_labels"]
        # Use the same conserved cross-line code-token mass as bounded_pool.
        tokens = [[t for t in line if feature["code_mask"][t]]
                  for line in feature["line_token_indices"]]
        copies = Counter(t for line in tokens for t in line)
        scale = math.lcm(*copies.values()) if copies else 1
        counts = [sum(scale // copies[t] for t in line) for line in tokens]
        target = sum(n for n, y in zip(counts, original) if y == 1)
        total = sum(counts)
        shuffled = [int(y == 1 and n > 0) for n, y in zip(counts, original)]
        eligible = 0 < target < total
        if eligible:
            # Random subset-sum: the original region proves the target is
            # reachable. Randomize traversal and backtracking, without using
            # original positive locations to choose the new region.
            complement = target > total - target
            budget = total - target if complement else target
            order = [i for i, n in enumerate(counts) if n > 0]
            rng.shuffle(order)
            reachable, history = 1, []
            mask = (1 << (budget + 1)) - 1
            for i in order:
                history.append(reachable)
                reachable |= (reachable << counts[i]) & mask
            if not (reachable >> budget) & 1:
                raise ValueError("original region token mass must be reachable")
            chosen, remaining = set(), budget
            for i, before in zip(reversed(order), reversed(history)):
                skip = (before >> remaining) & 1
                take = remaining >= counts[i] and (before >> (remaining - counts[i])) & 1
                if take and (not skip or rng.getrandbits(1)):
                    chosen.add(i)
                    remaining -= counts[i]
            if remaining:
                raise ValueError("random region failed to preserve token mass")
            shuffled = [int((i not in chosen) if complement else (i in chosen)) if n else 0
                        for i, n in enumerate(counts)]
        feature["line_labels"] = shuffled
        actual = sum(n for n, y in zip(counts, shuffled) if y == 1)
        if actual != target:
            raise ValueError("random and Diff supervision token masses differ")
        overlap = sum(n for n, a, b in zip(counts, original, shuffled) if a == b == 1)
        records.append(dict(eligible=eligible, code_tokens=total / scale,
            diff_region_tokens=target / scale, random_region_tokens=actual / scale,
            overlap_tokens=overlap / scale, unchanged=original == shuffled,
            diff_region_lines=sum(y == 1 and n > 0 for n, y in zip(counts, original)),
            random_region_lines=sum(y == 1 and n > 0 for n, y in zip(counts, shuffled))))
    return records


def restrict_region_supervision(features, target):
    """Disable only the other side's auxiliary targets; retain function labels."""
    if target not in (0, 1):
        raise ValueError("region supervision target must be 0 or 1")
    for feature in features:
        if feature["labels"] != target:
            feature["line_labels"] = [0] * len(feature["line_labels"])


def region_supervision_statistics(features, target=None):
    """Count original Diff coverage and enabled loss mass, conserving cross-line tokens."""
    if target not in (None, 0, 1):
        raise ValueError("region supervision target must be None, 0 or 1")
    fields = ("functions", "eligible_functions", "supervised_functions", "code_tokens",
              "visible_region_tokens", "supervised_region_tokens", "supervised_background_tokens",
              "empty_region_functions", "all_region_functions")
    sides = {str(y): dict.fromkeys(fields, 0) for y in (0, 1)}
    for feature in features:
        y = int(feature["labels"])
        side = sides[str(y)]
        tokens = [[t for t in line if feature["code_mask"][t]]
                  for line in feature["line_token_indices"]]
        copies = Counter(t for line in tokens for t in line)
        scale = math.lcm(*copies.values()) if copies else 1
        counts = [sum(scale // copies[t] for t in line) for line in tokens]
        foreground = sum(n for n, label in zip(counts, feature["line_labels"]) if label == 1)
        total = sum(counts)
        eligible = 0 < foreground < total
        enabled = eligible and (target is None or target == y)
        side["functions"] += 1
        side["eligible_functions"] += int(eligible)
        side["supervised_functions"] += int(enabled)
        side["code_tokens"] += total / scale
        side["visible_region_tokens"] += foreground / scale
        side["supervised_region_tokens"] += foreground / scale if enabled else 0
        side["supervised_background_tokens"] += (total - foreground) / scale if enabled else 0
        side["empty_region_functions"] += int(foreground == 0)
        side["all_region_functions"] += int(total > 0 and foreground == total)
    return dict(target=target, by_target=sides,
                **{key: sum(s[key] for s in sides.values()) for key in fields})


def train(backbone, dataset, method, seed, data_dir, prepared_dir, run, *, device="auto",
          model_source=None, resume=False, save_every=500):
    check_run(backbone, dataset, seed, method)
    if save_every < 1:
        raise ValueError("Checkpoint interval must be positive")
    run = Path(run)
    if resume != (run / "last.pt").is_file() or (not resume and run.exists() and any(run.iterdir())):
        raise ValueError("Use an empty run directory or explicitly resume its last.pt")
    device = resolve_device(device)
    with_regions = method in REGION_METHODS
    train_rows = load_rows(data_dir, prepared_dir, dataset, "train", with_diff=with_regions)
    valid_rows = load_rows(data_dir, prepared_dir, dataset, "valid")
    weight = class_weight(prepared_dir, dataset)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    model = build_model(backbone, method, weight, model_source).to(device)
    tokenizer = load_tokenizer(backbone, model_source)
    features = [make_feature(row, tokenizer) for row in train_rows]
    run.mkdir(parents=True, exist_ok=True)
    supervision = None
    if with_regions:
        target = SIDE_TARGETS.get(method)
        supervision = region_supervision_statistics(features, target)
        if target is not None:
            restrict_region_supervision(features, target)
        if method == "random_region":
            randomized = randomize_regions(features, seed)
            write_jsonl(run / "train.random_regions.jsonl",
                        [dict(sample_key=row["sample_key"], **r) for row, r in zip(train_rows, randomized)])
        write_jsonl(run / "train.attention_regions.jsonl", [
            dict(sample_key=row["sample_key"], line_numbers=f["line_numbers"], labels=f["line_labels"])
            for row, f in zip(train_rows, features)
        ])
    loader = data_loader(features, train=True)
    valid_loader = data_loader([make_feature(row, tokenizer) for row in valid_rows])
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optimizer, EPOCHS * len(loader) * WARMUP_FRACTION,
                                               EPOCHS * len(loader))
    metric = selected_by(dataset)
    source, options = pretrained_source(backbone, model_source)
    meta = dict(backbone=backbone, dataset=dataset, method=method, seed=seed,
                pretrained=dict(source=str(source), **options), max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, adam_epsilon=ADAM_EPSILON, weight_decay=0.,
                warmup_fraction=WARMUP_FRACTION, gradient_clip=MAX_GRAD_NORM,
                max_emphasis_factor=MAX_EMPHASIS_FACTOR, class_weight=weight, soft_regional_loss_weight=REGIONAL_LOSS_WEIGHT if with_regions else 0.,
                region_supervision=supervision, selection_metric=metric, threshold=.5, comparison=">",
                training_members=[r["sample_key"] for r in train_rows],
                validation_members=[r["sample_key"] for r in valid_rows])
    state = dict(epoch=1, next_batch=0, global_step=0, best=-math.inf if metric == "mcc" else -1., chosen=None,
                 running=dict(loss=0., bce=0., soft_regional_loss=0.), epoch_rng=None, training_seconds=0.)
    if resume:
        state, saved_rng = load_training_state(run, model, optimizer, meta, scheduler=scheduler)
        restore_rng(saved_rng)
    history = TrainingProgress(run / "best.pt", resume=resume)
    for epoch in range(state["epoch"], EPOCHS + 1):
        model.train()
        started = time.monotonic()
        previous_time = state["training_seconds"]
        if state["next_batch"]:
            restore_rng(state["epoch_rng"])
        else:
            state["epoch_rng"] = rng_state()
        iterator = iter(loader)
        if state["next_batch"]:
            for _ in range(state["next_batch"]):
                next(iterator)
            restore_rng(saved_rng)
        for step, batch in enumerate(iterator, start=state["next_batch"]):
            values = forward_batch(model, batch, device, with_labels=True, with_regions=with_regions)
            loss = values["loss"]
            if not torch.isfinite(loss):
                raise ValueError(f"{method}: nonfinite loss at epoch {epoch}, step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["global_step"] += 1
            state["next_batch"] = step + 1
            for key in state["running"]:
                state["running"][key] += float(values[key].detach())
            global_step = state["global_step"]
            if global_step == 1 or global_step % save_every == 0 or step + 1 == len(loader):
                state["training_seconds"] = previous_time + time.monotonic() - started
                save_training_state(run, model, optimizer, meta, state, scheduler=scheduler)
        training_seconds = previous_time + time.monotonic() - started
        validation, predictions = predict_rows(model, valid_rows, valid_loader, dataset, device)
        components = {key: value / len(loader) for key, value in state["running"].items()}
        history.add(dict(event="epoch", epoch=epoch, train_loss=components.pop("loss"),
                         train_components=components, optimizer_steps=len(loader), global_step=state["global_step"],
                         validation=validation, validation_threshold=.5, training_seconds=training_seconds,
                         epoch_seconds=previous_time + time.monotonic() - started))
        score = round(validation[metric], 4) if metric == "p_c" else validation[metric]
        if score > state["best"]:
            state.update(best=score, chosen=epoch)
            save_checkpoint(run / "best.safetensors", model, backbone=backbone, dataset=dataset,
                            method=method, seed=seed, class_weight=weight, epoch=epoch,
                            selected_by=metric)
            write_json(run / "selection.json", dict(epoch=epoch, validation=validation))
            write_json(run / "valid.metrics.json", dict(selected_epoch=epoch, metrics=validation))
            write_jsonl(run / "valid.predictions.jsonl", predictions)
        state.update(epoch=epoch + 1, next_batch=0, epoch_rng=None,
                     running=dict.fromkeys(state["running"], 0.), training_seconds=0.)
        save_training_state(run, model, optimizer, meta, state, scheduler=scheduler)
    summary = dict(meta, completed_epochs=EPOCHS, selected_epoch=state["chosen"], optimizer_steps=state["global_step"])
    write_json(run / "training.metrics.json", summary)
    return summary
