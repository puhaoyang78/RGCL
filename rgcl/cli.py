"""Command line entry points for training, evaluation, and frozen analyses."""

import argparse
import json
from pathlib import Path

from .config import BACKBONES, DATASETS, METHODS, SEEDS, result_path


def data_arguments(parser):
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))


def model_arguments(parser):
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", type=Path, help="Optional local copy of the exact pretrained revision")


def analysis_arguments(parser, name):
    data_arguments(parser)
    parser.add_argument("--model", type=Path, help="Optional local copy of the UniXcoder revision")
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / name)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rgcl", description="Repair-Guided Contextual Learning")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Build paper membership and Diff indices")
    data_arguments(p)
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")

    p = commands.add_parser("train", help="Train one paper experiment")
    data_arguments(p)
    model_arguments(p)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--backbone", choices=BACKBONES, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=500)

    p = commands.add_parser("evaluate", help="Evaluate an explicitly selected checkpoint")
    data_arguments(p)
    model_arguments(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("valid", "test"), default="test")
    p.add_argument("--output-dir", type=Path, required=True)

    p = commands.add_parser("predict", help="Predict from a single function's source")
    model_arguments(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)

    analysis_arguments(commands.add_parser("rq3", help="Paired classification transitions"), "rq3")
    analysis_arguments(commands.add_parser("rq4", help="Compare line emphasis on repair regions"), "rq4")
    p = commands.add_parser("summarize", help="Summarize results across seeds")
    p.add_argument("--family", choices=("main", "rq2"), default="main")
    p.add_argument("--results-dir", type=Path, default=Path("runs"))
    return dispatch(parser.parse_args(argv))


def dispatch(args):
    if args.command == "prepare":
        from .data import prepare
        datasets = DATASETS if args.dataset == "all" else (args.dataset,)
        result = {name: prepare(args.data_dir, args.prepared_dir, name) for name in datasets}
    elif args.command == "train":
        from .training import train
        run = args.run_dir or result_path("runs", args.backbone, args.dataset, args.seed, args.method)
        train(args.backbone, args.dataset, args.method, args.seed, args.data_dir, args.prepared_dir, run,
              device=args.device, model_source=args.model, resume=args.resume, save_every=args.save_every)
        result = dict(checkpoint=str(run / "best.safetensors"), metrics=str(run / "training.metrics.json"))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.checkpoint, args.data_dir, args.prepared_dir, args.split, args.output_dir,
                          device=args.device, model_source=args.model)
    elif args.command == "predict":
        from .evaluation import predict
        result = predict(args.checkpoint, args.source.read_text(encoding="utf-8"),
                         device=args.device, model_source=args.model)
    elif args.command in ("rq3", "rq4"):
        from .analysis import rq3, rq4
        analysis = rq3 if args.command == "rq3" else rq4
        result = analysis(args.data_dir, args.prepared_dir, args.results_dir, args.output_dir, args.model)
    elif args.command == "summarize":
        from .reporting import summarize
        result = summarize(args.results_dir, args.family)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
