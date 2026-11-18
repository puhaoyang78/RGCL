"""Append-only training history shared by RGCL and the upstream adapters."""

import json
from pathlib import Path


class TrainingProgress:
    def __init__(self, checkpoint, *, resume=False):
        self.path = Path(checkpoint).with_suffix(".training.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.path.write_text("", encoding="utf-8")

    def add(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if row.get("event") == "epoch":
            print(json.dumps(row, allow_nan=False), flush=True)
