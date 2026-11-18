# RGCL — Repair-Guided Contextual Learning

RGCL classifies individual functions as vulnerable or non-vulnerable. A
repair pair gives us two versions of a function, one before and one after
the fix. During training, RGCL learns to give more weight to the lines
changed by the fix. Once trained, it can score a function from its source
code alone.

## Install and prepare the data

Use Python 3.11 and a CUDA-enabled PyTorch installation for GPU training.
Run the commands below in Bash from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Start with the original paired PrimeVul release. [The data guide](data/README.md)
lists the exact downloads and where to put them. Prepare each dataset once.
For SecVulEval or CleanVul, change the dataset name in the command to
`secvuleval` or `cleanvul`.

```bash
DATA_DIR=data
DEVICE=cuda:0
python -m rgcl prepare --data-dir "$DATA_DIR" --dataset primevul_paired
```

Set `DATA_DIR` to the directory holding your downloaded datasets. The
prepared indices go to `data/prepared/`. Training downloads the pinned
UniXcoder, CodeBERT, or CodeT5 backbone for the selected run. [Model revisions](rgcl/config.py)

## Train and test one model

This example trains RGCL with UniXcoder and seed 42, then evaluates the
checkpoint selected on the validation split.

```bash
run=runs/main/unixcoder/primevul_paired/seed42/rgcl
python -m rgcl train --data-dir "$DATA_DIR" --dataset primevul_paired \
  --backbone unixcoder --method rgcl --seed 42 --device "$DEVICE"
python -m rgcl evaluate --data-dir "$DATA_DIR" --checkpoint "$run/best.safetensors" \
  --split test --output-dir "$run" --device "$DEVICE"
```

Training writes `best.safetensors`, `selection.json`, and
`valid.metrics.json` in the run directory. Evaluation writes
`test.metrics.json` and `test.predictions.jsonl` in the same directory.
Rerun an interrupted training command with `--resume`.

To score one function, save it as `function.c` and run:

```bash
python -m rgcl predict --checkpoint "$run/best.safetensors" \
  --source function.c --device "$DEVICE"
```

## RQ1 — Main comparison

The main comparison covers three datasets, three backbones, and seeds
`42`, `1234`, and `123456`.

| Method | CLI value | What it does |
|---|---|---|
| PLM classifier | `plm_classifier` | Classifies the function with the backbone's standard readout. |
| RGCL w/o repair guidance | `rgcl_no_repair_guidance` | Weights lines using the function context and trains on function labels. |
| RGCL | `rgcl` | Also uses changed lines from both versions of a repair pair to guide the line weights. |

All runs use 512 tokens, batch size 8, 16 epochs, AdamW at 2e-5, and 10%
linear warmup. Checkpoint selection uses validation P-C on PrimeVul and
validation MCC on SecVulEval and CleanVul. P-C is the fraction of repair
pairs in which both functions are classified correctly. It is rounded to four
decimal places for checkpoint selection. Ties go to the earliest epoch.
The classification rule throughout is `probability > 0.5`.

Prepare all three datasets before running the full table. Define this
helper once in the same Bash session for the RQ1 and RQ2 commands. It
reuses completed runs in their usual directories.

```bash
run_experiment() {
  local group="$1" dataset="$2" backbone="$3" seed="$4" method="$5"
  local out="runs/$group/$backbone/$dataset/seed$seed/$method"
  if [ ! -f "$out/training.metrics.json" ]; then
    python -m rgcl train --data-dir "$DATA_DIR" --dataset "$dataset" \
      --backbone "$backbone" --seed "$seed" --method "$method" \
      --run-dir "$out" --device "$DEVICE" || return 1
  fi
  if [ ! -f "$out/test.metrics.json" ] || [ ! -f "$out/test.predictions.jsonl" ]; then
    python -m rgcl evaluate --data-dir "$DATA_DIR" --checkpoint "$out/best.safetensors" \
      --split test --output-dir "$out" --device "$DEVICE"
  fi
}
```

The full RQ1 comparison has 81 runs. This command writes the mean and
standard deviation across seeds to `runs/rq1-summary.json`.

```bash
(
  set -e
  for dataset in primevul_paired secvuleval cleanvul; do
    for backbone in unixcoder codebert codet5; do
      for seed in 42 1234 123456; do
        for method in plm_classifier rgcl_no_repair_guidance rgcl; do
          run_experiment main "$dataset" "$backbone" "$seed" "$method"
        done
      done
    done
  done
  python -m rgcl summarize --family main > runs/rq1-summary.json
)
```

## RQ2 — Ablation study

RQ2 uses UniXcoder on PrimeVul. It reuses the three RGCL and three
RGCL w/o repair guidance runs from RQ1, then adds four variants:

| Variant | CLI value | What changes |
|---|---|---|
| Random-region | `random_region` | Uses a random region with the same effective token mass as the repair region. |
| Vulnerable-only | `vulnerable_only` | Applies repair guidance only to vulnerable functions. |
| Repaired-only | `repaired_only` | Applies repair guidance only to repaired functions. |
| RGCL w/o interaction | `rgcl_no_interaction` | Removes the interaction between each line and the function context. |

The command below also works when starting with RQ2. It trains any of
the six comparison runs that are still missing.

```bash
(
  set -e
  for seed in 42 1234 123456; do
    run_experiment main primevul_paired unixcoder "$seed" rgcl
    run_experiment main primevul_paired unixcoder "$seed" rgcl_no_repair_guidance
    for method in random_region vulnerable_only repaired_only rgcl_no_interaction; do
      run_experiment rq2 primevul_paired unixcoder "$seed" "$method"
    done
  done
  python -m rgcl summarize --family rq2 > runs/rq2-summary.json
)
```

## RQ3 — Which repair pairs change classification?

Run the analysis after evaluating RGCL and RGCL w/o repair guidance
on PrimeVul with UniXcoder for all three seeds. The transition matrix
and summary are saved to `runs/rq3/classification_transitions.csv`
and `runs/rq3/classification_summary.csv`. Repair-region coverage is
saved to `runs/rq3/coverage.json`.

```bash
python -m rgcl rq3 --data-dir "$DATA_DIR"
```

## RQ4 — How much weight does RGCL give to repair lines?

This analysis also needs the three Random-region runs. It divides the
mean line emphasis on repair lines by the mean on surrounding lines.
Both averages use effective token mass. The tables go to
`runs/rq4/region_summary.csv` and
`runs/rq4/paired_changes.csv`.

```bash
python -m rgcl rq4 --data-dir "$DATA_DIR"
```

## Use an existing checkpoint

Run the training command above to get a fine-tuned checkpoint. If you
already have one, pass its local `.safetensors` path to
`predict --checkpoint` to score a function. To evaluate a dataset split,
prepare that dataset and use the same path with `evaluate --checkpoint`.

To recreate the RQ tables from saved runs, place each model in its run
directory with `selection.json` and `valid.metrics.json`. Write Test
metrics and predictions to that directory with the evaluation command.
RQ3 and RQ4 use the selection record to match predictions to the chosen
epoch.

For paired evaluation and the external comparison, see the author
repositories for [PrimeVul](https://github.com/DLVulDet/PrimeVul) and
[LOSVER](https://github.com/waroad/losver). RGCL code is under the
[MIT license](LICENSE).
