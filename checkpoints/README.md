# Model checkpoints

The 93 validation-selected checkpoints are available on [ModelScope](https://modelscope.cn/models/udisns/RGCL).

| Experiment | Checkpoints |
| --- | --- |
| RQ1 | 81 runs covering `plm_classifier`, `rgcl_no_repair_guidance`, and `rgcl` across all three backbones and datasets |
| RQ2 | 12 additional UniXcoder runs on paired PrimeVul covering `random_region`, `vulnerable_only`, `repaired_only`, and `rgcl_no_interaction` |

Files follow the path `<backbone>/<dataset>/seed<seed>/<method>.safetensors`.
Use `unixcoder`, `codebert`, or `codet5` for the backbone and
`primevul_paired`, `secvuleval`, or `cleanvul` for the dataset.
The available seeds are `42`, `1234`, and `123456`.

## Download one checkpoint

Run these commands from the RGCL repository root. The example downloads
RGCL with UniXcoder on paired PrimeVul for seed 42.

```bash
python -m pip install modelscope
checkpoint_file=unixcoder/primevul_paired/seed42/rgcl.safetensors
modelscope download --model udisns/RGCL "$checkpoint_file" --local_dir checkpoints
```

## Run inference or evaluation

After installing RGCL, save a function as `function.c` and run the commands
below. [RGCL installation](../README.md#install-and-prepare-the-data)

```bash
checkpoint="checkpoints/$checkpoint_file"
DEVICE=cuda:0
python -m rgcl predict --checkpoint "$checkpoint" \
  --source function.c --device "$DEVICE"
```

To evaluate the same checkpoint on the PrimeVul test set, first download
the dataset. [Data download guide](../data/README.md)

```bash
python -m rgcl prepare --data-dir data --dataset primevul_paired
python -m rgcl evaluate --data-dir data --checkpoint "$checkpoint" \
  --split test --output-dir runs/checkpoint_eval --device "$DEVICE"
```
