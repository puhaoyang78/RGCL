# Download the datasets

Put the source files under `data/`, or set `DATA_DIR` in the main README
to a directory of your choice. Start with PrimeVul for the single-model
example and for RQ2–RQ4. The full RQ1 comparison uses all three datasets.

## PrimeVul

Use the three paired files from the original release. They give 4,354
training pairs, 562 validation pairs, and 564 test pairs. Each pair is
ordered vulnerable function first, repaired function second. The v0.1
release has different members. [Author release notes](https://github.com/DLVulDet/PrimeVul#original-release)
· [Original download](https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY?usp=drive_link)

Put the downloaded files in this layout:

```text
data/PrimeVul_original/
  primevul_train_paired.jsonl
  primevul_valid_paired.jsonl
  primevul_test_paired.jsonl
```

## SecVulEval

Use the pinned C/C++ snapshot with 25,440 functions and the original
binary labels. [Authors' repository](https://github.com/basimbd/SecVulEval)
· [Dataset snapshot](https://huggingface.co/datasets/arag0rn/SecVulEval/tree/bde3e7c3225cc61c9ad727133737255428a2ea3d)

```bash
DATA_DIR=data
hf download arag0rn/SecVulEval data/train-00000-of-00001.parquet \
  --repo-type dataset --revision bde3e7c3225cc61c9ad727133737255428a2ea3d \
  --local-dir "$DATA_DIR/SecVulEval"
```

## CleanVul

Use score tiers 3 and 4 from the pinned release. Preprocessing keeps
the C/C++ records and takes one function before and one after each of the
3,136 repairs. [Dataset snapshot](https://huggingface.co/datasets/yikun-li/CleanVul/tree/74b2fda1c51eea0f99987df6924ca60c6e9f92d2)

```bash
DATA_DIR=data
hf download yikun-li/CleanVul vulnerability_score_3.csv vulnerability_score_4.csv \
  --repo-type dataset --revision 74b2fda1c51eea0f99987df6924ca60c6e9f92d2 \
  --local-dir "$DATA_DIR/cleanvul"
```

## Prepare the paper splits

Run the [preparation command](../README.md#install-and-prepare-the-data)
once for each downloaded dataset. It reads the source files in place
and writes membership lists and repair annotations to `data/prepared/`.
Use these counts to check the output:

| Dataset | Train functions | Validation functions | Test functions |
|---|---:|---:|---:|
| PrimeVul | 8,708 | 1,124 | 1,128 |
| SecVulEval | 20,352 | 2,544 | 2,544 |
| CleanVul | 5,016 | 628 | 628 |

PrimeVul keeps the author splits. SecVulEval and CleanVul are split once
with seed 42. Groups keep shared CVEs, repair commits, linked functions,
and whitespace-normalized duplicate source together.

Preprocessing compares both versions of each repair and records the
nonblank changed lines on each side. SecVulEval uses a repair link when
the commit, file, and function identity match and the source changes.
When a repaired function has several valid predecessors, the changed
lines must agree on the repaired side.

The input contains the first 510 source tokens, plus start and end tokens
and padding. Tokens that span several lines share a total mass of one.
Whitespace-only tokens take up space in the input but give no weight to
a line. PrimeVul uses the visible part of a changed line that gets cut off.
SecVulEval and CleanVul use complete visible edit lines with reliable repair
links. The region loss needs both a marked region and surrounding code
with positive token mass. Every function keeps its classification label.
