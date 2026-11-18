"""Frozen paired classification and repair-region attention analyses."""

from collections import Counter
import csv
import math
from pathlib import Path
import statistics

import numpy as np

from .artifacts import read_json, read_jsonl, write_json, write_jsonl
from .config import SEEDS, SPLITS, result_path
from .data import load_rows, make_feature
from .evaluation import regional_emphasis
from .metrics import paired_metrics
from .model import load_tokenizer

BACKBONE = "unixcoder"
DATASET = "primevul_paired"
METHODS = ("rgcl_no_repair_guidance", "rgcl")
STATES = ("p_c", "p_v", "p_b", "p_r")
VISIBILITY = ("full", "partial", "invisible", "no_direct_diff")
REGION_METHODS = ("rgcl_no_repair_guidance", "random_region", "rgcl")


def line_token_mass(feature):
    """Match bounded_pool: a cross-line Token contributes 1/copies per line."""
    members = [set(t for t in ts if feature["code_mask"][t])
               for ts in feature["line_token_indices"]]
    copies = Counter(t for ts in members for t in ts)
    if set(copies) != {t for t, valid in enumerate(feature["code_mask"]) if valid}:
        raise ValueError("physical lines do not cover the valid code Tokens")
    return [math.fsum(1 / copies[t] for t in ts) for ts in members]


def coverage_record(row, feature):
    gt = set(row["lines_ground_truth"])
    numbers = feature["line_numbers"]
    complete = {n for n, ok in zip(numbers, feature["line_complete"]) if ok}
    visible = gt.intersection(numbers)
    category = ("no_direct_diff" if not gt else "full" if gt <= complete
                else "partial" if visible else "invisible")
    counts = line_token_mass(feature)
    foreground = math.fsum(c for n, c in zip(numbers, counts) if n in gt)
    total = math.fsum(counts)
    actual = math.fsum(c for c, y in zip(counts, feature["line_labels"]) if y == 1)
    if not math.isclose(actual, foreground, abs_tol=1e-9):
        raise ValueError("PrimeVul effective labels differ from the historical Diff rule")
    eligible = foreground > 0 and total - foreground > 0
    selected_tokens = {t for n, ts in zip(numbers, feature["line_token_indices"])
                       if n in gt for t in ts if feature["code_mask"][t]}
    return dict(sample_key=row["sample_key"], pair_key=row["pair_key"], target=row["target"],
        visibility=category, source_tokens=feature["source_tokens"], code_tokens=total,
        truncated=feature["source_tokens"] > 510, diff_lines=sorted(gt),
        visible_diff_lines=sorted(visible), fully_visible_diff_lines=sorted(gt & complete),
        partial_diff_lines=sorted(visible - complete), line_numbers=numbers,
        line_complete=feature["line_complete"], effective_token_mass=counts,
        visible_region_token_mass=foreground, visible_region_distinct_tokens=len(selected_tokens),
        background_token_mass=total - foreground, eligible=eligible,
        effective_region_token_mass=foreground if eligible else 0.,
        effective_background_token_mass=total - foreground if eligible else 0.)


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, std=None, quantiles={})
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std(ddof=0)),
                quantiles={str(q): float(np.quantile(a, q)) for q in (0, .05, .25, .5, .75, .95, 1)})


def summarize_coverage(records):
    result = {}
    for side, target in (("all", None), ("vulnerable", 1), ("repaired", 0)):
        rs = [r for r in records if target is None or r["target"] == target]
        categories = {k: sum(r["visibility"] == k for r in rs) for k in VISIBILITY}
        labeled = len(rs) - categories["no_direct_diff"]
        result[side] = dict(functions=len(rs), direct_diff_functions=labeled,
            category_counts=categories,
            category_fraction_all={k: n / len(rs) if rs else None for k, n in categories.items()},
            category_fraction_nonempty={k: categories[k] / labeled if labeled else None
                                       for k in VISIBILITY[:3]},
            eligible_functions=sum(r["eligible"] for r in rs),
            eligible_with_partial_diff_line=sum(r["eligible"] and bool(r["partial_diff_lines"]) for r in rs),
            all_visible_code_is_region=sum(r["visible_region_token_mass"] > 0 and not r["eligible"] for r in rs),
            truncated_functions=sum(r["truncated"] for r in rs),
            diff_lines=sum(len(r["diff_lines"]) for r in rs),
            visible_diff_lines=sum(len(r["visible_diff_lines"]) for r in rs),
            fully_visible_diff_lines=sum(len(r["fully_visible_diff_lines"]) for r in rs),
            partial_diff_lines=sum(len(r["partial_diff_lines"]) for r in rs),
            token_sums={k: math.fsum(r[k] for r in rs) for k in (
                "code_tokens", "visible_region_token_mass", "visible_region_distinct_tokens",
                "effective_region_token_mass", "effective_background_token_mass")},
            eligible_region_tokens=distribution([r["effective_region_token_mass"] for r in rs if r["eligible"]]))
    return result


def align_predictions(rows, predictions, coverage):
    """Join unique sample IDs and check full pair keys, labels and source mapping."""
    if len(rows) != len(coverage) or any(r["sample_key"] != c["sample_key"] for r, c in zip(rows, coverage)):
        raise ValueError("coverage records do not match the source cohort")
    expected = {r["sample_key"] for r in rows}
    by_key = {r["sample_key"]: r for r in predictions}
    if len(expected) != len(rows) or len(by_key) != len(predictions) or set(by_key) != expected:
        raise ValueError("prediction IDs are duplicated, missing or outside the official cohort")
    result = []
    for row, cov in zip(rows, coverage):
        p = by_key[row["sample_key"]]
        if p["pair_key"] != row["pair_key"] or p["target"] != row["target"]:
            raise ValueError("prediction full pair_key/label differs")
        value = p["probability"]
        if not math.isfinite(value) or not 0 <= value <= 1 or p["prediction"] != int(value > .5):
            raise ValueError("invalid probability or changed fixed decision rule")
        if (p["line_numbers"] != cov["line_numbers"] or
                len(p["effective_token_mass"]) != len(cov["effective_token_mass"]) or
                not np.allclose(p["effective_token_mass"], cov["effective_token_mass"], rtol=1e-6, atol=1e-5)):
            raise ValueError("prediction physical lines/Token counts differ from source encoding")
        result.append(p)
    paired_metrics(rows, [p["probability"] for p in result])
    return result


def pair_state(a, b):
    return "p_c" if a > .5 and b <= .5 else "p_v" if a > .5 else "p_r" if b > .5 else "p_b"


def classification_records(rows, predictions, seed):
    """One official repair pair per row; no ranking or gap-based decisions."""
    for method in METHODS:
        if len(predictions[method]) != len(rows):
            raise ValueError("complete prediction cohort required")
        paired_metrics(rows, [p["probability"] for p in predictions[method]])
    records, keys = [], set()
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        if a["pair_key"] in keys:
            raise ValueError("duplicate full pair_key")
        keys.add(a["pair_key"])
        record = dict(seed=seed, pair_key=a["pair_key"], sample_keys=[a["sample_key"], b["sample_key"]])
        for method in METHODS:
            ps = predictions[method][i:i+2]
            if any(p["sample_key"] != r["sample_key"] or p["pair_key"] != r["pair_key"] or
                   p["target"] != r["target"] or p["prediction"] != int(p["probability"] > .5)
                   for p, r in zip(ps, (a, b))):
                raise ValueError("prediction member, pair, label or fixed threshold differs")
            pv, pr = [p["probability"] for p in ps]
            record[method] = dict(vulnerable_probability=pv, repaired_probability=pr,
                vulnerable_prediction=int(pv > .5), repaired_prediction=int(pr > .5), state=pair_state(pv, pr))
        record["transition"] = record["rgcl_no_repair_guidance"]["state"] + "->" + record["rgcl"]["state"]
        records.append(record)
    return records


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_classification(records):
    """Count pair states and function error corrections on ONE seed/cohort."""
    if not records or len({r["seed"] for r in records}) != 1:
        raise ValueError("summarize exactly one nonempty seed, never pooled seed repetitions")
    n = len(records)
    if len({r["pair_key"] for r in records}) != n:
        raise ValueError("duplicate repair pairs")
    matrix = [[0] * 4 for _ in STATES]
    events = Counter(corrected_fp=0, new_fp=0, corrected_fn=0, new_fn=0)
    for r in records:
        c, g = [r[m] for m in METHODS]
        for p in (c, g):
            if (p["state"] != pair_state(p["vulnerable_probability"], p["repaired_probability"]) or
                p["vulnerable_prediction"] != int(p["vulnerable_probability"] > .5) or
                p["repaired_prediction"] != int(p["repaired_probability"] > .5)):
                raise ValueError("pair state differs from strict function classifications")
        matrix[STATES.index(c["state"])][STATES.index(g["state"])] += 1
        events.update(corrected_fp=int(c["repaired_prediction"] == 1 and g["repaired_prediction"] == 0),
            new_fp=int(c["repaired_prediction"] == 0 and g["repaired_prediction"] == 1),
            corrected_fn=int(c["vulnerable_prediction"] == 0 and g["vulnerable_prediction"] == 1),
            new_fn=int(c["vulnerable_prediction"] == 1 and g["vulnerable_prediction"] == 0))
    row_totals = list(map(sum, matrix))
    column_totals = [sum(row[j] for row in matrix) for j in range(4)]
    methods = {}
    for method, counts in zip(METHODS, (row_totals, column_totals)):
        methods[method] = dict(pair_counts=dict(zip(STATES, counts)), p_c=counts[0] / n,
            recall=(counts[0]+counts[1]) / n, patched_fpr=(counts[1]+counts[3]) / n,
            false_positives=counts[1]+counts[3], false_negatives=counts[2]+counts[3])
    gains = {state: matrix[i][0] for i, state in enumerate(STATES) if i}
    losses = {state: matrix[0][j] for j, state in enumerate(STATES) if j}
    gained, lost = sum(gains.values()), sum(losses.values())
    before, after = [methods[m] for m in METHODS]
    if (sum(row_totals) != n or sum(column_totals) != n or
        gained-lost != column_totals[0]-row_totals[0] or
        events["new_fp"]-events["corrected_fp"] != after["false_positives"]-before["false_positives"] or
        events["new_fn"]-events["corrected_fn"] != after["false_negatives"]-before["false_negatives"]):
        raise ValueError("transition/error accounting is inconsistent")
    opportunities = dict(corrected_fp=before["false_positives"], new_fp=n-before["false_positives"],
                         corrected_fn=before["false_negatives"], new_fn=n-before["false_negatives"])
    return dict(seed=records[0]["seed"], pairs=n, matrix=matrix, row_totals=row_totals,
        column_totals=column_totals, methods=methods,
        fraction_all=[[value/n for value in row] for row in matrix],
        fraction_source=[[fraction(value, row_totals[i]) for value in row] for i, row in enumerate(matrix)],
        pc=dict(gained=gained, lost=lost, net=gained-lost, retained=matrix[0][0],
            gained_fraction=gained/n, lost_fraction=lost/n, net_fraction=(gained-lost)/n,
            loss_fraction_of_original_pc=fraction(lost, row_totals[0]),
            gain_sources=gains, gain_source_shares={k:fraction(v, gained) for k,v in gains.items()},
            loss_destinations=losses),
        errors={k:dict(count=events[k], fraction_all=events[k]/n, opportunities=opportunities[k],
                       fraction_opportunity=fraction(events[k], opportunities[k])) for k in events},
        net_fp=events["new_fp"]-events["corrected_fp"], net_fn=events["new_fn"]-events["corrected_fn"])


def mean_sd(values):
    # An undefined conditional rate is not imputed or silently removed.
    return dict(mean=statistics.mean(values), std=statistics.stdev(values)) if all(
        v is not None for v in values) else dict(mean=None, std=None)


def aggregate_classification(summaries, records):
    if sorted(r["seed"] for r in summaries) != list(SEEDS):
        raise ValueError("aggregate exactly seeds 42, 1234 and 123456")
    cohorts = [{(r["pair_key"], tuple(r["sample_keys"])) for r in records if r["seed"] == seed} for seed in SEEDS]
    if not cohorts[0] or any(c != cohorts[0] for c in cohorts[1:]) or any(r["pairs"] != len(cohorts[0]) for r in summaries):
        raise ValueError("the three seeds must repeat exactly the same cohort")
    return dict(seeds=list(SEEDS), pairs_per_seed=len(cohorts[0]), std_definition="sample SD, ddof=1; n=3 seeds on the same cohort",
        matrix=[[mean_sd([r["matrix"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        fraction_all=[[mean_sd([r["fraction_all"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        fraction_source=[[mean_sd([r["fraction_source"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        methods={m:{k:mean_sd([r["methods"][m][k] for r in summaries])
                    for k in ("p_c", "recall", "patched_fpr", "false_positives", "false_negatives")} for m in METHODS},
        pc={k:mean_sd([r["pc"][k] for r in summaries]) for k in ("gained", "lost", "net", "retained",
            "gained_fraction", "lost_fraction", "net_fraction", "loss_fraction_of_original_pc")},
        gain_sources={s:mean_sd([r["pc"]["gain_sources"][s] for r in summaries]) for s in STATES[1:]},
        gain_source_shares={s:mean_sd([r["pc"]["gain_source_shares"][s] for r in summaries]) for s in STATES[1:]},
        errors={k:{field:mean_sd([r["errors"][k][field] for r in summaries])
                   for field in ("count", "fraction_all", "fraction_opportunity")} for k in summaries[0]["errors"]},
        net_fp=mean_sd([r["net_fp"] for r in summaries]), net_fn=mean_sd([r["net_fn"] for r in summaries]))


def write_tables(root, result):
    with (root/"classification_transitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seed", "from_state", "to_state", "count", "all_pairs", "fraction_all", "source_pairs", "fraction_source"])
        for r in result["summaries"]:
            for i, before in enumerate(STATES):
                for j, after in enumerate(STATES):
                    writer.writerow([r["seed"], before, after, r["matrix"][i][j], r["pairs"], r["fraction_all"][i][j],
                                     r["row_totals"][i], r["fraction_source"][i][j]])
    with (root/"classification_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        keys = ("gained", "lost", "net", "gained_fraction", "lost_fraction", "net_fraction")
        writer.writerow(["seed", "pairs", *[f"pc_{k}" for k in keys],
            *[f"{m}_{k}" for m in METHODS for k in ("p_c", "recall", "patched_fpr")],
            *result["summaries"][0]["errors"], "net_fp", "net_fn"])
        for r in result["summaries"]:
            writer.writerow([r["seed"], r["pairs"], *[r["pc"][k] for k in keys],
                *[r["methods"][m][k] for m in METHODS for k in ("p_c", "recall", "patched_fpr")],
                *[v["count"] for v in r["errors"].values()], r["net_fp"], r["net_fn"]])


def region_attention_records(rows, mapping, predictions, seed, method):
    """Partition already predicted multipliers with the formal Token-weighted statistic."""
    if method not in REGION_METHODS:
        raise ValueError("RQ4 compares RGCL w/o repair guidance, Random-region and RGCL")
    aligned = align_predictions(rows, predictions, mapping)
    records = []
    for row, cov, pred in zip(rows, mapping, aligned):
        value = regional_emphasis(dict(row, lines_ground_truth=cov["diff_lines"]), pred)
        if value["eligible"] != cov["eligible"]:
            raise ValueError("model output changed the source-defined eligible cohort")
        if value["eligible"]:
            records.append(dict(seed=seed, method=method, sample_key=row["sample_key"],
                pair_key=row["pair_key"], target=row["target"],
                region_token_mass=value["region_token_mass"],
                background_token_mass=value["background_token_mass"],
                ratio=value["emphasis_ratio"], region_margin=value["region_margin"]))
    return records


def summarize_region_attention(records):
    """Repeat one fixed function cohort; aggregate seeds, never 3*n independent rows."""
    groups = {(s, m): [r for r in records if r["seed"] == s and r["method"] == m]
              for s in SEEDS for m in REGION_METHODS}
    if sum(map(len, groups.values())) != len(records):
        raise ValueError("unexpected RQ4 seed or method")
    cohort = None
    for rs in groups.values():
        keys = {(r["sample_key"], r["pair_key"], r["target"], r["region_token_mass"],
                 r["background_token_mass"]) for r in rs}
        if not rs or len(keys) != len(rs) or len({r["sample_key"] for r in rs}) != len(rs):
            raise ValueError("nonempty unique function cohort required for every seed/method")
        if cohort is not None and keys != cohort:
            raise ValueError("all nine runs must use exactly the same cohort and Token masses")
        cohort = keys
        if any(not math.isfinite(r["ratio"]) or not .2 <= r["ratio"] <= 5
               or min(r["region_token_mass"], r["background_token_mass"]) <= 0 for r in rs):
            raise ValueError("invalid K=5 region ratio or empty region/background")
    summaries, comparisons = [], []
    for seed in SEEDS:
        for method in REGION_METHODS:
            values = [r["ratio"] for r in groups[seed, method]]
            summaries.append(dict(seed=seed, method=method, **distribution(values),
                median=float(np.median(values)), fraction_above_one=sum(x>1 for x in values)/len(values),
                fraction_at_least_two=sum(x>=2 for x in values)/len(values)))
        rgcl = {r["sample_key"]: r["ratio"] for r in groups[seed, "rgcl"]}
        for other in REGION_METHODS[:2]:
            differences = [rgcl[r["sample_key"]]-r["ratio"] for r in groups[seed, other]]
            comparisons.append(dict(seed=seed, comparison="rgcl-"+other, **distribution(differences),
                median=float(np.median(differences)),
                fraction_increased=sum(x>0 for x in differences)/len(differences)))
    return dict(functions_per_run=len(cohort), summaries=summaries, comparisons=comparisons,
        aggregate=dict(std_definition="sample SD, ddof=1, n=3 seeds on the same functions",
            methods={m: {k: mean_sd([r[k] for r in summaries if r["method"]==m])
                         for k in ("mean", "median", "fraction_above_one", "fraction_at_least_two")}
                     for m in REGION_METHODS},
            comparisons={"rgcl-"+m: {k: mean_sd([r[k] for r in comparisons if r["comparison"]=="rgcl-"+m])
                                    for k in ("mean", "median", "fraction_increased")}
                         for m in REGION_METHODS[:2]}))


def write_region_tables(root, result):
    """Summaries of the complete function cohort for each seed."""

    columns = ("seed", "method", "n", "mean", "median", "std", "fraction_above_one", "fraction_at_least_two")
    with (root/"region_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*columns, "q05", "q25", "q75", "q95", "min", "max"])
        for r in result["summaries"]:
            writer.writerow([*[r[k] for k in columns], *[r["quantiles"][q] for q in ("0.05", "0.25", "0.75", "0.95", "0", "1")]])
    with (root/"paired_changes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        columns = ("seed", "comparison", "n", "mean", "median", "fraction_increased")
        writer.writerow(columns)
        writer.writerows([r[k] for k in columns] for r in result["comparisons"])


def prime_predictions(results_dir, rows, methods):
    """Load the fixed Test predictions and verify their recorded selection and counts."""
    snapshots = {}
    for seed in SEEDS:
        for method in methods:
            root = result_path(results_dir, BACKBONE, DATASET, seed, method)
            selected = read_json(root / "selection.json")
            saved = read_json(root / "test.metrics.json")
            predictions = read_jsonl(root / "test.predictions.jsonl")
            if saved["selected_epoch"] != selected["epoch"]:
                raise ValueError(f"{root}: Test and Valid selection disagree")
            # align_predictions will validate the physical coordinates after tokenization.
            snapshot = {p["sample_key"]: p for p in predictions}
            if len(snapshot) != len(predictions) or snapshot.keys() != {r["sample_key"] for r in rows}:
                raise ValueError(f"{root}: Test membership differs")
            predictions = [snapshot[r["sample_key"]] for r in rows]
            values = paired_metrics(rows, [p["probability"] for p in predictions])
            if any(values[k] != saved["metrics"][k] for k in values):
                raise ValueError(f"{root}: saved Test metrics differ")
            snapshots[seed, method] = dict(predictions=predictions, metrics=values)
    return snapshots


def rq3(data_dir, prepared_dir, results_dir, output_dir, model_source=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(BACKBONE, model_source)
    coverage = {}
    for split in SPLITS:
        rows = load_rows(data_dir, prepared_dir, DATASET, split, with_diff=True)
        features = [make_feature(row, tokenizer) for row in rows]
        mapping = [coverage_record(row, feature) for row, feature in zip(rows, features, strict=True)]
        pair_indices = range(0, len(rows), 2)
        coverage[split] = dict(summarize_coverage(mapping), pairs=len(rows) // 2,
            identical_visible_pairs=sum(features[i]["input_ids"] == features[i + 1]["input_ids"] for i in pair_indices),
            both_eligible_pairs=sum(mapping[i]["eligible"] and mapping[i + 1]["eligible"] for i in pair_indices),
            either_eligible_pairs=sum(mapping[i]["eligible"] or mapping[i + 1]["eligible"] for i in pair_indices))
        del features
    if len(rows) != 1128 or len({r["pair_key"] for r in rows}) != 564:
        raise ValueError("RQ3 requires all 564 official Test pairs")
    snapshots = prime_predictions(results_dir, rows, METHODS)
    all_records, summaries = [], []
    for seed in SEEDS:
        predictions = {m: align_predictions(rows, snapshots[seed, m]["predictions"], mapping) for m in METHODS}
        records = classification_records(rows, predictions, seed)
        summary = summarize_classification(records)
        for method in METHODS:
            original = snapshots[seed, method]["metrics"]
            if any(summary["methods"][method][k] != original[k]
                   for k in ("pair_counts", "p_c", "recall", "patched_fpr")):
                raise ValueError("Transition margins differ from the recorded paired metrics")
        all_records.extend(records)
        summaries.append(summary)
    result = dict(states=STATES, summaries=summaries, aggregate=aggregate_classification(summaries, all_records),
                  coverage=coverage, threshold=.5, comparison=">")
    write_json(root / "coverage.json", coverage)
    write_json(root / "rq3.json", result)
    write_jsonl(root / "paired_classifications.jsonl", all_records)
    write_tables(root, result)
    return dict(pairs=564, seeds=list(SEEDS), pc_changes=[r["pc"]["net"] for r in summaries], output=str(root))


def rq4(data_dir, prepared_dir, results_dir, output_dir, model_source=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(BACKBONE, model_source)
    rows = load_rows(data_dir, prepared_dir, DATASET, "test", with_diff=True)
    snapshots = prime_predictions(results_dir, rows, REGION_METHODS)
    mapping = [coverage_record(row, make_feature(row, tokenizer)) for row in rows]
    records = []
    for (seed, method), snapshot in snapshots.items():
        records.extend(region_attention_records(rows, mapping, snapshot["predictions"], seed, method))
    result = summarize_region_attention(records)
    write_json(root / "rq4.json", result)
    write_jsonl(root / "function_regions.jsonl", records)
    write_region_tables(root, result)
    return dict(functions_per_run=result["functions_per_run"], output=str(root))
