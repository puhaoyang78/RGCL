"""Original release membership, one 512-token view, and direct repair masks."""

from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from difflib import Differ
import math
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .artifacts import read_json, read_jsonl, write_json, write_jsonl
from .config import BATCH_SIZE, DATASETS, MAX_LENGTH, SPLITS

CVE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)
CC = {"c", "h", "cpp", "cc", "cxx", "c++", "hh", "hpp", "hxx"}
SECVUL_FILE = Path("SecVulEval/data/train-00000-of-00001.parquet")


def modified_lines(source, counterpart):
    """Return direct edits on one side of bidirectional edit extraction."""
    position, result = 0, []
    for line in Differ().compare(source.splitlines(keepends=True), counterpart.splitlines(keepends=True)):
        if line.startswith("- ") and line[2:].strip():
            result.append(position)
        if line.startswith(("- ", "  ")):
            position += 1
    return result


def prime_file(root, split):
    return Path(root) / "PrimeVul_original" / f"primevul_{split}_paired.jsonl"


def read_secvuleval(root, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(Path(root) / SECVUL_FILE, columns=columns).to_pylist()


def read_cleanvul(root, score):
    csv.field_size_limit(100_000_000)
    with (Path(root) / "cleanvul" / f"vulnerability_score_{score}.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)
        required = {"func_before", "func_after", "commit_url", "commit_msg",
                    "file_name", "vulnerability_score", "extension", "cve_id", "is_test"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing CleanVul columns")
        return list(reader)


def prime_source_rows(root, split):
    raw = read_jsonl(prime_file(root, split))
    count = DATASETS["primevul_paired"][SPLITS.index(split)]
    if len(raw) != count or [row["target"] for row in raw] != [1, 0] * (count // 2):
        raise ValueError(f"{split}: use the original PrimeVul paired release")
    prefix = f"primevul_paired:{split}:"
    for i, row in enumerate(raw):
        if not isinstance(row["func"], str) or not row["func"].strip():
            raise ValueError(f"{split}:{i + 1}: empty function")
        row.update(sample_key=prefix + str(i + 1), pair_key=prefix + str(i // 2), source_row=i + 1)
    return raw


def prepare_primevul(root):
    splits = {}
    for split in SPLITS:
        raw = prime_source_rows(root, split)
        out = []
        for i, row in enumerate(raw):
            index = {k: row.get(k) for k in
                     ("sample_key", "pair_key", "target", "idx", "source_row", "project", "commit_id")}
            index["lines_ground_truth"] = modified_lines(row["func"], raw[i ^ 1]["func"])
            out.append(index)
        splits[split] = out
    return splits


def secvuleval_members(root):
    native = read_secvuleval(root, ["idx", "fixed_func_idx", "is_vulnerable", "project_url",
        "filepath", "func_name", "commit_id", "func_body", "cve_list"])
    by_id = {r["idx"]: r for r in native}
    if len(by_id) != len(native):
        raise ValueError("SecVulEval idx must be unique")
    reliable, reverse = {}, defaultdict(list)
    for row in native:
        if type(row["is_vulnerable"]) is not bool:
            raise ValueError("SecVulEval requires native boolean labels")
        if not row["is_vulnerable"]:
            continue
        fixed = row["fixed_func_idx"]
        other = by_id.get(int(fixed)) if fixed is not None and math.isfinite(fixed) and fixed == int(fixed) else None
        if (other is not None and not other["is_vulnerable"]
            and all(row[k] == other[k] for k in ("project_url", "filepath", "func_name", "commit_id"))
            and row["func_body"] != other["func_body"]):
            reliable[row["idx"]] = other["idx"]
            reverse[other["idx"]].append(row["idx"])
    rows, counterparts = [], {}
    for row in native:
        idx = row["idx"]
        target = int(row["is_vulnerable"])
        partner = reliable.get(idx)
        if partner is None and idx in reverse:
            candidates = reverse[idx]
            if len(candidates) == 1 or len({
                tuple(modified_lines(row["func_body"], by_id[i]["func_body"])) for i in candidates
            }) == 1:
                partner = candidates[0]
        key = f"secvuleval:{idx}"
        code = row["func_body"]
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"{key}: empty function")
        if not row["filepath"].endswith((".c", ".h", ".C", ".cc", ".cpp", ".cxx",
                                          ".hh", ".hpp", ".hxx", ".c++")):
            raise ValueError(f"{key}: unexpected language")
        rows.append(dict(sample_key=key, pair_key=key, target=target, func=code,
            project=row["project_url"], commit_id=row["commit_id"],
            cves=sorted(set(row["cve_list"] or [])), file_name=row["filepath"],
            func_name=row["func_name"], source_row=idx,
            normal_kind="vulnerable" if target else "repaired" if idx in reverse else "ordinary",
            repair_key=None if partner is None else f"secvuleval:{partner}"))
        if partner is not None:
            counterparts[key] = by_id[partner]["func_body"]
    return rows, counterparts


def cleanvul_members(root):
    rows, counterparts = [], {}
    commit_refs = defaultdict(set)
    # Only these published score tiers affect the paper membership.
    for score in (3, 4):
        for number, row in enumerate(read_cleanvul(root, score), 1):
            if int(row["vulnerability_score"]) != score:
                raise ValueError("CleanVul score differs from filename")
            language = row["extension"].lower().lstrip(".")
            if language not in CC:
                continue
            before, after = row["func_before"], row["func_after"]
            url = urlsplit(row["commit_url"])
            parts = url.path.strip("/").split("/")
            if (not before.strip() or not after.strip()
                or row["is_test"].lower() != "false"
                or url.scheme not in ("http", "https") or url.hostname != "github.com"
                or len(parts) != 4 or parts[2] != "commit" or not row["file_name"].strip()
                or not re.fullmatch(r"[0-9a-fA-F]{4,40}", parts[3])):
                raise ValueError(f"CleanVul score {score}, row {number}: malformed release record")
            project = f"https://github.com/{parts[0]}/{parts[1]}"
            commit = parts[3].lower()
            commit_refs[project].add(commit)
            pair = f"cleanvul:{score}:{number}"
            cves = sorted({m.upper() for m in CVE.findall(row["cve_id"] + "\n" + row["commit_msg"])})
            for target, side, code, other, mate in (
                (1, "before", before, after, "after"), (0, "after", after, before, "before")
            ):
                key = f"{pair}:{side}"
                rows.append(dict(sample_key=key, pair_key=pair, target=target, func=code,
                    project=project, commit_id=commit, cves=cves, file_name=row["file_name"],
                    source_row=number, vulnerability_score=score, language=language,
                    normal_kind="vulnerable" if target else "repaired", repair_key=f"{pair}:{mate}"))
                counterparts[key] = other
    for row in rows:
        refs = commit_refs[row["project"]]
        prefix = min((h for h in refs if row["commit_id"].startswith(h)), key=len)
        row["commit_group"] = row["project"] + "@" + prefix
    return rows, counterparts
def split_rgcl_groups(rows, counterparts, seed=42):
    """80/10/10 class targets; connected CVE/commit/repair/duplicate groups."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    import hashlib
    index, keys, duplicate_codes = {}, {r["sample_key"]: i for i, r in enumerate(rows)}, Counter()
    for i, r in enumerate(rows):
        values = [("cve", c.upper()) for c in r["cves"] if c]
        if r["commit_id"]:
            values.append(("commit", r["commit_id"]))
        for code in (r["func"], counterparts.get(r["sample_key"])):
            if code is not None:
                digest = hashlib.sha256(" ".join(code.split()).encode()).digest()
                values.append(("code", digest))
        duplicate_codes[hashlib.sha256(" ".join(r["func"].split()).encode()).digest()] += 1
        for value in values:
            if value in index:
                union(i, index[value])
            else:
                index[value] = i
        if r["repair_key"] in keys:
            union(i, keys[r["repair_key"]])
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -len(groups[k]))
    total = Counter(r["target"] for r in rows)
    targets = {s: {y: total[y] * fraction for y in (0, 1)} for s, fraction in zip(SPLITS, (.8, .1, .1))}
    counts = {s: Counter() for s in SPLITS}
    for key in order:
        labels = Counter(rows[i]["target"] for i in groups[key])
        cost = lambda s: sum(((counts[s][y] + labels[y] - targets[s][y]) ** 2 -
                              (counts[s][y] - targets[s][y]) ** 2) / max(1, targets[s][y]) for y in (0, 1))
        split = min(SPLITS, key=cost)
        counts[split].update(labels)
        for i in groups[key]:
            rows[i].update(split=split, group_key=f"group:{key}")
    if any(not counts[s][0] or not counts[s][1] for s in SPLITS):
        raise ValueError("connected groups cannot provide both labels in every split")
    # Recheck every grouping key; no CVE, commit, counterpart or duplicate may cross.
    for i, r in enumerate(rows):
        assert rows[find(i)]["split"] == r["split"]
        if r["repair_key"] in keys:
            assert r["split"] == rows[keys[r["repair_key"]]]["split"]
    return dict(seed=seed, target_ratios=[.8, .1, .1], groups=len(groups),
        largest_groups=sorted((len(g) for g in groups.values()), reverse=True)[:10],
        grouping="transitive union of CVE, fix commit, repair link and whitespace-normalized source (including counterparts)",
        cross_split_group_overlap=0, members_removed=0,
        duplicate_code_groups=sum(n > 1 for n in duplicate_codes.values()))



def prepare_native(root, dataset):
    rows, counterparts = secvuleval_members(root) if dataset == "secvuleval" else cleanvul_members(root)
    grouped = (
        [dict(row, commit_id=row["commit_group"]) for row in rows] if dataset == "cleanvul" else rows
    )
    split_rgcl_groups(grouped, counterparts, seed=42)
    splits = {split: [] for split in SPLITS}
    for row, assigned in zip(rows, grouped):
        other = counterparts.get(row["sample_key"])
        index = {key: value for key, value in row.items() if key != "func"}
        index.update(split=assigned["split"], group_key=assigned["group_key"], diff_reliable=other is not None,
                     lines_ground_truth=modified_lines(row["func"], other) if other is not None else [])
        splits[index["split"]].append(index)
    return splits


def prepare(data_dir, prepared_dir, dataset):
    splits = prepare_primevul(data_dir) if dataset == "primevul_paired" else prepare_native(data_dir, dataset)
    expected = DATASETS[dataset]
    if tuple(len(rows) for rows in splits.values()) != expected:
        raise ValueError(f"{dataset}: split sizes differ from the paper's release {expected}")
    dest = Path(prepared_dir) / dataset
    if any((dest / f"{split}{suffix}.jsonl").exists()
           for split in SPLITS for suffix in ("", ".edits")):
        raise FileExistsError(f"{dest}: choose a new prepared directory")
    counts = {split: dict(Counter(r["target"] for r in rows)) for split, rows in splits.items()}
    edit_fields = {"lines_ground_truth", "diff_reliable"}
    for split, rows in splits.items():
        # Inference reads membership only. Training and post-hoc analyses join edits by key.
        write_jsonl(dest / f"{split}.jsonl",
                    [{k: v for k, v in row.items() if k not in edit_fields} for row in rows])
        write_jsonl(dest / f"{split}.edits.jsonl",
                    [{k: v for k, v in row.items() if k in edit_fields | {"sample_key"}} for row in rows])
    result = dict(dataset=dataset, counts=counts,
                  classification_positive_weight=counts["train"][0] / counts["train"][1])
    write_json(dest / "statistics.json", result)
    return result


def load_rows(data_dir, prepared_dir, dataset, split, *, with_diff=False):
    """Resolve prepared indices against the original release without storing source copies."""
    rows = read_jsonl(Path(prepared_dir) / dataset / f"{split}.jsonl")
    if len(rows) != DATASETS[dataset][SPLITS.index(split)] or len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError(f"{dataset}/{split}: incomplete or duplicate membership")
    if dataset == "primevul_paired":
        raw = read_jsonl(prime_file(data_dir, split))
        if len(raw) != len(rows):
            raise ValueError("Wrong PrimeVul release")
        for row, source in zip(rows, raw):
            if (row["idx"], row["target"]) != (source["idx"], source["target"]):
                raise ValueError(f"{row['sample_key']}: original member differs")
            row["func"] = source["func"]
    elif dataset == "secvuleval":
        raw = {r["idx"]: r for r in read_secvuleval(data_dir, ["idx", "is_vulnerable", "commit_id", "func_body"])}
        for row in rows:
            source = raw[row["source_row"]]
            if row["target"] != int(source["is_vulnerable"]) or row["commit_id"] != source["commit_id"]:
                raise ValueError(f"{row['sample_key']}: original member differs")
            row["func"] = source["func_body"]
    else:
        raw = {score: read_cleanvul(data_dir, score) for score in (3, 4)}
        for row in rows:
            source = raw[row["vulnerability_score"]][row["source_row"] - 1]
            if row["file_name"] != source["file_name"] or urlsplit(source["commit_url"]).path.strip("/").split("/")[-1].lower() != row["commit_id"]:
                raise ValueError(f"{row['sample_key']}: original member differs")
            row["func"] = source["func_before" if row["target"] else "func_after"]
    if with_diff:
        edits = read_jsonl(Path(prepared_dir) / dataset / f"{split}.edits.jsonl")
        for row, annotation in zip(rows, edits, strict=True):
            if annotation["sample_key"] != row["sample_key"]:
                raise ValueError(f"{dataset}/{split}: edit annotations differ from membership")
            row.update(annotation)
    return rows


def class_weight(prepared_dir, dataset):
    return read_json(Path(prepared_dir) / dataset / "statistics.json")["classification_positive_weight"]


def encode_source(code, tokenizer):
    """First 510 source tokens, visible parts, and zero-based physical line IDs."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with source offsets is required")
    encoded = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=False, verbose=False)
    starts, position = [], 0
    for line in code.splitlines(keepends=True):
        starts.append(position)
        position += len(line)
    memberships = [[] for _ in starts]
    special_ids = set(tokenizer.all_special_ids)
    for token, (token_id, (start, end)) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
        if token_id in special_ids or end <= start:
            continue
        line = max(0, bisect_right(starts, start) - 1)
        while line < len(starts) and starts[line] < end:
            a = max(start, starts[line])
            b = min(end, starts[line + 1] if line + 1 < len(starts) else len(code))
            if code[a:b].strip():
                memberships[line].append(token)
            line += 1
    source_count = len(encoded["input_ids"])
    visible_count = min(MAX_LENGTH - 2, source_count)
    input_ids = [tokenizer.cls_token_id, *encoded["input_ids"][:visible_count], tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids) + [0] * (MAX_LENGTH - len(input_ids))
    input_ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(input_ids))
    numbers, indices, complete = [], [], []
    code_mask = [False] * MAX_LENGTH
    for line, tokens in enumerate(memberships):
        visible = [t + 1 for t in tokens if t < visible_count]
        if visible:
            numbers.append(line)
            indices.append(visible)
            complete.append(tokens[-1] < visible_count)
            for t in visible:
                code_mask[t] = True
    if not any(code_mask):
        raise ValueError("no non-whitespace source tokens in the input")
    return dict(input_ids=input_ids, attention_mask=attention_mask, code_mask=code_mask,
                line_numbers=numbers, line_token_indices=indices, line_complete=complete,
                source_tokens=source_count)


def make_feature(row, tokenizer, *, encoding=None):
    f = encode_source(row["func"], tokenizer) if encoding is None else dict(encoding)
    f["labels"] = row["target"]
    if "lines_ground_truth" in row:
        changed = set(row["lines_ground_truth"])
        numbers = f["line_numbers"]
        f["line_labels"] = [int(n in changed) for n in numbers]
        if "diff_reliable" in row:
            complete = {n for n, ok in zip(numbers, f["line_complete"]) if ok}
            visible = changed.intersection(numbers)
            if not (row["diff_reliable"] and visible and visible.issubset(complete)):
                f["line_labels"] = [0] * len(numbers)
    return f


def collate(features):
    width = max(len(f["line_numbers"]) for f in features)
    numbers = torch.full((len(features), width), -1, dtype=torch.long)
    membership = torch.zeros(len(features), width, MAX_LENGTH, dtype=torch.bool)
    batch = {key: torch.tensor([f[key] for f in features])
             for key in ("input_ids", "attention_mask", "code_mask", "labels")}
    if "line_labels" in features[0]:
        batch["line_labels"] = torch.full((len(features), width), -1.)
    for i, feature in enumerate(features):
        numbers[i, :len(feature["line_numbers"])] = torch.tensor(feature["line_numbers"])
        for j, tokens in enumerate(feature["line_token_indices"]):
            membership[i, j, tokens] = True
        if "line_labels" in batch:
            batch["line_labels"][i, :len(feature["line_numbers"])] = torch.tensor(feature["line_labels"])
    return dict(batch, line_numbers=numbers, line_token_mask=membership)


def data_loader(features, *, train=False):
    sampler = RandomSampler(features) if train else SequentialSampler(features)
    return DataLoader(features, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=collate,
                      num_workers=4, pin_memory=True)
