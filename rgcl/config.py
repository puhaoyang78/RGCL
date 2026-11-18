"""The fixed configurations used in the paper."""

from pathlib import Path

BACKBONES = {
    "unixcoder": ("microsoft/unixcoder-base-nine", "5f0dc256eb0904af496d427451da5d7f90261676"),
    "codebert": ("microsoft/codebert-base", "3b0952feddeffad0063f274080e3c23d75e7eb39"),
    "codet5": ("Salesforce/codet5-base", "02cd2d31bb7c6d0e4d91156167b2de044989c733"),
}
DATASETS = {
    "primevul_paired": (8708, 1124, 1128),
    "secvuleval": (20352, 2544, 2544),
    "cleanvul": (5016, 628, 628),
}
SPLITS = ("train", "valid", "test")
SEEDS = (42, 1234, 123456)
MAIN_METHODS = ("plm_classifier", "rgcl_no_repair_guidance", "rgcl")
SIDE_TARGETS = {"vulnerable_only": 1, "repaired_only": 0}
ABLATIONS = ("random_region", *SIDE_TARGETS, "rgcl_no_interaction")
METHODS = (*MAIN_METHODS, *ABLATIONS)
REGION_METHODS = ("rgcl", *ABLATIONS)
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 16
LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8
WARMUP_FRACTION = .1
MAX_GRAD_NORM = 1.
REGIONAL_LOSS_WEIGHT = .1
MAX_EMPHASIS_FACTOR = 5


def check_run(backbone, dataset, seed, method):
    if backbone not in BACKBONES or dataset not in DATASETS or seed not in SEEDS or method not in METHODS:
        raise ValueError("Unknown paper experiment")
    if method in ABLATIONS and (backbone, dataset) != ("unixcoder", "primevul_paired"):
        raise ValueError("The formal ablations use UniXcoder and original paired PrimeVul")


def selected_by(dataset):
    return "p_c" if dataset == "primevul_paired" else "mcc"


def result_path(root, backbone, dataset, seed, method):
    family = "rq2" if method in ABLATIONS else "main"
    return Path(root) / family / backbone / dataset / f"seed{seed}" / method
