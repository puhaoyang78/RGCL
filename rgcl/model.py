"""The paper's PLM classifier and context-conditioned line pooling."""

import json
import math

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, RobertaModel, RobertaTokenizerFast, T5Model
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from .config import BACKBONES, MAX_EMPHASIS_FACTOR, REGION_METHODS, REGIONAL_LOSS_WEIGHT, check_run, selected_by


def pretrained_source(backbone, model_source=None):
    model_id, revision = BACKBONES[backbone]
    return (model_source, {}) if model_source is not None else (model_id, {"revision": revision})


def load_tokenizer(backbone, model_source=None):
    source, options = pretrained_source(backbone, model_source)
    return RobertaTokenizerFast.from_pretrained(source, trim_offsets=False, **options)


class Detector(nn.Module):
    def __init__(self, encoder, config, method, class_weight):
        super().__init__()
        if not math.isfinite(class_weight) or class_weight <= 0:
            raise ValueError("Positive class weight must be finite and greater than zero")
        self.encoder = encoder
        self.config = config
        self.method = method
        self.class_weight = class_weight
        self.is_plm_classifier = method == "plm_classifier"
        self.line_function_interaction = method not in ("plm_classifier", "rgcl_no_interaction")
        # Keep the original classifier draw before initializing the method-specific heads.
        self.classifier = nn.Linear(config.hidden_size, 1)
        if self.is_plm_classifier:
            with torch.random.fork_rng(devices=[]):
                if config.model_type == "t5":
                    self.classifier = nn.Linear(config.hidden_size, 2)
                else:
                    head_config = type(config).from_dict(config.to_dict())
                    head_config.num_labels = 1
                    self.classifier = RobertaClassificationHead(head_config)
                    for layer in self.classifier.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.normal_(layer.weight, std=config.initializer_range)
                            nn.init.zeros_(layer.bias)
        else:
            with torch.random.fork_rng(devices=[]):
                self.line_scorer = nn.Linear(config.hidden_size, 1)
            nn.init.zeros_(self.line_scorer.weight)
            nn.init.zeros_(self.line_scorer.bias)
            if self.line_function_interaction:
                with torch.random.fork_rng(devices=[]):
                    self.context_u = nn.Linear(config.hidden_size, 32, bias=False)
                    self.context_v = nn.Linear(config.hidden_size, 32, bias=False)
                nn.init.zeros_(self.context_v.weight)

    def bounded_pool(self, states, code_mask, line_token_mask, *, return_line_states=False):
        """Pool physical lines with effective token mass n_i and emphasis factor r_i."""
        membership = (line_token_mask & code_mask[:, None].bool()).to(states.dtype)
        copies = membership.sum(1)
        if not torch.equal(copies > 0, code_mask.bool()) or not code_mask.any(-1).all():
            raise ValueError("Physical lines must cover every valid code token")
        # Tokens shared by physical lines contribute a total mass of one.
        membership = membership / copies.clamp_min(1)[:, None]
        counts = membership.sum(-1)
        lines = torch.bmm(membership, states) / counts.clamp_min(1e-12)[..., None]
        scores = self.line_scorer(lines).squeeze(-1)
        if self.line_function_interaction:
            valid = code_mask.to(states.dtype)
            global_function_context = (states * valid[..., None]).sum(1) / valid.sum(-1, keepdim=True)
            queries = self.context_u(lines)
            context = self.context_v(global_function_context)[:, None]
            scores = scores + (queries * context).sum(-1) / math.sqrt(32)
        rates = (1 + (MAX_EMPHASIS_FACTOR - 1) * scores.sigmoid()).masked_fill(counts == 0, 0)
        mass = counts * rates
        attention = mass / mass.sum(-1, keepdim=True)
        pooled_representation = (lines * attention[..., None]).sum(1)
        token_pooling_weights = (membership * (attention / counts.clamp_min(1e-12))[..., None]).sum(1)
        details = dict(line_emphasis_factors=rates, effective_token_mass=counts, pooling_weights=attention,
                       token_pooling_weights=token_pooling_weights)
        if return_line_states:
            details.update(line_states=lines, attention_scores=scores)
        return pooled_representation, details

    @staticmethod
    def soft_regional_loss(rates, counts, labels):
        selected = (labels == 1).to(rates.dtype)
        region = counts * selected
        background = counts * (1 - selected)
        nr, nb = region.sum(-1), background.sum(-1)
        valid = (nr > 0) & (nb > 0)
        mean_r = (region * rates).sum(-1) / nr.clamp_min(1e-12)
        mean_b = (background * rates).sum(-1) / nb.clamp_min(1e-12)
        margin = F.relu(math.log(2) - mean_r.clamp_min(1e-12).log() + mean_b.clamp_min(1e-12).log())
        return (margin[valid].mean() if valid.any() else rates.sum() * 0), valid.sum()

    def forward(self, input_ids, attention_mask, code_mask, line_token_mask,
                *, labels=None, line_labels=None, return_line_states=False):
        t5 = self.config.model_type == "t5"
        decoder = dict(decoder_input_ids=self.encoder._shift_right(input_ids),
                       decoder_attention_mask=attention_mask, use_cache=False) if t5 else {}
        states = self.encoder(input_ids, attention_mask=attention_mask, **decoder)[0]
        details = {}
        if not self.is_plm_classifier:
            pooled, details = self.bounded_pool(states, code_mask, line_token_mask,
                                                return_line_states=return_line_states)
        elif t5:
            eos = input_ids.eq(self.config.eos_token_id) & attention_mask.bool()
            if not eos.any(-1).all():
                raise ValueError("CodeT5 classification requires the source EOS")
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            last = positions.expand_as(input_ids).masked_fill(~eos, -1).max(-1).values
            pooled = states[torch.arange(len(input_ids), device=input_ids.device), last]
        else:
            pooled = states[:, 0]
        logits = self.classifier(pooled[:, None] if self.is_plm_classifier and not t5 else pooled)
        if self.is_plm_classifier and t5:
            logits = logits[:, 1:2] - logits[:, 0:1]
        details.update(logits=logits[:, 0], probability=logits.sigmoid()[:, 0])
        if labels is not None:
            y = labels.float()
            bce = F.binary_cross_entropy_with_logits(logits[:, 0], y, weight=y * self.class_weight + (1 - y))
            region, effective = bce * 0, logits.new_zeros((), dtype=torch.long)
            if self.method in REGION_METHODS:
                if line_labels is None:
                    raise ValueError("Region training requires line labels")
                region, effective = self.soft_regional_loss(details["line_emphasis_factors"], details["effective_token_mass"], line_labels)
            details.update(bce=bce, soft_regional_loss=region, region_functions=effective,
                           loss=bce + REGIONAL_LOSS_WEIGHT * region)
        return details


def build_model(backbone, method, class_weight=1., model_source=None, *, pretrained=True):
    source, options = pretrained_source(backbone, model_source)
    config = AutoConfig.from_pretrained(source, **options)
    config._attn_implementation = "eager"
    t5 = config.model_type == "t5"
    if config.model_type != ("t5" if backbone == "codet5" else "roberta"):
        raise ValueError("Model type differs from the requested backbone")
    cls = T5Model if t5 else RobertaModel
    kwargs = {} if t5 else {"add_pooling_layer": False}
    encoder = (cls.from_pretrained(source, config=config, attn_implementation="eager", **options, **kwargs)
               if pretrained else cls(config, **kwargs))
    return Detector(encoder, config, method, class_weight)


def checkpoint_metadata(checkpoint):
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        meta = f.metadata()
    required = ("backbone", "dataset", "seed", "method", "epoch", "selected_by", "class_weight", "aliases")
    if not meta or any(key not in meta for key in required):
        raise ValueError(f"Missing paper checkpoint metadata in {checkpoint}")
    check_run(meta["backbone"], meta["dataset"], int(meta["seed"]), meta["method"])
    if meta["selected_by"] != selected_by(meta["dataset"]):
        raise ValueError(f"Incorrect checkpoint selection metric in {checkpoint}")
    return meta


def load_checkpoint(checkpoint, *, device="cpu", model_source=None):
    meta = checkpoint_metadata(checkpoint)
    state = load_file(checkpoint, device="cpu")
    for alias, source in json.loads(meta["aliases"]).items():
        state[alias] = state[source]
    model = build_model(meta["backbone"], meta["method"], float(meta["class_weight"]),
                        model_source, pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_checkpoint(path, model, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors, aliases, owners = {}, {}, {}
    for key, value in model.state_dict().items():
        identity = (value.untyped_storage().data_ptr(), value.storage_offset(),
                    tuple(value.shape), value.stride(), value.dtype)
        if identity in owners:
            aliases[key] = owners[identity]
        else:
            owners[identity] = key
            tensors[key] = value.detach().cpu().contiguous()
    temp = path.with_suffix(".tmp")
    save_file(tensors, temp, metadata=dict({k: str(v) for k, v in metadata.items()},
                                         aliases=json.dumps(aliases, sort_keys=True)))
    temp.replace(path)


def forward_batch(model, batch, device, *, with_labels=False, with_regions=False, return_line_states=False):
    inputs = {key: batch[key].to(device, non_blocking=True) for key in
              ("input_ids", "attention_mask", "code_mask", "line_token_mask")}
    if with_labels:
        inputs["labels"] = batch["labels"].to(device, non_blocking=True)
    if with_regions:
        inputs["line_labels"] = batch["line_labels"].to(device, non_blocking=True)
    return model(**inputs, return_line_states=return_line_states)
