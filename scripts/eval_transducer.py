"""Sequence-level evaluation of LZW transducer checkpoints on held-out data.

Training reports TOKEN-level loss/accuracy: averaged over positions, a model can
sit at 90% while never once producing a correct compressed sequence. RQ1 asks
whether the model executes the algorithm, which is a per-sample yes/no. This
script answers that, for every checkpoint of a --mode compress run:

  strict   the generated sequence equals the target token for token
  relaxed  the generated sequence EXPANDS to the same base tokens as the target
           (a valid compression that segments differently from canonical LZW)

strict implies relaxed, always -- the script asserts it.

Cost note. Strict match needs no generation at all. Under greedy decoding,
"free-running generation reproduces the target" is EQUIVALENT to "teacher-forced
argmax equals the label at every supervised position": while every token so far
is correct the generated prefix IS the ground-truth prefix, so both settings feed
the model identical inputs; at the first mismatch both emit the same wrong token
and both verdicts are failure. So strict costs ONE forward per batch instead of
one per generated token. Relaxed does need real generation (a diverged sequence
may still expand correctly), but only for the samples strict already failed, and
it stops the moment the expansion disagrees in base space. --verify_generation
checks the equivalence empirically rather than asking you to trust it.

Usage:
    python scripts/eval_transducer.py --run_dir /path/to/zip2zip_Phi-20M_compress_ms4_scratch
    python scripts/eval_transducer.py --run_dir ... --steps 1000,6000 --n_samples 64
    python scripts/eval_transducer.py --run_dir ... --verify_generation 8
"""

import argparse
import dataclasses
import glob
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.data import Zip2ZipDataset, remap_collate_fn
from zip2zip_core.disabled_ids import compute_disabled_ids
from zip2zip_core.model import Zip2ZipLlama3Model, rope_cache_len


def log(msg):
    print(f"[eval-transducer] {msg}", flush=True)


# ---------------------------------------------------------------- checkpoints


def find_checkpoints(run_dir, steps=None):
    """Checkpoint dirs under run_dir, ordered by step."""
    found = {}
    for path in glob.glob(os.path.join(run_dir, "step_*")):
        name = os.path.basename(path)
        if not os.path.isfile(os.path.join(path, "model.pt")):
            continue
        try:
            found[int(name.split("_", 1)[1])] = path
        except ValueError:
            continue
    if not found:
        raise ValueError(f"no step_*/model.pt under {run_dir}")
    if steps:
        missing = [s for s in steps if s not in found]
        if missing:
            raise ValueError(f"requested steps {missing} not in {sorted(found)}")
        return [(s, found[s]) for s in steps]
    return sorted(found.items())


def load_train_args(ckpt_dir):
    meta = torch.load(os.path.join(ckpt_dir, "meta.pt"), map_location="cpu",
                      weights_only=False)
    args = meta.get("args")
    if not args:
        raise ValueError(f"{ckpt_dir}/meta.pt has no train args to rebuild from")
    return dict(args), int(meta["step"])


def build_model(train_args, device, disable_varlen=False):
    """Rebuild the trained model from meta.pt args.

    Mirrors train.py's construction. Anything read from train_args rather than
    re-derived here is a setting that must not drift between train and eval --
    the same class of mismatch documented in docs/evaluation.md.
    """
    config = zip2zip_llama_configs[train_args["model_config"]]
    replace_kwargs = dict(
        max_codebook_size=train_args["max_codebook_size"],
        max_subtokens=train_args["max_subtokens"],
        hyper_encoder_type=train_args.get("hyper_encoder_type", "flat"),
        token_type_loss_weight=train_args.get("token_type_loss_weight", 0.0),
        tie_hyper_encoder=not train_args.get("untied_hyper_encoder", False),
        share_hyper_encoder_weights=train_args.get("share_hyper_encoder_weights", False),
        base_token_positions=train_args.get("base_token_positions", False),
        two_axis_rope=train_args.get("two_axis_rope", False),
        gated_compressed_rope=train_args.get("gated_compressed_rope", False),
        gated_rope_start_layer=train_args.get("gated_rope_start_layer", 0),
        gated_rope_start_pair=train_args.get("gated_rope_start_pair", 0),
        zero_init_encoder_output=train_args.get("zero_init_encoder_output", False),
        rope=dataclasses.replace(
            config.rope,
            max_seq_len=rope_cache_len(
                train_args["seq_len"],
                train_args["max_subtokens"],
                train_args.get("base_token_positions", False),
            ),
        ),
    )
    for field in ("encoder_dim", "encoder_intermediate_size",
                  "encoder_n_heads", "encoder_n_layers"):
        if train_args.get(field) is not None:
            replace_kwargs[field] = train_args[field]
    config = dataclasses.replace(config, **replace_kwargs)

    model = config.build()
    # The hyper-encoder's varlen path calls flash-attention, which has no CPU
    # kernel. Same padded fallback the trainer exposes as --disable_varlen: it
    # computes the same attention, so verdicts are unaffected, but it is a
    # different kernel and the two devices are not bit-comparable.
    if disable_varlen or train_args.get("disable_varlen"):
        model.hyper_encoder.disable_varlen = True
        if getattr(model, "hyper_output", None) is not None:
            model.hyper_output.disable_varlen = True
    if train_args.get("no_encoder_residual"):
        model.encoder_residual = False
    with torch.no_grad():
        model.init_weights()
    return model, config


def load_weights(model, ckpt_dir, device):
    from zip2zip_core.checkpoint import strip_wrapper_prefixes

    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    state = strip_wrapper_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"state dict mismatch for {ckpt_dir}: missing={list(missing)[:5]} "
            f"unexpected={list(unexpected)[:5]}. The config rebuilt from meta.pt "
            "does not describe these weights."
        )
    # Device only -- NEVER a dtype cast. The RoPE cache is a COMPLEX buffer, and
    # .to(dtype=float32) silently drops its imaginary part ("Casting complex
    # values to real discards the imaginary part"), which destroys positional
    # encoding and makes a fully-trained checkpoint score at chance. Training
    # keeps params in fp32 and gets bf16 compute from autocast; mirror that.
    return model.to(device=device).eval()


# ------------------------------------------------------------------ eval data


def build_eval_batches(train_args, config, n_samples, batch_size,
                       shard_index, offset, tokenizer_name):
    """Held-out samples built exactly like training samples.

    Same LZW settings, same markers, same direction -- only the position in the
    corpus differs. Training reads each rank's shards from offset 0 and a 6000
    step run consumes a few hundred million tokens per rank, so a late shard at a
    large offset was never touched. The caller is responsible for that choice;
    check_unseen() below turns the arithmetic into an assertion.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    disabled_ids = compute_disabled_ids(
        tok, config.vocab_size,
        disable_digit_ids=train_args.get("disable_digit_ids", False),
    )
    dataset = Zip2ZipDataset(
        data_dir=train_args["data_dir"],
        seq_len=train_args["seq_len"],
        max_subtokens=train_args["max_subtokens"],
        max_codebook_size=train_args["max_codebook_size"],
        initial_vocab_size=config.vocab_size,
        pad_token_id=config.pad_token_id,
        disabled_ids=disabled_ids,
        rank=0,
        world_size=1,
        mode="compress",
        remap_codebook=not train_args.get("no_remap_codebook", False),
        max_active_codebook_size=train_args["max_active_codebook_size"],
        compress_token_id=train_args["compress_token_id"],
        decompress_token_id=train_args["decompress_token_id"],
        direction=train_args.get("direction", "both"),
    )
    n_shards = len(dataset.shard_files)
    dataset._shard_idx = n_shards - 1 if shard_index < 0 else shard_index
    dataset._offset = offset
    log(f"held-out stream: shard {dataset._shard_idx}/{n_shards - 1} "
        f"offset {offset:,} direction={train_args.get('direction', 'both')}")

    samples = list(itertools.islice(iter(dataset), n_samples))
    if len(samples) < n_samples:
        raise ValueError(
            f"held-out region yielded {len(samples)} of {n_samples} samples; "
            "lower --eval_offset or use an earlier shard."
        )
    collate = lambda batch: remap_collate_fn(
        batch,
        pad_token_id=config.pad_token_id,
        max_subtokens=train_args["max_subtokens"],
        max_active_codebook_size=train_args["max_active_codebook_size"],
    )
    return [collate(samples[i:i + batch_size])
            for i in range(0, len(samples), batch_size)]


def check_unseen(train_args, shard_index, offset, n_shards):
    """Refuse to score a checkpoint on data it may have trained on.

    Per rank the stream advances by at most seq_len base tokens per sample, so
    steps * accum * local_batch * seq_len bounds the reach into a rank's FIRST
    shard; later shards in a rank's slice are only reached once the first is
    exhausted. Shards are handed out as shard_files[rank::world_size], so shard
    index >= world_size is never a rank's first shard.
    """
    world_size = train_args.get("world_size") or 4
    per_rank_samples = (train_args["steps"]
                        * train_args["gradient_accumulation_steps"]
                        * train_args["local_batch_size"])
    max_reach = per_rank_samples * train_args["seq_len"]
    resolved = n_shards - 1 if shard_index < 0 else shard_index
    first_shard_of_some_rank = resolved < world_size
    if first_shard_of_some_rank and offset < max_reach:
        raise ValueError(
            f"shard {resolved} is rank {resolved}'s first shard and offset "
            f"{offset:,} is inside the at-most {max_reach:,} tokens training "
            "could have consumed there. Pick a later shard or a larger "
            "--eval_offset (pass --allow_seen_data to override)."
        )
    return {"world_size_assumed": world_size, "max_reach_per_rank": max_reach,
            "shard": resolved, "offset": offset}


# -------------------------------------------------------------------- scoring


def supervised_spans(labels):
    """(start, end) of the supervised block of every sample in a batch.

    The block is contiguous by construction (data.py sets loss_mask[start:end]),
    so first and last supervised index define it. Deriving it from the labels
    rather than re-locating the markers keeps this correct even if a marker id
    ever appears inside the source text.
    """
    valid = labels != -100
    if not bool(valid.any(dim=1).all()):
        raise ValueError("a held-out sample has no supervised position")
    idx = torch.arange(labels.shape[1], device=labels.device)
    start = torch.where(valid, idx, torch.full_like(idx, labels.shape[1])).min(dim=1).values
    end = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values + 1
    if not bool((valid.sum(dim=1) == (end - start)).all()):
        raise ValueError("supervised block is not contiguous")
    return start, end


def slice_batch(batch, n):
    """First n samples of a collated batch, so verification cost is bounded."""
    inputs, labels = batch
    return ({k: v[:n] for k, v in inputs.items()}, labels[:n])


@torch.no_grad()
def score_teacher_forced(model, batch, vocab_size, hyper_causal_mask, device):
    """Per-sample strict verdict plus the token stats that come free with it."""
    inputs, labels = batch
    x = inputs["input"].to(device)
    cb = inputs["codebook"].to(device)
    labels = labels.to(device)

    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        logits = model(x, codebook=cb, hyper_causal_mask=hyper_causal_mask)
    logits = logits.float()

    valid = labels != -100
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & valid
    nll = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(0, 1),
        reduction="none", ignore_index=-100,
    ).view(labels.shape)

    n_valid = valid.sum(dim=1)
    return {
        # All supervised positions right <=> greedy generation reproduces the
        # target (see module docstring).
        "strict": (correct.sum(dim=1) == n_valid),
        "n_valid": n_valid,
        "n_correct": correct.sum(dim=1),
        "nll_sum": (nll * valid).sum(dim=1),
        "n_base_tokens": inputs["n_base_tokens"].to(device),
        "preds": preds,
    }


def expand_to_base(tokens, cb_row_table, vocab_size, pad_token_id):
    """Expand a token sequence to base tokens through one sample's codebook.

    Rows hold the full base expansion right-padded with pad_token_id, and
    pad_token_id is in disabled_ids so it never occurs as a real element -- the
    trailing padding is therefore unambiguous.
    """
    out = []
    for tok in tokens:
        t = int(tok)
        if t < vocab_size:
            out.append(t)
            continue
        row = cb_row_table[t - vocab_size]
        for sub in row:
            s = int(sub)
            if s == pad_token_id:
                break
            out.append(s)
    return out


@torch.no_grad()
def generate_relaxed(model, batch, strict, vocab_size, pad_token_id,
                     hyper_causal_mask, device, force_all=False):
    """Free-running greedy decode, scored in BASE-token space.

    Only samples that failed strict are decoded: strict success already implies
    the expansions match. Decoding stops for a sample as soon as its expanded
    prefix disagrees with the target's, because expansion is prefix-monotone --
    no continuation can repair an earlier base token.
    """
    inputs, labels = batch
    x = inputs["input"].to(device).clone()
    cb = inputs["codebook"].to(device)
    labels = labels.to(device)
    start, end = supervised_spans(labels)
    B, T = x.shape

    cb_cpu = cb.cpu().tolist()
    targets_base, active = [], []
    for b in range(B):
        tgt = labels[b, start[b]:end[b]].cpu().tolist()
        targets_base.append(expand_to_base(tgt, cb_cpu[b], vocab_size, pad_token_id))
        active.append(force_all or not bool(strict[b]))
    produced = [[] for _ in range(B)]
    verdict = [None] * B
    for b in range(B):
        if not active[b]:
            verdict[b] = True  # strict pass; expansions are equal by definition

    pos = int(start.min())
    last = int(end.max())
    while pos < last and any(active):
        # Samples in a batch start their supervised block at different offsets.
        # Stepping past a position where no live sample is inside its own window
        # costs a full forward and changes nothing, and once every sample has
        # diverged the remaining positions are pure waste.
        if not any(active[b] and int(start[b]) <= pos < int(end[b])
                   for b in range(B)):
            pos += 1
            continue
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits = model(x[:, :pos + 1], codebook=cb,
                           hyper_causal_mask=hyper_causal_mask)
        nxt = logits[:, -1].float().argmax(dim=-1)
        for b in range(B):
            if not active[b] or not (start[b] <= pos < end[b]):
                continue
            tok = int(nxt[b])
            produced[b].append(tok)
            if pos + 1 < T:
                x[b, pos + 1] = tok
            expanded = expand_to_base(produced[b], cb_cpu[b], vocab_size, pad_token_id)
            target = targets_base[b]
            if expanded != target[:len(expanded)]:
                verdict[b], active[b] = False, False   # base-space divergence
            elif pos + 1 == int(end[b]):
                verdict[b], active[b] = (expanded == target), False
        pos += 1
    for b in range(B):
        if verdict[b] is None:
            verdict[b] = False
    return torch.tensor(verdict, device=device), produced


# ------------------------------------------------------------------- wandb


def open_wandb_run(train_args, args):
    """Resume the training run this checkpoint came from, to log into it.

    meta.pt records wandb_name AFTER train.py appended its random suffix, so the
    name identifies the run uniquely and no id has to be carried around by hand.
    """
    import wandb
    from zip2zip_core.project import WANDB_ENTITY, WANDB_PROJECT

    entity = args.wandb_entity or train_args.get("wandb_entity") or WANDB_ENTITY
    project = args.wandb_project or train_args.get("wandb_project") or WANDB_PROJECT
    run_id = args.wandb_id or train_args.get("wandb_id")
    if not run_id:
        name = train_args.get("wandb_name")
        if not name:
            raise ValueError("meta.pt has no wandb_name; pass --wandb_id")
        matches = list(wandb.Api().runs(f"{entity}/{project}",
                                        filters={"display_name": name}))
        if len(matches) != 1:
            raise ValueError(
                f"{len(matches)} wandb runs named {name!r} in {entity}/{project}"
                f" ({[m.id for m in matches]}); pass --wandb_id to disambiguate"
            )
        run_id = matches[0].id
    log(f"logging into wandb run {entity}/{project}/{run_id}")
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    # Own x-axis instead of the run's step counter: the training run already
    # advanced past 6000, and re-logging at earlier steps would be dropped.
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    return run


# ----------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Training output dir holding step_*/ checkpoints.")
    p.add_argument("--steps", default=None,
                   help="Comma-separated steps to score (default: all).")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_shard", type=int, default=-1,
                   help="Shard index for held-out data; -1 = last (default).")
    p.add_argument("--eval_offset", type=int, default=1_000_000_000,
                   help="Token offset into that shard. Must be past anything "
                        "training could have read; see check_unseen().")
    p.add_argument("--allow_seen_data", action="store_true")
    p.add_argument("--relaxed", action="store_true",
                   help="Also score base-space (relaxed) match. Needs real "
                        "generation, so it costs one forward per token.")
    p.add_argument("--verify_generation", type=int, default=0,
                   help="Decode N strict-PASSING samples free-running and assert "
                        "they reproduce the target, i.e. check the equivalence "
                        "this script's cheap strict path relies on.")
    p.add_argument("--disable_varlen", action="store_true",
                   help="Padded hyper-encoder attention instead of flash varlen. "
                        "Forced on for non-CUDA devices, which have no kernel.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None)
    p.add_argument("--output_jsonl", default=None)
    p.add_argument("--wandb", action="store_true",
                   help="Log the curve into the ORIGINAL training run, resolved "
                        "from the wandb_name recorded in meta.pt.")
    p.add_argument("--wandb_id", default=None,
                   help="Skip the name lookup and write to this run id.")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    args = p.parse_args()

    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    checkpoints = find_checkpoints(args.run_dir, steps)
    log(f"{len(checkpoints)} checkpoints under {args.run_dir}")

    train_args, _ = load_train_args(checkpoints[0][1])
    if train_args.get("mode") != "compress":
        raise ValueError(f"run mode is {train_args.get('mode')!r}, not 'compress'")
    direction = train_args.get("direction", "both")

    disable_varlen = args.disable_varlen or not args.device.startswith("cuda")
    if disable_varlen and not args.device.startswith("cuda"):
        log("non-CUDA device: falling back to the padded hyper-encoder attention "
            "(flash varlen is CUDA-only)")
    _, config = build_model(train_args, args.device, disable_varlen)
    if not args.allow_seen_data:
        n_shards = len(glob.glob(os.path.join(train_args["data_dir"], "shard_*.npy")))
        info = check_unseen(train_args, args.eval_shard, args.eval_offset, n_shards)
        log(f"held-out check passed ({n_shards} shards): {info}")

    batches = build_eval_batches(
        train_args, config, args.n_samples, args.batch_size,
        args.eval_shard, args.eval_offset, train_args["tokenizer"],
    )
    log(f"{args.n_samples} held-out samples in {len(batches)} batches")

    wandb_run = open_wandb_run(train_args, args) if args.wandb else None

    records, summaries = [], []
    for step, ckpt_dir in checkpoints:
        t0 = time.time()
        model, _ = build_model(train_args, args.device, disable_varlen)
        model = load_weights(model, ckpt_dir, args.device)

        strict_hits = relaxed_hits = n_seen = 0
        tok_correct = tok_valid = 0
        for bi, batch in enumerate(batches):
            out = score_teacher_forced(
                model, batch, config.vocab_size,
                train_args.get("hyper_causal_mask", False), args.device,
            )
            strict = out["strict"]
            relaxed = strict
            if args.relaxed:
                relaxed, _ = generate_relaxed(
                    model, batch, strict, config.vocab_size, config.pad_token_id,
                    train_args.get("hyper_causal_mask", False), args.device,
                )
                if bool((strict & ~relaxed).any()):
                    raise AssertionError(
                        "strict pass with relaxed failure — impossible unless "
                        "the expansion or the generation loop is wrong"
                    )
            for b in range(strict.shape[0]):
                records.append({
                    "step": step, "batch": bi, "sample": b,
                    "strict": bool(strict[b]), "relaxed": bool(relaxed[b]),
                    "n_valid": int(out["n_valid"][b]),
                    "token_acc": float(out["n_correct"][b] / out["n_valid"][b]),
                    "loss_per_target_token": float(out["nll_sum"][b] / out["n_valid"][b]),
                    "loss_per_base_token": float(out["nll_sum"][b] / out["n_base_tokens"][b]),
                })
            strict_hits += int(strict.sum())
            relaxed_hits += int(relaxed.sum())
            tok_correct += int(out["n_correct"].sum())
            tok_valid += int(out["n_valid"].sum())
            n_seen += strict.shape[0]

        # The equivalence being checked is a property of the implementation, not
        # of the weights, so verifying once is enough -- and it is the expensive
        # path (a strict-PASSING sample decodes the whole target, one full
        # forward per token). Do it on the last checkpoint, the one most likely
        # to have a passing sample and therefore to exercise that branch at all.
        if args.verify_generation and step == checkpoints[-1][0]:
            n_verify = min(args.verify_generation, args.batch_size)
            verify_batch = slice_batch(batches[0], n_verify)
            out = score_teacher_forced(
                model, verify_batch, config.vocab_size,
                train_args.get("hyper_causal_mask", False), args.device)
            _, produced = generate_relaxed(
                model, verify_batch, out["strict"], config.vocab_size,
                config.pad_token_id, train_args.get("hyper_causal_mask", False),
                args.device, force_all=True)
            inputs, labels = verify_batch
            s, e = supervised_spans(labels.to(args.device))
            checked = 0
            for b in range(min(args.verify_generation, labels.shape[0])):
                tgt = labels[b, s[b]:e[b]].cpu().tolist()
                agree = (produced[b] == tgt) == bool(out["strict"][b])
                if not agree:
                    raise AssertionError(
                        f"step {step} sample {b}: free-running generation and the "
                        "teacher-forced strict verdict disagree"
                    )
                checked += 1
            log(f"  verified generation/teacher-forcing equivalence on {checked} samples")

        summary = {
            "step": step,
            "model_config": train_args["model_config"],
            "direction": direction,
            "n_samples": n_seen,
            "strict_exact_match": strict_hits / n_seen,
            "relaxed_base_match": relaxed_hits / n_seen if args.relaxed else None,
            "pooled_token_acc": tok_correct / tok_valid,
            "seconds": round(time.time() - t0, 1),
        }
        summaries.append(summary)
        if wandb_run is not None:
            # Scope the keys by direction. The two directions are separate models
            # in separate runs, so a shared key would look fine on a single run
            # and silently average the two the moment runs are grouped (by model
            # size, say) -- and the mean of a compression score and a
            # decompression score is not a quantity.
            payload = {
                "eval/step": step,
                f"eval/{direction}/strict_exact_match": summary["strict_exact_match"],
                f"eval/{direction}/pooled_token_acc": summary["pooled_token_acc"],
            }
            if args.relaxed:
                payload[f"eval/{direction}/relaxed_base_match"] = summary["relaxed_base_match"]
            wandb_run.log(payload)
        log(f"step {step:>6}: strict={summary['strict_exact_match']:.4f}"
            + (f" relaxed={summary['relaxed_base_match']:.4f}" if args.relaxed else "")
            + f" | pooled_token_acc={summary['pooled_token_acc']:.4f}"
            + f" ({summary['seconds']}s)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_json = args.output_json or os.path.join(args.run_dir, "eval_transducer.json")
    with open(out_json, "w") as f:
        json.dump({"run_dir": args.run_dir, "direction": direction,
                   "model_config": train_args["model_config"],
                   "eval_shard": args.eval_shard, "eval_offset": args.eval_offset,
                   "n_samples": args.n_samples, "summaries": summaries}, f, indent=2)
    log(f"wrote {out_json}")

    out_jsonl = args.output_jsonl or os.path.join(args.run_dir, "eval_transducer_samples.jsonl")
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {out_jsonl} ({len(records)} sample records)")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
