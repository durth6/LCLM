"""Train an explicit multi-head role-aware latent Selector.

Encoder, Adapter and Decoder remain frozen.  Role labels are offline training
supervision only; the saved selector consumes a short query and cached latent
vectors at answer time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from latent_context import QuestionEncoder, from_pretrained
from latent_context.role_selector import (
    ROLE_NAMES,
    RoleAwareSelector,
    save_role_selector_checkpoint,
)
from train.train_selector import (
    _MemoryLRU,
    _build_examples,
    _positive_mask,
    _role_masks,
    balanced_bce_loss,
    group_coverage_loss,
    multi_positive_ranking_loss,
    role_group_masks,
    split_examples_by_document,
)


def _target_mask(
    role: str,
    roles: dict[str, torch.Tensor],
    positives: torch.Tensor,
) -> torch.Tensor:
    preferences = {
        "relevance": ("key_answer", "support", "all_answer"),
        "key": ("key", "key_answer"),
        "answer": ("all_answer", "answer"),
        "support": ("support", "key_answer"),
    }[role]
    available = [roles[name] for name in preferences if name in roles]
    if not available:
        return positives
    target = available[0].clone()
    if role == "relevance":
        for mask in available[1:]:
            target |= mask
    return target


def _fused_scores(
    selector: RoleAwareSelector,
    question: torch.Tensor,
    role_scores: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    gates = selector.gate_weights(question)
    fused = torch.zeros_like(role_scores[ROLE_NAMES[0]])
    for index, role in enumerate(ROLE_NAMES):
        fused = fused + gates[:, index:index + 1].to(fused.dtype) * role_scores[role]
    return fused, gates


@torch.no_grad()
def evaluate(
    examples: list[dict[str, Any]],
    memory_lru: _MemoryLRU,
    question_encoder: QuestionEncoder,
    selector: RoleAwareSelector,
    tokenizer,
    device: str,
    selection_ratio: float,
) -> dict[str, float]:
    selector.eval()
    answer_recall: list[float] = []
    joint_group_recall: list[float] = []
    key_group_recall: list[float] = []
    support_group_recall: list[float] = []
    for example in examples:
        memory = memory_lru.get(example["document_id"], device)
        encoded = tokenizer(
            [example["question"]], padding=True, return_tensors="pt", add_special_tokens=False
        )
        question = question_encoder(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        )
        k = max(1, math.ceil(selection_ratio * memory.num_latents))
        selected = set(selector(question, memory, k).indices[0].tolist())
        positives = {value for value in example["positives"] if 0 <= value < memory.num_latents}
        answer_recall.append(len(selected & positives) / len(positives))
        roles = example.get("roles", {})

        def group_fraction(prefix: str) -> float | None:
            groups = [
                {int(value) for value in values if 0 <= int(value) < memory.num_latents}
                for name, values in roles.items()
                if isinstance(name, str)
                and name.startswith(prefix + "_")
                and name[len(prefix) + 1:].isascii()
                and name[len(prefix) + 1:].isdecimal()
                and isinstance(values, list)
            ] if isinstance(roles, dict) else []
            groups = [group for group in groups if group]
            return sum(bool(selected & group) for group in groups) / len(groups) if groups else None

        key_value = group_fraction("key_answer")
        support_value = group_fraction("support")
        if key_value is not None:
            key_group_recall.append(key_value)
        if support_value is not None:
            support_group_recall.append(support_value)
        active = [value for value in (key_value, support_value) if value is not None]
        if active:
            joint_group_recall.append(min(active))
    selector.train()
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "answer_recall": mean(answer_recall),
        "key_answer_group_recall": mean(key_group_recall),
        "support_group_recall": mean(support_group_recall),
        "joint_role_recall": mean(joint_group_recall),
        "examples": float(len(examples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--role-oracle-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retrieval-dim", type=int, default=128)
    parser.add_argument("--min-evidence-fraction", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--group-weight", type=float, default=0.5)
    parser.add_argument("--gate-balance-weight", type=float, default=0.05)
    parser.add_argument("--selection-ratio", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--split-seed", type=int, default=6201)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.epochs <= 0 or args.grad_accumulation <= 0:
        raise ValueError("epochs and grad accumulation must be positive")
    if not 0.0 < args.selection_ratio <= 1.0:
        raise ValueError("selection ratio must be in (0, 1]")
    if min(args.rank_weight, args.bce_weight, args.fused_weight, args.group_weight,
           args.gate_balance_weight) < 0:
        raise ValueError("loss weights must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = _build_examples(args.queries_jsonl, args.role_oracle_jsonl)
    train_examples, validation_examples, split_manifest = split_examples_by_document(
        examples, args.validation_fraction, args.split_seed
    )
    model, _, _ = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval()
    hidden_size = model.encoder.embed_model.config.hidden_size
    question_encoder = QuestionEncoder(
        model.encoder.embed_model, hidden_size, freeze_backbone=True
    ).to(args.device)
    question_encoder.eval()
    selector = RoleAwareSelector(
        hidden_size,
        args.retrieval_dim,
        min_evidence_fraction=args.min_evidence_fraction,
    ).to(args.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=0.01)
    memory_lru = _MemoryLRU(args.cache_dir)
    tokenizer = model.encoder.embed_tokenizer
    fingerprint = model.memory_fingerprint()
    history: list[dict[str, Any]] = []
    start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_examples)
        selector.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("loss", "role", "fused", "group", "gate")}
        for item_index, example in enumerate(train_examples, start=1):
            memory = memory_lru.get(example["document_id"], args.device)
            if memory.fingerprint != fingerprint:
                raise ValueError("training cache was built by an incompatible Encoder")
            encoded = tokenizer(
                [example["question"]], padding=True, return_tensors="pt", add_special_tokens=False
            )
            question = question_encoder(
                encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device)
            )
            scores = selector.role_scores(question, memory)
            fused, gates = _fused_scores(selector, question, scores)
            positives = _positive_mask(memory, example["positives"], args.device)
            role_masks = _role_masks(memory, example.get("roles"), args.device)
            role_loss = fused.new_zeros(())
            for role in ROLE_NAMES:
                target = _target_mask(role, role_masks, positives)
                role_loss = role_loss + (
                    args.rank_weight * multi_positive_ranking_loss(scores[role][0], target)
                    + args.bce_weight * balanced_bce_loss(scores[role][0], target)
                )
            role_loss = role_loss / len(ROLE_NAMES)
            fused_loss = (
                args.rank_weight * multi_positive_ranking_loss(fused[0], positives)
                + args.bce_weight * balanced_bce_loss(fused[0], positives)
            )
            selection_k = max(1, math.ceil(args.selection_ratio * memory.num_latents))
            group_loss = group_coverage_loss(
                scores["key"][0], role_group_masks(role_masks, "key_answer"),
                selection_k=selection_k,
            ) + group_coverage_loss(
                scores["answer"][0], role_group_masks(role_masks, "answer"),
                selection_k=selection_k,
            ) + group_coverage_loss(
                scores["support"][0], role_group_masks(role_masks, "support"),
                selection_k=selection_k,
            )
            target_gate = torch.full_like(gates, 1.0 / len(ROLE_NAMES))
            gate_loss = F.kl_div(gates.clamp_min(1e-8).log(), target_gate, reduction="batchmean")
            loss = (
                role_loss
                + args.fused_weight * fused_loss
                + args.group_weight * group_loss
                + args.gate_balance_weight * gate_loss
            )
            (loss / args.grad_accumulation).backward()
            if item_index % args.grad_accumulation == 0 or item_index == len(train_examples):
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name, value in (
                ("loss", loss), ("role", role_loss), ("fused", fused_loss),
                ("group", group_loss), ("gate", gate_loss),
            ):
                totals[name] += float(value.detach())
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / len(train_examples) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in evaluate(
                validation_examples, memory_lru, question_encoder, selector, tokenizer,
                args.device, args.selection_ratio,
            ).items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    split_payload = json.dumps(split_manifest, indent=2) + "\n"
    (output / "data_split.json").write_text(split_payload)
    role_digest = hashlib.sha256(Path(args.role_oracle_jsonl).read_bytes()).hexdigest()
    metadata = {
        "training_stage": "role_aware_multi_head_v1",
        "document_split_sha256": hashlib.sha256(split_payload.encode()).hexdigest(),
        "role_sidecar_sha256": role_digest,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "selection_ratio": args.selection_ratio,
        "loss_weights": {
            "rank": args.rank_weight,
            "bce": args.bce_weight,
            "fused": args.fused_weight,
            "group": args.group_weight,
            "gate_balance": args.gate_balance_weight,
        },
        "input_contract": "cached latent vectors + short query; role labels offline training only",
    }
    save_role_selector_checkpoint(output, selector, fingerprint, metadata)
    (output / "training_metrics.json").write_text(json.dumps({
        "metadata": metadata,
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(validation_examples),
        "elapsed_seconds": perf_counter() - start,
        "history": history,
    }, indent=2) + "\n")
    print(json.dumps({"saved": str(output), "final": history[-1]}), flush=True)


if __name__ == "__main__":
    main()
