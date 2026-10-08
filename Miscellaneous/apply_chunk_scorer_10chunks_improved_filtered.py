#!/usr/bin/env python3
"""
Apply the improved chunk scorer with stricter filtering (10-CHUNK VERSION):
- Only keep chunks whose relation matches the gold relation (fallback to best if none).
- Prefer chunks that contain both head and tail surface strings.
- Keep a small top-K (default: 2) to avoid prompt bloat.

This is a drop-in variant of apply_chunk_scorer.py for the 10-chunk improved pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Make the trained chunk scorer loader importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "1_train_chuck_scorer_model"))
from load_trained_model import load_trained_chunk_scorer  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _clean_instruction(base_instruction: str) -> str:
    """Strip any stale Examples section so we can rebuild from scorer output."""
    if "Examples:" in base_instruction:
        return base_instruction.split("Examples:", 1)[0].strip()
    return base_instruction.strip()


def _format_prompt(
    instruction: str,
    example_str: str,
    context: str,
    response: str,
) -> str:
    """Match the training formatting used by chunk scorer SFTTrainer."""
    if example_str:
        instruction = f"{instruction} Examples: {example_str}"
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input: \n"
        f"{context}\n\n"
        "### Response: \n"
        f"{response}"
    )
    return prompt


def _split_logits_on_response_token(
    logits: torch.Tensor, input_ids: torch.Tensor, response_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split logits into (instruction/context part, response part) using the
    last occurrence of the Response token. This mirrors the training-time
    cosine similarity computation in the custom trainer.
    """
    positions = (input_ids[0] == response_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        split_idx = logits.shape[1] // 2  # fallback to midpoint
    else:
        split_idx = positions[-1].item()

    # Ensure we always have at least one token on each side
    split_idx = max(0, min(split_idx, logits.shape[1] - 2))
    left = logits[:, : split_idx + 1, :]
    right = logits[:, split_idx + 1 :, :]
    return left, right


def score_candidate(
    model,
    tokenizer,
    response_token_id: int,
    base_instruction: str,
    context: str,
    response: str,
    chunk_text: str,
    relation: str,
    max_length: int,
) -> float:
    """
    Compute the cosine score for a single candidate chunk using the trained scorer.
    """
    example = f"Context: {chunk_text.strip()} Response: {relation.strip()}"
    prompt = _format_prompt(base_instruction, example, context, response)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)
        logits = outputs.logits  # (1, seq_len, vocab)

    left, right = _split_logits_on_response_token(logits, encoded["input_ids"], response_token_id)
    score = F.cosine_similarity(left.mean(dim=1), right.mean(dim=1)).item()
    return score


def _parse_gold(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = [p.strip() for p in (response or "").split("|")]
    if len(parts) >= 3:
        return parts[0].lower(), parts[1].lower(), parts[2].lower()
    return None, None, None


def rerank_split(
    split_name: str,
    model,
    tokenizer,
    response_token_id: int,
    topn_path: Path,
    knn_path: Path,
    output_path: Path,
    *,
    max_examples: int,
    max_length: int,
    limit: Optional[int] = None,
) -> None:
    """
    Re-rank one split (train/test) with relation/head-tail filtering.
    """
    print(f"\n{'='*60}")
    print(f"🔄 {split_name.upper()}: Starting chunk scoring and re-ranking (filtered)")
    print(f"{'='*60}")
    print(f"📂 Input files:")
    print(f"   • TopN chunks: {topn_path}")
    print(f"   • KNN demo: {knn_path}")
    print(f"📤 Output file: {output_path}")
    print(f"⚙️  Settings: max_examples={max_examples}, max_length={max_length}")
    if limit:
        print(f"⚠️  Limited to {limit} examples (debugging mode)")

    start_time = time.time()
    examples_written = 0
    total_candidates_scored = 0
    total_candidates_skipped = 0

    # Count total examples first (for progress estimation)
    total_examples = sum(1 for _ in _read_jsonl(topn_path))
    if limit:
        total_examples = min(total_examples, limit)

    with output_path.open("w") as out_f:
        for idx, (topn_obj, knn_obj) in enumerate(zip(_read_jsonl(topn_path), _read_jsonl(knn_path))):
            if limit is not None and idx >= limit:
                break

            base_instruction = _clean_instruction(knn_obj.get("instruction", ""))
            context = knn_obj.get("context") or topn_obj.get("sentence", "")
            response = knn_obj.get("response", "")
            gold_h, gold_r, gold_t = _parse_gold(response)

            raw_candidates = topn_obj.get("topn_sim_chuck", {}) or {}
            scored: List[Tuple[str, str, float]] = []

            for chunk_text, relation in raw_candidates.items():
                chunk_text = (chunk_text or "").strip()
                relation = (relation or "").strip()
                if not chunk_text or not relation or chunk_text == "{}" or relation == "{}":
                    total_candidates_skipped += 1
                    continue

                try:
                    score = score_candidate(
                        model,
                        tokenizer,
                        response_token_id,
                        base_instruction,
                        context,
                        response,
                        chunk_text,
                        relation,
                        max_length,
                    )
                    scored.append((chunk_text, relation, score))
                    total_candidates_scored += 1
                except Exception as e:
                    print(f"   ⚠️  Error scoring candidate at example {idx+1}: {e}")
                    total_candidates_skipped += 1
                    continue

            # Relation filter (case-insensitive)
            if gold_r:
                rel_filtered = [
                    s for s in scored if s[1].lower() == gold_r
                ]
            else:
                rel_filtered = []

            # Prefer chunks containing both head and tail
            def has_both_entities(text: str) -> bool:
                if not gold_h or not gold_t:
                    return False
                t = text.lower()
                return gold_h in t and gold_t in t

            candidates: List[Tuple[str, str, float]]
            if rel_filtered:
                with_entities = [s for s in rel_filtered if has_both_entities(s[0])]
                if with_entities:
                    candidates = sorted(with_entities, key=lambda x: x[2], reverse=True)
                else:
                    candidates = sorted(rel_filtered, key=lambda x: x[2], reverse=True)
            elif scored:
                candidates = sorted(scored, key=lambda x: x[2], reverse=True)
            else:
                candidates = []

            trimmed = candidates[:max_examples]

            examples_str = " ".join(
                f"{i}. Context: {chunk} Response: {relation}"
                for i, (chunk, relation, _) in enumerate(trimmed, start=1)
            )
            top_chunks = [
                {"chunk": chunk, "relation": relation, "score": score}
                for chunk, relation, score in trimmed
            ]

            record = {
                "instruction": f"{base_instruction} Examples: {examples_str}".strip()
                if examples_str
                else base_instruction,
                "context": context,
                "response": response,
                "category": knn_obj.get("category", "triplet_extraction"),
                "retrieved_chunks": top_chunks,
            }
            out_f.write(json.dumps(record))
            out_f.write("\n")
            examples_written += 1

            if (idx + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"   • Progress: {idx + 1}/{total_examples} ({100*(idx+1)/total_examples:.1f}%) | "
                      f"Rate: {rate:.2f} ex/s")

    elapsed_total = time.time() - start_time
    elapsed_minutes = elapsed_total / 60

    print(f"\n{'='*60}")
    print(f"✅ {split_name.upper()}: Completed successfully")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   • Examples processed: {examples_written}")
    print(f"   • Total candidates scored: {total_candidates_scored}")
    print(f"   • Candidates skipped: {total_candidates_skipped}")
    print(f"   • Total time: {elapsed_minutes:.2f} minutes ({elapsed_total:.1f} seconds)")
    print(f"   • Average time per example: {elapsed_total/examples_written:.2f} seconds")
    print(f"📁 Output written to: {output_path}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_topn_train = REPO_ROOT / "0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS" / "train_data_topn_chucks_10_improved.json"
    default_topn_test = REPO_ROOT / "0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS" / "test_data_topn_chucks_10_improved_deduplicated.json"
    default_knn_train = base_dir / "KNN_demo_train.json"
    default_knn_test = base_dir / "KNN_demo_test.json"
    default_checkpoint = REPO_ROOT / "0_make_relation_chuck_and_scorer_data" / "1_train_chuck_scorer_model" / "Our_model_improved" / "checkpoint-1000"

    parser = argparse.ArgumentParser(description="Apply the trained chunk scorer (10-chunk improved) with relation/entity filtering.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to chunk scorer checkpoint dir.")
    parser.add_argument("--topn-train", type=Path, default=default_topn_train, help="train_data_topn_chucks_10_improved.json path")
    parser.add_argument("--topn-test", type=Path, default=default_topn_test, help="test_data_topn_chucks_10_improved_deduplicated.json path")
    parser.add_argument("--knn-train", type=Path, default=default_knn_train, help="KNN_demo_train.json path")
    parser.add_argument("--knn-test", type=Path, default=default_knn_test, help="KNN_demo_test.json path")
    parser.add_argument("--output-train", type=Path, default=base_dir / "train_triplet_chunk_scorer_10chunks_improved_filtered.json", help="Output JSONL for train")
    parser.add_argument("--output-test", type=Path, default=base_dir / "test_triplet_chunk_scorer_10chunks_improved_filtered.json", help="Output JSONL for test")
    parser.add_argument("--max-examples", type=int, default=2, help="Top-K chunks to keep in the instruction examples")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenization max_length for scorer inputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of records (debugging)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "="*60)
    print("🚀 Chunk Scorer Application (10-Chunk Improved, Filtered)")
    print("="*60)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"📋 Configuration:")
    print(f"   • Checkpoint: {args.checkpoint}")
    print(f"   • Max examples per instruction: {args.max_examples}")
    print(f"   • Max token length: {args.max_length}")
    if args.limit:
        print(f"   • ⚠️  DEBUG MODE: Limited to {args.limit} examples per split")

    print(f"\n📥 Loading chunk scorer from {args.checkpoint} ...")
    model, tokenizer = load_trained_chunk_scorer(str(args.checkpoint))
    model.eval()
    response_token_id = tokenizer.encode(" Response:", add_special_tokens=False)[-1]
    print(f"   ✅ Model loaded. Response token ID: {response_token_id}")

    # Process train split
    rerank_split(
        "train",
        model,
        tokenizer,
        response_token_id,
        args.topn_train,
        args.knn_train,
        args.output_train,
        max_examples=args.max_examples,
        max_length=args.max_length,
        limit=args.limit,
    )

    # Process test split
    rerank_split(
        "test",
        model,
        tokenizer,
        response_token_id,
        args.topn_test,
        args.knn_test,
        args.output_test,
        max_examples=args.max_examples,
        max_length=args.max_length,
        limit=args.limit,
    )

    print("\n" + "="*60)
    print("🎉 Script completed successfully!")
    print("="*60)
    print(f"📁 Output files:")
    print(f"   • Train: {args.output_train}")
    print(f"   • Test:  {args.output_test}")
    print("="*60)


if __name__ == "__main__":
    import time

    main()



