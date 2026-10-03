"""
evaluate_model.py
-----------------
1. Warns if best_model.pt was trained with old flat-LR code.
2. Writes dev_greedy.jsonl and dev_beam.jsonl in WikiSQL evaluator format.
3. Writes test_beam.jsonl for the final test evaluation.
4. Computes component accuracy (sel, agg, WHERE) on the dev set.
5. Plots the decoder cross-attention map for one dev example.
6. Updates results.json with accuracy numbers.

Run from inside starter/:
    python evaluate_model.py

Then evaluate with the official evaluator:
    python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl WikiSQL/data/dev.db dev_beam.jsonl
    python WikiSQL/evaluate.py WikiSQL/data/test.jsonl WikiSQL/data/test.db test_beam.jsonl
"""

import json
import os
import re
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm

from transformer import TransformerModel
from tokenizer import BOS_ID, EOS_ID, PAD_ID
from data_prep import AGG_OPS, COND_OPS, load_split, encode_source
from decode import (
    greedy_decode,
    beam_search,
    parse_prediction,
    encode_source_str,
    model,
    sp,
    DEVICE,
)


# ---------------------------------------------------------------------------
# Warn if using an old (flat-LR) checkpoint
# ---------------------------------------------------------------------------

def _check_checkpoint(path="best_model.pt"):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — run train.py first.")
        return
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "step" not in ck:
            print("=" * 60)
            print("WARNING: best_model.pt was trained with the OLD flat-LR code.")
            print("  Accuracy will be ~14%. Run  python train.py  to retrain")
            print("  with the correct Noam schedule before evaluating.")
            print("=" * 60)
        else:
            ep = ck.get("epoch", "?")
            vl = ck.get("val_loss", "?")
            st = ck.get("step", "?")
            print(f"Checkpoint: epoch={ep}, step={st}, val_loss={vl}  ✓ (Noam schedule)")
    except Exception as e:
        print(f"Could not inspect checkpoint: {e}")

_check_checkpoint()

# ---------------------------------------------------------------------------
# Load dev + test data
# ---------------------------------------------------------------------------

dev_examples,  dev_tables  = load_split("dev")
test_examples, test_tables = load_split("test")
print(f"Dev examples:  {len(dev_examples)}")
print(f"Test examples: {len(test_examples)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source(example, tables):
    table  = tables[example["table_id"]]
    header = table["header"]
    return encode_source(example["question"], header), header


def decode_greedy(src_tensor):
    ids = greedy_decode(src_tensor)
    return sp.decode(ids)


def decode_beam(src_tensor, beam_size=4):
    ids = beam_search(src_tensor, beam_size=beam_size)
    return sp.decode(ids)


def conds_match(pred_conds, gold_conds):
    """Order-independent comparison of condition lists."""
    def norm(c):
        return (int(c[0]), int(c[1]), str(c[2]).strip().lower())
    return set(norm(c) for c in pred_conds) == set(norm(c) for c in gold_conds)


# ---------------------------------------------------------------------------
# Write prediction files  (WikiSQL evaluator format)
# {"query": {"sel": int, "agg": int, "conds": [[col, op, val], ...]}}
# or {"error": "parse"} on failure
# ---------------------------------------------------------------------------

def write_predictions(split_examples, split_tables, out_path, mode="beam"):
    """
    mode: "greedy" or "beam"
    Writes one JSON line per example in WikiSQL evaluator format.
    Returns the list of parsed predictions (None where parse failed).
    """
    predictions = []
    parse_errors = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(split_examples):
            table  = split_tables[ex["table_id"]]
            header = table["header"]

            src_str = encode_source(ex["question"], header)
            src     = encode_source_str(src_str)

            if mode == "greedy":
                text = decode_greedy(src)
            else:
                text = decode_beam(src, beam_size=4)

            parsed = parse_prediction(text)

            if parsed is None:
                fout.write(json.dumps({"error": "parse"}) + "\n")
                predictions.append(None)
                parse_errors += 1
            else:
                fout.write(json.dumps({"query": parsed}) + "\n")
                predictions.append(parsed)

            if (i + 1) % 500 == 0:
                print(f"  [{mode}] {i+1}/{len(split_examples)}  "
                      f"parse errors so far: {parse_errors}")

    rate = parse_errors / len(split_examples) * 100
    print(f"  [{mode}] Done. Parse-failure rate: {rate:.2f}%  → {out_path}")
    return predictions, parse_errors


# ---------------------------------------------------------------------------
# Component accuracy
# ---------------------------------------------------------------------------

def component_accuracy(predictions, gold_examples):
    """
    Returns dict with sel%, agg%, where% over non-parse-error examples.
    """
    sel_correct  = agg_correct = where_correct = 0
    total        = 0

    for pred, ex in zip(predictions, gold_examples):
        if pred is None:
            continue
        gold = ex["sql"]
        total += 1
        if pred["sel"] == gold["sel"]:
            sel_correct += 1
        if pred["agg"] == gold["agg"]:
            agg_correct += 1
        if conds_match(pred["conds"], gold["conds"]):
            where_correct += 1

    if total == 0:
        return {"sel": 0.0, "agg": 0.0, "where": 0.0, "total": 0}

    return {
        "sel":   sel_correct  / total * 100,
        "agg":   agg_correct  / total * 100,
        "where": where_correct / total * 100,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Cross-attention visualization
# ---------------------------------------------------------------------------

def plot_cross_attention(example, tables, example_idx=0, save_path="cross_attention.png"):
    """
    Plots the last-layer decoder cross-attention (averaged over heads)
    for one dev example.
    """
    table  = tables[example["table_id"]]
    header = table["header"]

    src_str = encode_source(example["question"], header)
    src     = encode_source_str(src_str)

    # Encode source
    memory, src_mask = model.encode(src)

    # Generate with greedy, collecting cross-attention
    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)
    cross_attn_frames = []

    model.eval()
    with torch.no_grad():
        for _ in range(63):
            logits, cross_attn = model.decode(
                generated, memory, src_mask, return_cross_attn=True
            )
            # cross_attn: (1, heads, T, S) — take the last generated position
            # average over heads → (T, S), then take the row for the new token
            attn_avg = cross_attn[0].mean(dim=0)    # (T, S)
            cross_attn_frames.append(attn_avg[-1].cpu())   # last position

            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

    # Stack frames: (gen_steps, S)
    attn_matrix = torch.stack(cross_attn_frames, dim=0).numpy()

    # Source token labels
    src_ids   = sp.encode(src_str.lower()) + [EOS_ID]
    src_tokens = [sp.id_to_piece(i) for i in src_ids]

    # Target token labels (skip BOS)
    tgt_ids    = generated[0].tolist()[1:]
    tgt_tokens = [sp.id_to_piece(i) for i in tgt_ids]

    fig, ax = plt.subplots(figsize=(max(12, len(src_tokens) * 0.35),
                                     max(4, len(tgt_tokens) * 0.45)))
    im = ax.imshow(attn_matrix, aspect="auto", cmap="Blues",
                   vmin=0, vmax=attn_matrix.max())
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=8)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_title(
        f"Cross-Attention (last decoder layer, avg over heads)\n"
        f"Q: {example['question']}"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Cross-attention map saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Writing greedy predictions (dev) …")
    greedy_preds, greedy_errors = write_predictions(
        dev_examples, dev_tables, "dev_greedy.jsonl", mode="greedy"
    )

    print("\nWriting beam predictions (dev) …")
    beam_preds, beam_errors = write_predictions(
        dev_examples, dev_tables, "dev_beam.jsonl", mode="beam"
    )

    print("\nWriting beam predictions (test) …")
    test_preds, test_errors = write_predictions(
        test_examples, test_tables, "test_beam.jsonl", mode="beam"
    )

    # ---- Component accuracy (dev, beam) ----
    print("\n--- Component Accuracy (dev, beam) ---")
    acc = component_accuracy(beam_preds, dev_examples)
    print(f"  sel  correct : {acc['sel']:.2f}%")
    print(f"  agg  correct : {acc['agg']:.2f}%")
    print(f"  where correct: {acc['where']:.2f}%")
    print(f"  (evaluated on {acc['total']} non-error examples)")

    print("\n--- Component Accuracy (dev, greedy) ---")
    acc_g = component_accuracy(greedy_preds, dev_examples)
    print(f"  sel  correct : {acc_g['sel']:.2f}%")
    print(f"  agg  correct : {acc_g['agg']:.2f}%")
    print(f"  where correct: {acc_g['where']:.2f}%")

    # ---- Cross-attention plot ----
    print("\nPlotting cross-attention for dev example 0 …")
    plot_cross_attention(dev_examples[0], dev_tables, save_path="cross_attention.png")

    # ---- Parse-failure rates ----
    n_dev  = len(dev_examples)
    n_test = len(test_examples)
    print(f"\nParse-failure rate — greedy: {greedy_errors/n_dev*100:.2f}%  "
          f"beam (dev): {beam_errors/n_dev*100:.2f}%  "
          f"beam (test): {test_errors/n_test*100:.2f}%")

    # ---- Update results.json with component accuracy ----
    if os.path.exists("results.json"):
        with open("results.json", encoding="utf-8") as f:
            results = json.load(f)
        results["component_accuracy_dev_beam"] = {
            "sel":   round(acc["sel"],   2),
            "agg":   round(acc["agg"],   2),
            "where": round(acc["where"], 2),
        }
        results["parse_failure_rate_beam_dev"]  = round(beam_errors  / n_dev  * 100, 2)
        results["parse_failure_rate_beam_test"] = round(test_errors  / n_test * 100, 2)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print("Updated results.json with component accuracy and parse-failure rates.")

    print("\nDone. Run the official evaluator with:")
    print("  python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl "
          "WikiSQL/data/dev.db dev_greedy.jsonl")
    print("  python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl "
          "WikiSQL/data/dev.db dev_beam.jsonl")
    print("  python WikiSQL/evaluate.py WikiSQL/data/test.jsonl "
          "WikiSQL/data/test.db test_beam.jsonl")