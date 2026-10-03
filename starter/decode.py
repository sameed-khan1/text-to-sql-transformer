import re
import torch
import sentencepiece as spm

from transformer import TransformerModel, make_padding_mask
from tokenizer import BOS_ID, EOS_ID, PAD_ID
from data_prep import AGG_OPS, COND_OPS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column token pattern:  <c0>, <c1>, …
_COL_RE  = re.compile(r"<c(\d+)>")
# AGG names (lower-case, in order matching AGG_OPS)
_AGG_MAP = {a.lower(): i for i, a in enumerate(AGG_OPS) if a}   # skip "" (none)
# COND operators
_OP_MAP  = {op: i for i, op in enumerate(COND_OPS)}

# Length-normalization exponent for beam search
_LEN_ALPHA = 0.6


# ---------------------------------------------------------------------------
# Load shared tokenizer + model (module-level, so callers just import)
# ---------------------------------------------------------------------------

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")

model = TransformerModel(
    vocab_size=sp.get_piece_size(),
    d_model=256,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=1024,
    dropout=0.1,
    pad_id=PAD_ID,
).to(DEVICE)

checkpoint = torch.load("best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------------------------------------------------------
# Helper: encode a source string to a tensor
# ---------------------------------------------------------------------------

def encode_source_str(source_str):
    """
    source_str: pre-built string like
        "what is X <sep> <c0> player <c1> no. ..."
    Returns (1, S) LongTensor ready for the model.
    """
    ids = sp.encode(source_str.lower().strip()) + [EOS_ID]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(src_tensor, max_len=64):
    """
    src_tensor: (1, S) LongTensor
    Returns list of token ids (includes BOS and EOS if generated).
    """
    memory, src_mask = model.encode(src_tensor)

    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len - 1):
        logits = model.decode(generated, memory, src_mask)   # (1, T, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, 1)
        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == EOS_ID:
            break

    return generated[0].tolist()


# ---------------------------------------------------------------------------
# Beam search decoding (encoder cached, length-normalized)
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(src_tensor, beam_size=4, max_len=64, alpha=_LEN_ALPHA):
    """
    src_tensor: (1, S) LongTensor
    Returns list of token ids for the best hypothesis.
    Length-normalized score: sum(log_prob) / length^alpha.
    """
    memory, src_mask = model.encode(src_tensor)

    # Expand memory for beam: (beam, S, d_model)
    memory_exp  = memory.expand(beam_size, -1, -1)
    src_mask_exp = src_mask.expand(beam_size, -1, -1, -1)

    # Each beam: (token_ids, cumulative_log_prob)
    beams     = [([BOS_ID], 0.0)]
    completed = []          # finished hypotheses

    for step in range(1, max_len):
        if not beams:
            break

        # Build batch decoder input for all live beams
        max_t = max(len(b[0]) for b in beams)
        tgt_batch = torch.full(
            (len(beams), max_t), PAD_ID, dtype=torch.long, device=DEVICE
        )
        for i, (tok, _) in enumerate(beams):
            tgt_batch[i, :len(tok)] = torch.tensor(tok, dtype=torch.long)

        # Use the right slice of expanded memory
        mem_slice  = memory_exp[:len(beams)]
        mask_slice = src_mask_exp[:len(beams)]

        logits = model.decode(tgt_batch, mem_slice, mask_slice)   # (B, T, V)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)   # (B, V)

        candidates = []
        for i, (tokens, score) in enumerate(beams):
            top_vals, top_ids = torch.topk(log_probs[i], beam_size)
            for val, idx in zip(top_vals.tolist(), top_ids.tolist()):
                new_tokens = tokens + [idx]
                new_score  = score + val
                candidates.append((new_tokens, new_score))

        # Sort and keep top beam_size
        candidates.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
        candidates = candidates[:beam_size * 2]

        beams = []
        for tokens, score in candidates:
            if tokens[-1] == EOS_ID:
                completed.append((tokens, score))
            else:
                beams.append((tokens, score))
            if len(beams) >= beam_size:
                break

        if len(completed) >= beam_size:
            break

    # If nothing completed, use live beams
    all_hyps = completed if completed else beams
    all_hyps.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
    return all_hyps[0][0]


# ---------------------------------------------------------------------------
# Parser: token string → WikiSQL structured query dict
# ---------------------------------------------------------------------------

def parse_prediction(text):
    """
    Convert model output string (e.g. "select count <c2> where <c0> = kim")
    into {"sel": int, "agg": int, "conds": [[col, op, val], ...]}.
    Returns None if the string cannot be parsed.
    """
    text = text.lower().strip()

    # ---- SELECT clause ----
    # Pattern: "select [agg] <cN>"
    sel_match = re.search(
        r"select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>",
        text,
    )
    if not sel_match:
        return None

    agg_str = sel_match.group(1)
    agg     = _AGG_MAP.get(agg_str, 0) if agg_str else 0
    sel     = int(sel_match.group(2))

    # ---- WHERE clause ----
    # Pattern: "where <cN> op value [and <cN> op value]*"
    conds = []
    where_part = text[sel_match.end():]

    # Split on the keyword boundary
    cond_re = re.compile(
        r"(?:where|and)\s+<c(\d+)>\s*(=|>|<)\s*(.*?)(?=\s+and\s+<c\d+>|$)"
    )
    for m in cond_re.finditer(where_part):
        col   = int(m.group(1))
        op    = _OP_MAP.get(m.group(2))
        value = m.group(3).strip()
        if op is None:
            continue
        conds.append([col, op, value])

    return {"sel": sel, "agg": agg, "conds": conds}


# ---------------------------------------------------------------------------
# Human-readable SQL (for the front end)
# ---------------------------------------------------------------------------

def to_readable_sql(query, header):
    """
    query:  {"sel": int, "agg": int, "conds": [[col, op, val], ...]}
    header: list of column name strings
    Returns a readable SQL string.
    """
    sel_col = header[query["sel"]] if query["sel"] < len(header) else f"col{query['sel']}"
    agg     = AGG_OPS[query["agg"]]

    if agg:
        select_part = f"{agg}({sel_col})"
    else:
        select_part = sel_col

    sql = f"SELECT {select_part} FROM table"

    if query["conds"]:
        parts = []
        for col, op, val in query["conds"]:
            col_name = header[col] if col < len(header) else f"col{col}"
            parts.append(f"{col_name} {COND_OPS[op]} '{val}'")
        sql += " WHERE " + " AND ".join(parts)

    return sql


# ---------------------------------------------------------------------------
# Quick demo (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = "what school did patrick o'bryant play for"
    source   = (
        question
        + " <sep> <c0> player <c1> no. <c2> nationality "
        + "<c3> position <c4> years in toronto <c5> school/club team"
    ).lower()

    src = encode_source_str(source)

    # Greedy
    greedy_ids = greedy_decode(src)
    greedy_txt = sp.decode(greedy_ids)
    print("Question :", question)
    print("Greedy   :", greedy_txt)

    parsed = parse_prediction(greedy_txt)
    header = ["Player", "No.", "Nationality", "Position",
              "Years in Toronto", "School/Club Team"]
    if parsed:
        print("Parsed   :", parsed)
        print("SQL      :", to_readable_sql(parsed, header))
    else:
        print("Parse failed")

    # Beam search
    beam_ids = beam_search(src, beam_size=4)
    beam_txt = sp.decode(beam_ids)
    print("\nBeam (4) :", beam_txt)

    parsed_b = parse_prediction(beam_txt)
    if parsed_b:
        print("Parsed   :", parsed_b)
        print("SQL      :", to_readable_sql(parsed_b, header))
    else:
        print("Parse failed")