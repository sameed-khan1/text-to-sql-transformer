import json
from pathlib import Path

DATA_DIR = Path("WikiSQL/data")
AGG_OPS = ["", "max", "min", "count", "sum", "avg"]
COND_OPS = ["=", ">", "<"]
MAX_COLS = 64


def load_split(split):
    tables = {}

    with open(DATA_DIR / f"{split}.tables.jsonl", encoding="utf-8") as f:
        for line in f:
            table = json.loads(line)
            tables[table["id"]] = table

    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    return examples, tables


def encode_source(question, header):
    cols = " ".join(
        f"<c{i}> {name}" for i, name in enumerate(header)
    )
    return f"{question.strip()} <sep> columns {cols}".lower()


def encode_target(sql):
    output = ["select"]

    if sql["agg"]:
        output.append(AGG_OPS[sql["agg"]])

    output.append(f"<c{sql['sel']}>")

    for i, (col, op, value) in enumerate(sql["conds"]):
        output.extend([
            "where" if i == 0 else "and",
            f"<c{col}>",
            COND_OPS[op],
            str(value)
        ])

    return " ".join(output).lower()


def build_pairs(split):
    examples, tables = load_split(split)
    pairs = []

    for example in examples:
        header = tables[example["table_id"]]["header"]

        pairs.append({
            "table_id": example["table_id"],
            "src": encode_source(example["question"], header),
            "tgt": encode_target(example["sql"])
        })

    return pairs


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        pairs = build_pairs(split)

        with open(
            f"{split}_pairs.jsonl", "w", encoding="utf-8"
        ) as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        print(f"{split}: {len(pairs)} pairs")
        print(pairs[0]["src"])
        print(pairs[0]["tgt"])