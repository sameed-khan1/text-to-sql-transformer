import json
import re


def parse_prediction(sql):
    sql = sql.lower().strip()

    select_match = re.search(
        r"select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>",
        sql
    )

    if not select_match:
        return None

    agg_names = {
        None: 0,
        "max": 1,
        "min": 2,
        "count": 3,
        "sum": 4,
        "avg": 5
    }

    agg = agg_names[select_match.group(1)]
    sel = int(select_match.group(2))

    conditions = []

    where_match = re.search(
        r"where\s+(.+)",
        sql
    )

    if where_match:
        where_text = where_match.group(1)

        parts = re.split(
            r"\s+and\s+",
            where_text
        )

        for part in parts:
            match = re.match(
                r"<c(\d+)>\s*(=|>|<)\s*(.+)",
                part.strip()
            )

            if not match:
                continue

            col = int(match.group(1))
            op_text = match.group(2)
            value = match.group(3).strip()

            op = {
                "=": 0,
                ">": 1,
                "<": 2
            }[op_text]

            conditions.append([
                col,
                op,
                value
            ])

    return {
        "sel": sel,
        "agg": agg,
        "conds": conditions
    }


with open(
    "test_predictions.jsonl",
    encoding="utf-8"
) as f:
    predictions = [
        json.loads(line)
        for line in f
    ]


with open(
    "test_predictions_eval.jsonl",
    "w",
    encoding="utf-8"
) as out:

    for prediction in predictions:

        query = parse_prediction(
            prediction["sql"]
        )

        if query is None:
            result = {
                "error": "Could not parse prediction"
            }
        else:
            result = {
                "query": query
            }

        out.write(
            json.dumps(result)
            + "\n"
        )


print("Saved: test_predictions_eval.jsonl")