
import json

count = 0

with open("train_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        if "what school" in item["src"] or "what school/club" in item["src"]:
            print("SOURCE:", item["src"])
            print("TARGET:", item["tgt"])
            print("-" * 60)

            count += 1

            if count == 15:
                break

print("Examples found:", count)