import argparse, json, os, re, random

ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)

def extract_gold_final(target: str) -> str:
    m = ANS_RE.search(target)
    if not m:
        return ""
    return m.group(1).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_samples", type=int, default=2000)  # 先小点跑通
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            gold = extract_gold_final(j["target"])
            if gold:
                rows.append({"prompt": j["prompt"], "gold_final": gold})

    random.seed(42)
    random.shuffle(rows)
    rows = rows[: args.max_samples]

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
