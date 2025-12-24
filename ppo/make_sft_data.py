import argparse, os, json
import pandas as pd

SYSTEM = (
    "You are a careful math solver. "
    "You must show your work, then give a final answer. "
    "Return exactly this format:\n"
    "<reasoning>...</reasoning>\n"
    "<answer>...</answer>\n"
    "Keep reasoning concise."
)

def format_example(question: str, gold_answer: str | None):
    # GSM8K gold answer usually ends with '#### <number>'
    # We'll keep full rationale in <reasoning> and parse final in <answer>
    prompt = (
        f"<task>\n{SYSTEM}\n</task>\n\n"
        f"<input>\n{question.strip()}\n</input>\n\n"
        f"<output_format>\n<reasoning>...</reasoning>\n<answer>...</answer>\n</output_format>\n"
    )

    if gold_answer is None:
        return {"prompt": prompt}

    ans = gold_answer.strip()

    # GSM8K 'main' split usually: rationale + '#### final'
    final = None
    if "####" in ans:
        parts = ans.split("####")
        reasoning = parts[0].strip()
        final = parts[-1].strip()
    else:
        reasoning = ans
        final = ans

    target = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{final}\n</answer>\n"
    return {"prompt": prompt, "target": target}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.002)  # 小验证集就够跑通
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_parquet(args.train_parquet)
    test_df  = pd.read_parquet(args.test_parquet)

    # Columns are typically: question, answer
    assert "question" in train_df.columns
    assert "answer" in train_df.columns
    assert "question" in test_df.columns

    # shuffle train
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_val = max(1, int(len(train_df) * args.val_ratio))
    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    tr_df  = train_df.iloc[n_val:].reset_index(drop=True)

    def dump_jsonl(df, path, has_target: bool):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                ex = format_example(row["question"], row["answer"] if has_target else None)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump_jsonl(tr_df,  os.path.join(args.out_dir, "train.jsonl"), True)
    dump_jsonl(val_df, os.path.join(args.out_dir, "val.jsonl"),   True)
    dump_jsonl(test_df,os.path.join(args.out_dir, "test.jsonl"),  False)

    print("Wrote:", args.out_dir)

if __name__ == "__main__":
    main()
