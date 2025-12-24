import argparse, json, os, random, re
from typing import Tuple

ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)

def extract_gold_final(target: str) -> str:
    m = ANS_RE.search(target)
    return m.group(1).strip() if m else ""

def corrupt_answer(ans: str) -> str:
    """
    生成一个“明显错误”的 answer，用于构造 rejected 样本。
    GSM8K 最终答案多是数字；我们尽量做数字扰动，否则做字符串扰动。
    """
    s = ans.strip().replace(",", "")
    try:
        x = float(s)
        # 扰动：加上一个随机偏移，避免等于原答案
        delta = random.choice([1, 2, 3, 5, 10, 17])
        y = x + delta
        # 整数尽量输出整数
        if abs(y - round(y)) < 1e-9:
            return str(int(round(y)))
        return str(y)
    except:
        # 非数字：简单加后缀扰动
        return ans + " (wrong)"

def build_chosen_rejected(prompt: str, target: str) -> Tuple[str, str, str]:
    """
    chosen：使用 gold target（含 reasoning+answer）
    rejected：保留 reasoning，但把 <answer> 换成错误答案（更贴近偏好学习）
    gold_final：正确 final answer
    """
    gold_final = extract_gold_final(target)
    if not gold_final:
        return "", "", ""

    chosen = target.strip()

    wrong = corrupt_answer(gold_final)
    # 用 wrong 替换 <answer> 内部
    rejected = re.sub(
        r"(<answer>\s*)(.*?)(\s*</answer>)",
        lambda m: f"{m.group(1)}{wrong}{m.group(3)}",
        chosen,
        flags=re.S,
    )

    return chosen, rejected, gold_final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train_jsonl", required=True, help="data/processed_gsm8k/train.jsonl")
    ap.add_argument("--out_dir", required=True, help="data/rlhf_gsm8k")
    ap.add_argument("--max_samples", type=int, default=5000, help="先跑通可设 2000~5000")
    args = ap.parse_args()

    random.seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    ppo_rows = []
    rm_rows = []

    with open(args.in_train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            prompt = j["prompt"]
            target = j["target"]
            chosen, rejected, gold_final = build_chosen_rejected(prompt, target)
            if not chosen:
                continue

            ppo_rows.append({
                "prompt": prompt,
                "gold_final": gold_final,
            })
            rm_rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    random.shuffle(ppo_rows)
    random.shuffle(rm_rows)

    ppo_rows = ppo_rows[: args.max_samples]
    rm_rows = rm_rows[: args.max_samples]

    ppo_path = os.path.join(args.out_dir, "ppo_prompts.jsonl")
    rm_path  = os.path.join(args.out_dir, "rm_pairs.jsonl")

    with open(ppo_path, "w", encoding="utf-8") as f:
        for r in ppo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(rm_path, "w", encoding="utf-8") as f:
        for r in rm_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote PPO prompts: {len(ppo_rows)} -> {ppo_path}")
    print(f"Wrote RM pairs:   {len(rm_rows)} -> {rm_path}")

if __name__ == "__main__":
    main()
