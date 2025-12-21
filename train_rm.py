import argparse, os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="base model path")
    ap.add_argument("--rm_pairs_jsonl", required=True, help="data/rlhf_gsm8k/rm_pairs.jsonl")
    ap.add_argument("--out_dir", required=True, help="outputs/rm_gsm8k")
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("json", data_files={"train": args.rm_pairs_jsonl})["train"]

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # num_labels=1 => 回归/打分
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
    )

    def pack(prompt, resp):
        # 把 prompt + resp 拼一起打分
        return f"{prompt}\n\n{resp}"

    def preprocess(ex):
        chosen_text = pack(ex["prompt"], ex["chosen"])
        rejected_text = pack(ex["prompt"], ex["rejected"])

        c = tok(chosen_text, truncation=True, max_length=args.max_len)
        r = tok(rejected_text, truncation=True, max_length=args.max_len)

        return {
            "chosen_input_ids": c["input_ids"],
            "chosen_attention_mask": c["attention_mask"],
            "rejected_input_ids": r["input_ids"],
            "rejected_attention_mask": r["attention_mask"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    collator = DataCollatorWithPadding(tok, return_tensors="pt")

    class PairwiseRMTrainer(Trainer):
        """
        Pairwise loss: -log(sigmoid(score_chosen - score_rejected))
        """
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            chosen = {
                "input_ids": inputs["chosen_input_ids"],
                "attention_mask": inputs["chosen_attention_mask"],
            }
            rejected = {
                "input_ids": inputs["rejected_input_ids"],
                "attention_mask": inputs["rejected_attention_mask"],
            }

            chosen_out = model(**chosen).logits.squeeze(-1)    # [bs]
            rejected_out = model(**rejected).logits.squeeze(-1) # [bs]

            loss = -F.logsigmoid(chosen_out - rejected_out).mean()
            return (loss, {"chosen": chosen_out, "rejected": rejected_out}) if return_outputs else loss

        def get_train_dataloader(self):
            # 让 DataCollatorWithPadding 同时 pad chosen/rejected
            # 简单做法：手动 pad 两套字段
            from torch.utils.data import DataLoader

            def _collate(features):
                chosen_feats = [{"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]} for f in features]
                rejected_feats = [{"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]} for f in features]
                c = collator(chosen_feats)
                r = collator(rejected_feats)
                return {
                    "chosen_input_ids": c["input_ids"],
                    "chosen_attention_mask": c["attention_mask"],
                    "rejected_input_ids": r["input_ids"],
                    "rejected_attention_mask": r["attention_mask"],
                }

            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=_collate,
            )

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
    )

    trainer = PairwiseRMTrainer(
        model=rm,
        args=targs,
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved RM to:", args.out_dir)

if __name__ == "__main__":
    main()
