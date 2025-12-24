import argparse, json, os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--deepspeed", type=str, default="ds_config.json")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4bit load (QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    def preprocess(ex):
        text = ex["prompt"] + ex["target"]
        out = tok(
            text,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tr = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)
    ev = ds["eval"].map(preprocess, remove_columns=ds["eval"].column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,   
        learning_rate=args.lr,
        weight_decay=0.01,                
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        deepspeed=args.deepspeed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr,
        eval_dataset=ev,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
