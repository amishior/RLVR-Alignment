import argparse
import json
import os
import re
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


# ----------------------------
# GSM8K 格式相关（你之前的格式）
# ----------------------------
ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)

def extract_answer(text: str) -> str:
    m = ANS_RE.search(text)
    return m.group(1).strip() if m else ""

def has_format(text: str) -> bool:
    return ("<reasoning>" in text and "</reasoning>" in text and "<answer>" in text and "</answer>" in text)

def normalize(s: str) -> str:
    return s.strip().replace(",", "")


# ----------------------------
# Policy + Value (Critic) 合体模型
# 不用 TRL wrapper，避免缺属性
# ----------------------------
class PolicyValueModel(nn.Module):
    """
    一个最小但“正宗”的 Actor-Critic：
    - base: CausalLM，产出 logits
    - value_head: 从最后一层 hidden state 预测每个 token 的 value（我们取最后 token 的 value 做序列级）
    """
    def __init__(self, base_lm: nn.Module):
        super().__init__()
        self.base = base_lm

        # 尽量稳健地拿 hidden size
        hs = None
        cfg = getattr(base_lm, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            hs = cfg.hidden_size
        if hs is None:
            # 兜底：从 embedding 维度推断
            hs = base_lm.get_input_embeddings().weight.shape[1]

        self.value_head = nn.Linear(hs, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits                                # [B, T, V]
        h = out.hidden_states[-1]                           # [B, T, H]
        values = self.value_head(h).squeeze(-1)             # [B, T]
        return logits, values

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)


# ----------------------------
# logprob / entropy 工具
# ----------------------------
def masked_logprob_and_entropy(
    logits: torch.Tensor,          # [B, T, V]
    input_ids: torch.Tensor,       # [B, T]
    response_start: torch.Tensor   # [B] 每个样本 response 第一个 token 在 input_ids 中的 index
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算“response token”的序列 logprob（sum）和 entropy（sum）
    - logits[i] 预测 input_ids[i+1]
    - response 第一个 token index = s
      对应 token_logp 的 index = s-1
    """
    B, T, V = logits.shape
    # shift
    shift_logits = logits[:, :-1, :]               # [B, T-1, V]
    shift_labels = input_ids[:, 1:]                # [B, T-1]

    logp = F.log_softmax(shift_logits, dim=-1)     # [B, T-1, V]
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # entropy per position: -sum(p*logp)
    p = logp.exp()
    token_ent = -(p * logp).sum(dim=-1)            # [B, T-1]

    # mask：positions >= (response_start - 1)
    idx = torch.arange(T - 1, device=logits.device).unsqueeze(0).expand(B, -1)
    start_pos = (response_start - 1).unsqueeze(1)  # [B, 1]
    mask = idx >= start_pos

    seq_logp = (token_logp * mask).sum(dim=-1)     # [B]
    seq_ent  = (token_ent  * mask).sum(dim=-1)     # [B]
    return seq_logp, seq_ent


# ----------------------------
# RM 打分
# ----------------------------
@torch.no_grad()
def rm_score(
    rm_model,
    tokenizer,
    prompt_texts: List[str],
    response_texts: List[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    reward_model: SequenceClassification(num_labels=1)
    输入 = prompt + response
    输出 = 标量 logit（越大越好）
    """
    texts = [f"{p}\n\n{r}" for p, r in zip(prompt_texts, response_texts)]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    out = rm_model(**batch).logits.squeeze(-1)   # [B]
    return out


# ----------------------------
# PPO 训练主逻辑（单卡/可多卡）
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--rm_dir", required=True)
    ap.add_argument("--ppo_prompts_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--ds_config", type=str, default="ds_ppo.json")
    ap.add_argument("--steps", type=int, default=200)

    ap.add_argument("--micro_batch_size", type=int, default=1)   # 每次生成/更新的样本数（先用 1 跑通）
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--rm_max_len", type=int, default=1536)

    # PPO 超参
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--kl_coef", type=float, default=0.05)

    # reward shaping（可开可关）
    ap.add_argument("--format_bonus", type=float, default=0.2)
    ap.add_argument("--correct_bonus", type=float, default=1.0)
    ap.add_argument("--missing_answer_penalty", type=float, default=0.2)

    # precision
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    # LoRA（可选，默认开，方便单卡）
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # deepspeed launcher 会传
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- deepspeed / device ----
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # if world_size > 1:
    #     deepspeed.init_distributed(dist_backend="nccl")
    deepspeed.init_distributed(dist_backend="nccl")

    # ---- load data ----
    ds = load_dataset("json", data_files={"train": args.ppo_prompts_jsonl})["train"]

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- dtype ----
    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # ---- load models ----
    # policy (trainable)
    base_policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=None,
    )
    base_policy.config.use_cache = False

    # （可选）LoRA：如果你想先跑通“全参”，就不加 --use_lora
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )
        base_policy = get_peft_model(base_policy, lora_cfg)

    model = PolicyValueModel(base_policy)

    # ref (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.config.use_cache = False

    # reward model (frozen)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_dir,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    rm_model.eval()
    for p in rm_model.parameters():
        p.requires_grad = False

    # ---- DeepSpeed config load & override ----
    with open(args.ds_config, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)

    ds_cfg["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_cfg["gradient_accumulation_steps"] = args.grad_accum
    # 同步 lr / wd（避免你忘了改 json）
    ds_cfg["optimizer"]["params"]["lr"] = args.lr
    ds_cfg["optimizer"]["params"]["weight_decay"] = args.weight_decay

    # precision switch
    if args.precision == "bf16":
        ds_cfg["bf16"] = {"enabled": True}
        ds_cfg["fp16"] = {"enabled": False}
    elif args.precision == "fp16":
        ds_cfg["bf16"] = {"enabled": False}
        ds_cfg["fp16"] = {"enabled": True}
    else:
        ds_cfg["bf16"] = {"enabled": False}
        ds_cfg["fp16"] = {"enabled": False}

    # ---- init deepspeed engine ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        config=ds_cfg,
        dist_init_required=False,   # ✅ 我们已手动 init，不让 DS 再 init 一遍
    )
    engine.train()

    # ---- generation kwargs ----
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # ---- training loop ----
    rng = random.Random(42)

    for step in range(args.steps):
        # 采一个 micro-batch（先用 batch=1 最稳；你也可设 >1）
        idxs = [rng.randrange(len(ds)) for _ in range(args.micro_batch_size)]
        batch_ex = [ds[i] for i in idxs]
        prompts = [ex["prompt"] for ex in batch_ex]
        golds   = [ex["gold_final"] for ex in batch_ex]

        # tokenize prompts with padding
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        prompt_lens = attn_mask.sum(dim=1)  # [B] 真实 prompt 长度（含 pad 过滤）

        # -------- rollout: generate with current policy --------
        engine.module.base.eval()  # 生成阶段 eval（更稳）
        with torch.no_grad():
            gen_out = engine.module.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                **gen_kwargs
            )
        engine.module.base.train()

        # response_only_ids：每个样本从 prompt_len 开始切
        # 由于 batch padding，prompt_len 每个样本不同，需要逐条切
        response_texts = []
        full_input_ids_list = []
        response_start_list = []
        for b in range(gen_out.size(0)):
            pl = int(prompt_lens[b].item())
            full_ids = gen_out[b:b+1, :]  # [1, T]
            resp_ids = full_ids[:, pl:]   # completion-only
            response_texts.append(tok.decode(resp_ids[0], skip_special_tokens=True))
            full_input_ids_list.append(full_ids)
            response_start_list.append(pl)

        # 拼成一个 batch（用 pad 对齐）
        # 这里为了简单可靠：把每个 full_ids pad 成同一长度
        maxT = max(x.size(1) for x in full_input_ids_list)
        full_input_ids = torch.full((len(full_input_ids_list), maxT), tok.pad_token_id, device=device, dtype=torch.long)
        full_attn_mask = torch.zeros((len(full_input_ids_list), maxT), device=device, dtype=torch.long)
        for i, ids in enumerate(full_input_ids_list):
            T = ids.size(1)
            full_input_ids[i, :T] = ids[0]
            full_attn_mask[i, :T] = 1
        response_start = torch.tensor(response_start_list, device=device, dtype=torch.long)  # [B]

        # -------- compute old logp/value and ref logp (no grad) --------
        with torch.no_grad():
            old_logits, old_values_all = engine.module(full_input_ids, attention_mask=full_attn_mask)
            old_logp, old_ent = masked_logprob_and_entropy(old_logits, full_input_ids, response_start)
            old_value = old_values_all[:, -1]  # [B] 序列末端 value（简化）

            ref_out = ref_model(full_input_ids, attention_mask=full_attn_mask, return_dict=True)
            ref_logits = ref_out.logits
            ref_logp, _ = masked_logprob_and_entropy(ref_logits, full_input_ids, response_start)

            # KL 近似：logp_policy - logp_ref（序列级 sum）
            kl = (old_logp - ref_logp)  # [B]

            # RM reward
            r_rm = rm_score(rm_model, tok, prompts, response_texts, args.rm_max_len, device)  # [B]

        # -------- shaping reward（可选，但建议 GSM8K 先开稳定） --------
        r_shape = torch.zeros_like(r_rm)
        for i, resp in enumerate(response_texts):
            if has_format(resp):
                r_shape[i] += args.format_bonus
            pred = extract_answer(resp)
            if not pred:
                r_shape[i] -= args.missing_answer_penalty
            else:
                if normalize(pred) == normalize(golds[i]):
                    r_shape[i] += args.correct_bonus

        # 总 reward：RM + shaping - KL penalty（这里把 KL 当作 reward penalty）
        # 这更贴近 RLHF 常用形式
        rewards = r_rm + r_shape - args.kl_coef * kl

        # advantage（简化为：A = R - V_old）
        adv = (rewards - old_value).detach()
        returns = rewards.detach()

        # -------- PPO update (with grad, deepspeed) --------
        # forward new
        new_logits, new_values_all = engine.module(full_input_ids, attention_mask=full_attn_mask)
        new_logp, new_ent = masked_logprob_and_entropy(new_logits, full_input_ids, response_start)
        new_value = new_values_all[:, -1]

        # PPO ratio
        ratio = torch.exp(new_logp - old_logp.detach())

        # clipped policy loss
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        # value loss
        value_loss = 0.5 * torch.mean((new_value - returns) ** 2)

        # entropy bonus（可选）
        ent_bonus = torch.mean(new_ent)

        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent_bonus

        engine.backward(loss)
        engine.step()

        if step % 10 == 0 and (args.local_rank == 0):
            avg_rm = r_rm.mean().item()
            avg_shape = r_shape.mean().item()
            avg_kl = kl.mean().item()
            avg_r = rewards.mean().item()
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"rm={avg_rm:.3f} shape={avg_shape:.3f} kl={avg_kl:.3f} reward={avg_r:.3f} "
                f"pred={extract_answer(response_texts[0])!r} gold={golds[0]!r}"
            )

        # checkpoint
        if step > 0 and step % 50 == 0 and (args.local_rank == 0):
            ckpt_dir = os.path.join(args.output_dir, f"ckpt_step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            # 保存 policy（如果是 LoRA，只会保存 adapter；否则保存全参）
            if hasattr(engine.module.base, "save_pretrained"):
                engine.module.base.save_pretrained(ckpt_dir)
            else:
                torch.save(engine.module.base.state_dict(), os.path.join(ckpt_dir, "policy_base.pt"))

            # 保存 value head
            torch.save(engine.module.value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))
            tok.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "train_args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            print("Saved ckpt:", ckpt_dir)

    # final save
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if hasattr(engine.module.base, "save_pretrained"):
            engine.module.base.save_pretrained(args.output_dir)
        else:
            torch.save(engine.module.base.state_dict(), os.path.join(args.output_dir, "policy_base.pt"))
        torch.save(engine.module.value_head.state_dict(), os.path.join(args.output_dir, "value_head.pt"))
        tok.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print("Saved final to:", args.output_dir)


if __name__ == "__main__":
    main()
