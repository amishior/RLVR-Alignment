# Single-GPU, hydra-safe, no-tilde model path
# bash zh_SFT_Qwen3-4B.sh sft_outputs_qwen3_4b
set -x

# 只要求 save_path
if [ "$#" -lt 1 ]; then
    echo "Usage: SFT_Qwen3-4B.sh <save_path> [other_configs...]"
    exit 1
fi

# 单卡默认 1 个进程（可用 NPROC_PER_NODE 覆盖）
nproc_per_node=${NPROC_PER_NODE:-1}

save_path=$1
shift 1  # 透传剩余 hydra overrides

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$HOME/data/gsm8k_zh/train.parquet \
  data.val_files=$HOME/data/gsm8k_zh/test.parquet \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys="['question']" \
  +data.response_dict_keys="['answer']" \
  data.micro_batch_size_per_gpu=8 \
  optim.lr=2e-5 \
  optim.weight_decay=0.05 \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.clip_grad=1.0 \
  optim.lr_scheduler=cosine \
  model.fsdp_config.model_dtype=bf16 \
  model.lora_rank=32 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  model.partial_pretrain=/home/amishor/jupyterlab/Qwen3-4B \
  trainer.default_local_dir=$save_path \
  trainer.project_name=gsm8k-zh-sft \
  trainer.experiment_name=gsm8k-zh-sft-Qwen3-4B \
  trainer.total_epochs=3 \
  trainer.logger='["console","wandb"]' \
  $@
