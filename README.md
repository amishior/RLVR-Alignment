# RLVR-Alignment

> **RLVR-Alignment** 是一个面向大语言模型的 **RLVR / GRPO 对齐实验仓库**，主要用于研究和复现基于 *Verifier / Reward* 的强化学习对齐流程，覆盖 **SFT → GRPO（RLVR）→ LoRA RLVR** 等典型训练路径。

本项目以 **Qwen3 系列模型** 为主要实验对象，强调：

* 可复现的工程化脚本
* 清晰的训练阶段拆分
* 易于修改和扩展的 RLVR 结构

---

## ✨ 项目特性

* ✅ **完整对齐链路**：SFT → GRPO / RLVR → LoRA-RLVR
* ✅ **轻量可跑**：0.6B 模型即可验证 RLVR 效果
* ✅ **工程友好**：Shell 脚本一键启动，参数清晰
* ✅ **研究导向**：方便插拔 verifier / reward / rewrite 逻辑

---

## 📂 项目结构

```text
RLVR-Alignment/
├── data/                     # 训练与评测数据（示例或自定义）
├── scripts/
│   ├── SFT_Qwen3-4B.sh        # Qwen3-4B 的监督微调（SFT）
│   ├── SFT_Qwen3-4B_zh.sh     # 中文数据 SFT 示例
│   ├── GRPO_Qwen3_0.6b.sh     # Qwen3-0.6B 的 GRPO / RLVR 训练
│   └── GRPO_lora_Qwen3_0.6b.sh# 基于 LoRA 的 GRPO / RLVR
├── src/
│   ├── trainer/              # SFT / GRPO Trainer 实现
│   ├── reward/               # Reward / Verifier 定义
│   ├── rollout/              # 生成与采样逻辑
│   └── utils/                # 通用工具
├── requirements.txt
└── README.md
```

---

## 🧠 方法简介：什么是 RLVR / GRPO？

**RLVR（Reinforcement Learning with Verifier/Reward）** 是一种不依赖人工偏好对比数据（如 RLHF 中的 pairwise preference），而是通过：

1. **模型生成多个候选回答（rollout）**
2. **Verifier / Reward Model 对候选进行自动评估**
3. **使用 GRPO（Group Relative Policy Optimization）进行更新**

来实现对齐的方法。

相比 PPO，GRPO 的特点是：

* 不需要单独的 value model
* 更适合多候选相对排序
* 工程实现更简洁，数值稳定性更好

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
pip install -r requirements.txt
```

建议环境：

* Python ≥ 3.9
* PyTorch ≥ 2.1
* CUDA ≥ 12.x

---

### 2️⃣ 监督微调（SFT）

以 Qwen3-4B 为例：

```bash
bash scripts/SFT_Qwen3-4B.sh
```

适用场景：

* 打基础能力
* 对齐领域数据（如数学 / 代码 / 中文）
* 作为 RLVR 的初始模型

---

### 3️⃣ GRPO / RLVR 训练

使用 0.6B 模型进行 RLVR：

```bash
bash scripts/GRPO_Qwen3_0.6b.sh
```

该阶段通常包含：

* rollout 生成（多样本）
* verifier / reward 打分
* 相对优势计算（group-based）
* GRPO 参数更新

---

### 4️⃣ LoRA + RLVR（推荐用于快速实验）

```bash
bash scripts/GRPO_lora_Qwen3_0.6b.sh
```

适合：

* 快速验证 reward / verifier 设计
* 小算力环境
* 多轮对齐实验

---

## ⚙️ 关键可调参数（建议重点关注）

* **rollout 数量（group size）**：每个 prompt 采样多少条回答
* **max_new_tokens**：生成长度，直接影响算力消耗
* **reward / verifier 逻辑**：对齐效果的核心
* **KL / clip 系数**：稳定性与收敛速度
* **LoRA rank / alpha**：参数效率与表达能力权衡

---

## 📊 实验建议

* ✅ 先用 **0.6B + LoRA + 短生成** 验证流程
* ✅ reward / verifier 不稳定时，先可视化打分分布
* ✅ 每次只改一个变量（K / reward / temperature）
* ❌ 不建议一开始就跑大模型长 rollout

---

## 📌 适用人群

* 对 **RLHF / RLAIF / RLVR** 感兴趣的研究者
* 想复现 **GRPO / group-based RL** 的工程人员
* 希望在小模型上验证对齐方法的团队

---

## 📄 License

本项目遵循原仓库 License（如 Apache-2.0 / MIT），具体以仓库声明为准。

---

## 🙌 致谢

* Qwen 系列模型
* GRPO / RLVR 相关研究与社区实践

欢迎 PR、Issue 与实验复现分享 ✨
