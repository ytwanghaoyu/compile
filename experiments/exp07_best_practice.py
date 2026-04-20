"""
exp07_best_practice.py — 综合最佳实践：四条黄金法则完整实现

目标：
  把前六个实验学到的所有知识整合成一个"可以直接拿去用"的训练模板：
    法则一：显式 mark_dynamic，第一次 call 就 dynamic 编译
    法则二：drop_last=True，从根源消除 0/1 specialization
    法则三：DDP(model) 再 torch.compile（本脚本演示单卡版，DDP 版见注释）
    法则四：诊断从 TORCH_LOGS=recompiles 开始，tlparse 整体验证

  模型：带 if-else 分支的 Transformer Encoder Block（贴近真实 LLM 层）
  场景：batch size 抖动 + 少量 data-independent 分支 + BF16 autocast

运行方式：
  # 标准运行
  python exp07_best_practice.py

  # CI 守门（超过 2 次 recompile 立即报错）
  python exp07_best_practice.py --ci

  # 生成 tlparse 可视化报告（需要 pip install tlparse）
  TORCH_TRACE=/tmp/trace python exp07_best_practice.py
  tlparse /tmp/trace --latest -o tl_out/

  # DDP 版（需要多卡）
  # torchrun --nproc_per_node=4 exp07_best_practice.py --ddp

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import argparse
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters

# ── 超参 ──────────────────────────────────────────────────────────────────────
DEVICE    = "cuda"
DTYPE     = torch.bfloat16
D_MODEL   = 512
N_HEADS   = 8
D_FF      = 2048
DROPOUT   = 0.0           # 推理友好，生产训练可改为 0.1
SEQ_LEN   = 128
MAX_BATCH = 64
MIN_BATCH = 4             # ≥2，绕开 0/1 specialization
WARMUP    = 5
TRAIN_STEPS = 40

# 模拟真实 DataLoader 的 batch size 序列（最后几步刻意有抖动）
BATCH_SCHEDULE = (
    [32] * 10 +
    [16] * 6  +
    [48] * 6  +
    [32] * 8  +
    [24] * 6  +
    [64] * 4
)[:TRAIN_STEPS]


# ══════════════════════════════════════════════════════════════════════════════
# 模型：Transformer Encoder Block（带 data-independent if-else）
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    标准 Pre-LN Transformer Block，包含两类 if-else：
      [A] data-independent（is_compiling / training mode）→ 安全，无 graph break
      [B] 没有 data-dependent 分支（已避免）
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:

        # [A-1] data-independent 分支：编译时 is_compiling()=True，整块 DCE
        if not torch.compiler.is_compiling():
            assert x.dim() == 3, f"expected (B, T, D), got {x.shape}"

        # Pre-LN self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=attn_mask,
                                need_weights=False)
        x = x + self.drop(attn_out)

        # Pre-LN FFN
        normed = self.norm2(x)

        # [A-2] data-independent 分支：training 状态在编译时已知
        if self.training:
            ff_out = self.drop(F.gelu(self.ff1(normed)))
        else:
            ff_out = F.gelu(self.ff1(normed))   # eval 下跳过 dropout

        x = x + self.ff2(ff_out)
        return x


class SmallTransformer(nn.Module):
    """2 层 Transformer Encoder + 分类头"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int = 2, vocab_size: int = 256,
                 n_classes: int = 10):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=DROPOUT)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)          # (B, T, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x[:, 0, :])         # CLS token
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# 全局配置（法则一的前置：配置 torch._dynamo）
# ══════════════════════════════════════════════════════════════════════════════

def configure_dynamo(ci_mode: bool = False):
    torch._dynamo.config.recompile_limit = 16          # 为边角 shape 留缓冲
    torch._dynamo.config.accumulated_recompile_limit = 256

    if ci_mode:
        # CI 守门：超过阈值立即报错，防止 recompile 回归
        torch._dynamo.config.recompile_limit = 2
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        print("  [CI 模式] recompile_limit=2，触顶直接抛错")


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def build_compiled_model() -> nn.Module:
    """
    法则三（单卡版）：
      单卡：torch.compile(model, ...)
      DDP 版（多卡）：
        ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=25,
                        static_graph=True, find_unused_parameters=False)
        compiled = torch.compile(ddp_model, ...)   ← DDP 必须在 compile 外层
    """
    model = SmallTransformer(
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF
    ).to(DEVICE)

    compiled = torch.compile(
        model,
        mode="default",          # 训练首选；max-autotune 留给 eval/inference
        fullgraph=False,         # 容忍 training mode 分支（data-independent）
        dynamic=None,            # automatic dynamic（配合 mark_dynamic 使用）
        options={
            "shape_padding":    True,   # BF16 Tensor Core 对齐
            "epilogue_fusion":  True,   # matmul + bias + act 融合
        },
    )
    return compiled


def train_step(compiled, optimizer, token_ids: torch.Tensor, labels: torch.Tensor):
    """
    法则一：mark_dynamic 必须在 compile 外部调用，且每次传入新 tensor 都需要标记。
    法则二：drop_last=True 已在 BATCH_SCHEDULE 构造时保证 batch ≥ MIN_BATCH。
    """
    # 法则一：显式标记 batch 维为 dynamic
    torch._dynamo.mark_dynamic(token_ids, 0, min=MIN_BATCH, max=MAX_BATCH)
    torch._dynamo.mark_dynamic(labels,    0, min=MIN_BATCH, max=MAX_BATCH)

    with torch.autocast("cuda", dtype=DTYPE):
        logits = compiled(token_ids)
        loss   = F.cross_entropy(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_training(ci_mode: bool = False):
    configure_dynamo(ci_mode)

    compiled = build_compiled_model()
    optimizer = torch.optim.AdamW(
        compiled.parameters(),
        lr=1e-4,
        fused=True,    # fused AdamW 降低 optimizer kernel launch 开销（CUDA 专用）
    )

    # ── Warmup（触发编译）────────────────────────────────────────────────────
    print("\n  [Warmup] 前几步会触发编译，耗时较长...")
    counters.clear()

    for step in range(WARMUP):
        bs = BATCH_SCHEDULE[step % len(BATCH_SCHEDULE)]
        tok = torch.randint(0, 256, (bs, SEQ_LEN), device=DEVICE)
        lbl = torch.randint(0, 10,  (bs,),          device=DEVICE)
        t0  = time.perf_counter()
        loss_val = train_step(compiled, optimizer, tok, lbl)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        ug = counters["stats"]["unique_graphs"]
        print(f"    warmup step={step}  batch={bs}  loss={loss_val:.4f}  "
              f"time={ms:.0f}ms  unique_graphs={ug}")

    # ── 稳态训练 ─────────────────────────────────────────────────────────────
    print("\n  [稳态训练] unique_graphs 应在此阶段稳定...")
    step_times = []

    for step in range(TRAIN_STEPS):
        bs = BATCH_SCHEDULE[step]
        tok = torch.randint(0, 256, (bs, SEQ_LEN), device=DEVICE)
        lbl = torch.randint(0, 10,  (bs,),          device=DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        loss_val = train_step(compiled, optimizer, tok, lbl)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        step_times.append(ms)

        ug = counters["stats"]["unique_graphs"]
        if step < 5 or step % 10 == 0:
            print(f"    train step={step:2d}  batch={bs:3d}  loss={loss_val:.4f}  "
                  f"time={ms:.1f}ms  unique_graphs={ug}")

    return step_times, counters["stats"]["unique_graphs"]


# ══════════════════════════════════════════════════════════════════════════════
# Eager baseline（对比用）
# ══════════════════════════════════════════════════════════════════════════════

def run_eager_baseline():
    model = SmallTransformer(
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step_times = []
    for step in range(TRAIN_STEPS):
        bs  = BATCH_SCHEDULE[step]
        tok = torch.randint(0, 256, (bs, SEQ_LEN), device=DEVICE)
        lbl = torch.randint(0, 10,  (bs,),          device=DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.autocast("cuda", dtype=DTYPE):
            loss = F.cross_entropy(model(tok), lbl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        step_times.append(start.elapsed_time(end))

    return step_times


# ══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def percentile(data, p):
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data)-1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci",  action="store_true", help="CI 守门模式")
    parser.add_argument("--ddp", action="store_true", help="（占位）DDP 多卡模式")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    print("=" * 65)
    print("  exp07: 综合最佳实践——四条黄金法则完整实现")
    print("=" * 65)
    print(f"  模型: SmallTransformer  D={D_MODEL} H={N_HEADS} FF={D_FF}")
    print(f"  Batch 序列: {BATCH_SCHEDULE[:8]}...")
    print(f"  Batch 范围: [{MIN_BATCH}, {MAX_BATCH}]")
    print(f"  BF16 autocast, fused AdamW, shape_padding=True")

    if args.ddp:
        print("""
  DDP 模式说明（需要 torchrun --nproc_per_node=N）：
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=25,
                    static_graph=True, find_unused_parameters=False)
    compiled = torch.compile(ddp_model, mode="default", ...)
    ← 法则三：DDP 必须在 compile 外层，让 Dynamo 识别 bucket 边界

  本次跳过 DDP 演示（单卡模式）。
""")

    section("Eager Baseline")
    eager_times = run_eager_baseline()
    eager_p50 = percentile(eager_times, 50)
    eager_p95 = percentile(eager_times, 95)
    print(f"\n  p50={eager_p50:.1f}ms  p95={eager_p95:.1f}ms")

    section("torch.compile 最佳实践版")
    compiled_times, final_ug = run_training(ci_mode=args.ci)

    # 去掉前几步（包含编译延迟）
    steady_times = compiled_times[WARMUP+2:]
    compiled_p50 = percentile(steady_times, 50)
    compiled_p95 = percentile(steady_times, 95)

    section("最终结果")
    print(f"  unique_graphs（稳态）: {final_ug}")
    print(f"  {'':20s}  {'p50(ms)':>8}  {'p95(ms)':>8}  {'speedup':>8}")
    print(f"  {'─'*50}")
    print(f"  {'eager':20s}  {eager_p50:>8.1f}  {eager_p95:>8.1f}  {'1.00x':>8}")
    print(f"  {'compiled（稳态）':20s}  {compiled_p50:>8.1f}  {compiled_p95:>8.1f}  "
          f"{eager_p50/compiled_p50:>7.2f}x")

    if final_ug <= 2:
        print(f"\n  ✓ unique_graphs={final_ug} — 四条黄金法则生效，recompile 控制在极少次")
    else:
        print(f"\n  ⚠ unique_graphs={final_ug} — 仍有 recompile，"
              f"运行 TORCH_LOGS=recompiles 定位原因")

    print("""
四条黄金法则回顾：
  ① mark_dynamic(x, 0, min=2, max=MAX)   → 第一次 call 就直接 dynamic，省一次 static 编译
  ② drop_last=True / batch ≥ 2           → 从根源消除 0/1 specialization 陷阱
  ③ torch.compile(DDP(model))            → DDPOptimizer 才能正常工作，allreduce 与 bwd overlap
  ④ TORCH_LOGS=recompiles 先看，tlparse 整体验证 → 诊断工具链成熟，用好比盲猜省时

可选进阶（确认稳定后再开）：
  - eval/inference 阶段：mode="max-autotune"（大 GEMM 再 +5-20%）
  - FP8 训练（H100+）：torchao.float8 + prologue fusion（PT 2.7+）
  - 长 sequence 变化（NLP）：bucketing 策略（见 exp04）
  - 激活检查点：torch._dynamo.config.optimize_ddp="python_reducer" + compiled_autograd

下一步实验方向（自由探索）：
  A. 在 exp07 基础上开启 max-autotune，观察编译时间 vs 稳态收益
  B. 把 SmallTransformer 换成 FlexAttention（PT 2.5+），观察 kernel fusion
  C. 用 torchvision.models.resnet50 + FSDP2 测试多卡扩展
""")


if __name__ == "__main__":
    main()
