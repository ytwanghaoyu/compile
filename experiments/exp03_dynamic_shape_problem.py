"""
exp03_dynamic_shape_problem.py — 故意制造 recompile，亲眼看到问题

目标：
  1. 演示 batch size 随机变化时 recompile 如何爆炸
  2. 理解 0/1 specialization 陷阱
  3. 读懂 TORCH_LOGS=recompiles 的输出格式
  4. 用 counters 量化 recompile 次数

运行方式：
  # 基础：只看 unique_graphs 统计
  python exp03_dynamic_shape_problem.py

  # 看每次 recompile 的第一个失败 guard（推荐先看这个）
  TORCH_LOGS=recompiles python exp03_dynamic_shape_problem.py

  # 看所有 cache entry 的所有失败 guard（更详细）
  TORCH_LOGS=recompiles_verbose,dynamic python exp03_dynamic_shape_problem.py

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters, compile_times

# ── 超参 ──────────────────────────────────────────────────────────────────────
DEVICE  = "cuda"
DTYPE   = torch.bfloat16
IN_DIM  = 512
OUT_DIM = 128
STEPS   = 30   # 步数足够触发多次 recompile

# ── 模型 ──────────────────────────────────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, 1024)
        self.fc2 = nn.Linear(1024, OUT_DIM)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ── 场景 A：连续变化的 batch size（最常见的真实问题）────────────────────────
def scenario_a_varying_batch(model_c):
    """
    模拟 DataLoader 没有 drop_last=True，最后几个 batch 大小不同的情况。
    batch_sizes 包含多种不同值，每个值都会触发一次 recompile，直到命中 cache_size_limit。
    """
    section("场景 A：随机 batch size（[4,8,12,16,20,24,28,32] 循环）")

    batch_sizes = [4, 8, 12, 16, 20, 24, 28, 32]
    counters.clear()

    print(f"  将依次使用 batch_sizes = {batch_sizes}，每种 shape 都触发一次 recompile")
    print(f"  cache_size_limit 默认 = 8（超限后 fallback eager）\n")

    for step, bs in enumerate(batch_sizes * (STEPS // len(batch_sizes) + 1)):
        if step >= STEPS:
            break
        x = torch.randn(bs, IN_DIM, device=DEVICE, dtype=DTYPE)
        y = torch.randn(bs, OUT_DIM, device=DEVICE, dtype=DTYPE)

        loss = F.mse_loss(model_c(x), y)
        loss.backward()
        for p in model_c.parameters():
            if p.grad is not None: p.grad = None

        # 每步打印当前 unique_graphs，观察它如何线性增长
        ug = counters["stats"]["unique_graphs"]
        if step < 10 or step % 5 == 0:
            print(f"  step={step:2d}  batch={bs:3d}  unique_graphs={ug}")

    final_ug = counters["stats"]["unique_graphs"]
    print(f"\n  ✗ 最终 unique_graphs = {final_ug}（应为 1，实际远超）")
    print(f"  每种不同 batch size 都触发了一次独立的 Dynamo recompile")


# ── 场景 B：0/1 specialization 陷阱 ─────────────────────────────────────────
def scenario_b_zero_one(model_c):
    """
    PT2 总是对 size=0 或 size=1 的维度 specialize，即使 mark_dynamic 也如此。
    batch=1 是最常见的触发器（eval 时单样本推理）。
    """
    section("场景 B：0/1 specialization 陷阱（batch=1 永远 recompile）")
    counters.clear()

    # 先用 batch=8 编译一次
    x8 = torch.randn(8, IN_DIM, device=DEVICE, dtype=DTYPE)
    y8 = torch.randn(8, OUT_DIM, device=DEVICE, dtype=DTYPE)
    F.mse_loss(model_c(x8), y8).backward()
    for p in model_c.parameters():
        if p.grad is not None: p.grad = None

    print(f"  step 0: batch=8  unique_graphs={counters['stats']['unique_graphs']}")

    # 切到 batch=16（automatic dynamic 会把 dim0 升为 symbolic，再重编一次）
    x16 = torch.randn(16, IN_DIM, device=DEVICE, dtype=DTYPE)
    y16 = torch.randn(16, OUT_DIM, device=DEVICE, dtype=DTYPE)
    F.mse_loss(model_c(x16), y16).backward()
    for p in model_c.parameters():
        if p.grad is not None: p.grad = None

    print(f"  step 1: batch=16 unique_graphs={counters['stats']['unique_graphs']}")

    # 现在用 batch=24，应该复用 dynamic 版本（unique_graphs 不再增加）
    x24 = torch.randn(24, IN_DIM, device=DEVICE, dtype=DTYPE)
    y24 = torch.randn(24, OUT_DIM, device=DEVICE, dtype=DTYPE)
    F.mse_loss(model_c(x24), y24).backward()
    for p in model_c.parameters():
        if p.grad is not None: p.grad = None

    print(f"  step 2: batch=24 unique_graphs={counters['stats']['unique_graphs']} (应与 step1 相同)")

    # 致命：batch=1（0/1 specialization 永远触发独立编译）
    x1 = torch.randn(1, IN_DIM, device=DEVICE, dtype=DTYPE)
    y1 = torch.randn(1, OUT_DIM, device=DEVICE, dtype=DTYPE)
    F.mse_loss(model_c(x1), y1).backward()
    for p in model_c.parameters():
        if p.grad is not None: p.grad = None

    print(f"  step 3: batch=1  unique_graphs={counters['stats']['unique_graphs']} ← 触发！")
    print("  ✗ batch=1 触发了额外 recompile——因为 size=1 被 PT2 特殊对待（广播规则）")
    print("  解决方案：DataLoader drop_last=True，或 mark_unbacked（见 exp04）")


# ── 场景 C：Python int 参数变化导致的 recompile ──────────────────────────────
def scenario_c_python_scalar(model_c):
    """
    每次把 Python int 作为参数传入，Dynamo 会为每个不同的值生成独立 guard。
    这是 LR scheduler 经典坑的最小复现。
    """
    section("场景 C：Python int 参数变化（LR scheduler 经典坑的最小复现）")
    counters.clear()

    @torch.compile(fullgraph=False)
    def fn_with_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
        # scale 是 Python int，每个不同值都会生成 EQUALS_MATCH guard
        return x * scale + 1.0

    x = torch.randn(32, IN_DIM, device=DEVICE, dtype=DTYPE)
    scales = [1, 2, 3, 4, 5]

    for s in scales:
        fn_with_scale(x, s)
        print(f"  scale={s}  unique_graphs={counters['stats']['unique_graphs']}")

    print("  ✗ 每个不同的 scale 值触发一次 recompile（EQUALS_MATCH guard 失败）")
    print("  解决方案：把 scale 包成 torch.tensor(scale)，或放到模型 attribute 里")


# ── 诊断统计 ─────────────────────────────────────────────────────────────────
def print_compile_times():
    section("compile_times() 输出（哪些 frame 花了多少编译时间）")
    try:
        ct = compile_times(output="str")
        print(ct if ct.strip() else "  （无数据，compile_times 可能需要重置 counters）")
    except Exception as e:
        print(f"  compile_times 不可用: {e}")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    print("=" * 65)
    print("  exp03: 故意制造 recompile——观察 unique_graphs 线性增长")
    print("=" * 65)
    print(f"""
推荐同时运行：
  TORCH_LOGS=recompiles python {__file__.split('/')[-1]}

你会看到类似：
  Recompiling function forward in experiments/exp03...
    triggered by the following guard failure(s):
      - 0/0: tensor 'L[\\'x\\']' size mismatch at index 0. expected 4, actual 8

'0/0' 表示 [frame_id/cache_entry_id]。
""")

    # 场景 A：每次不同的 batch size
    model_a = SimpleNet().to(DEVICE, dtype=DTYPE)
    compiled_a = torch.compile(model_a, mode="default", fullgraph=False,
                                dynamic=False)  # 关闭 automatic dynamic，每种 shape 单独编译
    scenario_a_varying_batch(compiled_a)

    # 场景 B：0/1 specialization（用 automatic dynamic，演示 batch=1 依然被 specialize）
    model_b = SimpleNet().to(DEVICE, dtype=DTYPE)
    compiled_b = torch.compile(model_b, mode="default", fullgraph=False,
                                dynamic=None)   # automatic dynamic（默认）
    scenario_b_zero_one(compiled_b)

    # 场景 C：Python int 参数变化
    scenario_c_python_scalar(None)  # fn_with_scale 内部自己 compile

    print_compile_times()

    print("""
═══════════════════════════════════════════════════════════════
  总结：三种常见 recompile 触发源
  A. batch size 随机变化 + dynamic=False → 每种 shape 独立编译
  B. batch=1/0（0/1 specialization）→ 即使 automatic dynamic 也无法复用
  C. Python int/float 作为参数 → EQUALS_MATCH guard 每个值重编

  下一步 → exp04_dynamic_shape_fix.py：三种解法逐一修复上述问题
═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
