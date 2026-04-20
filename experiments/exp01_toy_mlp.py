"""
exp01_toy_mlp.py — torch.compile 入门：Toy MLP baseline

目标：
  1. 建立 eager vs compiled 的 timing 直觉
  2. 观察首次编译延迟 vs 稳态延迟
  3. 用 TORCH_LOGS=graph_code 看 Inductor 产出的 Triton kernel
  4. 用 counters 验证只编译了一次

运行方式：
  # 基础运行
  python exp01_toy_mlp.py

  # 看 Inductor 产出的 Triton kernel（输出较多）
  TORCH_LOGS=graph_code python exp01_toy_mlp.py

  # 看 guard 树
  TORCH_LOGS=guards python exp01_toy_mlp.py

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters

# ── 超参 ──────────────────────────────────────────────────────────────────────
DEVICE    = "cuda"
DTYPE     = torch.bfloat16   # L20 原生支持，无需 GradScaler
BATCH     = 256
IN_DIM    = 1024
HIDDEN    = 4096
OUT_DIM   = 512
WARMUP    = 5                 # 稳定 GPU clock 的预热步数
MEASURE   = 50                # 正式计时步数

# ── 模型定义 ──────────────────────────────────────────────────────────────────
class ToyMLP(nn.Module):
    """三层 MLP，足够产生有意义的 kernel fusion，又简单到容易解读"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


# ── 计时工具 ──────────────────────────────────────────────────────────────────
def time_step(fn, x, y, n: int) -> float:
    """计时 n 步 forward+backward+zero_grad 的平均墙钟时间（ms）"""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        loss = F.mse_loss(fn(x), y)
        loss.backward()
        # 不调 optimizer.step，专注测 fwd+bwd kernel 时间
        for p in fn.parameters():
            p.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n  # ms/step


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    model = ToyMLP(IN_DIM, HIDDEN, OUT_DIM).to(DEVICE, dtype=DTYPE)
    x = torch.randn(BATCH, IN_DIM,  device=DEVICE, dtype=DTYPE)
    y = torch.randn(BATCH, OUT_DIM, device=DEVICE, dtype=DTYPE)

    # ── 1. Eager baseline ────────────────────────────────────────────────────
    section("1. Eager baseline（无 compile）")
    for _ in range(WARMUP):
        F.mse_loss(model(x), y).backward()
        for p in model.parameters(): p.grad = None

    eager_ms = time_step(model, x, y, MEASURE)
    print(f"  eager  fwd+bwd: {eager_ms:.3f} ms/step")

    # ── 2. 首次编译延迟 ──────────────────────────────────────────────────────
    section("2. 首次编译延迟（compile + 第 1 次 call）")

    # 重建一个相同结构的模型，避免 eager warm-up 状态干扰
    model_c = ToyMLP(IN_DIM, HIDDEN, OUT_DIM).to(DEVICE, dtype=DTYPE)
    model_c.load_state_dict(model.state_dict())

    counters.clear()

    compiled = torch.compile(model_c, mode="default", fullgraph=False)

    t0 = time.perf_counter()
    loss = F.mse_loss(compiled(x), y)
    loss.backward()
    for p in model_c.parameters(): p.grad = None
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000

    print(f"  首次 call（含编译）: {first_call_ms:.1f} ms")
    print(f"  unique_graphs after 1st call: {counters['stats']['unique_graphs']}")

    # ── 3. 编译后稳态 ────────────────────────────────────────────────────────
    section("3. 编译后稳态（连续 forward+backward）")
    for _ in range(WARMUP):
        F.mse_loss(compiled(x), y).backward()
        for p in model_c.parameters(): p.grad = None

    compiled_ms = time_step(compiled, x, y, MEASURE)
    speedup = eager_ms / compiled_ms
    print(f"  compiled fwd+bwd: {compiled_ms:.3f} ms/step")
    print(f"  speedup vs eager: {speedup:.2f}x")
    print(f"  unique_graphs after {WARMUP+MEASURE} calls: {counters['stats']['unique_graphs']}")
    # unique_graphs 应该还是 1（shape 没变，没有 recompile）

    # ── 4. 不同 mode 对比 ────────────────────────────────────────────────────
    section("4. mode 对比（default / reduce-overhead / max-autotune-no-cudagraphs）")

    modes = [
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",   # 跳过 cudagraphs 避免静态 shape 限制
    ]

    results = {}
    for mode in modes:
        m = ToyMLP(IN_DIM, HIDDEN, OUT_DIM).to(DEVICE, dtype=DTYPE)
        m.load_state_dict(model.state_dict())
        c = torch.compile(m, mode=mode, fullgraph=False)

        t0 = time.perf_counter()
        F.mse_loss(c(x), y).backward()
        for p in m.parameters(): p.grad = None
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000

        for _ in range(WARMUP):
            F.mse_loss(c(x), y).backward()
            for p in m.parameters(): p.grad = None

        steady_ms = time_step(c, x, y, MEASURE)
        results[mode] = (compile_ms, steady_ms)
        print(f"  [{mode:36s}]  首次={compile_ms:7.0f}ms  稳态={steady_ms:.3f}ms")

    # ── 5. 小结 ──────────────────────────────────────────────────────────────
    section("5. 小结")
    print(f"  eager baseline : {eager_ms:.3f} ms/step")
    for mode, (ctime, stime) in results.items():
        sx = eager_ms / stime
        print(f"  {mode:36s}: 稳态 {stime:.3f} ms  ({sx:.2f}x)  首次编译 {ctime:.0f} ms")

    print("""
关键结论：
  - 首次 call 包含 Dynamo trace + AOTAutograd + Inductor codegen，耗时远高于稳态
  - 稳态下 unique_graphs == 1，说明 shape 不变时不会 recompile
  - max-autotune 首次编译明显更长，但稳态 GEMM 选了最优 Triton tile 配置
  - reduce-overhead 靠 CUDA Graphs 降低 CPU dispatch 开销，对小 batch 效果最显著

下一步 → exp02_resnet.py：在 ResNet-18 上重复实验，看更接近真实的 CNN 场景
""")


if __name__ == "__main__":
    main()
