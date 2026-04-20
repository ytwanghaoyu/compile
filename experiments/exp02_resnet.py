"""
exp02_resnet.py — ResNet-18 上的四档 mode 对比

目标：
  1. 在接近真实的 CNN 上观察 torch.compile 的收益
  2. 对比四档 mode 的编译时间 vs 稳态 throughput（samples/sec）
  3. 测量 AMP（BF16）是否与 compile 正交叠加
  4. 用 regional compilation 直觉理解"重复层只编一次"

运行方式：
  python exp02_resnet.py

  # 看 graph break（ResNet 一般是 0，确认一下）
  TORCH_LOGS=graph_breaks python exp02_resnet.py

  # 看 Inductor 分配的 kernel
  TORCH_LOGS=perf_hints python exp02_resnet.py

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
依赖：pip install torchvision
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch._dynamo.utils import counters

# ── 超参 ──────────────────────────────────────────────────────────────────────
DEVICE    = "cuda"
DTYPE     = torch.bfloat16
BATCH     = 64
IMG_SIZE  = 224
NUM_CLS   = 1000
WARMUP    = 3
MEASURE   = 20

# ── 工具 ──────────────────────────────────────────────────────────────────────
def make_batch():
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=DTYPE)
    y = torch.randint(0, NUM_CLS, (BATCH,), device=DEVICE)
    return x, y


def time_throughput(fn, n_steps: int) -> float:
    """返回 samples/sec（fwd + bwd）"""
    x, y = make_batch()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_steps):
        with torch.autocast("cuda", dtype=DTYPE):
            loss = F.cross_entropy(fn(x), y)
        loss.backward()
        for p in fn.parameters():
            if p.grad is not None: p.grad = None
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return (BATCH * n_steps) / (total_ms / 1000)   # samples/sec


def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    # ── 1. Eager baseline ────────────────────────────────────────────────────
    section("1. Eager baseline（ResNet-18, BF16 autocast）")

    base_model = tvm.resnet18(weights=None).to(DEVICE)

    # warmup
    for _ in range(WARMUP):
        x, y = make_batch()
        with torch.autocast("cuda", dtype=DTYPE):
            F.cross_entropy(base_model(x), y).backward()
        for p in base_model.parameters():
            if p.grad is not None: p.grad = None

    eager_tput = time_throughput(base_model, MEASURE)
    print(f"  eager: {eager_tput:.1f} samples/sec")

    # ── 2. 四档 mode 对比 ────────────────────────────────────────────────────
    section("2. torch.compile 四档 mode 对比")

    # max-autotune-no-cudagraphs: 获得 autotune 的 GEMM 选择但不需要静态 shape
    modes = [
        ("default",                      {}),
        ("reduce-overhead",              {}),
        ("max-autotune-no-cudagraphs",   {}),
        # max-autotune 含 cudagraphs，需要静态 shape，先不开，留给 exp07
        # ("max-autotune",               {}),
    ]

    results = {}
    ref_state = base_model.state_dict()

    for mode, extra_opts in modes:
        m = tvm.resnet18(weights=None).to(DEVICE)
        m.load_state_dict(ref_state)
        counters.clear()

        # 编译并记录首次 call 时间
        compiled = torch.compile(m, mode=mode, fullgraph=False)

        x, y = make_batch()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=DTYPE):
            F.cross_entropy(compiled(x), y).backward()
        for p in m.parameters():
            if p.grad is not None: p.grad = None
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000

        # warmup 稳定
        for _ in range(WARMUP):
            x, y = make_batch()
            with torch.autocast("cuda", dtype=DTYPE):
                F.cross_entropy(compiled(x), y).backward()
            for p in m.parameters():
                if p.grad is not None: p.grad = None

        tput = time_throughput(compiled, MEASURE)
        graphs = counters["stats"]["unique_graphs"]
        results[mode] = (first_ms, tput, graphs)

        print(f"  [{mode:35s}]  首次={first_ms:6.0f}ms  "
              f"稳态={tput:7.1f} smp/s  graphs={graphs}")

    # ── 3. shape_padding 效果 ────────────────────────────────────────────────
    section("3. shape_padding=True 的额外效果（BF16 Tensor Core 对齐）")

    m_pad = tvm.resnet18(weights=None).to(DEVICE)
    m_pad.load_state_dict(ref_state)
    compiled_pad = torch.compile(
        m_pad,
        mode="default",
        options={"shape_padding": True, "epilogue_fusion": True},
    )

    x, y = make_batch()
    with torch.autocast("cuda", dtype=DTYPE):
        F.cross_entropy(compiled_pad(x), y).backward()
    for p in m_pad.parameters():
        if p.grad is not None: p.grad = None
    for _ in range(WARMUP):
        x, y = make_batch()
        with torch.autocast("cuda", dtype=DTYPE):
            F.cross_entropy(compiled_pad(x), y).backward()
        for p in m_pad.parameters():
            if p.grad is not None: p.grad is None

    tput_pad = time_throughput(compiled_pad, MEASURE)
    default_tput = results["default"][1]
    print(f"  default                    : {default_tput:.1f} smp/s")
    print(f"  default + shape_padding    : {tput_pad:.1f} smp/s  "
          f"({tput_pad/default_tput:.2f}x)")
    print("  shape_padding 把最后一维 pad 到 8/16 对齐，命中 Tensor Core 更多路径")

    # ── 4. 小结 ──────────────────────────────────────────────────────────────
    section("4. 小结")
    print(f"  eager baseline : {eager_tput:.1f} smp/s")
    for mode, (fms, tput, g) in results.items():
        sx = tput / eager_tput
        print(f"  {mode:35s}: {tput:7.1f} smp/s  ({sx:.2f}x)  首次 {fms:.0f}ms  graphs={g}")

    print("""
关键结论：
  - ResNet-18 这类 conv 密集型网络，compile 在稳态有明显收益
  - max-autotune 首次编译时间远长于 default，生产训练一般从 default 起步
  - shape_padding 对 BF16 GEMM/conv 有额外帮助，几乎没有编译时间代价
  - unique_graphs 应该保持在 1（固定 shape + 固定 batch）

下一步 → exp03_dynamic_shape_problem.py：batch size 随机变化，观察 recompile 爆炸
""")


if __name__ == "__main__":
    main()
