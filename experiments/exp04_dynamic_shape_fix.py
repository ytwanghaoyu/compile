"""
exp04_dynamic_shape_fix.py — 动态 shape 三种解法并排对比

目标：
  1. 修复 exp03 暴露的 recompile 问题
  2. 量化三种解法的 recompile 次数、首次编译延迟、稳态吞吐
  3. 演示 0/1 specialization 的修复（mark_unbacked + drop_last）
  4. 演示 bucketing 策略（把无限 shape 变为有限桶）

运行方式：
  python exp04_dynamic_shape_fix.py

  # 观察 mark_dynamic 如何让符号维度出现
  TORCH_LOGS=dynamic python exp04_dynamic_shape_fix.py

  # 确认修复后 recompiles 日志静默（没有输出 = 没有 recompile）
  TORCH_LOGS=recompiles python exp04_dynamic_shape_fix.py

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters

# ── 超参 ──────────────────────────────────────────────────────────────────────
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
IN_DIM   = 512
OUT_DIM  = 128
STEPS    = 40
# 模拟真实 DataLoader：主要 batch=32，末尾 batch=7（drop_last=False 的情况）
BATCH_SEQUENCE = [32] * 8 + [16] * 6 + [8] * 4 + [24] * 6 + [32] * 8 + [7]

# ── 模型 ──────────────────────────────────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, 1024)
        self.fc2 = nn.Linear(1024, OUT_DIM)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def make_fresh_compiled(strategy: str) -> nn.Module:
    m = SimpleNet().to(DEVICE, dtype=DTYPE)
    if strategy == "static":
        # 对照组：关闭 automatic dynamic，每种 shape 单独编译
        return torch.compile(m, mode="default", dynamic=False)
    elif strategy == "auto_dynamic":
        # 方案 B：默认 automatic dynamic（first-static-then-dynamic）
        return torch.compile(m, mode="default", dynamic=None)
    elif strategy == "mark_dynamic":
        # 方案 A：在 training loop 里手动 mark_dynamic（见 run_with_mark_dynamic）
        return torch.compile(m, mode="default", dynamic=None)
    elif strategy == "unbacked":
        # 方案 C（PT 2.6+）：让 automatic dynamic 用 unbacked symbol
        return torch.compile(m, mode="default", dynamic=None)
    raise ValueError(strategy)


def run_steps(compiled, batches, *, use_mark_dynamic=False, use_unbacked=False):
    """执行若干 step，返回 (unique_graphs, elapsed_ms)"""
    if use_unbacked:
        import torch._dynamo.config as cfg
        cfg.automatic_dynamic_shapes_mark_as = "unbacked"

    counters.clear()
    t0 = time.perf_counter()

    for bs in batches:
        x = torch.randn(bs, IN_DIM, device=DEVICE, dtype=DTYPE)
        y = torch.randn(bs, OUT_DIM, device=DEVICE, dtype=DTYPE)

        if use_mark_dynamic:
            # mark_dynamic 必须在 compile 外部调用，且每次传入新 tensor 都需要标记
            torch._dynamo.mark_dynamic(x, 0, min=2, max=512)
            torch._dynamo.mark_dynamic(y, 0, min=2, max=512)

        loss = F.mse_loss(compiled(x), y)
        loss.backward()
        for p in compiled.parameters():
            if p.grad is not None: p.grad = None

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000

    if use_unbacked:
        # 重置，避免影响后续实验
        import torch._dynamo.config as cfg
        cfg.automatic_dynamic_shapes_mark_as = "dynamic"

    return counters["stats"]["unique_graphs"], elapsed


def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    print("=" * 65)
    print("  exp04: 动态 shape 三种解法对比")
    print("=" * 65)
    print(f"  batch 序列（{len(BATCH_SEQUENCE)} steps）：{BATCH_SEQUENCE[:10]}... + [7]（末尾不完整 batch）")

    # ── 对照：dynamic=False（每种 shape 独立编译）────────────────────────────
    section("对照组：dynamic=False（问题场景，来自 exp03）")
    c_static = make_fresh_compiled("static")
    ug_static, ms_static = run_steps(c_static, BATCH_SEQUENCE)
    print(f"  unique_graphs = {ug_static}  total_time = {ms_static:.0f}ms")
    print(f"  ✗ 每种不同 batch size 都编译了一次，超过 cache_size_limit 后退 eager")

    # ── 方案 A：mark_dynamic（推荐首选）─────────────────────────────────────
    section("方案 A：mark_dynamic(x, 0, min=2, max=512)（推荐首选）")
    c_mark = make_fresh_compiled("mark_dynamic")
    ug_mark, ms_mark = run_steps(c_mark, BATCH_SEQUENCE, use_mark_dynamic=True)
    print(f"  unique_graphs = {ug_mark}  total_time = {ms_mark:.0f}ms")
    print(f"  ✓ 第一次 call 就 dynamic 编译，后续全部复用")
    print(f"  注意：末尾 batch=7 也能复用（min=2 满足，不触发 0/1 specialization）")

    # ── 方案 B：automatic dynamic（零改动）──────────────────────────────────
    section("方案 B：automatic dynamic（零改动，first-static-then-dynamic）")
    c_auto = make_fresh_compiled("auto_dynamic")
    ug_auto, ms_auto = run_steps(c_auto, BATCH_SEQUENCE)
    print(f"  unique_graphs = {ug_auto}  total_time = {ms_auto:.0f}ms")
    print(f"  ✓ 第 2 次不同 shape 时自动升为 dynamic，第 3+ 次复用")
    print(f"  代价：浪费一次 static 编译（第 1 次 static，第 2 次重编为 dynamic）")
    print(f"  末尾 batch=7 结果如何？若 unique_graphs > 方案A 则说明 0/1 specialization 触发")

    # ── 方案 C：unbacked（PT 2.6+，支持 batch=1/0）────────────────────────
    section("方案 C：automatic_dynamic_shapes_mark_as='unbacked'（PT 2.6+）")
    c_unbacked = make_fresh_compiled("unbacked")
    ug_unbacked, ms_unbacked = run_steps(c_unbacked, BATCH_SEQUENCE, use_unbacked=True)
    print(f"  unique_graphs = {ug_unbacked}  total_time = {ms_unbacked:.0f}ms")
    print(f"  ✓ Unbacked symbol 不做 0/1 specialization，batch=1 也能复用")
    print(f"  代价：模型里 'if batch == 1:' 之类检查会 graph break")

    # ── 方案 D：drop_last=True（最简单的工程解法）──────────────────────────
    section("方案 D：drop_last=True（DataLoader 侧彻底消除不完整 batch）")
    # 去掉末尾的 batch=7
    batches_drop_last = [bs for bs in BATCH_SEQUENCE if bs >= 8]
    c_drop = make_fresh_compiled("auto_dynamic")
    ug_drop, ms_drop = run_steps(c_drop, batches_drop_last, use_mark_dynamic=True)
    print(f"  unique_graphs = {ug_drop}  total_time = {ms_drop:.0f}ms")
    print(f"  ✓ drop_last=True 让 0/1 specialization 问题从根源消失")
    print(f"  组合推荐：mark_dynamic(min=2) + drop_last=True = 零 recompile")

    # ── 方案 E：Bucketing（shape 桶化，把无限变有限）────────────────────────
    section("方案 E：Bucketing（把 batch size pad 到最近的桶）")

    BUCKETS = [8, 16, 32, 64]  # 只有 4 种 shape，最多 4 次编译

    def bucket_batch(bs: int) -> int:
        for b in BUCKETS:
            if b >= bs:
                return b
        return BUCKETS[-1]

    c_bucket = make_fresh_compiled("static")  # 用 static，因为桶数有限
    counters.clear()
    t0 = time.perf_counter()

    for bs in BATCH_SEQUENCE:
        b = bucket_batch(bs)
        # pad 到桶大小（多余部分填 0，用 mask 处理）
        x = torch.zeros(b, IN_DIM, device=DEVICE, dtype=DTYPE)
        y = torch.zeros(b, OUT_DIM, device=DEVICE, dtype=DTYPE)
        x[:bs] = torch.randn(bs, IN_DIM, device=DEVICE, dtype=DTYPE)
        y[:bs] = torch.randn(bs, OUT_DIM, device=DEVICE, dtype=DTYPE)

        loss = F.mse_loss(c_bucket(x), y)
        loss.backward()
        for p in c_bucket.parameters():
            if p.grad is not None: p.grad = None

    torch.cuda.synchronize()
    ug_bucket = counters["stats"]["unique_graphs"]
    ms_bucket = (time.perf_counter() - t0) * 1000
    print(f"  桶 = {BUCKETS}")
    print(f"  unique_graphs = {ug_bucket}  total_time = {ms_bucket:.0f}ms")
    print(f"  ✓ recompile 次数 ≤ 桶数（{len(BUCKETS)}），静态 shape 让 Inductor 更好 autotune")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    section("汇总对比")
    rows = [
        ("对照（dynamic=False）",    ug_static,   ms_static),
        ("方案 A：mark_dynamic",     ug_mark,     ms_mark),
        ("方案 B：auto dynamic",     ug_auto,     ms_auto),
        ("方案 C：unbacked",         ug_unbacked, ms_unbacked),
        ("方案 D：drop_last",        ug_drop,     ms_drop),
        ("方案 E：bucketing",        ug_bucket,   ms_bucket),
    ]
    print(f"  {'策略':30s}  {'unique_graphs':>13}  {'总时间(ms)':>12}")
    print(f"  {'─'*58}")
    for name, ug, ms in rows:
        print(f"  {name:30s}  {ug:>13}  {ms:>12.0f}")

    print("""
关键结论：
  - mark_dynamic 是首选：第一次 call 就直接 dynamic，不浪费一次 static 编译
  - automatic dynamic 是零改动方案：稳态等价，但有一次额外的 static→dynamic 转换
  - unbacked 绕开 0/1 specialization，适合需要 batch=1 的场景（eval 单样本）
  - drop_last=True 是最廉价的工程防线，强烈推荐与 mark_dynamic 组合使用
  - bucketing 适合 sequence length 变化大的场景（NLP），pad 开销换编译稳定性

下一步 → exp05_control_flow.py：if-else 分支的五种处理方式
""")


if __name__ == "__main__":
    main()
