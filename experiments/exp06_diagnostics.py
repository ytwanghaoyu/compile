"""
exp06_diagnostics.py — 完整诊断工具链实战

目标：
  故意写一个"有多种问题"的训练循环，然后用工具链逐步定位并修复：
    Step 1 → counters 发现 unique_graphs 异常
    Step 2 → TORCH_LOGS=recompiles 定位失败 guard
    Step 3 → dynamo.explain() 数 graph break
    Step 4 → TORCH_LOGS=recompiles_verbose,dynamic 深度分析
    Step 5 → 逐一修复，验证 unique_graphs 回到 1

内置的"坏味道"（刻意制造的问题）：
  [P1] batch size 每步随机变化（dynamic=False）
  [P2] 每步修改 self.qk_mask（ID_MATCH guard 失败）
  [P3] Python int scale 作为 forward 参数（EQUALS_MATCH guard 失败）
  [P4] forward 内有 data-dependent print（graph break）

运行方式：
  # 阶段一：先跑有问题的版本
  python exp06_diagnostics.py --phase broken

  # 阶段一 + TORCH_LOGS
  TORCH_LOGS=recompiles python exp06_diagnostics.py --phase broken

  # 阶段二：修复后验证
  python exp06_diagnostics.py --phase fixed

  # 对比所有阶段
  python exp06_diagnostics.py --phase all

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters

DEVICE  = "cuda"
DTYPE   = torch.bfloat16
IN_DIM  = 256
OUT_DIM = 64
STEPS   = 25


def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# 有问题的模型（故意埋坑）
# ══════════════════════════════════════════════════════════════════════════════

class BrokenModel(nn.Module):
    """
    故意埋入四种常见问题：
      P1: dynamic=False 搭配变化 batch（在 training loop 侧）
      P2: 每 step 赋新 tensor 给 self.qk_mask → ID_MATCH 每次失败
      P3: forward 接受 Python int scale → EQUALS_MATCH 每个值重编
      P4: forward 内有 data-dependent 控制流 + print → graph break
    """

    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(IN_DIM, OUT_DIM)
        # P2：每步会被替换成新 tensor，导致 ID_MATCH guard 失败
        self.qk_mask = torch.ones(1, device=DEVICE, dtype=DTYPE)

    def forward(self, x: torch.Tensor, scale: int = 1) -> torch.Tensor:
        # P3：scale 是 Python int，每个不同值都触发 EQUALS_MATCH guard 失败
        h = self.fc(x) * scale

        # P4：data-dependent 控制流 → graph break
        if h.abs().max().item() > 10.0:    # .item() 是 data-dependent
            print(f"[WARN] large activation: {h.abs().max().item():.2f}")
            h = h.clamp(-10.0, 10.0)

        # P2 的效果：self.qk_mask 的 id 每步变 → Dynamo 的 ID_MATCH guard 失败
        h = h * self.qk_mask
        return h.mean(dim=-1)


def run_broken(steps: int):
    """运行有问题的版本，统计 unique_graphs"""
    model = BrokenModel().to(DEVICE, dtype=DTYPE)
    # P1: dynamic=False 搭配变化 batch
    compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=False)

    counters.clear()
    t0 = time.perf_counter()
    scales = [1, 2, 3, 1, 2, 3, 1, 2]   # P3: 不同 scale 值

    for i in range(steps):
        bs     = random.choice([8, 16, 24, 32])  # P1: batch size 随机变化
        x      = torch.randn(bs, IN_DIM, device=DEVICE, dtype=DTYPE)
        y      = torch.randn(bs, OUT_DIM, device=DEVICE, dtype=DTYPE)
        scale  = scales[i % len(scales)]           # P3

        # P2: 每步新建 qk_mask tensor → 不同 id → ID_MATCH 失败
        model.qk_mask = torch.ones(1, device=DEVICE, dtype=DTYPE) * (i % 3 + 1)

        try:
            loss = F.mse_loss(compiled(x, scale), y)
            loss.backward()
        except Exception as e:
            print(f"  step={i} error: {e}")
        finally:
            for p in model.parameters():
                if p.grad is not None: p.grad = None

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    ug = counters["stats"]["unique_graphs"]
    return ug, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 修复后的模型
# ══════════════════════════════════════════════════════════════════════════════

class FixedModel(nn.Module):
    """
    修复版本：
      P1 fix: training loop 里用 mark_dynamic + automatic dynamic
      P2 fix: qk_mask 用 register_buffer，in-place 更新（不换 id）
      P3 fix: scale 包成 torch.tensor，不作为 Python int 传入
      P4 fix: 用 is_compiling() 保护副作用；clamp 改为无条件执行
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(IN_DIM, OUT_DIM)
        # P2 fix: register_buffer 预先分配，后续 in-place 更新
        self.register_buffer("qk_mask", torch.ones(1, dtype=DTYPE))

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # P3 fix: scale 是 0-dim tensor，不触发 EQUALS_MATCH guard
        h = self.fc(x) * scale

        # P4 fix: 副作用用 is_compiling() 保护，clamp 改为无条件
        if not torch.compiler.is_compiling():
            if h.abs().max().item() > 10.0:
                print(f"[WARN] large activation detected (eager mode only)")

        h = h.clamp(-10.0, 10.0)   # 直接 clamp，无 data-dependent 判断

        # P2 fix: qk_mask 的 id 不变（in-place 更新）
        h = h * self.qk_mask
        return h.mean(dim=-1)


def run_fixed(steps: int):
    """运行修复版本，统计 unique_graphs"""
    model = FixedModel().to(DEVICE, dtype=DTYPE)
    # P1 fix: automatic dynamic（default dynamic=None）
    compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=None)

    counters.clear()
    t0 = time.perf_counter()
    scales_raw = [1, 2, 3, 1, 2, 3, 1, 2]

    for i in range(steps):
        bs    = random.choice([8, 16, 24, 32])    # P1 fix: mark_dynamic 处理
        x     = torch.randn(bs, IN_DIM, device=DEVICE, dtype=DTYPE)
        y     = torch.randn(bs, OUT_DIM, device=DEVICE, dtype=DTYPE)
        scale = torch.tensor(scales_raw[i % len(scales_raw)],  # P3 fix: 0-dim tensor
                             device=DEVICE, dtype=DTYPE)

        # P1 fix: mark_dynamic 让 batch 维从第一次就是 dynamic
        torch._dynamo.mark_dynamic(x, 0, min=2, max=128)
        torch._dynamo.mark_dynamic(y, 0, min=2, max=128)

        # P2 fix: in-place 更新（qk_mask id 不变）
        model.qk_mask.fill_(float(i % 3 + 1))

        loss = F.mse_loss(compiled(x, scale), y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None: p.grad = None

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    ug = counters["stats"]["unique_graphs"]
    return ug, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# dynamo.explain 诊断
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_with_explain():
    section("Step 3：dynamo.explain() 诊断 graph break")

    model = BrokenModel().to(DEVICE, dtype=DTYPE)
    x     = torch.randn(16, IN_DIM, device=DEVICE, dtype=DTYPE)

    # explain 在 eager 下模拟一次 trace，返回 break 统计
    exp = torch._dynamo.explain(model)(x, 2)

    print(f"  graph_count       = {exp.graph_count}   （子图数量）")
    print(f"  graph_break_count = {exp.graph_break_count}  （break 次数）")
    print(f"  op_count          = {exp.op_count}   （捕获的 op 总数）")
    print()
    print("  break_reasons（每处 break 的原因）:")
    for i, reason in enumerate(exp.break_reasons):
        print(f"    [{i}] {reason}")
    print()
    print("  修复建议：")
    print("    - 含 .item() → 用 is_compiling() 保护 / 改为无条件 clamp")
    print("    - 含 print   → 用 is_compiling() 保护")


# ══════════════════════════════════════════════════════════════════════════════
# Guard failure 诊断提示
# ══════════════════════════════════════════════════════════════════════════════

def print_guard_tips():
    section("Step 4：TORCH_LOGS 读法速查")
    print("""
  典型日志片段（TORCH_LOGS=recompiles）：

    Recompiling function forward in exp06_diagnostics.py:XX
      triggered by the following guard failure(s):
        - 0/0: tensor 'L['x']' size mismatch at index 0. expected 8, actual 16
        → [P1] batch size 变化，fix：mark_dynamic(x, 0, min=2, max=128)

        - 0/1: ___check_obj_id(L['self'].qk_mask, 7616288)
        → [P2] qk_mask 每步新建（id 变），fix：register_buffer + in-place update

        - 0/2: L['scale'] == 2
        → [P3] Python int 参数变化（EQUALS_MATCH），fix：改成 torch.tensor(scale)

  TORCH_LOGS=dynamic 片段（成功 mark_dynamic 后应看到）：
    create_symbol s0 = 16 for L['x'].size()[0] [2, 128]
    ← 说明 dim0 被成功标为 symbolic，范围 [2, 128]

  TORCH_LOGS=graph_breaks 片段：
    Graph break in user code at exp06.py:XX
    Reason: .item() call was found [...]
    ← 直接定位到 .item() 的行号
""")


# ══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["broken", "fixed", "all"], default="all")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    print("=" * 65)
    print("  exp06: 完整诊断工具链实战")
    print("=" * 65)

    if args.phase in ("broken", "all"):
        section("Step 1 & 2：先跑有问题版本，观察 unique_graphs 异常")
        print("  [P1] batch size 随机变化 + dynamic=False")
        print("  [P2] self.qk_mask 每步新建（ID_MATCH 失败）")
        print("  [P3] Python int scale（EQUALS_MATCH 失败）")
        print("  [P4] data-dependent print → graph break")
        print()

        ug_broken, ms_broken = run_broken(STEPS)
        print(f"\n  ⇒ unique_graphs = {ug_broken}  total_time = {ms_broken:.0f}ms")
        if ug_broken > 5:
            print(f"  ✗ unique_graphs={ug_broken} 远超 1，明显有 recompile 问题")
            print(f"  下一步：TORCH_LOGS=recompiles python exp06_diagnostics.py --phase broken")
        else:
            print(f"  unique_graphs={ug_broken}，问题不明显（可能 cache_size_limit 已触顶）")

        diagnose_with_explain()
        print_guard_tips()

    if args.phase in ("fixed", "all"):
        section("Step 5：修复后验证")
        print("  [P1 fix] automatic dynamic + mark_dynamic(x, 0, min=2, max=128)")
        print("  [P2 fix] register_buffer + qk_mask.fill_(v)（in-place）")
        print("  [P3 fix] scale = torch.tensor(v)（0-dim tensor）")
        print("  [P4 fix] is_compiling() 保护 print，clamp 改为无条件")
        print()

        ug_fixed, ms_fixed = run_fixed(STEPS)
        print(f"\n  ⇒ unique_graphs = {ug_fixed}  total_time = {ms_fixed:.0f}ms")
        if ug_fixed <= 2:
            print(f"  ✓ unique_graphs={ug_fixed}（≤2 = 1次 static + 1次 dynamic 升级，正常）")
        else:
            print(f"  还有剩余 recompile，检查 TORCH_LOGS=recompiles 输出")

    if args.phase == "all":
        section("对比汇总")
        print(f"  有问题版本: unique_graphs={ug_broken}  time={ms_broken:.0f}ms")
        print(f"  修复版本:   unique_graphs={ug_fixed}   time={ms_fixed:.0f}ms")
        improvement = ms_broken / ms_fixed if ms_fixed > 0 else 0
        print(f"  速度提升: {improvement:.2f}x（主要来自消除 recompile 开销）")

    print("""
诊断工具链速查：
  1. counters["stats"]["unique_graphs"]  → 快速 sanity check
  2. TORCH_LOGS=recompiles              → 找哪个 frame、第一个失败 guard
  3. dynamo.explain(fn)(*args)          → graph break 数和原因
  4. TORCH_LOGS=recompiles_verbose      → 所有 cache entry 的所有 guard
  5. TORCH_LOGS=dynamic                → symbolic shape 创建过程
  6. tlparse（离线）: TORCH_TRACE=/tmp/t python ... && tlparse /tmp/t --latest

下一步 → exp07_best_practice.py：所有最佳实践的综合完整版
""")


if __name__ == "__main__":
    main()
