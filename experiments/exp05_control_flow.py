"""
exp05_control_flow.py — if-else 控制流的五种处理方式

目标：
  1. 理解 data-independent vs data-dependent 分支的本质区别
  2. 演示 graph break 的代价（fusion 机会丢失）
  3. 掌握 torch.cond / is_compiling() / @disable 三种修复工具
  4. 用 dynamo.explain() 快速诊断 graph break 数

运行方式：
  python exp05_control_flow.py

  # 看每处 graph break 的位置和原因
  TORCH_LOGS=graph_breaks python exp05_control_flow.py

  # 看 torch.cond 是否产生了 subgraph
  TORCH_LOGS=graph_code python exp05_control_flow.py

硬件假设：NVIDIA L20 / CUDA, PyTorch 2.8
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters

DEVICE = "cuda"
DTYPE  = torch.bfloat16
BATCH  = 32
DIM    = 512

def section(title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


def count_graph_breaks(fn, *args) -> dict:
    """用 dynamo.explain 快速拿到 graph break 统计"""
    exp = torch._dynamo.explain(fn)(*args)
    return {
        "graph_count":       exp.graph_count,
        "graph_break_count": exp.graph_break_count,
        "op_count":          exp.op_count,
        "break_reasons":     [str(r) for r in exp.break_reasons],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Case A：data-independent 分支（shape/dtype 判断）
# ══════════════════════════════════════════════════════════════════════════════

class CaseA_DataIndependent(nn.Module):
    """
    分支依赖 tensor.shape，是 data-independent 的。
    Dynamo 在编译时就能确定走哪条路，生成 guard 然后 specialize，
    不产生 graph break。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == DIM:               # data-independent：编译时可知
            x = F.gelu(self.fc(x))
        else:
            x = x * 2.0
        if x.dtype == torch.bfloat16:        # data-independent：编译时可知
            x = x.float()
        return x.mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Case B：data-dependent 分支（触发 graph break）
# ══════════════════════════════════════════════════════════════════════════════

class CaseB_DataDependent(nn.Module):
    """
    分支依赖 tensor 的值（.item() 或 bool(tensor)），是 data-dependent 的。
    Dynamo 无法在编译时知道走哪支，触发 graph break。
    中间 tensor 被实体化到 HBM，丢失跨界 fusion 机会。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        # ↓ data-dependent：依赖 tensor 的运行时值，Dynamo 无法静态推断
        if h.sum().item() > 0:               # .item() 触发 graph break
            h = F.relu(h)
        else:
            h = F.gelu(h)
        return h.mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Case C：torch.cond 改写（保持单图）
# ══════════════════════════════════════════════════════════════════════════════

class CaseC_TorchCond(nn.Module):
    """
    把 data-dependent if-else 改写为 torch.cond HOP。
    两条分支作为 FX subgraph，整个函数保持单图，支持 fullgraph=True。

    限制（PyTorch 2.7 及之前）：
      - true_fn / false_fn 返回的 tensor 必须 shape/dtype 完全相同
      - 分支内不能有 in-place mutation（有梯度时）
      - 分支内不能引用闭包变量，需显式通过 operands 传入
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)

        # pred 可以是 0-dim bool tensor（data-dependent）
        pred = h.sum() > 0

        # true_fn / false_fn 接收相同签名，返回相同 shape/dtype
        h_out = torch.cond(
            pred,
            lambda t: F.relu(t),    # true branch
            lambda t: F.gelu(t),    # false branch
            (h,),                   # operands，显式传入闭包变量
        )
        return h_out.mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Case D：torch.compiler.is_compiling() 保护副作用
# ══════════════════════════════════════════════════════════════════════════════

class CaseD_IsCompiling(nn.Module):
    """
    编译态走图友好路径（纯张量运算）；
    eager 态保留 assertion / NaN 检查 / print 等副作用。
    is_compiling() 在编译时返回 True，让整个 if 块在编译图中被 DCE 掉，
    不产生任何 graph break，也不影响 eager 下的调试体验。
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            # 这整块在编译时被 DCE，不产生 graph break
            assert x.dim() == 2, f"expected 2D input, got shape {x.shape}"
            if torch.isnan(x).any():
                print(f"[WARNING] NaN detected in input, shape={x.shape}")

        h = F.gelu(self.fc(x))

        if not torch.compiler.is_compiling():
            if torch.isnan(h).any():
                print("[WARNING] NaN detected after fc+gelu")

        return h.mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Case E：@torch.compiler.disable 隔离 data-dependent 外层
# ══════════════════════════════════════════════════════════════════════════════

@torch.compile(fullgraph=True, mode="default")
def compiled_inner(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """纯张量函数，fullgraph=True，可以放心 max-autotune"""
    return F.linear(x, w, b).relu()


def step_with_disable(x: torch.Tensor, fc: nn.Linear) -> torch.Tensor:
    """
    外层含有 data-dependent 控制流，留在 eager 中执行；
    内层纯张量计算交给 compiled_inner。
    这样 data-dependent 不会污染 compiled_inner 的 graph，
    compiled_inner 依然能 fullgraph=True。
    """
    # data-dependent 决策留在 eager（外层）
    if x.sum().item() > 0:
        out = compiled_inner(x, fc.weight, fc.bias)
    else:
        out = compiled_inner(-x, fc.weight, fc.bias)
    return out.mean(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(fn, x, n=30) -> float:
    """返回平均 forward 时间 ms"""
    # warmup
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main():
    torch.manual_seed(42)
    x = torch.randn(BATCH, DIM, device=DEVICE, dtype=DTYPE)

    print("=" * 65)
    print("  exp05: if-else 控制流的五种处理方式")
    print("=" * 65)

    # ── Case A ───────────────────────────────────────────────────────────────
    section("Case A：data-independent 分支（shape/dtype 判断）")
    model_a = CaseA_DataIndependent().to(DEVICE, dtype=DTYPE)

    info_a = count_graph_breaks(model_a, x)
    print(f"  graph_count = {info_a['graph_count']}")
    print(f"  graph_break_count = {info_a['graph_break_count']}  ← 应为 0")
    print(f"  op_count = {info_a['op_count']}")
    print(f"  ✓ shape/dtype 判断在编译时确定，Dynamo 生成 guard 后 specialize，无 break")

    compiled_a = torch.compile(model_a, fullgraph=False)
    t_a = benchmark(compiled_a, x)
    print(f"  稳态 forward: {t_a:.3f} ms")

    # ── Case B ───────────────────────────────────────────────────────────────
    section("Case B：data-dependent 分支（.item() 触发 graph break）")
    model_b = CaseB_DataDependent().to(DEVICE, dtype=DTYPE)

    info_b = count_graph_breaks(model_b, x)
    print(f"  graph_count = {info_b['graph_count']}  ← 应 > 1（每个 break 切出新 subgraph）")
    print(f"  graph_break_count = {info_b['graph_break_count']}  ← 应 ≥ 1")
    print(f"  break_reasons = {info_b['break_reasons'][:2]}")
    print(f"  ✗ graph break 导致：")
    print(f"     - 中间 tensor h 实体化到 HBM（丢失 fc+activation fusion）")
    print(f"     - backward 分段（两段 subgraph 分别生成 backward）")
    print(f"     - CUDA Graphs 失效（reduce-overhead 模式无法捕获）")

    compiled_b = torch.compile(model_b, fullgraph=False)
    t_b = benchmark(compiled_b, x)
    print(f"  稳态 forward: {t_b:.3f} ms  （对比 Case C 看 cond 的代价）")

    # ── Case C ───────────────────────────────────────────────────────────────
    section("Case C：torch.cond 改写（保持单图，fullgraph=True）")
    model_c = CaseC_TorchCond().to(DEVICE, dtype=DTYPE)

    info_c = count_graph_breaks(model_c, x)
    print(f"  graph_count = {info_c['graph_count']}  ← 应为 1（整函数单图）")
    print(f"  graph_break_count = {info_c['graph_break_count']}  ← 应为 0")
    print(f"  op_count = {info_c['op_count']}")

    compiled_c = torch.compile(model_c, fullgraph=True)
    t_c = benchmark(compiled_c, x)
    print(f"  稳态 forward: {t_c:.3f} ms")
    print(f"  ✓ torch.cond 把分支编码为 HOP subgraph，整体单图，支持 fullgraph=True")
    print(f"  注意：PyTorch 2.7 的 cond backward 仍是 prototype，生产训练需做回归测试")

    # ── Case D ───────────────────────────────────────────────────────────────
    section("Case D：is_compiling() 保护副作用（零代价调试代码）")
    model_d = CaseD_IsCompiling().to(DEVICE, dtype=DTYPE)

    info_d = count_graph_breaks(model_d, x)
    print(f"  graph_count = {info_d['graph_count']}")
    print(f"  graph_break_count = {info_d['graph_break_count']}  ← 应为 0")

    compiled_d = torch.compile(model_d, fullgraph=True)
    t_d = benchmark(compiled_d, x)
    print(f"  稳态 forward: {t_d:.3f} ms")
    print(f"  ✓ assert/print 在编译图中被 DCE，eager 下仍正常执行")
    print(f"  适合：保留 NaN 检查、shape 断言等调试代码而不影响编译性能")

    # ── Case E ───────────────────────────────────────────────────────────────
    section("Case E：隔离策略（data-dependent 留 eager，内层 fullgraph=True）")
    fc = nn.Linear(DIM, DIM).to(DEVICE, dtype=DTYPE)

    # compiled_inner 已经在模块顶层用 @torch.compile(fullgraph=True) 装饰
    t_e = benchmark(lambda inp: step_with_disable(inp, fc), x)
    print(f"  稳态 forward: {t_e:.3f} ms")
    print(f"  ✓ 外层 data-dependent 决策在 eager 执行，内层纯计算享受 fullgraph 优化")
    print(f"  适合：routing、模型并行调度等外层控制逻辑 + 内层大 GEMM 的场景")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    section("汇总：graph break 数 vs 稳态延迟")
    rows = [
        ("A. data-independent（guard）", info_a["graph_break_count"], t_a),
        ("B. data-dependent（break）",   info_b["graph_break_count"], t_b),
        ("C. torch.cond（单图）",        info_c["graph_break_count"], t_c),
        ("D. is_compiling()（DCE）",     info_d["graph_break_count"], t_d),
        ("E. 隔离（外层 eager）",        0,                           t_e),
    ]
    print(f"  {'策略':35s}  {'breaks':>6}  {'fwd(ms)':>8}")
    print(f"  {'─'*55}")
    for name, breaks, ms in rows:
        print(f"  {name:35s}  {breaks:>6}  {ms:>8.3f}")

    print("""
关键结论：
  - data-independent 分支（shape/dtype）：Dynamo 自动 specialize，零代价
  - data-dependent 分支（.item()/bool）：必须主动处理，否则 graph break 降性能
  - torch.cond：最彻底，保持单图，但两分支 output 必须严格一致，backward 有坑
  - is_compiling()：保留调试代码的最佳实践，编译图中完全 DCE
  - 隔离策略：最保守但最稳定，data-dep 留 eager，内层计算享受完整优化

推荐决策树：
  1. 分支依赖 shape/dtype/config？→ 直接写，Dynamo 自动处理
  2. 只是 print/assert？→ 用 is_compiling() 保护
  3. 少量 data-dep 分支（<5）？→ 容忍 graph break（fullgraph=False）
  4. 核心 data-dep 分支且输出 shape 一致？→ 尝试 torch.cond + 回归测试
  5. 外层调度 + 内层计算？→ 隔离策略（Case E）

下一步 → exp06_diagnostics.py：完整诊断工具链实战
""")


if __name__ == "__main__":
    main()
