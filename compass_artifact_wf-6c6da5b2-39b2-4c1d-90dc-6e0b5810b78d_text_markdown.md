# PyTorch torch.compile 训练加速深度解析

**torch.compile 在 2026 年已经成为 PyTorch 训练加速的默认路径，但它的威力只有在正确配置动态 shape、控制 recompile 触发源、并与 DDP/AMP 正确协同时才能完全释放。** 本报告面向已有 PyTorch 经验、正在把 `torch.compile` 投入单卡或 DDP 多卡训练的工程师，系统性地梳理 TorchDynamo + AOTAutograd + TorchInductor 的技术栈。核心痛点是用户在训练中存在 **prefix batch 维度抖动**（B1×C×H×W → B2×C×H×W）和 **Python 控制流分支**，二者是 recompile 的两大触发源。本报告的结论是：通过"默认 automatic dynamic + 显式 `mark_dynamic(x, 0, min=2, max=MAX)` + `drop_last=True` + `DDP(model)` 再 `torch.compile`"这一组合，可以在保留 Inductor 全部优化的前提下把 recompile 次数压到个位数。

后文按"历史背景 → 核心技术栈 → 参数详解 → 动态 shape → recompile 诊断 → 控制流 → DDP 集成 → 周边生态 → 端到端最佳实践"九章展开，每个技术点遵循"总体逻辑 → 实现原理 → 优劣势 → 适用场景"四维结构。

---

## 1. 旧版 JIT 的问题与向 torch.compile 的迁移

`torch.jit.trace` 与 `torch.jit.script` (TorchScript) 是 PyTorch 1.x 时代的 JIT 方案，但 **二者已在 2.x 的源码中被标注 `DeprecationWarning`**：官方 `torch/jit/_script.py` 明确建议切换到 `torch.compile` 或 `torch.export`，文档页 `jit_language_reference.html` 写明 "TorchScript is deprecated, please use torch.export instead"。

`torch.jit.trace` 的根本缺陷是**只记录一次运行时发生过的 op**，遇到 data-dependent 控制流（`if tensor.sum() > 0:`、循环次数依赖输入）只追踪当时那一条路径，**并且静默不报错**——模型换一条输入就行为错误。PyTorch 官方 Dynamo Deep-Dive 的原话是："The issue with torch.jit.trace is that it just works if the traced program is not data dependent."

`torch.jit.script` 虽然支持控制流，但要求源码落在 TorchScript 语言子集里：静态类型、不支持继承、不支持 `*args/**kwargs` 任意转发、不支持大部分动态元编程。最致命的是它的 **all-or-nothing** 语义——源码中一处不支持的语法整个模块编译不过，必须手工重写加 `@torch.jit.ignore`、`@torch.jit.unused`，且报错信息晦涩。PT2 论文（ASPLOS'24, Ansel et al.）在第 2 节点名批评："scripting requires significant code changes and will raise errors when unsupported Python is used; tracing silently produces wrong results on data-dependent control flow."

**`torch.compile` 的定位正是解决这两套工具无法同时具备"动态 Python 支持 + 无侵入 + 正确性"的矛盾**：通过 PEP 523 Frame Evaluation API 在 CPython 字节码层拦截、把能追踪的部分编译、把不能追踪的部分交回 Python 解释器（graph break），并用 guards 机制保证复用安全。新代码不应再写 TorchScript；长期需要 C++ 零 Python 部署的场景现在走 `torch.export` + AOTInductor。

---

## 2. torch.compile 核心技术栈

`torch.compile` 的完整流水线是：**Python 源码 → TorchDynamo（字节码层 JIT）→ AOTAutograd（联合 fwd/bwd trace + functionalization + partition）→ TorchInductor（FX → Triton/C++ codegen）**。三者解耦、可独立替换 backend。

### 2.1 TorchDynamo：字节码层的符号解释器

**总体逻辑**。`@torch.compile` 本身不编译任何东西，它只是挂一个 hook。当被装饰函数被调用时，CPython 通过 PEP 523 把"这个 frame 交给谁执行"的权利让给 Dynamo。Dynamo 用一个 **Python 实现的 CPython 字节码解释器**（`torch/_dynamo/symbolic_convert.py::InstructionTranslatorBase`，约 200 个字节码 handler）逐条"符号执行"字节码：不真执行 tensor 操作，而是把 PyTorch 调用记录进一张 FX Graph，同时收集 **guards**（输入必须满足的假设）。这是 Dynamo 相对 jit 的核心飞跃：**无侵入、支持任意 Python**，遇到不支持的结构通过 **graph break** 切成"前段子图 + Python resume 函数 + 后段子图"的拼装，而不是报错。

**实现原理**。Dynamo 内部栈上流动的不是真实 Python 对象，而是 `VariableTracker`：`ConstantVariable`、`TensorVariable`（生成 fx.Proxy）、`ListVariable`、`NNModuleVariable`、`SymNodeVariable`（承载 SymInt/SymFloat）等。每个 VariableTracker 关联一个 `Source`（`LocalSource('x')`、`GetItemSource(LocalSource('y'), 0)`），说明"从原 frame 如何重建这个值"——只有有 Source 的变量才能生成 guard。当符号解释器执行到 PyTorch op（例如 `torch.add`）时，`output_graph.create_proxy()` 把该 op 插入 FX Graph 并返回新的 TensorVariable；当执行到 Python 控制流指令（`POP_JUMP_IF_FALSE`）时，如果判据是 SymInt 比较（data-independent），就向 ShapeEnv 注册 symbolic guard 然后 specialize 到实际那一支；如果是 tensor 值（data-dependent），就触发 graph break。

Guard 机制是 Dynamo 保证复用安全的核心。常见 guard 类型包括 **TENSOR_MATCH**（检查 type、dtype、device、requires_grad、dispatch_key、size、stride）、**TYPE_MATCH**（`___check_type_id`）、**ID_MATCH**、**EQUALS_MATCH**（小 int/str/tuple）、**SHAPE_ENV**（SymInt 算术表达式）、**GLOBAL_STATE**（grad_enabled、autocast、inference_mode、TF32、deterministic）、**TORCH_FUNCTION_MODE_STACK**。**PyTorch 2.4–2.5 把 guards 从 Python lambda 链重构为 C++ `TREE_GUARD_MANAGER` 树**（`torch/csrc/dynamo/guards.cpp`），一次遍历检查全部 guard，eval latency 从毫秒级降到 150–300 μs。典型 `TORCH_LOGS=guards` 输出：

```
TREE_GUARD_MANAGER:
+- RootGuardManager
|  +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None
|  +- GLOBAL_STATE: ___check_global_state()
|  +- GuardManager: source=L['x']
|  |  +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, ...),
|  |                   torch.float32, size=[3, 3], stride=[3, 1])
Guard eval latency = 155.78 us
```

Graph break 的实现本身非常精妙。考虑：

```python
@torch.compile
def fn(a):
    b = a + 2
    print("Hi")       # graph break
    return b + a
```

Dynamo 把 `fn` 的字节码改写为：调用 `__compiled_fn_0(a)` 得到 `b`→ 原生执行 `print("Hi")` → 调用 `__resume_at_14_1(b, a)` 继续追踪。每段子图有独立 guards 和独立 cache entry，存储在 `PyCodeObject.co_extra` 槽位里组成链表。每次调用时 Dynamo 的 eval_frame 回调依次检查链表中每个 entry 的 guard，第一个通过的就运行它的重写字节码；都不过就触发新一轮追踪，新 entry 插入链表头，上限由 `torch._dynamo.config.cache_size_limit`（默认 8）控制。

**优劣势**。相较 jit.script/jit.trace，Dynamo 优势是 **动态 Python 完整支持 + 无需示例输入 + 无需改代码 + 原生 dynamic shape（SymInt）+ 与 AOTAutograd/Inductor/TensorRT/OpenXLA 等 backend 解耦**。劣势是：首次编译延迟大（大模型可能分钟级）；recompile 风险（`int` 参数变化、shape 0/1 specialization、module 实例身份变化、全局 state 切换都会 guard 失败）；调试复杂（官方 Deep-Dive 自嘲 "those insane backtraces"）；某些 Python 特性仍不支持（`async`、复杂 C 扩展、部分 tensor subclass）。

**适用场景**。训练大型模型的稳态阶段（LLM、Diffusion）；推理服务（配 AOTInductor 或 PT 2.7 的 Mega Cache）；有重复层的模型（配 PT 2.5 默认的 regional compilation）；需要 FlexAttention 的 attention 实验。**不适合**：短生命周期脚本（compile 开销回不了本）；模型内部极度动态（每步 shape/dtype 都变 + 大量 `.item()`）；需要纯 C++ 部署（此时走 `torch.export`）。

### 2.2 AOTAutograd：联合 fwd/bwd 追踪 + functionalization + min-cut 切分

**总体逻辑**。AOTAutograd（位于 `torch/_functorch/aot_autograd.py`）负责在编译时**同时生成前向与反向的 FX 图**——不是"先 trace 前向，运行时靠 autograd 引擎即时构造反向"，而是把前后向作为一张 **joint graph** 一次性拿到，再 functionalize、再切分成 `(fw_module, bw_module)`。这样做的好处是：后端（Inductor）对一张纯函数图做全局优化；无需运行时 autograd hook；不同 backend（Inductor / nvFuser / TVM / MAIA）共享同一份 fwd+bwd IR。

**实现原理**。AOTAutograd 通过 `__torch_dispatch__` 截获算子调用（不是 FX symbolic tracing）：用 FakeTensor 构造 `requires_grad=True` 的 primals → 调用用户函数得到输出 → 用 `autograd.grad(out, primals, tangents)` 以 VJP 形式 trace 反向 → 得到输入为 `(primals, tangents)`、输出为 `(fw_outs, gradients)` 的 joint graph。Horace He 在 PyTorch Developer Podcast 中强调："min-cut partitioner doesn't actually do a split; instead, it is deciding how much of the joint graph should be used for backwards"——**先生成一张完整联合图，再决策哪些节点归属反向**。

**Functionalization** 处理 PyTorch eager 语义里破坏纯函数性的三类东西：in-place（`x.add_(y)`）、view（`x.view(-1)`）、alias（两个 tensor 共享 storage）。具体做法：用 `FunctionalTensor` 包住原 tensor；将 `x.add_(y)` 重写为 `x_new = x + y`，在 graph 末尾（epilogue）对真实输入 `copy_(x_new)`；对视图用 **view-replay**——不把 view 保留在图中，而记录 view 信息在运行时用 `as_strided` / `view` / `slice` 重新生成。例外是 scatter/index_put：纯函数化会把 O(n) 变 O(n²)，Inductor 明确保留这类 mutation 作为 IR 原语。

**Partitioner** 决定哪些中间 tensor 作为 saved-for-backward。`min_cut_rematerialization_partition`（`torch/_functorch/partitioners.py`）的算法：

1. 把 joint graph 转成有向图，每个 fx.Node 拆成 `v_in / v_out`，间连一条容量 = "存下来该 tensor 的成本"的边。
2. 节点权重：`mem_size * (1.1 ** dist_from_bw)`——越靠近反向越偏向重算；matmul/conv 等贵的 op 不在重算白名单内。
3. 用 `networkx.minimum_cut` 求最小割。切集就是 saved tensors。
4. 前向图输出改为 saved set；反向图中对未 saved 的节点插入重算。

**min-cut 与传统 gradient checkpointing 的关键差别**：checkpointing 是"以时间换显存"，min-cut 在 Inductor 这种 fusion-friendly 后端上**可以同时降低延迟和显存**——因为重算两个小 pointwise op 融进一个 Triton kernel 比从 HBM 读巨大 activation 更快。functorch tutorial 的 benchmark 上，同一函数：eager fwd=740μs / bwd=1560μs，default partition bwd=909μs，**min-cut bwd=791μs** 且保存的中间 tensor 更少。

**Decompositions** 把 aten 的 2000+ op 展开成 **core aten opset**（约 180 个基础纯函数 op），后端只需支持这个小集合。PyTorch 官方文档定义：core aten opset is fully functional, no inplace or _out variants。AOTAutograd 通过 `make_fx(fn, decompositions=...)` 自动套用。

**优劣势与场景**。优点：训练路径一次拿到完整 fwd/bwd；支持 tensor subclass（DTensor/FP8/NJT）desugaring；min-cut 自动 checkpointing 效果。缺点：对 data-dependent shape/控制流不友好；某些自定义 `autograd.Function` 需特殊处理。**训练场景默认全部启用**——你无法在 `torch.compile` 下关闭 AOTAutograd 而让训练正常工作。

### 2.3 TorchInductor：默认 codegen backend

**总体逻辑**。Inductor（`torch/_inductor/`，Jason Ansel 主导）把 AOTAutograd 产出的 FX 图 lowering 到 **loop-level IR**（约 50 个核心 op），再 codegen 成 Triton（GPU）或 C++/OpenMP（CPU）。设计原则是 "**Pythonic, define-by-run, loop-level**"：IR 的 compute body 是一个 Python callable 接受 sympy 索引返回 `ops.*` 调用链——通过重载 `ops.*` handler，一份 IR 可以解释成 Triton、C++、Halide、MPS。

**实现原理**。流水线是：**FX post-grad passes（pattern match SDPA、conv+bn 融合、常量折叠）→ Lowering（注册 ~433 个 aten op，未注册走 FallbackKernel）→ Scheduler（依赖建图、buffer 规划、fusion 决策）→ Codegen（Triton / C++）→ Wrapper codegen（Python 胶水代码，管理 CUDA Graphs）**。

Fusion 分 vertical 和 horizontal：前者把生产者-消费者两个 pointwise op 融成一个 kernel，避免 global memory round-trip，是 Inductor 最常见的融合；后者把并行的独立 kernel 合并 launch 以减少 overhead，利好小 kernel。`Scheduler.score_fusion` 按 fusion 类别 → 节省的 memory traffic bytes → 节点间距离打分。

Triton codegen 的一个 `torch.log2(x)` 典型输出：

```python
@triton.jit
def triton_poi_fused_log2_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    xmask   = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tmp1 = tl.log2(tmp0)
    tl.store(out_ptr0 + xindex, tmp1, xmask)
```

Matmul 走 **Triton template + CUTLASS backend**（PT 2.5+），`mode="max-autotune"` 时对 `BLOCK_M/N/K, num_stages, num_warps` 组合实跑 benchmark 选最快。PT 2.8 blog 报告 bmm 和 fp8 kernel CUTLASS backend 比 Triton/cuBLAS 快 10–16%。**Epilogue fusion**（PT 2.0+）把 matmul 后的 bias+relu 融入 matmul kernel；**Prologue fusion**（PT 2.7 正式引入）把 matmul 之前的 pointwise（如 fp8 cast、dequant）融入 matmul，显著减少读 A/B 矩阵时的 HBM 带宽——这对 FP8 训练至关重要。

**CUDA Graphs 集成**由 `mode="reduce-overhead"` 触发，底层是 **CUDA Graph Trees**（PT 2.2+）：同一张 compiled graph 不同 branch 共享内存池，支持多 graph 共享、forward+backward 各自捕获。缺点是 CUDA Graph 要求静态 shape 和静态内存地址，动态 shape 会自动 fallback 到非 cudagraph 路径，动态分配的 tensor 需 `mark_static_address`。

**PyTorch 2.4-2.7 Inductor 重要改进**：2.4 引入 BF16 symbolic shape Beta、XPU（Intel GPU）Triton backend、AOTInductor CPU freezing；2.5 引入 **FlexAttention**（自动生成融合 FlashAttention kernel）、cuDNN SDPA（H100 上比 FAv2 快 75%）、**Regional compilation**（重复 nn.Module 只编一次）、CPP GEMM max-autotune template（FP32/BF16/FP16/INT8 + epilogue fusion）；2.6 引入 FlexAttention CPU、FP16 CPU Beta、Python 3.13 支持；2.7 引入 **Prologue fusion**、**Mega Cache**（跨机编译产物缓存）、Foreach Map、Native Context Parallel。

**优劣势与场景**。优点：Python 可 hack、IR 简洁、跨 GPU/CPU、fusion 质量接近手写。缺点：编译时间长（max-autotune 每个 shape 可能几分钟）、Triton 对复杂 mask+indirect load 生成质量不稳、动态 shape 下 CUDA Graph 失效。训练场景默认 `mode="default"` + 按需打开 `max_autotune` 和 `shape_padding` 即可。

---

## 3. torch.compile 参数详解

完整签名：`torch.compile(model=None, *, fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False)`。下面逐参数展开。

### 3.1 mode：四档性能/编译时间权衡

`mode` 与 `options` 互斥。运行 `torch._inductor.list_mode_options("max-autotune")` 可查实际映射。

| mode | 关键 config | 编译时间 | 运行时特征 | 适用场景 |
|---|---|---|---|---|
| `default` | 默认 Inductor；不开 CUDA Graphs、不 autotune | 最短 | 均衡 | 训练/推理通用起点；**训练首选** |
| `reduce-overhead` | `triton.cudagraphs=True` | 中等 | 小 batch/短 kernel 显著降 CPU 开销；workspace 占额外显存；不支持 input mutation | 推理、小 batch 训练 |
| `max-autotune` | `triton.cudagraphs=True` + `max_autotune=True` + `max_autotune_gemm=True` | **显著更长**（首次可能分钟级） | 大 GEMM/conv 可能 +5–30%；依赖 SM 数够的 GPU | 生产推理、固定 shape、长跑作业 |
| `max-autotune-no-cudagraphs` | `max_autotune=True`，不开 CUDA Graphs | 长 | 得到 autotune 的 GEMM 选择但保留输入可变 | 输入 shape 有变、含 control flow 但想 autotune |

**训练场景特别注意**：`reduce-overhead` 下 CUDA Graphs 对 **动态 batch 会频繁 re-record**（`cudagraph_unexpected_rerecord_limit` 默认 8），且与 DDPOptimizer 切出的多 subgraph 各自申请 workspace **极易 OOM**。**用户场景（batch dim 变化 + DDP）应优先用 `mode="default"`**，把 `max_autotune` 留给生产推理。

### 3.2 fullgraph：严格单图 vs 允许 break

`fullgraph=False`（默认）允许 graph break；`fullgraph=True` 要求整函数捕获成单一 FX 图，遇到任何 break 立即抛 `Unsupported` 异常，并隐式打开 `capture_scalar_outputs` 和 `capture_dynamic_output_shape_ops`。**必须开启的场景**：`torch.export`、生产部署配 CUDA Graphs（graph break 破坏连续捕获区段）、CI 检测 break 退化。**训练场景用户有 if-else 分支时**，如果分支是 data-dependent（`if x.sum() > 0`），`fullgraph=True` 会直接报错——推荐先用 `fullgraph=False` 跑通看 break 数，再决定是否用 `torch.cond` 重写到 `fullgraph=True`。

### 3.3 dynamic：三种 shape 策略

- **`dynamic=None`**（默认，automatic dynamic）：**first-static-then-dynamic**——第 1 次按静态编译，第 2 次不同 shape 触发重编并把变化的 dim 升为 symbolic，第 3 次起复用 dynamic 版本不再重编。为生产推荐。
- **`dynamic=True`**：尽可能全符号化。某些 op 仍强制 specialize。对大模型可能编译慢 + 性能差，官方文档原话 "may crash or be slower without benefit"。
- **`dynamic=False`**：关 automatic dynamic，每种 shape 单独静态编译（受 `cache_size_limit=8` 限制）。适合 shape 基本不变、追求极致性能、愿意禁用 shape 变化的场景。

### 3.4 backend：主后端与调试 backend

`torch._dynamo.list_backends()` 返回稳定列表：`inductor`, `cudagraphs`, `onnxrt`, `openxla`, `openxla_eval`, `tvm`。调试 backend：`eager`（Dynamo 捕获后直接 eager 执行，不 lower）、`aot_eager`（加 AOTAutograd 但不 lower）、`aot_eager_decomp_partition`（加 decomp + partition）。**调试三连**：`eager` → `aot_eager` → `inductor`，逐层复现错误定位问题在哪一层。

### 3.5 options：训练相关关键项

`options` 不能与 `mode` 同时传。训练场景最有价值的几个：

- `triton.cudagraphs`：等同 `reduce-overhead` 的核心
- `max_autotune` / `max_autotune_gemm`：autotune pointwise / GEMM
- `epilogue_fusion`（默认 True，max-autotune 下生效）：matmul 后 bias+relu 融入模板
- `coordinate_descent_tuning`：对已选 Triton config 再做坐标下降微调，额外 +5–15% 但编译更长
- **`shape_padding`**：把 tensor 最后一维 pad 成 8/16 对齐以命中 Tensor Core，**对 BF16/FP16 GEMM 训练尤其有益**
- `fallback_random`：调试精度时用 eager rng
- `emulate_precision_casts`：排查 BF16/FP16 融合导致的精度差异

训练推荐组合：

```python
compiled = torch.compile(model, options={
    "shape_padding": True,
    "epilogue_fusion": True,
})
```

### 3.6 torch._dynamo.config 关键字段

源码 `torch/_dynamo/config.py`。2.6+ 把 `cache_size_limit` 重命名为 `recompile_limit`，旧名通过 `Config(alias=...)` 保留。

| 字段 | 默认 | 语义 |
|---|---|---|
| `recompile_limit` / `cache_size_limit` | 8 | 单 frame 可重编译次数，超限 fallback eager |
| `accumulated_recompile_limit` | 256 | 跨 frame 累计编译上限 |
| `fail_on_recompile_limit_hit` | False | 超限直接抛错（CI 用） |
| `suppress_errors` | False | 开启后 Dynamo 异常静默回退 eager |
| `assume_static_by_default` | True | 首次按静态处理 |
| `automatic_dynamic_shapes` | True | guard 失败时自动升 dynamic |
| `automatic_dynamic_shapes_mark_as` | "dynamic"（2.6+ 可选 "unbacked"） | 控制自动 dynamic 用 backed 还是 unbacked symbol |
| `capture_scalar_outputs` | False | 允许追踪 `.item()`（`fullgraph=True` 隐式开） |
| `capture_dynamic_output_shape_ops` | False | 允许追踪 `nonzero`/`unique` |
| `optimize_ddp` | True / "ddp_optimizer" | DDP 按 bucket 切分 |
| `force_parameter_static_shapes` | True | `nn.Parameter` 始终静态 |

---

## 4. 动态 shape 深度处理（用户核心痛点）

这是本报告最关键的章节。用户场景：每个 batch 的第 0 维（batch dim）变化，其他维（C, H, W）固定。

### 4.1 符号 shape 原理：SymInt / ShapeEnv / SymPy

PyTorch 2 用 **SymInt / SymFloat / SymBool** 表示未知维度，运算时产生新的符号节点（两个 SymInt 相加 → 新 SymInt 内部记录这次加法）。`SymNode` 类型擦除承载 SymPy 表达式，方便混合类型运算。**`ShapeEnv`**（`torch/fx/experimental/symbolic_shapes.py`）是单次编译上下文，挂在 FakeTensorMode 上，记录所有自由符号、guards、替换规则，**只有 Python 实现，C++ 侧没有 ShapeEnv**（便于调试）。

编译流水线：进入 frame → 分配 ShapeEnv → 按 policy 为输入分配符号 size（backed 为 `s0, s1, ...`，unbacked 为 `u0, u1, ...`）→ 通过 meta kernel 传播符号过整个图，同时维护 FX IR 和 SymPy 表达式 → 条件分支触发 guard → guard 化简 SymPy → 结束 trace 后把 guards 与编译产物绑定。**只有所有 guard 都成立时编译产物才能复用**，否则 recompile。

**核心设计：hint-guided branch specialization**。PT2 不是完全符号系统，**总是走实际运行时值对应的那条分支**：

```python
def f(x, y):
    z = torch.cat([x, y])
    if z.size(0) > 2:       # 查实际值选一支
        return z.mul(2)
    return z.add(2)
```

编译期根据 hint（真实值）决定走哪条支，另一条支直接消失，在 guards 列表里记下 `z.size(0) > 2`。下次调用先评估 guards 全为真才能复用。这让 symbolic formula 保持简单但让 guards 多。

**Value ranges**。每个符号带值范围（`ShapeEnv.var_to_range`）。默认：backed SymInt 范围为 **`[2, int_oo]`**（注意下界是 2 不是 0 或 1，因为 0/1 被 specialize），size-like unbacked 为 `[0, Inf]`，普通 unbacked 为 `[-Inf, Inf]`。通过 `torch._check(x < 100)` 可收缩范围。

### 4.2 Automatic Dynamic Shapes（PT 2.1+）

核心机制："**Static by default, recompile to dynamic**"：

1. 第 1 次调用：假设全部静态，编译出 guard 都是精确值（`s0 == B1`）的图。
2. 第 2 次换 batch：guard 失败 → 触发 recompile → **Dynamo 检查哪些维变化，自动标成 dynamic**，重编通用图（`s0 ∈ [2, int_oo]`）。
3. 第 3 次起复用 dynamic 版本，不再 recompile。

**代价**：浪费一次 static 编译，首次切换有一次卡顿。关键 config：

```python
torch._dynamo.config.automatic_dynamic_shapes = True     # 默认
torch._dynamo.config.assume_static_by_default = True     # 默认
# PT 2.6+：让自动动态走 unbacked（绕开 0/1 specialization）
torch._dynamo.config.automatic_dynamic_shapes_mark_as = "unbacked"
```

`TORCH_LOGS=dynamic` 看日志：

```
create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo]
eval Eq(s77, 5) [guard added]   ← 发生了 specialization
```

每次看到 `[guard added]` 都对应一次潜在的 recompile 源。

### 4.3 手动标记 API

**`torch._dynamo.mark_dynamic(tensor, dim, min=None, max=None)`** 是用户场景的首选。强约束：告诉 Dynamo 该维必须 dynamic，后续被 specialize 会抛 `ConstraintViolationError`。

```python
x = torch.randn(10, 3, 224, 224)
torch._dynamo.mark_dynamic(x, 0, min=2, max=128)  # batch ∈ [2, 128]

@torch.compile
def f(x):
    return x.mean(dim=(2, 3))
f(x)  # 第一次就 dynamic 编译，省掉 static 阶段
```

**调用必须在 `torch.compile` 外部**（forward 内调用会 `AssertionError: Attempt to trace forbidden callable`）。**min/max 作用**：收缩值范围让 SymPy 化简更多；更紧的 guard 让 Inductor autotune、TensorRT 选更优 kernel；但过紧会 ConstraintViolationError。经验值：`min=max(2, 真实最小), max=实际最大 × 1.5`。

**`maybe_mark_dynamic(tensor, dim)`** 是弱提示，被 specialize 也不报错，适合"不确定是否能动态"的场景。

**`mark_static(tensor, dim)`** 强制静态，仅在你同时开启 `dynamic=True` 或 `TORCH_COMPILE_DYNAMIC_SOURCES` 通配时才需要。

**`mark_unbacked(tensor, dim)`** 从一开始把该维当 unbacked SymInt（`u0`）。Unbacked 的两个关键特性：**不做 0/1 specialization**；任何 size 判断/比较会报 `GuardOnDataDependentSymNode`，需要 `torch._check()` 帮它过关。PT 2.6+ 大幅优化了 unbacked 性能，2026 年已与 backed 持平。

### 4.4 0/1 Specialization 陷阱（最反直觉的坑）

**规则**：PT2 总是对大小为 0 或 1 的维自动 specialize，不管 `mark_dynamic` 与否。原因：size=1 有广播规则特殊性，size=0 有一堆 contiguity/stride 边界。**表现**：

```python
torch._dynamo.mark_dynamic(x, 0, min=1, max=4)
f(x)                                # x.shape=(4,...)  编出 s0∈[2, int_oo]（下界 2 不是 1!）
f(torch.rand(1, ...))               # ❌ recompile！
```

日志会看到 `L['x'].size()[0] >= 2`。**解决方案**：
- 用 `mark_unbacked`：unbacked 没有 0/1 specialization；
- 数据侧保证 batch ≥ 2（DataLoader `drop_last=True`）；
- PT 2.6+ 开 `automatic_dynamic_shapes_mark_as="unbacked"`。

### 4.5 对用户场景（prefix batch dim 变化）的最佳实践

**方案 A（推荐首选）**：显式 `mark_dynamic` 搭配合理 min/max：

```python
model_c = torch.compile(model, mode="default", fullgraph=False)

def train_step(batch):
    x = batch["x"].cuda(non_blocking=True)
    y = batch["y"].cuda(non_blocking=True)
    # 核心：batch 维动态，min≥2 绕开 0/1 specialization
    torch._dynamo.mark_dynamic(x, 0, min=2, max=512)
    torch._dynamo.mark_dynamic(y, 0, min=2, max=512)
    loss = F.cross_entropy(model_c(x), y)
    loss.backward()
```

优势：第一次调用就直接 dynamic 编译，没有"先 static 再重编"的浪费。

**方案 B**：零代码改动依赖 automatic dynamic。稳定阶段（第 3 个 batch 起）性能等同方案 A，适合原型验证。

**方案 C**（PT 2.6+）：让 automatic 走 unbacked：

```python
import torch._dynamo.config as cfg
cfg.automatic_dynamic_shapes_mark_as = "unbacked"
model_c = torch.compile(model)
```

副作用：模型里 `if batch == 1:` 类检查会 graph break。换来的是支持 batch=0/1。

**其他维处理**：**通常不需要 `mark_static` C/H/W**——`assume_static_by_default=True` 的默认已让它们静态，只在全局开了 `dynamic=True` 时才需要显式 mark_static 避免误标。

**DataLoader / collate / bucketing 最佳实践**。`drop_last=True` **强烈建议**——最后一个不完整 batch 会触发 recompile。如果 batch 极度不规律，可做 bucketing 把 batch 聚到少数桶（如 `[8, 16, 32, 64, 128]`），pad 到最近桶，把 recompile 次数上限钉在桶数。

### 4.6 常见陷阱清单

| # | 陷阱 | 表现 | 解法 |
|---|---|---|---|
| 1 | 0/1 specialization | batch=1 永远 recompile | `mark_unbacked` 或 `drop_last=True` |
| 2 | view/reshape 带常量 | 引入整除 guard | 用 `flatten` / `unflatten` |
| 3 | 跨 graph break 丢动态 | 后段 region 从 static 开始 | 所有相关 input 都 `mark_dynamic`，或用 `TORCH_COMPILE_DYNAMIC_SOURCES` |
| 4 | Python int 作为参数 | 每个值重编 | `dynamic_sources` 白名单 |
| 5 | `y` 没和 `x` 一起标 | constraint violation | 同 batch tensor 一起标 |
| 6 | forward 内调 `mark_dynamic` | AssertionError | 挪到 training loop 顶层 |
| 7 | `.item()` / `nonzero` | graph break | 开 `capture_scalar_outputs` + `torch._check` |
| 8 | `LayerNorm` backward specialize batch（#172822） | 训练反向每种 batch recompile | 用 `F.rms_norm` 或只对 forward 动态 |
| 9 | `cache_size_limit` 触顶 | 编译放弃退 eager | 调到 16/64 |
| 10 | `mark_dynamic` min/max 过紧 | ConstraintViolationError | 放宽 max 或用 `maybe_mark_dynamic` |

---

## 5. Recompile 诊断与优化

### 5.1 TORCH_LOGS：首选诊断入口

`TORCH_LOGS` 是 PyTorch 2.x 统一的结构化日志入口，也是 `tlparse` 工具链的输入源。关键 artifact：

- **`recompiles`**：触发 recompile 时打印**第一个**失败的 guard（定位首选）
- **`recompiles_verbose`**：打印**所有** cache 条目的所有失败 guard（深度定位）
- **`guards`**：完整 guard 树（TREE_GUARD_MANAGER）
- **`dynamic`**：symbolic shape 创建、`produce_guards`、specialization
- **`graph_breaks`**：每个 graph break 的位置与原因
- **`graph_code`** / `output_code`：Dynamo FX / Inductor 最终代码
- `+dynamo` / `+aot` / `+inductor`：组件级 DEBUG
- `perf_hints`：Inductor/CUDA Graph 性能提示（`skipping cudagraphs due to ...`）
- `compiled_autograd` / `compiled_autograd_verbose`

两种开启方式：

```bash
TORCH_LOGS="recompiles,guards,dynamic" python train.py
```

```python
import logging, torch
torch._logging.set_logs(recompiles=True, recompiles_verbose=True,
                        dynamic=logging.DEBUG, guards=True)
```

注意环境变量**完全覆盖** `set_logs` 调用。典型 `TORCH_LOGS="recompiles"` 输出：

```
Recompiling function forward in model.py:57
  triggered by the following guard failure(s):
    - 0/0: tensor 'L['x']' size mismatch at index 0. expected 3, actual 4
```

`0/0` 的含义是 `[frame_id/cache_entry_id]`。`recompiles_verbose` 会列出所有 cache entry 的所有失败 guard，便于看到历史演变。

**推荐诊断流程**：`recompiles` 找帧 → `recompiles_verbose` 找全部 → 加 `dynamic` 看 SymInt 创建 → 加 `guards` 看 guard 树全貌 → 遇到全局问题（`___check_global_state()`）用 `TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED`。

### 5.2 Cache Size Limit 配置

`torch._dynamo.config.recompile_limit`（旧名 `cache_size_limit`，默认 8）：单 frame 最多重编次数。超限发警告 **`torch._dynamo hit config.recompile_limit (8)`** 并 fallback eager。`accumulated_recompile_limit`（默认 256）是全局累积上限，原本用来拦截"给每个 Module 都 compile 一次"的灾难性场景。

什么时候调大：多种 shape pattern 并存（LLM 推理 KV cache 各种 seq_len）、显式保留多个静态 buckets、DDP 多 rank 共享同一 frame。vLLM / SGLang 生产实践会抬到 1024+。调大的代价：每个 cache entry 存独立 Triton kernel + AOT graph + guard 函数（几十 MB 级）、冷启动变慢、guard check 线性开销。

`fail_on_recompile_limit_hit=True` 是 CI 守门员：一旦发生重编立即抛错：

```python
torch._dynamo.config.recompile_limit = 1
torch._dynamo.config.fail_on_recompile_limit_hit = True
```

### 5.3 Guard Failure 诊断

Guard 类型速查：`TENSOR_MATCH`（dtype/device/size/stride/requires_grad）、`TYPE_MATCH`、`ID_MATCH`（对象身份）、`EQUALS_MATCH`（Python scalar 值）、`GLOBAL_STATE`（grad/autocast/inference_mode/TF32/deterministic）、`SHAPE_ENV`（符号约束）、`TORCH_FUNCTION_MODE_STACK`。日志读法：

- `- 0/2: L['self'].param_groups[0]['lr'] == 0.008` → EQUALS_MATCH 失败（LR scheduler 最经典坑）
- `tensor 'L['x']' stride mismatch at index 0. expected 985, actual 1011` → stride 失败（channels_last vs contiguous）
- `___check_obj_id(L['self'].qk_mask, 7616288)` → ID_MATCH 失败（每次 forward 重新赋值 self.attr）
- `___check_global_state()` 失败 → autocast / inference_mode / grad_enabled 切换

### 5.4 常见 Recompile 原因及解决方案

1. **Shape 变化**（用户主场景）：`mark_dynamic` 或依赖 automatic dynamic（见第 4 章）。
2. **Dtype 变化**：用 `torch.autocast` 上下文而非手动 `.to(dtype)`。
3. **Module attribute 变化**（Issue #121504）：`self.qk_mask = new_tensor` 每次 id 变 → 改用 `register_buffer` 预创建，in-place 写入。
4. **调用签名变化**：LIST_LENGTH / DICT_KEYS 失败 → 稳定调用签名。
5. **Stride / layout 变化**：统一 `.contiguous(memory_format=...)` 或 `mark_dynamic` stride。
6. **Python scalar 被 guard**（官方经典例子）：LR scheduler 每 step 设不同 lr → 把 lr 包成 `torch.tensor(0.01)` 再给 scheduler。
7. **多次 compile 同一函数**：对相同结构的 module 复用同一 compiled 函数；2.5+ regional compilation 自动处理。
8. **GLOBAL_STATE 切换**：稳定调用上下文。2.6+ 可用 `torch.compiler.set_stance("fail_on_recompile")` 在 CI 中检测。

### 5.5 explain / TORCH_COMPILE_DEBUG / depyf / tlparse

**`torch._dynamo.explain(fn)(*args)`** 返回 `ExplainOutput`：graph_count、graph_break_count、op_count、break_reasons、ops_per_graph、out_guards、compile_times。一次性诊断 graph break 的首选。

**`TORCH_COMPILE_DEBUG=1`** 生成 `torch_compile_debug/run_<ts>-pid_<pid>/` 目录，含 torchdynamo/debug.log、`fx_graph_readable.py`（人可读 FX）、`fx_graph_runnable.py`（可单独执行复现）、`ir_pre_fusion.txt` / `ir_post_fusion.txt`、`output_code.py`（最终 Triton/C++）。开销大，仅离线调试用。

**`tlparse`**（官方 2024 年后主推可视化工具）：

```bash
pip install tlparse
TORCH_TRACE="/tmp/trace" python train.py
tlparse /tmp/trace --latest -o tl_out/
```

HTML 报告里能看到 stack trie、每次 compile 的方块表示（`[0/0]`, `[0/1]`）、recompile 集中区域。支持分布式（每个 rank 一目录）。

**`depyf`**（`pip install depyf`，官方博客推荐）：把 Dynamo 产生的 bytecode 反编译为可读源码并落盘，关键能力是断点式调试 compile 后代码：

```python
with depyf.prepare_debug("./dump_src_dir"):
    for _ in range(100): toy(x, y)
with depyf.debug():
    toy(x, y)   # 在 dump_src_dir 里打断点
```

**`torch._dynamo.utils.counters`** 快速统计：

```python
from torch._dynamo.utils import counters
counters.clear()
compiled_fn(x)
print(counters["stats"]["unique_graphs"])
print(dict(counters["graph_break"]))
```

### 5.6 端到端诊断流程示例

```python
# Step 1: profile 确认是 recompile
from torch._dynamo.utils import counters, compile_times
counters.clear()
for i, batch in enumerate(loader):
    compiled_model(batch["x"])
    if i == 50: break
print("unique_graphs =", counters["stats"]["unique_graphs"])   # 若 > 8 怀疑
print(compile_times())
```

观察到 `unique_graphs=15` + `compile_times` 出现大段 `_compile` → Step 2 打开 `TORCH_LOGS="recompiles_verbose,dynamic"` 复现，日志显示 `size mismatch at index 0` → Step 3 应用 `mark_dynamic(x, 0, min=2, max=64)` → Step 4 验证 `TORCH_LOGS="dynamic,recompiles"` 应看到 `create_symbol s0 = 16 for L['x'].size()[0] [2, 64]` 且后续无 `Recompiling function`。最终 `tlparse` 整体验证：frame `[0/0]` 只出现一次。

---

## 6. 控制流处理（if-else 分支）

### 6.1 Data-independent vs Data-dependent 控制流

Dynamo 遇到分支指令（`POP_JUMP_IF_FALSE`）时，决定是否能 trace 取决于判据是否 data-independent：

- **Data-independent**（可 specialize + guard）：依赖 Python 常量、tensor 的 shape/dtype/device/rank、非 tensor 变量、`torch.compiler.is_compiling()`。例：`if x.dim() == 4`、`if cfg.training`。
- **Data-dependent**（通常 graph break）：依赖 tensor value，如 `.item()`、`tensor.sum() > 0`、`bool(tensor)`、`if tensor_scalar`。

Dynamo 核心假设是"一次编译只追踪一条路径，通过 guards 保证复用正确性"。Data-dependent 无法在不重新执行的情况下知道走哪支，只能把判据和后续分支还原到 Python 执行。

### 6.2 Graph Break 的代价

每次 break 的代价：**CPU 开销**（返回 CPython 解释器执行原始字节码，重新进入新编译 frame 评估 guards，百微秒量级）+ **丢失跨界 fusion 机会**（两段子图无法合并成一个 Triton kernel，中间 tensor 实体化到 HBM）+ **CUDA Graphs 失效**（`reduce-overhead` 需连续可捕获区段，break 会让整块 region fallback 到非 cudagraph）+ **训练下 backward 分段**（AOTAutograd 对每段前向分别生成后向子图，N 段前向 → N 段 backward，无法跨段融合）。

观察工具：`TORCH_LOGS="graph_breaks,recompiles"` 或 `dynamo.explain(fn)(*args)`。

### 6.3 高阶算子：torch.cond / torch.while_loop

HOP 把控制流编码成特殊 op，分支体作为 FX sub-GraphModule 作为参数，从而把语义保留在单一 graph 里，`fullgraph=True` 友好。

**`torch.cond(pred, true_fn, false_fn, operands=())`**：

- `pred` 可以是 Python bool / 0-维 bool tensor / SymBool（如 `x.shape[0] > 4`）。若 Python 常量会 specialize。
- `true_fn` / `false_fn` 必须接受相同签名、返回完全相同 pytree 结构+shape+dtype、无 closure（闭包变量需作为 operands 显式传入）、推理下可 in-place mutation 输入但有梯度时禁止 mutation、返回 tensor 不能与输入别名。

示例：

```python
@torch.compile(fullgraph=True)
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )
```

**重要警告**：PyTorch 2.7 以前文档仍标注 `torch.cond` 为 prototype，训练支持到 2.8 左右才完善；2.4-2.7 中 cond 下的 backward 可能不稳定。

**`torch._higher_order_ops.while_loop`**：语义等价 `while cond_fn(*val): val = body_fn(*val)`，输入输出 pytree 一致。prototype，至 2.7 不支持 training。

**`torch.map` / `torch.scan` / `associative_scan`**：向量化 map、带累积状态的 map、满足结合律时走并行前缀和。

### 6.4 对含 if-else 训练代码的优化策略

| 策略 | 做法 | 优势 | 劣势 |
|---|---|---|---|
| A. `torch.cond` 重写 | 把 data-dep 判断改写为 HOP | 单图，可 kernel fusion，支持 export | 2.4-2.7 prototype，两分支 output 必须严格一致，backward 有坑 |
| B. 允许 graph break 但控制数量 | `fullgraph=False`，用 `TORCH_LOGS=graph_breaks` 审计，只保留少数（<5）不可避免的 break | 改动小，容忍 print / 日志 / 极少 data-dep | 丢 CUDA Graphs，首次编译变多，backward 分段 |
| C. 隔离 data-dependent 部分 | data-dep 段用 `@torch.compiler.disable`，只 compile 内层纯张量函数 | 清晰分层，内层可 `fullgraph=True` + `max-autotune` | 外层 Python 开销 |
| D. `torch.compiler.is_compiling()` | 编译态走图友好路径，eager 态走易读路径 | 保留 eager 的 assertion/print，不影响 compile | 双分支需维护 |

策略 D 模板（在编译图中整段 DCE，不产生 graph break）：

```python
def forward(self, x):
    if not torch.compiler.is_compiling():
        assert x.dim() == 3, f"bad shape {x.shape}"
        if torch.isnan(x).any(): print("NaN detected")
    return self.linear(x)
```

策略 C 模板（隔离）：

```python
@torch.compile(fullgraph=True, mode="max-autotune")
def compiled_block(x, w):
    return F.linear(x, w).relu()

def step(x, w):
    if x.sum().item() > 0:         # data-dep 留在外层 eager
        x = compiled_block(x, w)
    else:
        x = compiled_block(-x, w)
    return x
```

**训练场景 if-else 的推荐决策**：如果分支仅依赖 shape/config，让 Dynamo 自动 specialize（产生 guard 但无 break）；如果 data-dependent 但少（<5 处），用默认 `fullgraph=False` 容忍 break；如果 data-dependent 多且核心（attention pattern、routing），尝试 `torch.cond` 但做回归测试确认 backward 正确；如果 data-dependent 在外层（模型控制/训练流程），用策略 C 隔离。

---

## 7. DDP 与 torch.compile 集成

### 7.1 DDPOptimizer 工作原理

**背景问题**（Will Constable 在 dev-discuss TorchDynamo Update 9）：Eager DDP 的性能优势来自**反向传播 allreduce 与 backward compute 重叠**——autograd hook 在某参数梯度就绪时立即触发对应 bucket 的 allreduce。但一旦 `torch.compile`，AOTAutograd 把 forward 和 backward 各自编译成一个大 graph，DDP 的 C++ autograd hook 只会在整个 backward graph 执行完毕后才集中触发，彻底打破 compute/comm 重叠。实测 naive `compile + DDP` 比 eager DDP **慢高达 25%**。

**核心思想 DDPOptimizer**（`torch/_dynamo/backends/distributed.py`）：Dynamo 在生成 FX forward graph 时检测到外层是 DDP，按 DDP `bucket_cap_mb` 大小在 forward graph 中**插入 graph break** 切成若干 subgraph。每个 subgraph 独立走 AOTAutograd → Inductor 生成独立 forward + backward 函数。backward 按 subgraph 逐段执行，每段结束后 DDP autograd hook 就能触发对应 bucket 的 allreduce，恢复 compute/comm 重叠。

**默认**：`torch._dynamo.config.optimize_ddp = True`（等价 `"ddp_optimizer"`，PT 2.0+ 默认）。Will Constable 报告启用后 DDP+inductor 在 64 GPU AWS EFA 基准上**不慢于 eager DDP 1%**，部分大模型快约 **15%**。调试用 `TORCH_LOGS="ddp_graphs,distributed,dist_ddp"`。

### 7.2 compile(DDP(model)) vs DDP(compile(model))：官方推荐

**官方文档明确推荐 `torch.compile(DDP(model))`**（`pytorch.org/docs/stable/notes/ddp.html`，2.4–2.11 一致）：

> "DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model, such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes."

```python
ddp_model = DDP(model, device_ids=[rank])
compiled = torch.compile(ddp_model)
```

**原因**：只有外层是 DDP 时 Dynamo 才能看到 DDP 包装识别 bucket 边界启动 DDPOptimizer。反之 `DDP(compile(model))` 时 Dynamo 看到的是 eager 模型，编译的是内部 forward，DDP 的 allreduce 在编译 graph 外触发——**Dynamo 无法识别 bucket 边界**，若编译得到整张 forward 大图，backward 也是一整块，allreduce 全部延迟到最后，失去 overlap。

### 7.3 配置与调优

`torch._dynamo.config.optimize_ddp` 取值：

| 取值 | 行为 |
|---|---|
| `"ddp_optimizer"` / `True`（默认） | 按 bucket 切分 forward graph，恢复 overlap；**不兼容 higher-order ops**（activation checkpointing、flex_attention） |
| `"python_reducer"`（实验） | 禁用 C++ Reducer 改用 Python Reducer，配合 **Compiled Autograd** 在 backward graph 中直接 trace allreduce，不需要 graph break。**未来方向** |
| `"python_reducer_without_compiled_forward"` | forward 走 eager，仅反向用 compiled autograd |
| `"no_optimization"` / `False` | 整张 graph + C++ Reducer，完全没有 overlap；不兼容 compiled_autograd |

**`static_graph=True` 交互**：早期不工作，**rohan-varma PR #103487（2023-11）修复后已兼容**。好处：bucket 重排、跳过 find_unused_parameters 扫描、支持 reentrant activation checkpointing。

**常见问题**：DDPOptimizer 不兼容 higher-order op（activation checkpointing、flex_attention，Issue #104674）——变通是 `optimize_ddp=False` 或 `"python_reducer"`。DDPOptimizer + CUDA Graphs（`reduce-overhead`）：每个 subgraph 独立申请 workspace **容易 OOM**，许多用户需关闭 cudagraphs。

**Compiled Autograd 的角色**（PT 2.4 prototype → 2.5 beta）：将整个 backward pass 推迟到 backward 执行时再 trace，能捕获 autograd hook（包括 DDP allreduce hook）进入 graph。与 `optimize_ddp="python_reducer"` 配合时可把 allreduce 直接纳入 backward graph，实现"不切图 + overlap"的理想态，但要求 `torch._dynamo.config.compiled_autograd=True`。

### 7.4 DDP + compile 完整训练脚本

```python
# torchrun --nproc_per_node=8 train.py
import os, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch._dynamo.config.optimize_ddp = "ddp_optimizer"   # 默认
torch._dynamo.config.recompile_limit = 16              # 为可能的动态预留

def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = MyModel().to(device)
    # 关键顺序：DDP 包装后再 compile
    ddp_model = DDP(model, device_ids=[rank],
                    bucket_cap_mb=25,
                    static_graph=True,
                    find_unused_parameters=False)
    compiled = torch.compile(ddp_model, mode="default")   # 避免 reduce-overhead OOM

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=torch.tensor(1e-4))
    loader = DataLoader(ds, batch_size=64, drop_last=True,
                        num_workers=8, pin_memory=True)

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # 核心：标记 batch 维动态
        torch._dynamo.mark_dynamic(x, 0, min=2, max=512)
        torch._dynamo.mark_dynamic(y, 0, min=2, max=512)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(compiled(x), y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dist.destroy_process_group()
```

---

## 8. 周边生态

### 8.1 FlashAttention / F.scaled_dot_product_attention

FlashAttention（Dao 2022）核心是 **IO-aware attention**：tiling + online softmax 让 Q/K/V 在 SRAM 分块计算，不 materialize 完整 N×N attention 矩阵到 HBM，显存 O(N²)→O(N)，同时减少 HBM 读写实现 2-4× 墙钟加速。

**SDPA 作为统一入口**：`F.scaled_dot_product_attention` 自动分派到多个后端：`FLASH_ATTENTION`（FA2，SM80+）、`EFFICIENT_ATTENTION`（xFormers memory-efficient）、**`CUDNN_ATTENTION`**（PT 2.5 加入，H100 上比 FA2 快 **75%**）、`MATH`（纯 C++ fallback）。用 `torch.nn.attention.sdpa_kernel([SDPBackend.CUDNN_ATTENTION])` 强制。SDPA 是原子 op，compile 会保留不 decompose。

**FlexAttention**（PT 2.5+）用 torch.compile 自动生成融合 FlashAttention kernel，支持自定义 `score_mod`/`mask_mod`。FA3（2024，Hopper 专用，利用 WGMMA + TMA + warp specialization）BF16 达 840 TFLOPs/s（85% 峰值），需 H100 + CUDA ≥ 12.3。

**使用建议**：优先 SDPA 让 PyTorch 自动选；H100 显式用 `CUDNN_ATTENTION` 常更优；只在需要变长/特殊 mask 时直接用 `flash-attn`；2026 年大多数训练用 SDPA 或 FlexAttention 即可。

### 8.2 AMP (torch.amp)

`torch.autocast("cuda", dtype=torch.bfloat16)` 在 forward 中将 matmul/conv 等白名单 op 降精度，reduction/softmax/BN 保持 FP32。`torch.amp.GradScaler("cuda")` 缩放 loss 避免 FP16 梯度 underflow。**与 torch.compile 配合**：compile 能看穿 autocast 上下文并将 dtype cast 融入 graph；Inductor 生成的 Triton kernel 遵循 autocast 策略。**BF16 vs FP16**：BF16 动态范围 ≈ FP32，无需 GradScaler（Ampere+ 支持）——LLM 预训练首选，2026 年事实标准。

### 8.3 CUDA Graphs

核心是把一段 CUDA kernel 序列一次性捕获为 graph，replay 时省去 CPU 提交开销。Inductor `mode="reduce-overhead"` 自动集成 **CUDA Graph Trees**（PT 2.2+，支持多 graph 共享内存池、forward+backward 各自捕获）。动态 shape 冲突：shape 变化就重新录制，超过 `cudagraph_unexpected_rerecord_limit`（默认 8）警告。**与用户场景（batch dim 动态 + DDP）互斥**：建议训练用 `mode="default"`。

### 8.4 xFormers

Meta 的库，最著名是 `memory_efficient_attention`。**与 SDPA 关系**：早期 memory-efficient attention kernel 已 upstream 到 SDPA 的 `EFFICIENT_ATTENTION` 后端。**2026 年定位**：维护模式，新项目首选 SDPA + FlexAttention；xFormers 仍常见于 HuggingFace diffusers 的 `pipe.enable_xformers_memory_efficient_attention()`。与 compile 兼容但视作 black-box kernel。

### 8.5 Liger Kernel

LinkedIn 开源的 Triton kernel 库（`linkedin/Liger-Kernel`），为 LLM 训练优化。核心算子（exact 计算，无近似）：fused RMSNorm（~3× 加速、3× 峰值显存降）、fused RoPE（~3×）、fused SwiGLU/GeGLU（~1.5-1.6× 显存）、**fused CrossEntropy**（vocab=163840 时 ~3× 加速、~5× 显存降）、FusedLinearCrossEntropy（显存节省高达 80%）。LLaMA3-8B + 8×A100 FSDP1 基准：**多 GPU 吞吐 +20%，显存 -60%**。

**与 compile 关系**：Liger 已是高度融合 Triton kernel，不需要 compile 再融合。用法是打补丁替换 HF 层：

```python
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
```

与 DDP/FSDP/FlashAttention 正交兼容。对 HF transformer-based LLM 训练建议直接启用。

### 8.6 其他

**Apex**（NVIDIA）：2026 年基本弃用（`torch.amp` 取代 amp，`torch.optim.AdamW(fused=True)` 取代 FusedAdam）。**Transformer Engine (NVIDIA TE)**：FP8 训练库（H100/Blackwell），`te.fp8_autocast` + `DelayedScaling` recipe；Megatron-LM 深度集成；小规模可用 torchao.float8 替代。**TorchAO**：原生设计为 torch.compile 友好；`torchao.float8` H100 上 Llama3 预训练 e2e 1.25-1.5× 加速，**必须 `torch.compile` 才能拿到性能**；与 FSDP2 + Async TP 组合（torchtitan 生态）。**`torch._foreach_*`**：`torch.optim.AdamW(..., fused=True)` 默认在 CUDA 启用，大幅降低 optimizer kernel launch 开销。

---

## 9. 端到端最佳实践（针对用户场景）

针对"单卡或 DDP + prefix batch dim 变化 + if-else 分支"场景的完整配置：

### 9.1 推荐初始配置

```python
import torch
import torch._dynamo

# --- 全局 config ---
torch._dynamo.config.recompile_limit = 16                       # 为边角 shape 留缓冲
torch._dynamo.config.accumulated_recompile_limit = 256          # 默认即可
# PT 2.6+（可选）：让自动 dynamic 产生 unbacked，支持 batch=1
# torch._dynamo.config.automatic_dynamic_shapes_mark_as = "unbacked"

# --- DDP 场景 ---
torch._dynamo.config.optimize_ddp = "ddp_optimizer"             # 默认
# 若用 activation checkpointing 或 flex_attention：
# torch._dynamo.config.optimize_ddp = "python_reducer"
# torch._dynamo.config.compiled_autograd = True

# --- 模型包装（DDP 必须在 compile 之前）---
if dist.is_initialized():
    model = DDP(model, device_ids=[rank], bucket_cap_mb=25,
                static_graph=True, find_unused_parameters=False)

compiled = torch.compile(
    model,
    mode="default",                                             # 避免 reduce-overhead OOM
    fullgraph=False,                                            # 容忍少量 break
    dynamic=None,                                               # automatic dynamic
    options={"shape_padding": True, "epilogue_fusion": True},
)

# --- DataLoader: drop_last=True 绕开 0/1 specialization ---
loader = DataLoader(ds, batch_size=64, drop_last=True,
                    num_workers=8, pin_memory=True)

# --- Training loop: 显式 mark_dynamic ---
for x, y in loader:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=512)
    torch._dynamo.mark_dynamic(y, 0, min=2, max=512)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(compiled(x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### 9.2 诊断流程

遇到性能异常或 recompile 告警时按如下顺序排查：

1. **用 counters 快速 sanity check**：`counters["stats"]["unique_graphs"]` 应在稳态后不再增长；若 >15 即怀疑 recompile。
2. **`TORCH_LOGS="recompiles"`**：找哪个 frame、哪次调用、第一个失败的 guard。
3. **`TORCH_LOGS="recompiles_verbose,dynamic"`**：看所有 cache entry + symbolic shape 创建过程。
4. **`torch._dynamo.explain(fn)(*args)`**：看 graph break 数和原因。
5. **`tlparse`**：`TORCH_TRACE=/tmp/trace python train.py && tlparse /tmp/trace --latest`，可视化整体。
6. **`TORCH_COMPILE_DEBUG=1` + `depyf`**：深度对比两次编译产物的差异。

### 9.3 优化步骤

1. **先用 `backend="eager"` → `"aot_eager"` → `"inductor"`** 逐层验证正确性和找问题出现的组件。
2. **降 graph break 到 0（或 <5 可接受数）**：data-dep 分支改 `torch.cond`（慎用，2.7 之前训练不稳）；副作用用 `torch.compiler.is_compiling()` 保护；极端段落用 `@torch.compiler.disable` 隔离。
3. **降 recompile 到 1**：`mark_dynamic(x, 0, min=2, max=MAX)` 所有带 batch 的 input；DataLoader `drop_last=True`；避免每步改 module attribute（用 `register_buffer` + in-place）；把 Python scalar 改成 `torch.tensor`（尤其 LR scheduler）。
4. **稳定上下文**：把 compile 区域放在固定 autocast/grad_enabled 上下文内，避免 `___check_global_state()` 失败。
5. **性能微调**：确认稳态后，可尝试 `options={"shape_padding": True}` 针对 Tensor Core 对齐；大矩阵训练在 inference 或 eval 阶段考虑 `mode="max-autotune"`；fp8 训练考虑 torchao + prologue fusion（PT 2.7+）。
6. **CI 守门**：`fail_on_recompile_limit_hit=True` + `recompile_limit=1` 或 `torch.compiler.set_stance("fail_on_recompile")` 防止回归。

### 9.4 结语：四条黄金法则

**第一，用默认 automatic dynamic，但显式 `mark_dynamic` 一次省一次编译**——用户场景下首次调用直接 dynamic 编译，避免"先 static 再重编"。

**第二，`drop_last=True` 是免费午餐**——0/1 specialization 是 PT2 最反直觉的设计约定，batch=1 会永远 recompile，业务上容忍 `drop_last` 远比诊断它便宜。

**第三，DDP 必须在 compile 之前**——`torch.compile(DDP(model))` 是启用 DDPOptimizer 的前提，反过来写会让 backward allreduce 退化到 graph 末尾丢失所有 overlap。

**第四，诊断永远从 `TORCH_LOGS="recompiles"` 开始，再到 `tlparse`**——工具链成熟度远超 2023 年的 PT2.0 时代，花 10 分钟学会这两个入口可以省掉几周的盲猜。

2026 年的 torch.compile 已经走出了 PT2.0 时代的边缘 case 不稳定期：unbacked dynamic 与 backed 性能持平、Compiled Autograd 解决 forward break 污染 backward 的历史问题、Mega Cache 允许跨机编译产物共享、regional compilation 让大模型冷启动从分钟级降到秒级。对训练工程师而言，`torch.compile` 已经从"可选的性能实验"变成了"默认应当启用的训练路径"——前提是掌握本报告中的动态 shape、recompile 诊断、DDP 集成三板斧。