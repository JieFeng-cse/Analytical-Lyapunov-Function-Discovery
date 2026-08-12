# GP-meld 并行化修复（2026-08-04）

状态：已实现，等待 smoke 验证（job 19653429）。用户要求实现并记录，供之后回看。

## 问题

GP-meld（DEAP 遗传编程）占 lossless6d 每个 epoch 墙钟时间的 **~60%**（实测
loop_32 日志：GP 平均 111 s / epoch 平均 186 s），而且一直是**单核串行**在跑——
SLURM 作业分配了 16 个 CPU 核 + 1 GPU，但 GP 适应度评估（每代整个种群在 50,000
个训练点上做常数优化 + reward 计算）只用 1 个核，其余 15 核和 GPU 全程闲置。

两处叠加的 bug：

1. `Lyapunov_test_dso.py:247` 硬编码 `pool = None` 传给 `GPController`
   （上游 dso.core.DeepSymbolicOptimizer 本来有 `Pool(n_cores_batch,
   initializer=set_task, ...)` 的逻辑，这个仓库改写入口时丢掉了）。
2. `libs/sd3/dso/dso/gp/gp_controller.py:212`：`__call__` 每次执行
   `self.toolbox.register("map", map)`，用串行内置 map 覆盖注册——即使传入
   pool 也会被丢弃（`__init__` 里把 pool.map 注册成 `"cmap"`，但
   `gp/base.py:91` 实际调用的是 `toolbox.map`，`"cmap"` 从未被读取）。

## 修复

1. `Lyapunov_test_dso.py`（原 247 行）：按 dso.core 的模式创建
   `multiprocessing.Pool(n_cores_gp, initializer=set_task, initargs=(config["task"],))`。
   核数取 `LYA_GP_CORES` > `SLURM_CPUS_PER_TASK` > cpu_count()；
   **设 `LYA_GP_CORES=1` 可完全恢复旧串行行为**（回滚开关）。
   Pool 在父进程初始化 CUDA 之前创建（fork 安全）。
2. `gp_controller.py:212`：注册
   `self.pool.map if self.pool is not None else map`。

两个仓库同步打了相同补丁：`~/Analytical-Lyapunov-Function-Discovery`
（diffusion-generator-design 分支）和 `~/ALFD-rotor`（rotor-benchmarks 分支）。

## 为什么这是安全的（不改变科学语义）

- 并行只作用于适应度评估的 map；演化算子（选择/交叉/变异）、随机种子、
  pool.map 的结果顺序都与串行一致，GP 轨迹应逐位等价（BFGS 常数优化确定性）。
- `Program.__getstate__/__setstate__`（program.py:279/302）就是为并行 reward
  设计的：worker 里 `p.r`（@cached_property）算完缓存，pickle 回传 tokens +
  reward + 优化后常数。`_eval_step` 用 `skip_cache=True`，缓存写回由主进程
  的 base.py 统一做，无竞争。
- worker 不触碰 torch/CUDA（reward 全 numpy/scipy）；sbatch 已设
  OMP_NUM_THREADS=1，无核争抢。PGD/SHGO 阶段本来就有自己的
  `mp.Pool(N_POOL)`，不受影响。

## 预期收益与验证

- 预期：GP 阶段 ~111s → 约 10s（16 核），lossless6d epoch ~186s → ~85s，
  同等 walltime 下吞吐约 2 倍。GPU 闲置占比相应下降。
- 验证：`repro/smoke_diffusion.sbatch`（job 19653429，gpu-debug，pendulum,
  20k samples）。对比基线 = 2026-06-29 的 `lya_diff_smoke_18718425.out`
  （GP ≈ 34 s/epoch）。看新日志里 `Genetic Porgramming Runtime` 行。
  同时核对 reward/成功行为与历史 smoke 一致（防止并行引入语义偏差）。

## 验证结果（2026-08-04，job 19653429）

冒烟（pendulum, 2 维）**功能通过、加速存疑**：
- 无崩溃，行为正常（2 epoch 内 reward 0.777→0.793，预算耗尽正常退出，
  与历史 smoke 同量级）。pool 路径工作正常。
- 但 GP 用时 38.3s / 45.3s per epoch，对比 6-29 串行基线 ~34s，**并没有变快**。
  可能原因：pendulum 每次适应度评估太便宜（~13k evals/epoch ÷ 34s ≈ 2.6ms/eval，
  15k 点、2 变量、常数优化很快），进程池的 pickle/IPC 开销吃掉了收益；外加
  不同节点的噪声。GP 的串行部分（选择/交叉/变异/约束检查，全在主进程）也不受
  pool 影响。
- **结论：修复对科学行为无害，但加速效果必须在 lossless6d（50k 点、6 维、
  贵的常数优化）上才能定论**——那里单次评估贵得多，eval 占比高，池化收益
  才会显现。待下一个 lossless6d 作业跑起来后对比 "Genetic Porgramming
  Runtime" 与 loop_32 的 111s/epoch 基线。若仍无收益，考虑：增大 chunksize、
  或把并行下沉到"每个个体内部的 50k 点求值"以外的粒度，或接受现状转向
  方案 2（异步流水线）。

## 后续可选（未实现，见会话讨论）

- 方案 2：GP 与 GPU 流水线化（异步专家，滞后一个 epoch）——需走 loop 协议 A/B。
- 方案 3：表达式求值搬上 GPU（大工程）。
- 方案 4：降 generations / 子采样适应度点——方案 1 生效后动机减弱。
- train.py:387 还有一处 `pool = None` 把 batch reward 预计算的并行路径关着，
  如需进一步压缩 CPU 阶段可一并启用（当前保持原样，控制变量）。

## 注意

- 补丁尚未 git commit（两个仓库都是工作树改动）。
- 自改进循环若恢复（rm experiments/LOOP_PAUSED），loop 分支从旧 commit 分出
  时**不含**本补丁；若希望 loop 变体也享受加速，需先把本补丁 commit 进
  diffusion-generator-design 再让循环从新 base 分支。
