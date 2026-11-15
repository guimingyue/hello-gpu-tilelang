# hello-gpu-tilelang 教程改造 Review 报告

- 审查日期：2026-11-14
- 审查范围：全部文档（docs/）、全部代码（code/part2-kernels/）、实验证据归档（evidence/）、README 与站点配置
- 参照蓝本：`~/github/hello-gpu-metal`（Metal 版，第 8–13 章直接来源）与 `~/github/hello-gpu`（AMD/ROCm 版上游）
- 审查方式：逐章通读文档与代码、evidence 数字逐项核对、文档内嵌路径逐一验证、实跑冒烟测试、执行 `npm run docs:build`

## 总体判断

**核心内容（第 2 篇第 8–13 章）已改造完成且质量很高**；剩余的是少量未改造干净的遗留表述与仓库层面的小问题（见下文问题清单）。

## 已验证完成的部分

| 项目 | 结果 |
| --- | --- |
| AMD/ROCm 术语残留 | 无。全库 grep `rocm/hipcc/gfx/rdna/cdna/wavefront/rocprof//opt/rocm` 等均无命中 |
| 章节数字与证据一致性 | 第 8–13 章文档表格中的中位数、进程范围、带宽、TFLOPS 与 `evidence/metal-rounds/summary/summary.csv` 逐项吻合 |
| 边界用例数 | 与 `boundary.json` 一致：第 8 章 65、第 9 章 66、第 10 章 12、第 11 章 8、第 12 章 16、第 13 章 24 |
| MLX 引用数字 | 与 hello-gpu-metal 同口径墙钟实验一致（已抽验第 8 章全部 5 行），`provenance=cited-hello-gpu-metal` 标注规范 |
| 代码与文档一致性 | 6 个 TileLang kernel、`common/rounds.py` 三进程框架、各章 `run_rounds/validate/summarize` 与文档片段及命令一致 |
| 文档内嵌路径 | 24 处 `<<< @/../...` 引用的源码与日志文件全部存在 |
| 图片 | 六章 `images/round-summary.png/svg` 全部存在 |
| 实跑验证 | `code/part2-kernels/chapter8/smoke_metal.py` 通过（`max_abs_error=0.0`） |
| 环境基线 | 实测与文档一致：TileLang 0.1.15、PyTorch 2.15.0、NumPy 2.5.4、Python 3.12.11、MPS 可用 |
| 站点构建 | `npm run docs:build` 通过，全部页面（part0/part1/part2/appendix/part3/part4）正常生成 |
| 导航/链接 | sidebar 所有条目均有对应页面；附录 D 编号与 Metal 版体系一致 |
| 计时口径 | 端到端墙钟（`wall-clock-full-operator`）全篇贯穿一致，负面结果（第 11 章 tile64 46 倍、第 10/13 章 T=1024 退化）如实记录 |

## 问题清单

### P1 — 内容矛盾/误导，必须修

1. **`docs/part3-agent/index.md:8`**：遗留 Metal 版评测器合同表述——「把候选 kernel 换成 TileLang（`target="metal"`）即可复用同一评测器合同（`MSL` 字面量 + `launch(lib,…)`）」。逻辑矛盾：换成 TileLang 后恰恰不能用 MSL 字面量 + launch(lib) 的合同（TileLang 是 Python 端 JIT 编译调用）。应改为描述 TileLang 候选对应的评测方式（TileLang 脚本 + 墙钟口径）。
2. **`docs/part2-kernels/chapter9/index.md:263`（另见 9.2.3）**：断言「`T.serial` 循环内加 `if` guard 会触发 LayoutInference 缺陷，必须补零 + 无条件读取」，与第 10 章、第 13 章及第 1 章/`smoke_metal.py` 的实际用法矛盾——后者均在 `T.serial`/`T.Parallel` 循环内使用 guard 且边界用例全过。缺陷的真实触发条件需要收窄（例如「在 grid-stride 累加写 `partials` 的模式下」）或给出最小复现。
3. **`docs/.vitepress/outline.mjs:5`（影响全站 30+ 处）**：所有以「hello-gpu-metal」名义的链接均指向 `https://github.com/datawhalechina/hello-gpu`（AMD/ROCm 版仓库），而引用的 MLX 对照数字实际来自本地 hello-gpu-metal 仓库。读者无法通过链接复现被引用的实验。若 Metal 版未公开发布，应链接 AMD 上游并注明数字出处（未发布的 Metal 版 / 私有仓库），或在 `mlx_reference.json` 中给出说明；另注：`upstreamUrl`/`metalUrl` 两个导出在站点代码中并未被使用。

### P2 — 表述不准确，建议修

4. **`README.md:21-32`**：README 声称「保留原课程 20 章 + 附录 + Notebook 的结构」，结构图列出 `code/part0-intro`、`part1-profiling`、`part3-agent`、`part4-models-agent`、`notebooks/`，但这些目录均不存在（`code/` 下只有 `part2-kernels`，`notebooks/` 完全没有）。建议改为「已发布：第 2 篇 6 章 + 第 1/5 章；规划中：其余章节」，与首页（只宣传第 2 篇，是准确的）对齐。
5. **`README.md:27`**：「TileLang + MSL/MLX 对照」与实际不符——仓库内没有任何 MSL 实现，实际是 TileLang（本机实测）+ MLX（引用数字）两条路线。应改为「TileLang + MLX 引用对照」。
6. **`docs/appendix/programming-models/index.md:10`**：附录 D「N=6，启动覆盖 8 个位置，最后两个越界」沿用自 Metal 版 8 线程示例，但 D.2 展示的代码是 `threads=256`（N=6 时 1 个 threadgroup 覆盖 256 个位置、250 个越界）。示例数字应对齐当前代码。

### P3 — 小问题/仓库卫生

7. **`docs/part2-kernels/chapter9/index.md:254`**：标题编号 9.6.3 重复（「其他规模验证」与「尾部正确性」），第二个应为 9.6.4。
8. **构建产物陈旧**：`docs/.vitepress/dist/` 与 `cache/` 原为旧构建（缺 part0/part1/appendix 页面、旧导航）。本次审查已重新执行 `npm run docs:build`，产物已刷新为最新；发布前记得重新构建即可。
9. **`threads > cols` 失效表述存疑**（第 10 章 10.7、第 13 章限制清单）：文中归因于「shared 合并读到未初始化槽位」，但最终代码中每个线程都执行 `s[tid] = ...`，所有 shared 槽位均有写入，无法从代码推出该失效模式（审查中未实测验证；边界脚本刻意用 `eff=min(T,cols)` 回避了）。建议补最小复现或修正归因表述——不影响教程正确性，因为实现已主动回避。

## 结论

第 8–13 章的改造（文档、代码、实验证据、图表）完整且自洽。修复上述 P1 三处（part3-agent 遗留表述、第 9 章 guard 结论与 10/13 章矛盾、hello-gpu-metal 引用链接指向错误仓库），并顺手处理 P2/P3（README 结构图、「MSL/MLX 对照」措辞、附录 D 示例数字、9.6.3 编号）后，教程即可视为改造完成。
