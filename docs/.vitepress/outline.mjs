// 站点目录：第 2 篇 6 章 + 首页。发布到自己的 Git 仓库后填写 repoUrl。
export const repoUrl = 'https://github.com/guimingyue/hello-gpu-tilelang'
export const repoBranch = 'main'
export const upstreamUrl = 'https://github.com/datawhalechina/hello-gpu'
export const metalUrl = 'https://github.com/datawhalechina/hello-gpu'

export const parts = [
  {
    prefix: '/part0-intro/',
    navText: '入门',
    title: '入门与第一个 TileLang kernel',
    readmeTitle: '第 0 篇：入门',
    landing: '/part0-intro/',
    chapters: [
      { number: 0, dir: 'chapter0', title: '课程地图：用 TileLang 在 Apple GPU 上学算子优化',
        summary: '课程结构、两条路线、M1 基线与口径', lead: '地图与约定，不写代码。' },
      { number: 1, dir: 'chapter1', title: '环境准备与第一个 TileLang kernel',
        summary: 'Apple Silicon 安装 TileLang（Metal 后端），跑通逐元素加法',
        lead: '装好环境并跑通第一个 kernel，建立写 kernel→校验→计时最小闭环。' },
      { number: 2, dir: 'chapter2', title: '执行层次：thread / SIMD-group / threadgroup / grid',
        summary: 'T.Kernel 与 T.Parallel 对应 Apple GPU 四级层次', lead: '建立「当前线程算哪部分数据」的图景。' },
      { number: 3, dir: 'chapter3', title: '内存层次：统一内存、threadgroup 内存与寄存器',
        summary: 'T.alloc_shared / T.alloc_fragment 与分块复用', lead: '数据在哪、搬运代价几何。' },
      { number: 4, dir: 'chapter4', title: '从 kernel 到实验：校验、计时与受控复测',
        summary: '正确性先于性能，三进程受控实验与证据归档', lead: '让每个数字有出处、可复跑。' },
    ],
  },
  {
    prefix: '/part1-profiling/',
    navText: '计时',
    title: '可信计时与性能分析',
    readmeTitle: '第 1 篇：计时',
    landing: '/part1-profiling/',
    chapters: [
      { number: 5, dir: 'chapter5', title: '可信计时：TileLang Metal 的口径',
        summary: '端到端墙钟 vs GPU 时间戳，三进程受控实验',
        lead: 'Metal 后端无 GPU 时间戳，统一墙钟口径与三进程受控实验。' },
      { number: 6, dir: 'chapter6', title: '剖析：KernelTrace 与 Instruments 的边界',
        summary: '核对启动规模，明确墙钟口径与硬件计数器边界', lead: '计时回答多久，剖析回答为什么。' },
      { number: 7, dir: 'chapter7', title: 'Roofline：算术强度与 Apple M1 参考线',
        summary: '实测 copy/GEMM 参考线，把算子映射为工作点', lead: '定性带宽/算力受限。' },
    ],
  },
  {
    prefix: '/part2-kernels/',
    navText: '算子实战',
    title: '经典算子与 Kernel 实战（TileLang Metal 版）',
    readmeTitle: '第 2 篇：经典算子与 Kernel 实战',
    landing: '/part2-kernels/',
    chapters: [
      {
        number: 8,
        dir: 'chapter8',
        title: 'Element-Wise：逐元素算子',
        summary: '向量加法：threadgroup 宽度与每线程元素数的受控扫描',
        lead: '从 C[i]=A[i]+B[i] 出发，用 TileLang Metal 后端实测 threadgroup 宽度与每线程元素数，再与 hello-gpu-metal 的 MLX 墙钟对照。',
      },
      {
        number: 9,
        dir: 'chapter9',
        title: 'Reduction：求和归约',
        summary: '两阶段 partial+final：threadgroup 数扫描与 float32 atomic 缺失的代价',
        lead: 'N=1,048,576 的 FP32 求和。Metal 后端无 float32 atomic、无 simd_sum，用 exclusive partial 两阶段归约，扫描 threadgroup 数。',
      },
      {
        number: 10,
        dir: 'chapter10',
        title: 'Normalization：行级 Softmax',
        summary: '稳定 Softmax 三遍扫描：threadgroup 宽度与 thread0 serial 合并的代价',
        lead: '4096×1024 的 shifted-dyadic 输入。一个 threadgroup 负责一行，三次行内归约都用 thread0 serial 合并，扫描 T=256/512/1024。',
      },
      {
        number: 11,
        dir: 'chapter11',
        title: 'Matmul：分块矩阵乘',
        summary: 'T.gemm 分块：tile=16/32/64 与 tile64 的 46 倍负面结果',
        lead: '1024³ 的 FP32 矩阵乘。T.gemm 在 Metal 后端可用但未用 SIMD-group 矩阵内建；tile=64 出现 46 倍 slowdown，如实记录。',
      },
      {
        number: 12,
        dir: 'chapter12',
        title: 'Fusion：Attention 前向',
        summary: '在线融合 vs 物化三阶段：128×64 教学版注意力的两条路线',
        lead: '128×64 的教学版 Attention。online 逐 key 更新 (m,l,acc)，materialized 先物化 scores；两者墙钟几乎相同，收益边界如实写。',
      },
      {
        number: 13,
        dir: 'chapter13',
        title: 'Fused RMSNorm：综合实战',
        summary: '平方和归约 + rsqrt 缩放：T=256/512/1024 与 MLX 三配置对照',
        lead: '4096×1024 的 Fused RMSNorm。行内平方和用 thread0 serial 合并；T=256 最快，T=1024 慢 9 倍，收束全篇方法。',
      },
    ],
  },
  {
    prefix: '/part3-agent/',
    navText: 'Agent',
    title: '算子优化 Agent',
    readmeTitle: '第 3 篇：算子优化 Agent',
    landing: '/part3-agent/',
    chapters: [
      { number: 14, dir: 'chapter14', title: 'Agent 入门：模型提修改，工具给证据',
        summary: '提出候选→校验计时→证据裁决的闭环', lead: '模型不直接碰 GPU，也不自报性能。' },
      { number: 15, dir: 'chapter15', title: '工具与合同：评测器只认 MSL 与 launch(lib,…)',
        summary: '候选白名单、隔离运行与权威计时', lead: '能不能跑在计时之前就有可读答案。' },
      { number: 16, dir: 'chapter16', title: '交互式 Agent：手动循环与在线规划',
        summary: 'step/call 与 run_online 共享守卫与 truthful 状态', lead: '两种驱动，同一套权威工具。' },
      { number: 17, dir: 'chapter17', title: '优化报告与独立复测：接受之后还要确认',
        summary: '配对改进、冻结基线复测、旧复测不可归到当前实现', lead: '接受不等于收益显著。' },
    ],
  },
  {
    prefix: '/part4-models-agent/',
    navText: '模型实战',
    title: '模型实战',
    readmeTitle: '第 4 篇：模型实战',
    landing: '/part4-models-agent/',
    chapters: [
      { number: 18, dir: 'chapter18', title: '模型实战：Core ML / MLX 跑 YOLO',
        summary: 'Apple Silicon 部署 YOLO，自定义算子可换 TileLang', lead: '模型层与 kernel DSL 解耦。' },
      { number: 19, dir: 'chapter19', title: 'LLM 解码：MLX-LM 与量化',
        summary: '4bit/8bit 量化、逐 token 计时、decode 带宽受限', lead: '把算子方法用到 LLM 解码。' },
    ],
  },
]

export const chapters = parts.flatMap((part) =>
  part.chapters.map((chapter) => ({
    ...chapter,
    part,
    path: `${part.prefix}${chapter.dir}/`,
    source: `docs${part.prefix}${chapter.dir}/index.md`,
  })),
)

export const appendices = [
  { title: '附录 A · 环境安装细节与常见坑', path: '/appendix/appendix-a-env-install/' },
  { title: '附录 B · 换一台 Mac：Apple Silicon 代际', path: '/appendix/appendix-b-switch-gpu/' },
  { title: '附录 C · 低层视角：TileLang 之下的 Metal 提交', path: '/appendix/appendix-c-metal-low-level/' },
  { title: '附录 D · TileLang 与 MLX 的编程范式', path: '/appendix/programming-models/' },
]

export const navItems = [
  { text: '首页', link: '/' },
  {
    text: '课程',
    activeMatch: '^/part',
    items: parts.map((part) => ({ text: part.readmeTitle, link: part.landing })),
  },
  {
    text: '附录',
    activeMatch: '^/appendix/',
    items: appendices.map((appendix) => ({ text: appendix.title, link: appendix.path })),
  },
  {
    text: '参考',
    items: [
      ...(repoUrl ? [{ text: 'GitHub', link: repoUrl }] : []),
      { text: 'Metal 版（对照）', link: metalUrl },
    ],
  },
]

export const sidebar = [
  ...parts.map((part) => ({
    text: part.readmeTitle,
    link: part.landing,
    collapsed: false,
    items: part.chapters.map((chapter) => ({
      text: `第 ${chapter.number} 章 ${chapter.title}`,
      link: `${part.prefix}${chapter.dir}/`,
    })),
  })),
  { text: '附录', collapsed: false, items: appendices.map((a) => ({ text: a.title, link: a.path })) },
]