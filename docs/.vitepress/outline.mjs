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
      { number: 1, dir: 'chapter1', title: '环境准备与第一个 TileLang kernel',
        summary: 'Apple Silicon 安装 TileLang（Metal 后端），跑通逐元素加法',
        lead: '装好环境并跑通第一个 kernel，建立写 kernel→校验→计时最小闭环。' },
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
  { title: '附录 D · TileLang 与 MLX 的编程范式', path: '/appendix/programming-models/' },
]

export const crossRef = [
  { title: '第 3 篇 · 算子优化 Agent（对照）', path: '/part3-agent/' },
  { title: '第 4 篇 · 模型实战（对照）', path: '/part4-models-agent/' },
]

export const navItems = [
  { text: '首页', link: '/' },
  { text: '入门', link: '/part0-intro/', activeMatch: '^/part0-intro/' },
  { text: '计时', link: '/part1-profiling/', activeMatch: '^/part1-profiling/' },
  { text: '算子实战', link: '/part2-kernels/', activeMatch: '^/part2-kernels/' },
  { text: '附录', link: '/appendix/appendix-a-env-install/', activeMatch: '^/appendix/' },
  ...(repoUrl ? [{ text: 'GitHub', link: repoUrl }] : []),
  { text: 'Metal 版（对照）', link: 'https://github.com/datawhalechina/hello-gpu' },
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
  { text: '对照篇目', collapsed: true, items: crossRef.map((a) => ({ text: a.title, link: a.path })) },
]