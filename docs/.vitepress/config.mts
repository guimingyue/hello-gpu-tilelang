import { defineConfig } from 'vitepress'
import footnote from 'markdown-it-footnote'
import figurePlugin from './markdown-figures.mjs'
import { navItems, sidebar, repoUrl, repoBranch } from './outline.mjs'

// VitePress 2 alpha 会把 MathJax 静态 SVG 的 viewBox 小写化；显式绑定以保留大小写。
function prepareMathSvgTemplate(value: string) {
  return value
    .replace(/^<mjx-container v-pre /, '<mjx-container ')
    .replace(/ viewbox="([^"]+)"/gi, (_, viewBox: string) => ` :viewBox="'${viewBox}'"`)
    .replace(/&amp;(lt|gt|amp);/g, '&$1;')
}

const configured = new WeakSet<object>()

export default defineConfig({
  lang: 'zh-CN',
  title: 'hello-gpu-tilelang',
  description: '用 TileLang（Metal 后端）学 GPU 算子优化：Apple M1 实测墙钟，与 hello-gpu-metal 的 MLX 同口径对照',
  base: '/hello-gpu-tilelang/',
  cleanUrls: true,
  appearance: 'dark',
  srcExclude: ['**/AGENTS.md', '**/CLAUDE.md'],

  vite: {
    resolve: {
      alias: {},
    },
  },

  transformPageData(pageData) {
    if (pageData.relativePath.startsWith('part2-kernels/')) {
      pageData.frontmatter.outline = [2, 2]
    }
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag: string) => tag.startsWith('mjx-'),
      },
    },
  },

  markdown: {
    math: true,
    headers: { level: [2] },
    async config(md) {
      if (configured.has(md)) return
      configured.add(md)
      md.use(footnote)
      md.use(figurePlugin)

      const mathInline = md.renderer.rules.math_inline!
      md.renderer.rules.math_inline = (tokens, idx, options, env, self) =>
        prepareMathSvgTemplate(mathInline(tokens, idx, options, env, self))
      const mathBlock = md.renderer.rules.math_block!
      md.renderer.rules.math_block = (tokens, idx, options, env, self) =>
        prepareMathSvgTemplate(mathBlock(tokens, idx, options, env, self))

      md.renderer.rules.footnote_caption = (tokens, idx) => `[${Number(tokens[idx].meta.id + 1)}]`

      const fence = md.renderer.rules.fence!
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const language = token.info.trim().split(/\s+/)[0]
        if (language === 'metal' || language === 'msl') {
          token.info = token.info.replace(/^\s*(metal|msl)/, 'cpp')
        }
        return fence(tokens, idx, options, env, self)
      }

      const image = md.renderer.rules.image!
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        if (token.attrIndex('loading') < 0) token.attrSet('loading', 'lazy')
        if (token.attrIndex('decoding') < 0) token.attrSet('decoding', 'async')
        return image(tokens, idx, options, env, self)
      }
    },
  },

  themeConfig: {
    nav: navItems,
    sidebar,
    sidebarMenuLabel: '全书目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换至浅色',
    darkModeSwitchTitle: '切换至深色',
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索章节、算子关键词…', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换' },
          },
        },
      },
    },
    ...(repoUrl
      ? { editLink: { pattern: `${repoUrl}/blob/${repoBranch}/docs/:path`, text: '在 GitHub 上编辑此页' } }
      : {}),
    outline: { level: [2, 3], label: '本页目录' },
    footer: {
      message: '改写自 <a href="https://github.com/datawhalechina/hello-gpu" target="_blank">Datawhale hello-gpu</a>（TileLang Metal 版）',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">CC BY-NC-SA 4.0</a> 进行许可',
    },
  },
})