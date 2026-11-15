// markdown-it 扩展（简化版）：::: figure <id> 容器 + @fig-id 交叉引用。
//
//   ::: figure fig-round-summary
//   ![逐轮比较](./images/round-summary.png)
//
//   柱长为三个独立进程 median 的中位数，误差线为进程范围。
//   :::
//
// 渲染为 <figure class="figure" id="...">…<figcaption>图 N.M　…</figcaption></figure>；
// 正文中的 @fig-id 替换为指向该 figure 的链接「图 N.M」。
import container from 'markdown-it-container'

const NAME = 'figure'
const REF = /@(fig-[a-z0-9][a-z0-9-]*)/g

function chapterOf(envPath) {
  const m = /chapter(\d+)/i.exec(envPath || '')
  return m ? Number(m[1]) : null
}

function closeIndex(tokens, open) {
  const level = tokens[open].level
  for (let i = open + 1; i < tokens.length; i++) {
    if (tokens[i].type === `container_${NAME}_close` && tokens[i].level === level) return i
  }
  return -1
}

function lastParagraph(tokens, open, close) {
  for (let i = close - 1; i > open; i--) {
    if (tokens[i].type !== 'paragraph_close') continue
    const level = tokens[i].level
    for (let j = i - 1; j > open; j--) {
      if (tokens[j].type === 'paragraph_open' && tokens[j].level === level) return [j, i]
    }
  }
  return null
}

export default function figurePlugin(md) {
  md.use(container, NAME, {
    validate: (params) => /^figure\s+[a-z0-9][a-z0-9-]*\s*$/i.test(params.trim()),
    render: (tokens, idx) => (tokens[idx].nesting === 1 ? '<figure class="figure">\n' : '</figure>\n'),
  })

  md.core.ruler.after('block', 'figure_numbering', (state) => {
    const env = state.env || {}
    const chapter = chapterOf(env.path || env.relativePath || '')
    const figures = {}
    let seq = 0
    const tokens = state.tokens
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i]
      if (t.type !== `container_${NAME}_open`) continue
      const m = /^figure\s+([a-z0-9][a-z0-9-]*)\s*$/i.exec(t.info.trim())
      if (!m) continue
      const id = m[1].toLowerCase()
      seq += 1
      const num = chapter != null ? `${chapter}.${seq}` : `${seq}`
      figures[id] = num
      t.meta = { ...(t.meta || {}), figureId: id, figureNum: num }
      const close = closeIndex(tokens, i)
      if (close < 0) continue
      tokens[close].meta = { ...(tokens[close].meta || {}), figureId: id }
      const pair = lastParagraph(tokens, i, close)
      if (pair) {
        tokens[pair[0]].meta = { ...(tokens[pair[0]].meta || {}), figureCaption: true, figureNum: num }
        tokens[pair[1]].meta = { ...(tokens[pair[1]].meta || {}), figureCaption: true }
      }
    }
    env._figures = figures
  })

  md.core.ruler.after('inline', 'figure_references', (state) => {
    const figures = (state.env || {})._figures || {}
    if (!Object.keys(figures).length) return
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue
      const next = []
      for (const child of token.children) {
        if (child.type !== 'text' || !REF.test(child.content)) {
          next.push(child)
          continue
        }
        REF.lastIndex = 0
        let last = 0
        let m
        const push = (type, content, attrs) => {
          const t = new state.Token(type, type === 'text' ? '' : 'a', type === 'text' ? 0 : type === 'link_open' ? 1 : -1)
          if (content != null) t.content = content
          if (attrs) for (const [k, v] of Object.entries(attrs)) t.attrSet(k, v)
          t.level = child.level
          next.push(t)
        }
        while ((m = REF.exec(child.content)) !== null) {
          if (m.index > last) push('text', child.content.slice(last, m.index))
          const num = figures[m[1]]
          if (num) {
            push('link_open', null, { href: `#${m[1]}`, class: 'figure-ref' })
            push('text', `图 ${num}`)
            push('link_close')
          } else {
            push('text', `@${m[1]}`)
          }
          last = m.index + m[0].length
        }
        if (last < child.content.length) push('text', child.content.slice(last))
      }
      token.children = next
    }
  })

  const open = md.renderer.rules[`container_${NAME}_open`]
  md.renderer.rules[`container_${NAME}_open`] = (tokens, idx, opts, env, self) => {
    const meta = tokens[idx].meta || {}
    return meta.figureId
      ? `<figure class="figure" id="${meta.figureId}">\n`
      : open(tokens, idx, opts, env, self)
  }
  const paragraphOpen = md.renderer.rules.paragraph_open
  md.renderer.rules.paragraph_open = (tokens, idx, opts, env, self) => {
    const meta = tokens[idx].meta || {}
    if (meta.figureCaption) return `<figcaption class="figure-caption">图 ${meta.figureNum}　`
    return paragraphOpen ? paragraphOpen(tokens, idx, opts, env, self) : self.renderToken(tokens, idx, opts)
  }
  const paragraphClose = md.renderer.rules.paragraph_close
  md.renderer.rules.paragraph_close = (tokens, idx, opts, env, self) => {
    if ((tokens[idx].meta || {}).figureCaption) return '</figcaption>\n'
    return paragraphClose ? paragraphClose(tokens, idx, opts, env, self) : self.renderToken(tokens, idx, opts)
  }
}