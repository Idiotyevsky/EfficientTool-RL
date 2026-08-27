import { defineConfig } from 'vitepress'

function normalizeBase(value: string): string {
  const trimmed = value.trim()
  if (!trimmed || trimmed === '/') return '/'
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}/`
}

function resolveBase(): string {
  if (process.env.VITEPRESS_BASE) return normalizeBase(process.env.VITEPRESS_BASE)
  if (process.env.GITHUB_ACTIONS === 'true' && process.env.GITHUB_REPOSITORY) {
    const repository = process.env.GITHUB_REPOSITORY.split('/')[1]
    return repository ? normalizeBase(repository) : '/'
  }
  return '/'
}

const base = resolveBase()

export default defineConfig({
  lang: 'zh-CN',
  title: 'MiniAgentRL',
  description: '从 Tool Calling 到真正 GRPO 参数更新的 Agentic RL 实战课程。',
  base,
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}logo.svg` }],
    ['meta', { name: 'theme-color', content: '#5b5ce2' }],
  ],
  markdown: {
    lineNumbers: true,
    math: true,
  },
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'MiniAgentRL',
    nav: [
      { text: 'Learn', link: '/learn/' },
      { text: 'Trajectories', link: '/playground/trajectories' },
      { text: 'Research', link: '/research/' },
      { text: 'GitHub', link: 'https://github.com/Idiotyevsky/EfficientTool-RL' },
    ],
    sidebar: {
      '/learn/': [
        {
          text: 'Learn Track',
          items: [
            { text: '课程地图', link: '/learn/' },
            { text: '00 · Start', link: '/learn/00-start' },
            { text: '01 · Tool Calling', link: '/learn/01-tool-calling' },
            { text: '02 · Real Qwen', link: '/learn/02-real-qwen' },
            { text: '03 · Multi-turn', link: '/learn/03-multiturn' },
            { text: '04 · ReAct + HotpotQA', link: '/learn/04-react-hotpot' },
            { text: '05 · Rollout & Reward', link: '/learn/05-rollout-reward' },
            { text: '06 · GRPO', link: '/learn/06-grpo' },
            { text: '07 · Real Update', link: '/learn/07-grpo-smoke' },
            { text: '08 · Efficient Tools', link: '/learn/08-efficient-tools' },
          ],
        },
      ],
      '/playground/': [
        {
          text: 'Playground',
          items: [{ text: 'Trajectory Explorer', link: '/playground/trajectories' }],
        },
      ],
      '/research/': [
        {
          text: 'Research Track',
          items: [
            { text: '研究总览', link: '/research/' },
            { text: '环境与复现', link: '/research/environment' },
            { text: '实验与状态', link: '/research/experiments' },
            { text: '指标定义', link: '/research/metrics' },
          ],
        },
      ],
    },
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Idiotyevsky/EfficientTool-RL' },
    ],
    editLink: {
      pattern: 'https://github.com/Idiotyevsky/EfficientTool-RL/edit/main/website/:path',
      text: '在 GitHub 上编辑此页',
    },
    outline: { level: [2, 3], label: '本页内容' },
    docFooter: { prev: '上一课', next: '下一课' },
    lastUpdated: { text: '最后更新' },
    footer: {
      message: 'Real implementation, bounded lessons, evidence-first research.',
      copyright: 'MiniAgentRL · EfficientTool-RL research project',
    },
  },
})
