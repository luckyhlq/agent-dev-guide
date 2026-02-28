import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: "Agent开发从入门到精通",
  description: "系统学习AI Agent开发，从基础概念到生产部署的完整教程",
  lang: 'zh-CN',
  
  head: [
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Agent开发教程',
    
    nav: [
      { text: '首页', link: '/' },
      { text: '基础入门', link: '/foundation/chapter1' },
      { text: '核心技能', link: '/core-skills/chapter4' },
      { text: '框架应用', link: '/frameworks/chapter7' },
      { text: '高级应用', link: '/advanced/chapter10' },
      { text: '前沿探索', link: '/frontier/chapter13' }
    ],

    sidebar: {
      '/foundation/': [
        {
          text: '第一阶段：基础入门',
          collapsible: true,
          items: [
            { text: '第1章：Agent开发概述', link: '/foundation/chapter1' },
            { text: '第2章：LLM基础与API使用', link: '/foundation/chapter2' },
            { text: '第3章：Agent核心概念', link: '/foundation/chapter3' }
          ]
        }
      ],
      '/core-skills/': [
        {
          text: '第二阶段：核心技能',
          collapsible: true,
          items: [
            { text: '第4章：高级Prompt技术', link: '/core-skills/chapter4' },
            { text: '第5章：Function Calling与工具使用', link: '/core-skills/chapter5' },
            { text: '第6章：记忆系统与向量数据库', link: '/core-skills/chapter6' }
          ]
        }
      ],
      '/frameworks/': [
        {
          text: '第三阶段：框架应用',
          collapsible: true,
          items: [
            { text: '第7章：LangChain框架', link: '/frameworks/chapter7' },
            { text: '第8章：CrewAI框架', link: '/frameworks/chapter8' },
            { text: '第9章：其他主流框架', link: '/frameworks/chapter9' }
          ]
        }
      ],
      '/advanced/': [
        {
          text: '第四阶段：高级应用',
          collapsible: true,
          items: [
            { text: '第10章：Agent架构设计', link: '/advanced/chapter10' },
            { text: '第11章：多Agent协作', link: '/advanced/chapter11' },
            { text: '第12章：生产部署', link: '/advanced/chapter12' }
          ]
        }
      ],
      '/frontier/': [
        {
          text: '第五阶段：前沿探索',
          collapsible: true,
          items: [
            { text: '第13章：Agent评估与优化', link: '/frontier/chapter13' },
            { text: '第14章：Agent安全与伦理', link: '/frontier/chapter14' },
            { text: '第15章：自主Agent与未来趋势', link: '/frontier/chapter15' },
            { text: '第16章：综合实战项目', link: '/frontier/chapter16' }
          ]
        }
      ],
      '/appendix/': [
        {
          text: '附录',
          collapsible: true,
          items: [
            { text: '附录A：常用工具与资源', link: '/appendix/appendix-a' },
            { text: '附录B：Prompt模板库', link: '/appendix/appendix-b' },
            { text: '附录C：代码仓库', link: '/appendix/appendix-c' },
            { text: '附录D：常见问题FAQ', link: '/appendix/appendix-d' },
            { text: '附录E：社区与交流', link: '/appendix/appendix-e' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-repo/agent-dev-guide' }
    ],

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2024-present Agent开发教程'
    },

    outline: {
      level: [2, 3]
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/your-repo/agent-dev-guide/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outlineTitle: '目录',
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  },

  markdown: {
    lineNumbers: true
  }
}))
