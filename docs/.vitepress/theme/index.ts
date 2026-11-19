import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import ImplementationTabs from '../../components/ImplementationTabs.vue'
import TileWorkDemo from '../../components/TileWorkDemo.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ImplementationTabs', ImplementationTabs)
    app.component('TileWorkDemo', TileWorkDemo)
  },
} satisfies Theme