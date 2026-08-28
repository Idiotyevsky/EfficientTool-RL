import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './custom.css'
import './landing.css'
import './homepage.css'

import AgentLoopDemo from '../../components/AgentLoopDemo.vue'
import BilingualLanding from '../../components/BilingualLanding.vue'
import ConceptFlow from '../../components/ConceptFlow.vue'
import CourseMap from '../../components/CourseMap.vue'
import GRPOGroupDemo from '../../components/GRPOGroupDemo.vue'
import LearningCheckpoint from '../../components/LearningCheckpoint.vue'
import LessonCard from '../../components/LessonCard.vue'
import MetricPill from '../../components/MetricPill.vue'
import TrajectoryExplorer from '../../components/TrajectoryExplorer.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('AgentLoopDemo', AgentLoopDemo)
    app.component('BilingualLanding', BilingualLanding)
    app.component('ConceptFlow', ConceptFlow)
    app.component('CourseMap', CourseMap)
    app.component('GRPOGroupDemo', GRPOGroupDemo)
    app.component('LearningCheckpoint', LearningCheckpoint)
    app.component('LessonCard', LessonCard)
    app.component('MetricPill', MetricPill)
    app.component('TrajectoryExplorer', TrajectoryExplorer)
  },
} satisfies Theme
