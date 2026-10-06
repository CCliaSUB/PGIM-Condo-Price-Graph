<template>
  <figure class="inset">
    <figcaption>
      <strong>Project-detail graph</strong>
      <span>Schematic, not to scale<template v-if="month">; graph snapshot for {{ formatMonth(month) }}</template>. Room-size nodes share the project's location, so they are drawn here instead of on the map.</span>
    </figcaption>
    <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="ariaLabel">
      <!-- size_project edges -->
      <g v-for="node in sizeLayout" :key="`se-${node.node_id}`">
        <line :x1="CX" :y1="CY" :x2="node.x" :y2="node.y" class="size-edge" :class="{ inactive: node.size_project_active === false }" />
      </g>
      <!-- project-to-project relation edges (one bundle per relation) -->
      <g v-for="hub in relationLayout" :key="`re-${hub.id}`" :class="{ faded: !hub.shown }">
        <line :x1="CX" :y1="CY" :x2="hub.x" :y2="hub.y" :stroke="hub.color" :stroke-dasharray="hub.dash" class="relation-edge" />
      </g>

      <!-- room-size nodes (squares) -->
      <g v-for="node in sizeLayout" :key="`sn-${node.node_id}`" :class="{ 'no-pred': !node.has_predictions, active: node.size_tier === activeTier }"
         class="size-node" @click="$emit('select-tier', node.size_tier)">
        <rect :x="node.x - 13" :y="node.y - 13" width="26" height="26" rx="4" />
        <text :x="node.x" :y="node.y + 4" text-anchor="middle">SZ{{ node.size_tier }}</text>
        <g v-if="node.similar_price_neighbours != null">
          <circle :cx="node.x - 15" :cy="node.y - 13" r="8" :fill="priceColor" class="badge" />
          <text :x="node.x - 15" :y="node.y - 10" text-anchor="middle" class="badge-text">{{ node.similar_price_neighbours }}</text>
        </g>
        <title>{{ node.label }} · graph node {{ node.node_id }}{{ node.has_predictions ? '' : ' · no held-out predictions' }}{{ node.size_project_active === false ? ' · size_project edge not active this month (project not yet built)' : '' }}{{ node.similar_price_neighbours != null ? ` · ${node.similar_price_neighbours} similar-price neighbours` : '' }}</title>
      </g>

      <!-- relation bundles (neighbour counts) -->
      <g v-for="hub in relationLayout" :key="`rh-${hub.id}`" :class="{ faded: !hub.shown }">
        <circle :cx="hub.x" :cy="hub.y" r="13" :fill="hub.color" class="hub" />
        <text :x="hub.x" :y="hub.y + 4" text-anchor="middle" class="hub-count">{{ hub.total > 99 ? '99+' : hub.total }}</text>
        <text :x="hub.x + 18" :y="hub.y - 1" class="hub-label">{{ hub.label }}</text>
        <text :x="hub.x + 18" :y="hub.y + 11" class="hub-sub">{{ hub.shown }} of {{ hub.total }} projects on map{{ hub.level === 'size' ? ' (via size nodes)' : '' }}</text>
        <title>{{ hub.label }}: {{ hub.total }} unique neighbouring projects, {{ hub.shown }} nearest drawn on the map</title>
      </g>

      <!-- selected project node -->
      <circle :cx="CX" :cy="CY" r="17" class="project-node" />
      <text :x="CX" :y="CY + 4" text-anchor="middle" class="project-label">P</text>
      <text :x="CX" :y="CY + 32" text-anchor="middle" class="project-name">node {{ selected.node_id }}</text>
    </svg>
    <p v-if="!relationLayout.length" class="hint">No project-to-project relations selected — only <code>size_project</code> edges remain.</p>
    <ul class="legend">
      <li><i class="lg-project"></i>Selected project node</li>
      <li><i class="lg-size"></i>Room-size node (hatched: no predictions)</li>
      <li><i class="lg-hub"></i>Neighbour bundle (count = unique projects)</li>
      <li><i class="lg-size-edge"></i><code>size_project</code> edge</li>
      <li v-if="hasPriceBadges"><i class="lg-badge" :style="{ background: priceColor }"></i>Similar-price neighbours of that room-size node</li>
      <li><i class="lg-rel-edge"></i>Project-to-project edge (styled per relation)</li>
    </ul>
  </figure>
</template>

<script>
import { formatMonth } from '../utils/format'

export default {
  name: 'GraphInset',
  props: {
    selected: { type: Object, required: true },
    sizeNodes: { type: Array, default: () => [] },
    relationSummary: { type: Array, default: () => [] },
    relations: { type: Array, default: () => [] },
    activeTier: { type: Number, default: null },
    month: { type: String, default: null },
  },
  data() { return { W: 380, H: 250, CX: 120, CY: 125 } },
  methods: { formatMonth },
  computed: {
    priceColor() { return (this.relationMeta.similar_price || {}).color || '#c0364a' },
    hasPriceBadges() { return this.sizeNodes.some(n => n.similar_price_neighbours != null) },
    relationMeta() { return Object.fromEntries(this.relations.map(r => [r.id, r])) },
    sizeLayout() {
      // Left-hand arc, evenly spaced
      const n = this.sizeNodes.length
      return this.sizeNodes.map((node, i) => {
        const angle = Math.PI * (n === 1 ? 1 : 0.62 + (0.76 * i) / (n - 1))
        return { ...node, x: this.CX + 82 * Math.cos(angle), y: this.CY - 82 * Math.sin(angle) }
      })
    },
    relationLayout() {
      const n = this.relationSummary.length
      return this.relationSummary.map((item, i) => {
        const meta = this.relationMeta[item.id] || {}
        const y = n === 1 ? this.CY : 22 + (i * (this.H - 44)) / (n - 1)
        return { id: item.id, level: item.level, label: meta.short || meta.label || item.id, color: meta.color, dash: meta.dash, total: item.neighbours_total, shown: item.neighbours_shown, x: 196, y }
      })
    },
    ariaLabel() {
      const sizes = this.sizeNodes.map(n => `SZ${n.size_tier}`).join(', ') || 'none'
      const rels = this.relationLayout.map(h => `${h.label} ${h.total}`).join('; ') || 'none'
      return `Selected project with room-size nodes ${sizes} via size_project edges; relation neighbour counts: ${rels}`
    },
  },
}
</script>

<style scoped>
.inset { margin: 0; }
figcaption { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; font-size: 12px; }
figcaption span { color: #56625f; font-size: 11px; line-height: 1.4; }
svg { width: 100%; height: auto; display: block; background: #f6f7f3; border: 1px solid #e0e4df; border-radius: 6px; }
.size-edge { stroke: #56625f; stroke-width: 2; }
.size-edge.inactive { stroke: #b9c1bc; stroke-dasharray: 2 3; }
.badge { stroke: #fff; stroke-width: 1.5; }
.badge-text { font-size: 8px; font-weight: 700; fill: #fff; }
.relation-edge { stroke-width: 2.5; }
.size-node { cursor: pointer; }
.size-node rect { fill: #fff; stroke: #182a3a; stroke-width: 2; }
.size-node text { font-size: 10px; font-weight: 700; fill: #182a3a; }
.size-node.no-pred rect { stroke-dasharray: 3 2; fill: #eceeea; }
.size-node.active rect { fill: #183d39; }
.size-node.active text { fill: #fff; }
.hub { stroke: #fff; stroke-width: 2; }
.hub-count { font-size: 9px; font-weight: 700; fill: #fff; }
.hub-label { font-size: 10px; fill: #182a3a; }
.hub-sub { font-size: 9px; fill: #56625f; }
.faded { opacity: .35; }
.project-node { fill: #d9432b; stroke: #fff; stroke-width: 3; }
.project-label { font-size: 12px; font-weight: 800; fill: #fff; }
.project-name { font-size: 9px; fill: #56625f; }
.hint { margin: 6px 0 0; font-size: 11px; color: #56625f; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 6px 0 0; padding: 0; list-style: none; font-size: 11px; color: #3d4a47; }
.legend i { display: inline-block; width: 11px; height: 11px; margin-right: 5px; vertical-align: -1px; }
.lg-project { background: #d9432b; border-radius: 50%; }
.lg-size { border: 2px solid #182a3a; border-radius: 2px; background: #fff; }
.lg-hub { background: #7b4fa8; border-radius: 50%; }
.lg-badge { border-radius: 50%; }
.lg-size-edge, .lg-rel-edge { height: 0 !important; width: 16px !important; border-top: 2px solid #56625f; vertical-align: 3px !important; }
.lg-rel-edge { border-top: 2px dashed #2567b3; }
</style>
