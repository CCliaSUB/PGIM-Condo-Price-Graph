<template>
  <div class="chart">
    <div class="legend" aria-hidden="true">
      <span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" class="actual-line" /><circle cx="11" cy="5" r="3" class="actual-dot" /></svg>Observed</span>
      <span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" class="prediction-line" /><rect x="8" y="2" width="6" height="6" class="prediction-dot" /></svg>Predicted</span>
    </div>
    <div class="plot" @mouseleave="hover = null">
      <svg
        ref="svg" :viewBox="`0 0 ${W} ${H}`" tabindex="0" role="img" :aria-label="ariaLabel"
        @mousemove="onMove" @keydown.left.prevent="step(-1)" @keydown.right.prevent="step(1)" @blur="hover = null"
      >
        <g v-for="tick in yTicks" :key="`y-${tick}`">
          <line :x1="PL" :x2="W - PR" :y1="y(tick)" :y2="y(tick)" class="gridline" />
          <text :x="PL - 6" :y="y(tick) + 3" text-anchor="end" class="axis-label">{{ tick.toFixed(yDigits) }}</text>
        </g>
        <g v-for="tick in xTicks" :key="`x-${tick}`">
          <line :x1="x(tick)" :x2="x(tick)" :y1="H - PB" :y2="H - PB + 4" class="axis" />
          <text :x="x(tick)" :y="H - PB + 15" text-anchor="middle" class="axis-label">{{ label(tick) }}</text>
        </g>
        <line :x1="PL" :x2="W - PR" :y1="H - PB" :y2="H - PB" class="axis" />
        <text :x="4" :y="PT - 6" class="axis-label">{{ unit }}</text>

        <!-- Lines break wherever consecutive months are missing -->
        <polyline v-for="(seg, i) in segments('actual')" :key="`a-${i}`" :points="seg" class="actual-line" />
        <polyline v-for="(seg, i) in segments('prediction')" :key="`p-${i}`" :points="seg" class="prediction-line" />
        <g v-for="p in plotted" :key="p.date">
          <circle v-if="p.actual != null" :cx="x(p.m)" :cy="y(p.actual)" r="2.6" class="actual-dot" />
          <rect v-if="p.prediction != null" :x="x(p.m) - 2.4" :y="y(p.prediction) - 2.4" width="4.8" height="4.8" class="prediction-dot" />
        </g>

        <g v-if="hoverPoint">
          <line :x1="x(hoverPoint.m)" :x2="x(hoverPoint.m)" :y1="PT" :y2="H - PB" class="hover-line" />
        </g>
      </svg>
      <div v-if="hoverPoint" class="tooltip" :style="tooltipStyle" role="status">
        <b>{{ formatMonth(hoverPoint.date) }}</b> (target month)<br />
        Observed: {{ formatValue(hoverPoint.actual) }}<br />
        Predicted: {{ formatValue(hoverPoint.prediction) }}<br />
        Error: {{ formatPct(hoverPoint.pct_error) }}
      </div>
    </div>
    <p class="gap-note" v-if="gapCount">{{ points.length }} held-out months with an observed value; {{ gapCount }} gap<span v-if="gapCount > 1">s</span> where no transaction was observed (lines are broken, not interpolated).</p>
    <details class="table-alt">
      <summary>Show data table ({{ points.length }} rows)</summary>
      <table>
        <thead><tr><th scope="col">Target month</th><th scope="col">Observed</th><th scope="col">Predicted</th><th scope="col">Error</th></tr></thead>
        <tbody>
          <tr v-for="p in points" :key="`t-${p.date}`">
            <td>{{ formatMonth(p.date) }}</td><td>{{ formatValue(p.actual) }}</td><td>{{ formatValue(p.prediction) }}</td><td>{{ formatPct(p.pct_error) }}</td>
          </tr>
        </tbody>
      </table>
    </details>
  </div>
</template>

<script>
import { monthIndex, formatMonth, formatMonthIndex, formatValue, formatPct } from '../utils/format'

export default {
  name: 'PredictionChart',
  props: { points: { type: Array, required: true }, unit: { type: String, default: '' }, title: { type: String, default: '' } },
  data() { return { W: 340, H: 190, PL: 44, PR: 10, PT: 18, PB: 26, hover: null } },
  computed: {
    plotted() { return this.points.map(p => ({ ...p, m: monthIndex(p.date) })) },
    xDomain() {
      if (!this.plotted.length) return [0, 1]
      const ms = this.plotted.map(p => p.m)
      const lo = Math.min(...ms), hi = Math.max(...ms)
      return lo === hi ? [lo - 1, hi + 1] : [lo, hi]
    },
    yDomain() {
      // One shared domain for observed and predicted values.
      const values = this.plotted.flatMap(p => [p.actual, p.prediction]).filter(v => v != null)
      if (!values.length) return [0, 1]
      let lo = Math.min(...values), hi = Math.max(...values)
      const pad = hi === lo ? Math.max(Math.abs(hi) * 0.05, 0.1) : (hi - lo) * 0.08
      lo -= pad; hi += pad
      return [lo, hi]
    },
    yTicks() {
      const [lo, hi] = this.yDomain
      const raw = (hi - lo) / 4
      const mag = Math.pow(10, Math.floor(Math.log10(raw)))
      const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) || raw
      const ticks = []
      for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(6))
      return ticks
    },
    yDigits() { const span = this.yDomain[1] - this.yDomain[0]; return span >= 50 ? 0 : span < 0.5 ? 2 : 1 },
    xTicks() {
      const [lo, hi] = this.xDomain
      const span = hi - lo
      if (this.plotted.length === 1) return [this.plotted[0].m]
      const stepMonths = span > 48 ? 24 : span > 18 ? 12 : span > 6 ? 3 : 1
      const ticks = []
      const start = stepMonths >= 12 ? Math.ceil(lo / 12) * 12 : lo
      for (let t = start; t <= hi; t += stepMonths) ticks.push(t)
      return ticks.length ? ticks : [lo, hi]
    },
    gapCount() {
      let gaps = 0
      for (let i = 1; i < this.plotted.length; i++) if (this.plotted[i].m - this.plotted[i - 1].m > 1) gaps++
      return gaps
    },
    hoverPoint() { return this.hover == null ? null : this.plotted[this.hover] },
    tooltipStyle() {
      if (!this.hoverPoint) return {}
      const left = (this.x(this.hoverPoint.m) / this.W) * 100
      return left > 55 ? { right: `${100 - left + 2}%` } : { left: `${left + 2}%` }
    },
    ariaLabel() {
      if (!this.plotted.length) return 'No data'
      const first = this.plotted[0], last = this.plotted[this.plotted.length - 1]
      return `${this.title}: observed versus predicted rent for ${this.plotted.length} held-out months from ${formatMonth(first.date)} to ${formatMonth(last.date)}. Use left and right arrow keys to inspect points, or open the data table below.`
    },
  },
  watch: { points() { this.hover = null } },
  methods: {
    formatMonth, formatValue, formatPct,
    label(m) { return this.xTicks.length > 1 && this.xTicks[1] - this.xTicks[0] >= 12 ? String(Math.floor(m / 12)) : formatMonthIndex(m) },
    x(m) { const [lo, hi] = this.xDomain; return this.PL + ((m - lo) / (hi - lo)) * (this.W - this.PL - this.PR) },
    y(v) { const [lo, hi] = this.yDomain; return this.H - this.PB - ((v - lo) / (hi - lo)) * (this.H - this.PT - this.PB) },
    segments(field) {
      const out = []
      let current = []
      let prev = null
      for (const p of this.plotted) {
        const value = p[field]
        if (value == null || (prev != null && p.m - prev > 1)) {
          if (current.length > 1) out.push(current.join(' '))
          current = []
        }
        if (value != null) current.push(`${this.x(p.m).toFixed(1)},${this.y(value).toFixed(1)}`)
        prev = p.m
      }
      if (current.length > 1) out.push(current.join(' '))
      return out
    },
    onMove(event) {
      if (!this.plotted.length) return
      const rect = this.$refs.svg.getBoundingClientRect()
      const px = ((event.clientX - rect.left) / rect.width) * this.W
      let best = 0
      this.plotted.forEach((p, i) => { if (Math.abs(this.x(p.m) - px) < Math.abs(this.x(this.plotted[best].m) - px)) best = i })
      this.hover = best
    },
    step(delta) {
      if (!this.plotted.length) return
      const current = this.hover == null ? (delta > 0 ? -1 : this.plotted.length) : this.hover
      this.hover = Math.min(this.plotted.length - 1, Math.max(0, current + delta))
    },
  },
}
</script>

<style scoped>
.chart { margin: 14px 0; }
.legend { display: flex; gap: 14px; justify-content: flex-end; font-size: 11px; color: #3d4a47; }
.legend span { display: inline-flex; align-items: center; gap: 4px; }
.plot { position: relative; }
.plot svg { width: 100%; height: auto; display: block; overflow: visible; }
.plot svg:focus-visible { outline: 3px solid #2567b3; outline-offset: 2px; }
.gridline { stroke: #e3e7e2; stroke-width: 1; }
.axis { stroke: #8c9592; stroke-width: 1; }
.axis-label { font-size: 9px; fill: #56625f; }
.actual-line, .prediction-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.actual-line { stroke: #183d39; }
.prediction-line { stroke: #d9432b; stroke-dasharray: 5 3; }
.actual-dot { fill: #183d39; }
.prediction-dot { fill: #fff; stroke: #d9432b; stroke-width: 1.5; }
.hover-line { stroke: #2567b3; stroke-width: 1; stroke-dasharray: 2 2; }
.tooltip { position: absolute; top: 8px; padding: 6px 8px; background: rgba(20,43,43,.94); color: #fff; font-size: 11px; line-height: 1.45; border-radius: 4px; pointer-events: none; white-space: nowrap; }
.gap-note { margin: 4px 0 0; font-size: 11px; color: #56625f; line-height: 1.4; }
.table-alt { margin-top: 8px; font-size: 11px; }
.table-alt summary { cursor: pointer; color: #1f5a99; }
.table-alt table { width: 100%; margin-top: 6px; border-collapse: collapse; }
.table-alt th, .table-alt td { padding: 3px 4px; border-bottom: 1px solid #eceeea; text-align: right; }
.table-alt th:first-child, .table-alt td:first-child { text-align: left; }
</style>
