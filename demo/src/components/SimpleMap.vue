<template>
  <main class="demo-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">PROPERTY GRAPH INTELLIGENCE MODEL</p>
        <h1>Singapore property graph</h1>
      </div>
      <div class="switchers" v-if="datasets.length">
        <div class="segmented" role="group" aria-label="Prediction target">
          <button v-for="t in targetOptions" :key="t.id" type="button" :class="{ active: task === t.id }" :aria-pressed="String(task === t.id)"
                  :disabled="!comboAvailable(t.id, coverage)" :title="comboTitle(t.id, coverage)" @click="setDataset(t.id, coverage)">{{ t.label }}</button>
        </div>
        <div class="segmented" role="group" aria-label="Geographic coverage">
          <button v-for="c in coverageOptions" :key="c.id" type="button" :class="{ active: coverage === c.id }" :aria-pressed="String(coverage === c.id)"
                  :disabled="!comboAvailable(task, c.id)" :title="comboTitle(task, c.id)" @click="setDataset(task, c.id)">{{ c.label }}</button>
        </div>
      </div>
      <div class="dataset-pill" :class="{ bad: metaError }">
        <span aria-hidden="true"></span>{{ metaError ? 'API unavailable' : (metadata.dataset_label || 'Loading dataset…') }}
      </div>
    </header>

    <div v-if="metaError" class="banner error-state" role="alert">{{ metaError }}</div>

    <section class="workspace">
      <aside class="controls panel" aria-label="Project and graph controls">
        <div class="section-heading">
          <span class="step">01</span>
          <div><h2>Select a condominium</h2><p>Search, or click a marker on the map. Only existing project nodes can be selected.</p></div>
        </div>
        <ProjectSearch :projects="projects" :disabled="!projects.length" @select="selectProject" />

        <div v-if="selectedProject" class="project-card">
          <p class="label">Selected project · graph node {{ graph ? graph.selected.node_id : '…' }}</p>
          <h3>{{ selectedProject.name }}</h3>
          <div class="facts">
            <span>{{ selectedProject.planning_area || 'Planning area unknown' }}</span>
            <span v-if="selectedProject.age_at_transaction" title="Condo age at the time of observed transactions">
              Age at transaction {{ ageText(selectedProject.age_at_transaction) }} yrs
            </span>
            <span v-if="!selectedProject.has_predictions">No exported predictions</span>
          </div>
        </div>

        <div class="section-heading compact">
          <span class="step">02</span>
          <div><h2>Graph relations</h2><p>{{ metadata.graph_mode === 'snapshot' ? 'Relations the model used. They change month to month; counts are unique neighbouring projects in the selected month.' : 'Project-to-project edge types. Counts are unique neighbouring projects.' }}</p></div>
        </div>
        <label v-if="metadata.graph_mode === 'snapshot' && metadata.months.length" class="month-picker">
          <span>Graph snapshot month</span>
          <select v-model="month">
            <option v-for="m in monthsDesc" :key="m" :value="m">{{ formatMonth(m) }}</option>
          </select>
        </label>
        <div class="relation-actions">
          <button type="button" @click="setAllRelations(true)">All</button>
          <button type="button" @click="setAllRelations(false)">None</button>
        </div>
        <fieldset class="relations">
          <legend class="visually-hidden">Visible relation types</legend>
          <label v-for="relation in metadata.relations" :key="relation.id" class="relation-row">
            <input v-model="activeRelations" type="checkbox" :value="relation.id" />
            <svg width="26" height="10" aria-hidden="true"><line x1="0" y1="5" x2="26" y2="5" :stroke="relation.color" stroke-width="3" :stroke-dasharray="relation.dash" /></svg>
            <span :title="relation.description">{{ relation.label }}</span>
            <b :title="relationTitle(relation.id)">{{ relationCount(relation.id) }}</b>
          </label>
        </fieldset>

        <div v-if="selectedProject" class="inset-wrap">
          <div v-if="graphError" class="error-state" role="alert">Graph: {{ graphError }}</div>
          <GraphInset
            v-else-if="graph" :selected="graph.selected" :size-nodes="graph.size_nodes"
            :relation-summary="graph.relation_summary" :relations="metadata.relations" :active-tier="selectedTier" :month="graph.month"
            @select-tier="selectedTier = $event"
          />
          <p v-else class="muted">Loading graph…</p>
        </div>
        <p v-else class="note">After selecting a project, a schematic shows its room-size nodes attached through <code>size_project</code> edges and the neighbour counts per relation.</p>
      </aside>

      <div class="map-panel panel">
        <l-map ref="map" :zoom="zoom" :center="center" :options="mapOptions" @ready="onMapReady">
          <l-tile-layer :url="tileUrl" :attribution="attribution" />
          <l-circle-marker
            v-for="project in backgroundProjects" :key="`bg-${project.project_id}`"
            :lat-lng="[project.latitude, project.longitude]" :radius="3" :weight="1"
            color="#5f6e6a" fill-color="#8c9a96" :fill-opacity="selectedProject ? 0.35 : 0.7"
            @click="selectProject(project)"
          ><l-tooltip>{{ project.name }}</l-tooltip></l-circle-marker>
          <l-polyline
            v-for="edge in mapEdges" :key="edge.key" :lat-lngs="edge.path"
            :color="edge.color" :dash-array="edge.dash" :weight="2.5" :opacity="0.85"
          ><l-tooltip>{{ edge.label }}: {{ edge.targetName }}<template v-if="edge.pairsText"><br />{{ edge.pairsText }}</template></l-tooltip></l-polyline>
          <l-circle-marker
            v-for="node in neighbourMarkers" :key="`nb-${node.project_id}`"
            :lat-lng="[node.latitude, node.longitude]" :radius="6" :weight="2"
            color="#ffffff" fill-color="#182a3a" :fill-opacity="0.95"
            @click="selectProject(projectById[node.project_id] || node)"
          ><l-tooltip>{{ node.name }} — {{ node.relations.map(relationLabel).join(', ') }}</l-tooltip></l-circle-marker>
          <l-circle-marker
            v-if="selectedHasLocation" :key="`sel-${selectedProject.project_id}`"
            :lat-lng="[selectedProject.latitude, selectedProject.longitude]" :radius="10" :weight="3"
            color="#ffffff" fill-color="#d9432b" :fill-opacity="1"
          ><l-tooltip :options="{ permanent: true, direction: 'top', offset: [0, -10] }">{{ selectedProject.name }}</l-tooltip></l-circle-marker>
        </l-map>
        <div class="map-key">
          <span><i class="k-sel"></i>Selected</span>
          <span><i class="k-nb"></i>Neighbour ({{ neighbourNodes.length }})</span>
          <span><i class="k-bg"></i>Other project ({{ backgroundProjects.length }})</span>
          <span>{{ mapEdges.length }} edges drawn<template v-if="truncated"> · nearest {{ graph.limit_per_relation }} per relation</template></span>
          <span v-if="unmappedCount">{{ unmappedCount }} project<template v-if="unmappedCount > 1">s</template> without coordinates (searchable, not drawn)</span>
        </div>
        <div v-if="loadingGraph" class="map-loading" role="status">Building neighbourhood…</div>
        <div v-else-if="selectedProject && !selectedHasLocation" class="map-loading" role="status">{{ selectedProject.name }} has no coordinates in this dataset, so it cannot be placed on the map.</div>
      </div>

      <aside class="predictions panel" aria-label="Price predictions">
        <div class="section-heading">
          <span class="step">03</span>
          <div><h2>Held-out predictions</h2><p>Observed vs PGIM prediction for each room-size node.</p></div>
        </div>
        <p class="eval-note">
          These are <b>test-set evaluation predictions</b> for months that already have an observed {{ metadata.task === 'sale' ? 'sale price' : 'rent' }}
          ({{ dateRangeText }}), not forecasts beyond the dataset.
        </p>
        <div class="horizons" role="group" aria-label="Forecast horizon">
          <button v-for="h in metadata.horizons" :key="h" type="button" :class="{ active: horizon === h }" :aria-pressed="String(horizon === h)" @click="setHorizon(h)">
            {{ h }} month<span v-if="h > 1">s</span> ahead
          </button>
        </div>

        <div v-if="!selectedProject" class="empty-state"><div aria-hidden="true">⌖</div><p>Select a condominium to inspect its graph and predictions.</p></div>
        <div v-else-if="predictionError" class="error-state" role="alert">Predictions: {{ predictionError }}</div>
        <div v-else-if="!predictions" class="empty-state" role="status"><div class="spinner"></div><p>Loading PGIM predictions…</p></div>
        <template v-else>
          <div :class="{ refreshing: loadingPredictions }">
            <div v-if="!predictions.has_predictions" class="info-state">
              This project is in the graph, but none of its room-size nodes have exported held-out rows for the {{ predictions.horizon }}-month artifact.
            </div>
            <template v-else>
              <div class="tier-tabs" role="tablist" aria-label="Room-size tier">
                <button
                  v-for="item in predictions.series" :key="item.size_tier" type="button" role="tab"
                  :aria-selected="String(activeSeries && activeSeries.size_tier === item.size_tier)"
                  :class="{ active: activeSeries && activeSeries.size_tier === item.size_tier, empty: !item.n_points }"
                  :title="item.label" @click="selectedTier = item.size_tier"
                >SZ{{ item.size_tier }}</button>
              </div>
              <template v-if="activeSeries">
                <p class="tier-label">{{ activeSeries.label }} · graph node {{ activeSeries.node_id }}</p>
                <div v-if="!activeSeries.n_points" class="info-state">No held-out rows for this room-size node at the {{ predictions.horizon }}-month horizon.</div>
                <template v-else>
                  <div class="latest-card">
                    <div>
                      <p>Last evaluated month · {{ formatMonth(last.date) }}</p>
                      <strong>{{ formatValue(last.prediction) }}</strong><small>predicted</small>
                      <p class="issued">Predicted {{ predictions.horizon }} month<span v-if="predictions.horizon > 1">s</span> ahead, from data up to {{ shiftMonths(last.date, -predictions.horizon) }}</p>
                    </div>
                    <div class="actual">
                      <p>Observed</p><strong>{{ formatValue(last.actual) }}</strong>
                      <small>{{ last.pct_error == null ? 'error n/a' : `${formatPct(last.pct_error)} error` }}</small>
                    </div>
                  </div>
                  <PredictionChart :points="activeSeries.points" :unit="predictions.unit" :title="activeSeries.label" />
                </template>
              </template>
              <table class="tier-summary">
                <caption>Last evaluated month per room-size node ({{ predictions.unit }})</caption>
                <thead><tr><th scope="col">Tier</th><th scope="col">Month</th><th scope="col">Pred.</th><th scope="col">Obs.</th><th scope="col">Error</th></tr></thead>
                <tbody>
                  <tr v-for="item in predictions.series" :key="`s-${item.size_tier}`" :class="{ current: activeSeries && item.size_tier === activeSeries.size_tier }" @click="selectedTier = item.size_tier">
                    <th scope="row"><button type="button" class="link" @click.stop="selectedTier = item.size_tier">SZ{{ item.size_tier }}</button></th>
                    <template v-if="item.last_evaluated">
                      <td>{{ formatMonth(item.last_evaluated.date) }}</td>
                      <td>{{ formatValue(item.last_evaluated.prediction) }}</td>
                      <td>{{ formatValue(item.last_evaluated.actual) }}</td>
                      <td>{{ formatPct(item.last_evaluated.pct_error) }}</td>
                    </template>
                    <td v-else colspan="4" class="muted">no held-out rows</td>
                  </tr>
                </tbody>
              </table>
            </template>
            <p class="disclaimer" v-if="predictions.provenance">
              Source: <code>{{ predictions.provenance.artifact }}</code> · seed {{ predictions.provenance.seed }} ·
              trial <code>{{ predictions.provenance.trial }}</code> · horizon {{ predictions.provenance.horizon_months }} mo ·
              dataset {{ predictions.provenance.dataset }}. {{ predictions.provenance.selection }}
              <template v-if="predictions.provenance.test_metrics">Whole test set at this horizon: R² {{ predictions.provenance.test_metrics.r2 }}, MAPE {{ predictions.provenance.test_metrics.mape_pct }}%, RMSE {{ predictions.provenance.test_metrics.rmse }}.</template>
              {{ coverageText }} Model estimates, not market quotations.
            </p>
          </div>
        </template>
      </aside>
    </section>
  </main>
</template>

<script>
import axios from 'axios'
import { LMap, LTileLayer, LCircleMarker, LPolyline, LTooltip } from 'vue2-leaflet'
import ProjectSearch from './ProjectSearch.vue'
import GraphInset from './GraphInset.vue'
import PredictionChart from './PredictionChart.vue'
import { formatMonth, formatValue, formatPct, shiftMonths, apiError, isCancel } from '../utils/format'

// Quadratic curve between two points; `bend` offsets parallel edges so they do not overplot.
function curve(a, b, bend) {
  if (!bend) return [a, b]
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
  const ctrl = [mid[0] - (b[1] - a[1]) * bend, mid[1] + (b[0] - a[0]) * bend]
  const pts = []
  for (let i = 0; i <= 16; i++) {
    const t = i / 16, u = 1 - t
    pts.push([u * u * a[0] + 2 * u * t * ctrl[0] + t * t * b[0], u * u * a[1] + 2 * u * t * ctrl[1] + t * t * b[1]])
  }
  return pts
}

export default {
  name: 'SimpleMap',
  components: { LMap, LTileLayer, LCircleMarker, LPolyline, LTooltip, ProjectSearch, GraphInset, PredictionChart },
  data() {
    return {
      center: [1.3521, 103.8198], zoom: 12, mapReady: false,
      mapOptions: { zoomControl: true, preferCanvas: true },
      tileUrl: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; OpenStreetMap contributors',
      metadata: { relations: [], horizons: [], months: [] }, metaError: '',
      datasets: [], datasetId: null, month: null,
      targetOptions: [{ id: 'rent', label: 'Rental' }, { id: 'sale', label: 'Sale' }],
      coverageOptions: [{ id: 'full', label: 'Full Singapore' }, { id: 'ccr', label: 'CCR' }],
      projects: [], selectedProject: null, activeRelations: [],
      graph: null, graphError: '', loadingGraph: false,
      horizon: 12, predictions: null, predictionError: '', loadingPredictions: false, selectedTier: null,
    }
  },
  computed: {
    currentDataset() { return this.datasets.find(d => d.id === this.datasetId) || null },
    task() { return this.currentDataset ? this.currentDataset.task : 'rent' },
    coverage() { return this.currentDataset ? this.currentDataset.coverage : 'full' },
    monthsDesc() { return [...(this.metadata.months || [])].reverse() },
    projectById() { return Object.fromEntries(this.projects.map(p => [p.project_id, p])) },
    relationMeta() { return Object.fromEntries(this.metadata.relations.map(r => [r.id, r])) },
    neighbourNodes() { return this.graph ? this.graph.nodes : [] },
    highlighted() {
      const ids = new Set(this.neighbourNodes.map(n => n.project_id))
      if (this.selectedProject) ids.add(this.selectedProject.project_id)
      return ids
    },
    mappable() { return this.projects.filter(p => p.latitude != null && p.longitude != null) },
    unmappedCount() { return this.projects.length - this.mappable.length },
    backgroundProjects() { return this.mappable.filter(p => !this.highlighted.has(p.project_id)) },
    neighbourMarkers() { return this.neighbourNodes.filter(n => n.latitude != null && n.longitude != null) },
    selectedHasLocation() { return !!this.selectedProject && this.selectedProject.latitude != null },
    mapEdges() {
      if (!this.graph || this.graph.selected.latitude == null) return []
      const sel = this.graph.selected
      const nodeMap = Object.fromEntries(this.graph.nodes.map(n => [n.project_id, n]))
      const byTarget = {}
      this.graph.edges.forEach(edge => { (byTarget[edge.target] = byTarget[edge.target] || []).push(edge) })
      const out = []
      Object.values(byTarget).forEach(group => {
        group.forEach((edge, i) => {
          const target = nodeMap[edge.target]
          if (target.latitude == null) return
          const meta = this.relationMeta[edge.relation] || {}
          const bend = group.length === 1 ? 0 : (i - (group.length - 1) / 2) * 0.22
          out.push({
            key: `${edge.relation}-${edge.target}`, color: meta.color, dash: meta.dash || null, label: meta.label || edge.relation,
            targetName: target.name,
            pairsText: edge.pairs ? edge.pairs.map(p => `SZ${p.from_tier} ↔ SZ${p.to_tier}`).join(', ') : '',
            path: curve([sel.latitude, sel.longitude], [target.latitude, target.longitude], bend),
          })
        })
      })
      return out
    },
    truncated() { return this.graph && this.graph.relation_summary.some(r => r.neighbours_shown < r.neighbours_total) },
    activeSeries() {
      if (!this.predictions) return null
      return this.predictions.series.find(item => item.size_tier === this.selectedTier) || this.predictions.series[0] || null
    },
    last() { return this.activeSeries && this.activeSeries.last_evaluated },
    dateRangeText() {
      const cov = this.metadata.coverage_check && this.metadata.coverage_check.per_horizon[String(this.horizon)]
      return cov && cov.date_range ? `${formatMonth(cov.date_range[0])} – ${formatMonth(cov.date_range[1])}` : 'test period'
    },
    coverageText() {
      const cov = this.metadata.coverage_check
      if (!cov) return ''
      return cov.identical_cells
        ? 'All horizon artifacts cover identical node/month cells, so horizons are directly comparable.'
        : `Horizon artifacts cover different cells (${cov.shared_cells} of ${cov.union_cells} shared); compare with care.`
    },
  },
  watch: {
    activeRelations() { this.loadGraph() },
    month() { this.loadGraph() },
  },
  created() {
    this.controllers = {}
  },
  async mounted() {
    try {
      const response = await axios.get('/api/datasets')
      this.datasets = response.data.datasets
      await this.loadDataset(response.data.default)
    } catch (error) { this.metaError = apiError(error) }
  },
  beforeDestroy() {
    Object.values(this.controllers).forEach(c => c.abort())
    if (this.resizeObserver) this.resizeObserver.disconnect()
  },
  methods: {
    formatMonth, formatValue, formatPct, shiftMonths,
    // Abort the previous request of the same kind; only the newest response may update state.
    beginRequest(kind) {
      if (this.controllers[kind]) this.controllers[kind].abort()
      const controller = new AbortController()
      this.controllers[kind] = controller
      return controller
    },
    isCurrent(kind, controller) { return this.controllers[kind] === controller },
    onMapReady(map) {
      this.mapReady = true
      // Leaflet caches its container size; re-measure when the responsive layout changes it.
      if (window.ResizeObserver) {
        this.resizeObserver = new ResizeObserver(() => map.invalidateSize())
        this.resizeObserver.observe(map.getContainer())
      }
    },
    comboFor(task, coverage) { return this.datasets.find(d => d.task === task && d.coverage === coverage) },
    comboAvailable(task, coverage) { const d = this.comboFor(task, coverage); return !!(d && d.available) },
    comboTitle(task, coverage) {
      const d = this.comboFor(task, coverage)
      if (!d) return 'Not configured'
      return d.available ? d.label : `${d.label} — missing: ${d.missing.join(', ')}`
    },
    setDataset(task, coverage) {
      const d = this.comboFor(task, coverage)
      if (d && d.available && d.id !== this.datasetId) this.loadDataset(d.id)
    },
    // Project IDs differ between datasets, so everything is reset; the same condo is re-selected by name when present.
    async loadDataset(id) {
      const controller = this.beginRequest('dataset')
      ;['graph', 'predictions'].forEach(kind => { if (this.controllers[kind]) this.controllers[kind].abort(); delete this.controllers[kind] })
      const previousName = this.selectedProject && this.selectedProject.name
      this.datasetId = id; this.metaError = ''
      this.selectedProject = null; this.graph = null; this.predictions = null; this.graphError = ''; this.predictionError = ''
      this.loadingGraph = false; this.loadingPredictions = false
      try {
        const params = { dataset: id }
        const [metadata, projects] = await Promise.all([
          axios.get('/api/metadata', { params, signal: controller.signal }),
          axios.get('/api/projects', { params, signal: controller.signal }),
        ])
        if (!this.isCurrent('dataset', controller)) return
        this.metadata = metadata.data
        this.projects = projects.data.projects
        this.month = this.metadata.months.length ? this.metadata.months[this.metadata.months.length - 1] : null
        this.activeRelations = this.metadata.relations.map(item => item.id)
        const match = previousName && this.projects.find(p => p.name === previousName)
        if (match) this.$nextTick(() => this.selectProject(match))
      } catch (error) {
        if (!isCancel(error) && this.isCurrent('dataset', controller)) { this.projects = []; this.metaError = apiError(error) }
      }
    },
    selectProject(project) {
      if (!project) return
      const changed = !this.selectedProject || this.selectedProject.project_id !== project.project_id
      this.selectedProject = project
      if (changed) { this.graph = null; this.predictions = null; this.selectedTier = null }
      this.loadGraph()
      this.loadPredictions()
      // Drive the map imperatively; also updating the bound `center` prop would interrupt flyTo.
      if (this.mapReady && this.$refs.map && project.latitude != null) {
        const map = this.$refs.map.mapObject
        map.flyTo([project.latitude, project.longitude], Math.max(map.getZoom(), 14), { duration: 0.6 })
      }
    },
    async loadGraph() {
      if (!this.selectedProject) return
      const controller = this.beginRequest('graph')
      this.loadingGraph = true; this.graphError = ''
      try {
        const response = await axios.get(`/api/graph/${this.selectedProject.project_id}`, {
          params: { dataset: this.datasetId, relations: this.activeRelations.join(','), ...(this.month ? { month: this.month } : {}) },
          signal: controller.signal,
        })
        if (this.isCurrent('graph', controller) && response.data.dataset === this.datasetId) this.graph = response.data
      } catch (error) {
        if (!isCancel(error) && this.isCurrent('graph', controller)) { this.graph = null; this.graphError = apiError(error) }
      } finally {
        if (this.isCurrent('graph', controller)) this.loadingGraph = false
      }
    },
    async loadPredictions() {
      if (!this.selectedProject) return
      const controller = this.beginRequest('predictions')
      const projectId = this.selectedProject.project_id
      this.loadingPredictions = true; this.predictionError = ''
      try {
        const response = await axios.get(`/api/predictions/${projectId}`, { params: { dataset: this.datasetId, horizon: this.horizon }, signal: controller.signal })
        if (!this.isCurrent('predictions', controller)) return
        const data = response.data
        if (data.project_id !== projectId || data.horizon !== this.horizon || data.dataset !== this.datasetId) return
        this.predictions = data
        const keep = data.series.find(s => s.size_tier === this.selectedTier && s.n_points)
        const firstWithData = data.series.find(s => s.n_points) || data.series[0]
        this.selectedTier = keep ? keep.size_tier : (firstWithData ? firstWithData.size_tier : null)
      } catch (error) {
        if (!isCancel(error) && this.isCurrent('predictions', controller)) { this.predictions = null; this.predictionError = apiError(error) }
      } finally {
        if (this.isCurrent('predictions', controller)) this.loadingPredictions = false
      }
    },
    setHorizon(h) { if (h !== this.horizon) { this.horizon = h; this.loadPredictions() } },
    setAllRelations(on) { this.activeRelations = on ? this.metadata.relations.map(r => r.id) : [] },
    summaryFor(id) { return this.graph ? this.graph.relation_summary.find(r => r.id === id) : null },
    relationCount(id) {
      if (!this.activeRelations.includes(id)) return 'off'
      const s = this.summaryFor(id)
      return s ? String(s.neighbours_total) : '–'
    },
    relationTitle(id) {
      const s = this.summaryFor(id)
      if (!s) return ''
      const extra = s.size_edges_total != null ? `; ${s.size_edges_total} room-size edges (${s.within_project_edges} within this project)` : ''
      return `${s.neighbours_total} unique neighbouring projects; ${s.neighbours_shown} nearest drawn on the map${extra}`
    },
    relationLabel(id) { return (this.relationMeta[id] || {}).label || id },
    ageText(age) { return age.min === age.max ? age.min : `${age.min}–${age.max}` },
  },
}
</script>

<style scoped>
* { box-sizing: border-box; }
.demo-shell { min-height: 100vh; background: #edf0ec; color: #182a3a; font-family: Inter, Avenir, Helvetica, Arial, sans-serif; text-align: left; }
.topbar { min-height: 76px; padding: 14px 24px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; background: #142b2b; color: #f5f2e8; }
.eyebrow { margin: 0 0 3px; color: #bcd3c9; font-size: 11px; letter-spacing: .15em; font-weight: 700; }
h1 { margin: 0; font-family: Georgia, serif; font-size: 27px; font-weight: 500; }
.switchers { display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; }
.segmented { display: inline-flex; border: 1px solid #5d7a76; border-radius: 6px; overflow: hidden; }
.segmented button { padding: 7px 12px; border: 0; background: transparent; color: #e6efe9; font: inherit; font-size: 12px; cursor: pointer; }
.segmented button + button { border-left: 1px solid #5d7a76; }
.segmented button.active { background: #f5f2e8; color: #142b2b; font-weight: 700; }
.segmented button:disabled { opacity: .45; cursor: not-allowed; }
.month-picker { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 10px; font-size: 12px; color: #33403d; }
.month-picker select { padding: 5px 8px; border: 1px solid #b9c1bc; border-radius: 4px; background: white; font: inherit; }
.dataset-pill { padding: 8px 13px; border: 1px solid #5d7a76; border-radius: 20px; font-size: 12px; }
.dataset-pill span { display: inline-block; width: 8px; height: 8px; margin-right: 8px; border-radius: 50%; background: #70c1a1; }
.dataset-pill.bad span { background: #f08a74; }
.banner { margin: 12px 12px 0; }
.workspace { display: grid; grid-template-columns: 320px minmax(380px, 1fr) 370px; gap: 12px; height: calc(100vh - 76px); padding: 12px; }
.panel { background: #fffdf7; border: 1px solid #d9ddd7; border-radius: 8px; overflow: hidden; box-shadow: 0 3px 14px rgba(30, 43, 43, .06); }
.controls, .predictions { padding: 18px; overflow-y: auto; }
.section-heading { display: flex; gap: 11px; margin-bottom: 12px; }
.section-heading.compact { margin-top: 22px; }
.step { color: #c2461f; font-size: 11px; font-weight: 800; letter-spacing: .1em; padding-top: 4px; }
h2 { margin: 0 0 3px; font-family: Georgia, serif; font-size: 19px; font-weight: 500; }
.section-heading p { margin: 0; color: #56625f; font-size: 12px; line-height: 1.4; }
.project-card { margin-top: 12px; padding: 12px 14px; border-left: 4px solid #d9432b; background: #f2f4ef; }
.label { margin: 0 0 5px; color: #56625f; font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.project-card h3 { margin: 0; font-size: 15px; }
.facts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.facts span { padding: 3px 7px; background: white; border: 1px solid #dde1dc; font-size: 11px; }
.relation-actions { display: flex; gap: 6px; margin-bottom: 4px; }
.relation-actions button, .horizons button, .tier-tabs button { border: 1px solid #b9c1bc; background: white; color: #33403d; cursor: pointer; font: inherit; }
.relation-actions button { padding: 3px 10px; border-radius: 12px; font-size: 11px; }
.relations { margin: 0; padding: 0; border: 0; }
.relation-row { display: grid; grid-template-columns: 16px 26px 1fr auto; align-items: center; gap: 8px; padding: 7px 0; border-bottom: 1px solid #eceeea; cursor: pointer; font-size: 12px; }
.relation-row b { color: #3d4a47; font-size: 11px; font-variant-numeric: tabular-nums; }
.inset-wrap { margin-top: 16px; }
.note, .muted { margin-top: 14px; color: #56625f; font-size: 12px; line-height: 1.45; }
.note code, .eval-note code { color: #b23b1c; }
.map-panel { position: relative; min-height: 300px; }
.map-panel > .vue2leaflet-map { height: 100%; width: 100%; background: #e8ece7; }
.map-key, .map-loading { position: absolute; z-index: 900; padding: 7px 11px; background: rgba(20,43,43,.92); color: white; border-radius: 4px; font-size: 11px; }
.map-key { bottom: 16px; left: 16px; right: 16px; display: flex; flex-wrap: wrap; gap: 4px 14px; width: fit-content; max-width: calc(100% - 32px); }
.map-key i { display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 50%; vertical-align: -1px; }
.k-sel { background: #d9432b; border: 2px solid #fff; }
.k-nb { background: #182a3a; border: 2px solid #fff; }
.k-bg { background: #8c9a96; }
.map-loading { top: 16px; left: 50%; transform: translateX(-50%); }
.eval-note { margin: 0 0 12px; padding: 9px 11px; background: #fff6e0; border-left: 3px solid #c99a1b; font-size: 12px; line-height: 1.45; }
.horizons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin-bottom: 16px; }
.horizons button { padding: 8px 2px; border-radius: 4px; font-size: 11px; }
.horizons button.active, .tier-tabs button.active { border-color: #183d39; background: #183d39; color: white; }
button:focus-visible, input:focus-visible, .link:focus-visible { outline: 3px solid #2567b3; outline-offset: 1px; }
.empty-state { min-height: 240px; display: grid; place-content: center; text-align: center; color: #56625f; }
.empty-state div:first-child { margin: auto; color: #c2461f; font-size: 34px; }
.empty-state p { max-width: 230px; font-size: 13px; line-height: 1.5; }
.spinner { width: 24px; height: 24px; border: 3px solid #dce5e1; border-top-color: #d9432b; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.error-state { padding: 11px 12px; background: #fff0ed; color: #9a321c; font-size: 12px; line-height: 1.5; border-left: 3px solid #d9432b; }
.info-state { padding: 11px 12px; background: #eef3f7; color: #24425f; font-size: 12px; line-height: 1.5; border-left: 3px solid #2567b3; margin-bottom: 10px; }
.refreshing { opacity: .55; transition: opacity .15s; }
.tier-tabs { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }
.tier-tabs button { min-width: 44px; height: 30px; border-radius: 15px; font-size: 12px; font-weight: 700; }
.tier-tabs button.empty:not(.active) { border-style: dashed; color: #6b7673; }
.tier-label { margin: 0 0 8px; font-size: 12px; color: #3d4a47; }
.latest-card { display: flex; justify-content: space-between; gap: 10px; padding: 14px; background: #183d39; color: white; border-radius: 6px; }
.latest-card p { margin: 0 0 5px; color: #cfe0da; font-size: 11px; }
.latest-card strong { margin-right: 5px; font: 28px Georgia, serif; }
.latest-card small { color: #cfe0da; font-size: 11px; }
.latest-card .issued { margin: 6px 0 0; font-size: 10px; }
.latest-card .actual { text-align: right; }
.latest-card .actual strong { display: block; margin: 0; font-size: 20px; }
.tier-summary { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
.tier-summary caption { text-align: left; font-size: 11px; color: #56625f; padding-bottom: 4px; }
.tier-summary th, .tier-summary td { padding: 6px 4px; border-bottom: 1px solid #eceeea; text-align: right; font-variant-numeric: tabular-nums; }
.tier-summary thead th { font-size: 10px; color: #56625f; font-weight: 700; }
.tier-summary th:first-child { text-align: left; }
.tier-summary tbody tr { cursor: pointer; }
.tier-summary tr.current { background: #eef3ef; }
.tier-summary td.muted { text-align: left; margin: 0; }
.link { border: 0; background: none; padding: 0; font: inherit; font-weight: 700; color: #1f5a99; cursor: pointer; }
.disclaimer { color: #56625f; font-size: 11px; line-height: 1.5; margin-top: 12px; word-break: break-word; }
.visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@media (max-width: 1150px) {
  .workspace { grid-template-columns: 300px 1fr; height: auto; }
  .controls { max-height: 780px; }
  .map-panel { min-height: 620px; }
  .predictions { grid-column: 1 / -1; }
}
@media (max-width: 700px) {
  .topbar { padding: 12px 16px; }
  h1 { font-size: 22px; }
  .dataset-pill { display: none; }
  .workspace { display: flex; flex-direction: column; padding: 8px; }
  .controls, .predictions { overflow: visible; max-height: none; }
  .map-panel { height: 440px; min-height: 0; order: 2; }
  .controls { order: 1; }
  .predictions { order: 3; }
}
</style>
