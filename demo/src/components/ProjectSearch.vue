<template>
  <div class="search-box" @focusout="onFocusOut">
    <label :for="inputId" class="visually-hidden">Search condominium by name</label>
    <input
      :id="inputId" ref="input" v-model="query" class="search" type="search" autocomplete="off"
      placeholder="Search condominium name" role="combobox" aria-autocomplete="list"
      :aria-expanded="String(open)" :aria-controls="listId"
      :aria-activedescendant="open && activeIndex >= 0 ? optionId(activeIndex) : null"
      :disabled="disabled"
      @input="onInput" @keydown.down.prevent="move(1)" @keydown.up.prevent="move(-1)"
      @keydown.enter.prevent="choose(activeIndex)" @keydown.esc="close"
    />
    <ul v-show="open" :id="listId" class="results" role="listbox" aria-label="Matching condominiums">
      <li
        v-for="(project, index) in results" :id="optionId(index)" :key="project.project_id"
        role="option" :aria-selected="String(index === activeIndex)" :class="{ active: index === activeIndex }"
        @mousedown.prevent="choose(index)" @mousemove="activeIndex = index"
      >
        <strong>{{ project.name }}</strong>
        <small>{{ project.planning_area || 'Planning area unknown' }}<span v-if="!project.has_predictions"> · no predictions</span><span v-if="project.latitude == null"> · no map location</span></small>
      </li>
      <li v-if="!results.length" class="no-match" role="option" aria-disabled="true">No matching condominium</li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'ProjectSearch',
  props: { projects: { type: Array, required: true }, disabled: Boolean },
  data() {
    return { query: '', open: false, activeIndex: -1, inputId: 'project-search', listId: 'project-search-results' }
  },
  computed: {
    results() {
      const q = this.query.trim().toLowerCase()
      if (!q) return []
      return this.projects.filter(p => p.name.toLowerCase().includes(q)).slice(0, 10)
    },
  },
  methods: {
    optionId(index) { return `project-option-${index}` },
    onInput() { this.open = this.query.trim().length > 0; this.activeIndex = this.results.length ? 0 : -1 },
    move(step) {
      if (!this.open) { this.onInput(); return }
      if (!this.results.length) return
      this.activeIndex = (this.activeIndex + step + this.results.length) % this.results.length
    },
    choose(index) {
      const project = this.results[index]
      if (!project) return
      this.$emit('select', project)
      this.query = ''
      this.close()
    },
    close() { this.open = false; this.activeIndex = -1 },
    onFocusOut(event) { if (!this.$el.contains(event.relatedTarget)) this.close() },
  },
}
</script>

<style scoped>
.search-box { position: relative; }
.search { width: 100%; padding: 11px 12px; border: 1px solid #b9c1bc; border-radius: 5px; background: white; color: #182a3a; font-size: 13px; }
.search:focus-visible { outline: 3px solid #2567b3; outline-offset: 1px; }
.results { position: absolute; z-index: 1100; left: 0; right: 0; margin: 4px 0 0; padding: 0; list-style: none; max-height: 300px; overflow-y: auto; border: 1px solid #b9c1bc; border-radius: 5px; background: white; box-shadow: 0 8px 22px rgba(0,0,0,.15); }
.results li { display: flex; flex-direction: column; padding: 9px 11px; border-bottom: 1px solid #eee; cursor: pointer; font-size: 12px; }
.results li.active { background: #e3eef8; }
.results small { margin-top: 2px; color: #56625f; }
.results .no-match { cursor: default; color: #56625f; }
.visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
</style>
