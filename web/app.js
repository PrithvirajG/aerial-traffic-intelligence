/*
 * Drone traffic analytics dashboard - vanilla JS + Canvas.
 *
 * Why no framework: the core of this view is a canvas overlay redrawn every animation frame
 * in lockstep with video playback. A virtual DOM contributes nothing to that (we never touch
 * DOM nodes per frame) while adding a build step; and there's no Node toolchain on this
 * machine, so a bundled React app couldn't be built here anyway. Plain canvas + one rAF loop
 * is both faster and simpler for this shape of problem.
 *
 * Sync model: the <video> element's own clock is the single source of truth. Every frame we
 * read video.currentTime, convert to a frame index, and draw that frame's detections in BOTH
 * panels. Nothing is interpolated or timer-driven, so the overlay and the flow panel can't
 * drift from the footage (which is exactly what st.video in the Streamlit version cannot do -
 * it exposes no playback clock to the server).
 *
 * Data comes from web/export_data.py as frame-indexed JSON - see that file for the format.
 */

// Dark-mode categorical palette, fixed order = the taxonomy order. Colour identifies a
// class and never gets reassigned as filters change which classes are on screen.
const CLASS_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#4fae4f', '#9085e9'];
const INK = '#ffffff', INK2 = '#c3c2b7', MUTED = '#898781', GRID = '#2c2c2a';

const TRAIL_FRAMES = 45;   // ~1.5s at 30fps - matches pipeline/trajectory_trails.py
// Band edges set from the ACTUAL distribution across the processed 10s combos (n=1395
// tracks: speed p25/50/75 = 7 / 22 / 78 px/s; accel p25/50/75 = 24 / 43 / 83 px/s²), so each
// band holds a meaningful share of traffic. First-guess accel edges of 2/10/30 put ~90% of
// tracks in "hard" - useless as a filter. Values are large because this is pixel-space at
// 4K, not m/s²; they're comparative bands, not physical ones (no geo-calibration yet).
const SPEED_BANDS = [
  { key: 'stationary', label: 'Stationary', min: 0, max: 5 },
  { key: 'slow', label: 'Slow', min: 5, max: 25 },
  { key: 'moderate', label: 'Moderate', min: 25, max: 80 },
  { key: 'fast', label: 'Fast', min: 80, max: Infinity },
];
const ACCEL_BANDS = [
  { key: 'steady', label: 'Steady', min: 0, max: 20 },
  { key: 'gentle', label: 'Gentle', min: 20, max: 45 },
  { key: 'moderate', label: 'Moderate', min: 45, max: 90 },
  { key: 'hard', label: 'Hard', min: 90, max: Infinity },
];

/* ---------------------------------------------------------------- categories
 * One active category at a time drives BOTH the visible filter chips and the colour of every
 * drawn object. Each category gets the palette type that matches what its data actually is,
 * which is the whole point of making this switchable rather than always colouring by class:
 *   - class / turn : categorical  (unordered identities -> distinct hues)
 *   - speed / accel: sequential   (ordered magnitude -> one hue, dim->bright)
 *   - direction    : cyclic       (a bearing wraps around, so hue wraps around too, and
 *                                  opposite headings land opposite on the colour wheel)
 *   - colour       : the swatch IS the datum (a red car's chip is red)
 * Selections are remembered per category, but only the ACTIVE category filters - so nothing
 * is ever hidden by a filter you can't currently see.
 */
/* Green (low) -> red (high) ramp, shared by Speed band and Acceleration band. Safe to share
 * because only one category is ever active, so the two can never be on screen together.
 *
 * Green->red is the classic red-green CVD failure case, so this ramp was constructed in OKLCH
 * with **strictly monotonic lightness** rather than picked by eye: L = 0.52 -> 0.685 in even
 * steps. That is what makes it survive colour blindness - the ordering is carried by
 * brightness as well as hue, so it still reads low->high under deuteranopia, protanopia, or
 * full monochromacy. Measured against the dark chart surface (#1a1a19):
 *   contrast  3.39 / 4.18 / 4.85 / 5.65   (all clear 3:1, needed for small dots and thin lines)
 *   adjacent  CVD dE 10.8 / 18.0 / 10.1   (all clear the >=8 floor)
 *   endpoints dE 33.6 normal, 47.9 deuteranopia - unmistakable either way
 * Hand-picked green->red attempts failed badly here: pure orange->red measured CVD dE 0.8-3.1,
 * i.e. indistinguishable, because hue alone was doing the work. Don't re-tune by eye.
 */
const LOW_HIGH_RAMP = ['#007f35', '#5d8900', '#c67300', '#f96050'];
const SPEED_RAMP = LOW_HIGH_RAMP;
const ACCEL_RAMP = LOW_HIGH_RAMP;
const TURN_COLORS = ['#3987e5', '#199e70', '#c98500', '#d55181'];  // through, left, right, u-turn

function cyclicHue(i, n) {
  // Evenly spaced around the wheel: encodes a bearing's wrap-around honestly.
  return `hsl(${Math.round((i / n) * 360)}, 62%, 60%)`;
}

const CATEGORIES = {
  class: {
    label: 'Vehicle class',
    note: 'Categorical: distinct hues for unordered identities.',
    buckets: (d) => d.classOrder.map((n, i) => ({
      key: i, label: n.replace('_candidate', ''), colour: CLASS_COLORS[i % CLASS_COLORS.length],
    })),
    of: (t) => t.c,
  },
  colour: {
    label: 'Colour',
    note: 'Each chip is painted its own colour family. Grey/silver dominate because from '
      + 'altitude you mostly see sunlit metallic roofs - saturation is tested before hue.',
    buckets: () => COLOUR_FAMILIES.map(([k, swatch]) => ({ key: k, label: k, colour: swatch })),
    of: (t) => colourFamily(t.col || '#898781'),
  },
  speed: {
    label: 'Speed band',
    note: 'Green = slow, red = fast. Lightness rises with the hue shift, so the low→high '
      + 'ordering still reads under red-green colour blindness. Bands are set from the real '
      + 'distribution across the processed clips, not round numbers.',
    buckets: () => SPEED_BANDS.map((b, i) => ({
      key: b.key, colour: SPEED_RAMP[i],
      label: `${b.label} ${b.max === Infinity ? `${b.min}+` : `${b.min}–${b.max}`}`,
    })),
    of: (t) => bandOf(SPEED_BANDS, t.sp),
  },
  accel: {
    label: 'Acceleration band',
    note: 'Green = steady, red = hard acceleration. Same green→red scale as speed — the two '
      + 'are never shown together, so they share one ramp.',
    buckets: () => ACCEL_BANDS.map((b, i) => ({
      key: b.key, colour: ACCEL_RAMP[i],
      label: `${b.label} ${b.max === Infinity ? `${b.min}+` : `${b.min}–${b.max}`}`,
    })),
    of: (t) => bandOf(ACCEL_BANDS, t.ac),
  },
  direction: {
    label: 'Direction',
    note: null,   // set per-combo: a real bearing only when telemetry geo-referencing exists
    buckets: (d) => [
      // Key -1 = stationary: a real observation (the queue), not missing data.
      { key: -1, label: 'stationary', colour: '#6b6a66' },
      ...(d.compass || []).map((n, i) => ({
        key: i, label: n, colour: cyclicHue(i, (d.compass || []).length),
      })),
    ],
    of: (t) => (t.dir === null || t.dir === undefined ? null : t.dir),
  },
  turn: {
    label: 'Turn',
    note: 'Categorical: through / left / right / u-turn, from entry vs exit heading. '
      + 'Stationary vehicles have no manoeuvre to classify and are bucketed as such.',
    buckets: (d) => [
      { key: -1, label: 'stationary', colour: '#6b6a66' },
      ...(d.turnTypes || []).map((n, i) => ({
        key: i, label: n, colour: TURN_COLORS[i % TURN_COLORS.length],
      })),
    ],
    of: (t) => (t.turn === null || t.turn === undefined ? null : t.turn),
  },
};
const CATEGORY_ORDER = ['class', 'colour', 'speed', 'accel', 'direction', 'turn'];

const state = {
  index: [], data: null, trackPoints: new Map(), perFrameCounts: null,
  activeCat: 'class',
  selections: {},          // category key -> Set of selected bucket keys (remembered per category)
  bucketOf: new Map(),     // trackId -> active category's bucket key (recomputed on change)
  colourOf: new Map(),     // trackId -> colour string (recomputed on change)
  visibleTracks: new Set(),
  availableBands: { speed: SPEED_BANDS.length, accel: ACCEL_BANDS.length },
  show: { boxes: true, trails: true, labels: false,
          flowPoints: true, flowTrails: true, flowAll: false, arrows: true },
  mapShow: { base: 'sat', roads: false, traj: true, points: true, arrows: true,
             desire: false, queues: true, history: false },
  rightView: 'flow',
  mapView: null,        // {cE, cN, mPerPx} in local ENU metres; null = auto-fit on next draw
  frameStride: 1,       // 1 = every frame detected; 3 = a 10 FPS run on 30 fps footage
  mapDirty: true,       // coalesces redraws into one per animation frame (see requestMapDraw)
  mapAnim: null,        // in-flight zoom animation, advanced by the main rAF loop
  tileIndex: null,
  frame: 0, hover: null,
};

function setRightView(view) {
  const hasMap = !!(state.data && state.data.mapLayers);
  if (view === 'map' && !hasMap) return;
  state.rightView = view;
  document.querySelectorAll('#rightViewTabs .cattab').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.view === view)));
  $('stageFlow').hidden = view !== 'flow';
  $('stageMap').hidden = view !== 'map';
  $('flowToggles').hidden = view !== 'flow';
  $('mapToggles').hidden = view !== 'map';
  $('mapAttrib').hidden = view !== 'map';
  $('flowNote').textContent = view === 'map'
    ? 'live-synced to playback · real coordinates on the real road network'
    : 'live-synced to playback';
  setMapAttribution();
}

function setMapAttribution() {
  const idx = state.tileIndex;
  const src = state.mapShow.base;
  const el = $('mapAttrib');
  if (!idx || src === 'none' || !idx.sources[src]) { el.textContent = 'No basemap'; return; }
  // Attribution is a licence condition of both tile services, not decoration.
  el.textContent = `Basemap: ${idx.sources[src].attribution} · roads: OpenStreetMap (ODbL) · `
    + 'tiles cached locally, no live tile requests';
}

/** Colour for a track under the active category. Falls back to a neutral for tracks the
 *  active category can't classify (e.g. a track too short to have a speed band) - drawing
 *  them in a real bucket colour would be a lie. */
function trackColour(tid) {
  return state.colourOf.get(tid) || MUTED;
}

const $ = (id) => document.getElementById(id);
const video = $('video');

/** Value-or-placeholder. Written as a helper rather than using `??` so the whole file stays
 *  parseable by the syntax checker available here (no Node/browser on this machine to lint
 *  with) - and it reads better at the call sites than a chain of ternaries either way.
 *  Note `||` would be wrong for these: a genuine 0 (stationary vehicle) must not become '—'. */
function val(v, dash) {
  const d = dash === undefined ? '—' : dash;
  return (v === null || v === undefined || Number.isNaN(v)) ? d : v;
}

/* ---------------------------------------------------------------- colour helpers */

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Buckets an arbitrary hex into a human-meaningful colour family. Vehicle roofs from
 *  altitude are mostly desaturated, so saturation is checked before hue - otherwise nearly
 *  every car lands in some arbitrary hue bucket and the filter becomes meaningless. */
function colourFamily(hex) {
  const [r, g, b] = hexToRgb(hex);
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2 / 255;
  const s = max === min ? 0 : (max - min) / (255 - Math.abs(max + min - 255));
  if (s < 0.16) {
    if (l > 0.72) return 'white';
    if (l > 0.45) return 'silver';
    if (l > 0.22) return 'grey';
    return 'black';
  }
  let h;
  const d = max - min;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
  else if (max === g) h = ((b - r) / d + 2) * 60;
  else h = ((r - g) / d + 4) * 60;
  if (h < 16 || h >= 336) return 'red';
  if (h < 45) return l < 0.4 ? 'brown' : 'orange';
  if (h < 70) return 'yellow';
  if (h < 165) return 'green';
  if (h < 260) return 'blue';
  return 'purple';
}

const COLOUR_FAMILIES = [
  ['white', '#e8e6e1'], ['silver', '#a8a6a1'], ['grey', '#6b6a66'], ['black', '#2a2a28'],
  ['red', '#d64545'], ['orange', '#e07a2f'], ['brown', '#8a5a34'], ['yellow', '#d4b52a'],
  ['green', '#3f9e4d'], ['blue', '#3d7fd6'], ['purple', '#8b6bd9'],
];

function bandOf(bands, v) {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  for (const b of bands) if (v >= b.min && v < b.max) return b.key;
  return bands[bands.length - 1].key;
}

/* ---------------------------------------------------------------- data loading */

async function boot() {
  state.index = await (await fetch('data/index.json')).json();
  // Cached basemap tiles are optional: without them the map still works, just with no
  // imagery behind the data, so a missing cache must not break boot.
  try {
    state.tileIndex = await (await fetch('tiles/index.json')).json();
  } catch (e) {
    state.tileIndex = null;
  }
  const sources = [...new Set(state.index.map((e) => e.source))];
  fillSelect($('selSource'), sources);
  onSourceChange();
  $('selSource').addEventListener('change', onSourceChange);
  $('selDuration').addEventListener('change', onDurationChange);
  $('selModel').addEventListener('change', loadCombo);
  wireControls();
  requestAnimationFrame(tick);
}

function fillSelect(sel, values, labels) {
  sel.innerHTML = '';
  values.forEach((v, i) => {
    const o = document.createElement('option');
    o.value = v; o.textContent = labels ? labels[i] : v;
    sel.appendChild(o);
  });
}

function onSourceChange() {
  const src = $('selSource').value;
  const durations = [...new Set(state.index.filter((e) => e.source === src).map((e) => e.duration))];
  fillSelect($('selDuration'), durations);
  onDurationChange();
}

function onDurationChange() {
  const src = $('selSource').value, dur = $('selDuration').value;
  const combos = state.index.filter((e) => e.source === src && e.duration === dur);
  fillSelect($('selModel'), combos.map((c) => c.key), combos.map((c) => c.model));
  loadCombo();
}

async function loadCombo() {
  const key = $('selModel').value;
  if (!key) return;
  $('runinfo').textContent = 'Loading ' + key + '…';
  const d = await (await fetch(`data/${key}.json`)).json();
  state.data = d;

  // Precompute per-track point arrays (ground-contact point = bbox bottom-centre, the
  // project-wide convention) so trails and hover-testing don't rescan frames.
  const pts = new Map();
  const frameNums = Object.keys(d.frames).map(Number).sort((a, b) => a - b);
  for (const f of frameNums) {
    for (const [tid, x1, y1, x2, y2] of d.frames[f]) {
      if (!pts.has(tid)) pts.set(tid, []);
      pts.get(tid).push([f, (x1 + x2) / 2, y2]);
    }
  }
  state.trackPoints = pts;
  state.frameStride = detectStride(d.frames);
  state.maxFrame = frameNums.length ? frameNums[frameNums.length - 1] : 0;

  // Same per-track history, but in map (ENU metres) space, for the map view's arrows -
  // heading needs a few frames of run-up or a single-frame delta is dominated by jitter,
  // same reasoning as the pixel-space version above.
  const mapPts = new Map();
  if (d.mapFrames) {
    const mapFrameNums = Object.keys(d.mapFrames).map(Number).sort((a, b) => a - b);
    for (const f of mapFrameNums) {
      for (const [tid, e10, n10] of d.mapFrames[f]) {
        if (!mapPts.has(tid)) mapPts.set(tid, []);
        mapPts.get(tid).push([f, e10 / 10, n10 / 10]);
      }
    }
  }
  state.mapTrackPoints = mapPts;

  // Per-frame per-class counts for the time chart - computed once, not per render.
  const nClasses = d.classOrder.length;
  const counts = Array.from({ length: state.maxFrame + 1 }, () => new Array(nClasses).fill(0));
  for (const f of frameNums) {
    for (const [tid] of d.frames[f]) {
      const t = d.tracks[String(tid)];
      if (t) counts[f][t.c]++;
    }
  }
  state.perFrameCounts = counts;

  buildFilters();
  const meta = d.meta || {};
  $('runinfo').innerHTML =
    `<strong>${d.source} · ${d.duration} · ${d.model}</strong> — weights <code>${meta.weights_source || '?'}</code>`
    + ` · tracker ${meta.tracker || '?'} · conf≥${val(meta.confidence_threshold, '?')}`
    + ` · SAHI ${val((meta.sliced_inference || {}).slice_size, '?')}px slices`
    + ` · ${val(meta.frames_processed, '?')} frames @ ${(meta.fps || 0).toFixed(1)} fps`
    + ` · <strong>${Object.keys(d.tracks).length}</strong> unique tracks`;

  video.src = d.clip ? `clips/${d.clip}` : '';
  video.load();
  state.frame = 0;
  // A combo without Level 4 output has no map to switch to - disable the tab rather than
  // offering a view that would render empty.
  state.mapView = null;      // different scene, different extent - refit rather than keeping
  const hasMap = !!d.mapLayers;
  $('tabMap').disabled = !hasMap;
  $('tabMap').title = hasMap ? '' : 'No spatial grounding for this run';
  if (!hasMap && state.rightView === 'map') setRightView('flow'); else setRightView(state.rightView);
  recomputeVisible();
}

/* ---------------------------------------------------------------- filters */

/** Buckets every track under one category, returning [bucketOf, counts, unclassified]. */
function bucketTracks(catKey) {
  const d = state.data;
  const cat = CATEGORIES[catKey];
  const bucketOf = new Map();
  const counts = new Map();
  let unclassified = 0;
  for (const [tidStr, t] of Object.entries(d.tracks)) {
    const b = cat.of(t);
    if (b === null || b === undefined) { unclassified++; continue; }
    bucketOf.set(Number(tidStr), b);
    counts.set(b, (counts.get(b) || 0) + 1);
  }
  return [bucketOf, counts, unclassified];
}

function buildFilters() {
  const d = state.data;
  // Fresh per-combo selections: bucket keys aren't comparable across combos.
  state.selections = {};
  const tabs = $('catTabs');
  tabs.innerHTML = '';
  for (const key of CATEGORY_ORDER) {
    const [, counts] = bucketTracks(key);
    const b = document.createElement('button');
    b.className = 'cattab';
    b.dataset.cat = key;
    b.textContent = CATEGORIES[key].label;
    if (counts.size === 0) {
      // Nothing to show for this category in this combo (e.g. no kinematics at all) -
      // disabled rather than offered as an empty view.
      b.disabled = true;
      b.title = 'No data for this category in this run';
    } else {
      b.addEventListener('click', () => setCategory(key));
    }
    tabs.appendChild(b);
  }
  $('btnAll').onclick = () => { setAllChips(true); };
  $('btnNone').onclick = () => { setAllChips(false); };

  // Default to class if usable, else the first category that has data.
  const firstUsable = CATEGORY_ORDER.find((k) => bucketTracks(k)[1].size > 0) || 'class';
  setCategory(bucketTracks('class')[1].size > 0 ? 'class' : firstUsable);
}

function setCategory(catKey) {
  state.activeCat = catKey;
  const d = state.data;
  const cat = CATEGORIES[catKey];
  const [bucketOf, counts, unclassified] = bucketTracks(catKey);
  state.bucketOf = bucketOf;

  const buckets = cat.buckets(d).filter((b) => counts.get(b.key));
  const palette = new Map(buckets.map((b) => [b.key, b.colour]));
  state.colourOf = new Map([...bucketOf].map(([tid, b]) => [tid, palette.get(b) || MUTED]));

  // Remembered selection for this category, defaulting to everything on.
  if (!state.selections[catKey]) state.selections[catKey] = new Set(buckets.map((b) => b.key));
  const sel = state.selections[catKey];

  document.querySelectorAll('.cattab').forEach((t) =>
    t.setAttribute('aria-pressed', String(t.dataset.cat === catKey)));

  const container = $('fActive');
  container.innerHTML = '';
  for (const b of buckets) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.dataset.key = String(b.key);
    btn.setAttribute('aria-pressed', String(sel.has(b.key)));
    btn.innerHTML = `<span class="dot" style="background:${b.colour}"></span>${b.label}`
      + `<span class="cnt">${counts.get(b.key)}</span>`;
    btn.addEventListener('click', () => {
      const on = btn.getAttribute('aria-pressed') === 'true';
      btn.setAttribute('aria-pressed', String(!on));
      if (on) sel.delete(b.key); else sel.add(b.key);
      recomputeVisible();
    });
    container.appendChild(btn);
  }

  const notes = [];
  if (catKey === 'direction') {
    notes.push(d.dirFrame === 'compass'
      ? 'Cyclic palette: hue wraps like a bearing, so opposite headings are opposite colours. '
        + 'Bearings are true compass directions, geo-referenced from drone telemetry.'
      : 'Cyclic palette: hue wraps like a bearing. ⚠ This run has no telemetry geo-referencing, '
        + 'so these are image-relative directions, NOT compass bearings.');
  } else if (cat.note) {
    notes.push(cat.note);
  }
  if (unclassified) {
    notes.push(`${unclassified} of ${Object.keys(d.tracks).length} tracks can't be classified `
      + `under this category (too short to resolve) — they are hidden while it is active, `
      + `rather than shown in a bucket they don't belong to.`);
  }
  $('filterNote').textContent = notes.join(' ');

  recomputeVisible();
}

function setAllChips(on) {
  const sel = state.selections[state.activeCat];
  document.querySelectorAll('#fActive .chip').forEach((c) => {
    c.setAttribute('aria-pressed', String(on));
  });
  sel.clear();
  if (on) {
    const [, counts] = bucketTracks(state.activeCat);
    CATEGORIES[state.activeCat].buckets(state.data)
      .filter((b) => counts.get(b.key)).forEach((b) => sel.add(b.key));
  }
  recomputeVisible();
}

/** Recomputes which track IDs pass the ACTIVE category's filter. Done once per change (not
 *  per frame) - the render loop then just tests set membership. */
function recomputeVisible() {
  const d = state.data;
  if (!d) return;
  const sel = state.selections[state.activeCat] || new Set();
  const vis = new Set();
  for (const [tid, bucket] of state.bucketOf) {
    if (sel.has(bucket)) vis.add(tid);
  }
  state.visibleTracks = vis;
  renderCatLegend();
  renderStats();
  renderCharts();
}

function renderCatLegend() {
  const d = state.data;
  const [, counts] = bucketTracks(state.activeCat);
  const sel = state.selections[state.activeCat] || new Set();
  const buckets = CATEGORIES[state.activeCat].buckets(d).filter((b) => counts.get(b.key));
  $('catLegend').innerHTML =
    `<span style="color:var(--ink-2)">${CATEGORIES[state.activeCat].label}:</span>`
    + buckets.map((b) => `<span style="opacity:${sel.has(b.key) ? 1 : 0.32}">`
      + `<i style="background:${b.colour}"></i>${b.label}</span>`).join('');
}

/* ---------------------------------------------------------------- controls */

function wireControls() {
  const toggle = (id, key) => $(id).addEventListener('click', () => {
    state.show[key] = !state.show[key];
    $(id).setAttribute('aria-pressed', String(state.show[key]));
  });
  toggle('tgBoxes', 'boxes'); toggle('tgTrails', 'trails'); toggle('tgLabels', 'labels');
  toggle('tgFlowPoints', 'flowPoints'); toggle('tgFlowTrails', 'flowTrails');
  toggle('tgFlowAll', 'flowAll'); toggle('tgArrows', 'arrows');

  const mapToggle = (id, key) => $(id).addEventListener('click', () => {
    state.mapShow[key] = !state.mapShow[key];
    $(id).setAttribute('aria-pressed', String(state.mapShow[key]));
    requestMapDraw();
  });
  mapToggle('mgRoads', 'roads'); mapToggle('mgTraj', 'traj');
  mapToggle('mgPoints', 'points'); mapToggle('mgArrows', 'arrows');
  mapToggle('mgDesire', 'desire'); mapToggle('mgQueues', 'queues');
  mapToggle('mgHistory', 'history');
  $('mapBase').addEventListener('change', (e) => {
    state.mapShow.base = e.target.value;
    setMapAttribution();
    requestMapDraw();
  });

  // Right-panel view switcher: the flow canvas and the map are two renderings of the same
  // moment, so switching between them must not change playback or filter state.
  document.querySelectorAll('#rightViewTabs .cattab').forEach((b) => {
    b.addEventListener('click', () => setRightView(b.dataset.view));
  });
  const fit = () => { state.mapView = null; state.mapAnim = null; requestMapDraw(); };
  $('mapReset').addEventListener('click', fit);
  $('mapFit').addEventListener('click', fit);
  const zoomBtn = (id, factor) => $(id).addEventListener('click', () => {
    const r = $('mapCanvas').getBoundingClientRect();
    zoomAbout(r.width / 2, r.height / 2, factor, r);
  });
  zoomBtn('mapZoomIn', 0.6);
  zoomBtn('mapZoomOut', 1 / 0.6);
  wireMapInteraction();

  $('btnPlay').addEventListener('click', () => (video.paused ? video.play() : video.pause()));
  video.addEventListener('play', () => ($('btnPlay').textContent = '❚❚'));
  video.addEventListener('pause', () => ($('btnPlay').textContent = '▶'));
  video.addEventListener('loadedmetadata', () => { $('seek').max = video.duration || 0; });
  $('seek').addEventListener('input', (e) => { video.currentTime = Number(e.target.value); });

  const flow = $('canvasFlow');
  flow.addEventListener('mousemove', (e) => {
    const r = flow.getBoundingClientRect();
    state.hover = { x: e.clientX - r.left, y: e.clientY - r.top, cx: e.clientX, cy: e.clientY, w: r.width, h: r.height };
  });
  flow.addEventListener('mouseleave', () => { state.hover = null; $('tip').style.display = 'none'; });

  // Chart canvases are sized from layout, so a resize invalidates the cached buffers.
  // Debounced: resize fires continuously while dragging a window edge.
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.data) renderCharts(); }, 150);
  });
}

/* ---------------------------------------------------------------- render loop */

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

/** Letterbox-aware mapping from source-video pixels to on-screen canvas pixels. The video
 *  uses object-fit:contain, so the drawn image is centred with bars on one axis - ignoring
 *  that is what makes overlays sit visibly off the vehicles. */
function projector(sw, sh, cw, ch) {
  const scale = Math.min(cw / sw, ch / sh);
  const ox = (cw - sw * scale) / 2, oy = (ch - sh * scale) / 2;
  return { scale, ox, oy, X: (x) => ox + x * scale, Y: (y) => oy + y * scale };
}

/** Detections for the current playback frame, tolerating a subsampled run.
 *
 * A clip processed at 10 FPS only has rows on every 3rd frame of the 30 fps timeline (see
 * kaggle/level4_90s/), so an exact lookup would return nothing for 2 frames out of every 3
 * and the overlay would strobe. This holds the most recent sample instead, which is honest:
 * it shows the last thing actually observed rather than inventing positions between samples.
 *
 * `stride` is derived once per combo from the data itself rather than assumed, so 30 fps runs
 * (stride 1) keep exact-frame behaviour and cost nothing extra.
 */
function detsAtFrame(map, frame, stride) {
  if (!map) return [];
  const exact = map[frame];
  if (exact) return exact;
  if (stride <= 1) return [];
  // Walk back at most one stride to the previous detected frame.
  for (let k = 1; k < stride + 1; k++) {
    const got = map[frame - k];
    if (got) return got;
  }
  return [];
}

/** Smallest gap between consecutive detected frames = the run's effective stride. */
function detectStride(frameMap) {
  const keys = Object.keys(frameMap || {}).map(Number).sort((a, b) => a - b);
  if (keys.length < 3) return 1;
  let best = Infinity;
  for (let i = 1; i < Math.min(keys.length, 60); i++) {
    best = Math.min(best, keys[i] - keys[i - 1]);
  }
  return Number.isFinite(best) && best > 0 ? best : 1;
}

function pointsUpTo(arr, frame, minFrame) {
  // arr is sorted by frame; binary-search the upper bound then walk back.
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m][0] <= frame) lo = m + 1; else hi = m; }
  let start = lo - 1;
  while (start > 0 && arr[start - 1][0] >= minFrame) start--;
  return [start, lo];
}

function tick(now) {
  requestAnimationFrame(tick);
  const d = state.data;
  if (!d) return;
  if (state.mapAnim) stepMapAnim(now || performance.now());

  const fps = d.fps || 30;
  state.frame = Math.min(state.maxFrame, Math.max(0, Math.round((video.currentTime || 0) * fps)));
  if (!Number.isNaN(video.duration)) {
    $('seek').value = video.currentTime || 0;
    $('tcode').textContent = `${(video.currentTime || 0).toFixed(2)} / ${(video.duration || 0).toFixed(2)}s`;
  }
  drawOverlay();
  // Only the visible right-hand view is drawn each frame - rendering both would double the
  // per-frame cost for something the user cannot see.
  if (state.rightView === 'map') {
    // Redraw only when the picture can actually have changed: while the video is advancing,
    // or when an interaction/tile-load flagged it. When paused and idle this skips the whole
    // map render, leaving the main thread free.
    const advancing = !video.paused && !video.ended;
    if (advancing || state.mapDirty) {
      state.mapDirty = false;
      drawMap();
    }
  } else {
    drawFlow();
  }
  drawTimeMarker();
}

/* --- colour-batched drawing -------------------------------------------------------------
 * Canvas state changes (strokeStyle, globalAlpha) and stroke() calls are the expensive part
 * of 2D rendering, far more than the number of line segments. The original per-segment trail
 * fade cost ~1,300 stroke() calls per frame on the 90s clips (measured: 88 tracks x 15 trail
 * segments), i.e. ~84,000 calls/second at 60fps - enough to starve video playback.
 *
 * Because colours come from a small categorical palette (one per class / lane / band), all
 * geometry sharing a colour can be accumulated into ONE Path2D and stroked once. That turns
 * ~1,300 calls per frame into roughly one per distinct colour on screen.
 *
 * The per-segment alpha gradient is the one thing this gives up; it is replaced by two passes
 * (older trail dim, recent trail bright), which keeps the "recent motion is dominant" reading
 * at 2 draw calls per colour instead of 15 per track.
 */
function strokeBatched(ctx, batches, lineWidth, alpha) {
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  for (const [colour, path] of batches) {
    ctx.strokeStyle = colour;
    ctx.stroke(path);
  }
  ctx.globalAlpha = 1;
}

function batchFor(batches, colour) {
  let path = batches.get(colour);
  if (!path) { path = new Path2D(); batches.set(colour, path); }
  return path;
}

/** Appends a track's recent trail into the per-colour batch, split into an older and a
 *  recent portion so the fade survives batching. */
function addTrailToBatch(older, recent, p, tid, frame, minFrame, colour) {
  const arr = state.trackPoints.get(tid);
  if (!arr) return;
  const [s, e] = pointsUpTo(arr, frame, minFrame);
  if (e - s < 2) return;
  const split = s + Math.floor((e - s) * 0.55);
  const push = (batches, from, to) => {
    if (to - from < 2) return;
    const path = batchFor(batches, colour);
    path.moveTo(p.X(arr[from][1]), p.Y(arr[from][2]));
    for (let i = from + 1; i < to; i++) path.lineTo(p.X(arr[i][1]), p.Y(arr[i][2]));
  };
  push(older, s, split + 1);
  push(recent, split, e);
}

function drawOverlay() {
  const d = state.data;
  const { ctx, w, h } = fitCanvas($('canvasOverlay'));
  ctx.clearRect(0, 0, w, h);
  const p = projector(d.frameWidth, d.frameHeight, w, h);
  const dets = detsAtFrame(d.frames, state.frame, state.frameStride);

  if (state.show.trails) {
    const older = new Map(), recent = new Map();
    for (const [tid] of dets) {
      if (!state.visibleTracks.has(tid)) continue;
      addTrailToBatch(older, recent, p, tid, state.frame,
                      state.frame - TRAIL_FRAMES, trackColour(tid));
    }
    strokeBatched(ctx, older, 1.6, 0.28);
    strokeBatched(ctx, recent, 1.6, 0.95);
  }

  if (state.show.boxes) {
    const boxes = new Map();
    for (const [tid, x1, y1, x2, y2] of dets) {
      if (!state.visibleTracks.has(tid)) continue;
      batchFor(boxes, trackColour(tid))
        .rect(p.X(x1), p.Y(y1), (x2 - x1) * p.scale, (y2 - y1) * p.scale);
    }
    strokeBatched(ctx, boxes, 1.6, 1);

    if (state.show.labels) {
      // Labels can't batch (each is at its own position), so they stay per-detection - which
      // is why they are opt-in rather than always on.
      ctx.font = '10px system-ui, sans-serif';
      for (const [tid, x1, y1] of dets) {
        if (!state.visibleTracks.has(tid)) continue;
        ctx.fillStyle = trackColour(tid);
        ctx.fillText(`#${tid}`, p.X(x1), Math.max(9, p.Y(y1) - 3));
      }
    }
  }
}

function drawTrail(ctx, p, tid, frame, minFrame) {
  const arr = state.trackPoints.get(tid);
  if (!arr) return;
  const [s, e] = pointsUpTo(arr, frame, minFrame);
  if (e - s < 2) return;
  const base = trackColour(tid);
  for (let i = s + 1; i < e; i++) {
    // Fade older segments so current motion stays visually dominant in dense traffic.
    ctx.globalAlpha = 0.15 + 0.85 * ((i - s) / Math.max(1, e - s));
    ctx.strokeStyle = base;
    ctx.beginPath();
    ctx.moveTo(p.X(arr[i - 1][1]), p.Y(arr[i - 1][2]));
    ctx.lineTo(p.X(arr[i][1]), p.Y(arr[i][2]));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** Batched form of drawArrow: appends the arrowhead triangle to a per-colour Path2D instead
 *  of issuing its own fill, so all arrows of one colour cost a single fill call. */
function addArrowToBatch(batches, p, tid, frame, colour) {
  const arr = state.trackPoints.get(tid);
  if (!arr) return;
  const [, e] = pointsUpTo(arr, frame, frame - TRAIL_FRAMES);
  if (e < 2) return;
  const tipPt = arr[e - 1];
  const backPt = arr[Math.max(0, e - 6)];
  let dx = tipPt[1] - backPt[1], dy = tipPt[2] - backPt[2];
  const len = Math.hypot(dx, dy);
  if (len < 1.5) return;       // effectively stationary: an arrow would point at noise
  dx /= len; dy /= len;

  const x = p.X(tipPt[1]), y = p.Y(tipPt[2]);
  const size = 7, spread = 0.42;
  const path = batchFor(batches, colour);
  path.moveTo(x + dx * size, y + dy * size);
  path.lineTo(x + (-dx * 0.55 - dy * spread) * size, y + (-dy * 0.55 + dx * spread) * size);
  path.lineTo(x + (-dx * 0.55 + dy * spread) * size, y + (-dy * 0.55 - dx * spread) * size);
  path.closePath();
}

/** Arrowhead at a track's current position, pointing along its recent heading.
 *  Heading comes from the last few trail points rather than two adjacent ones: consecutive
 *  bbox centres jitter enough that a 1-frame delta makes the arrow spin on the spot. */
function drawArrow(ctx, p, tid, frame, colour) {
  const arr = state.trackPoints.get(tid);
  if (!arr) return;
  const [, e] = pointsUpTo(arr, frame, frame - TRAIL_FRAMES);
  if (e < 2) return;
  const tipPt = arr[e - 1];
  const backIdx = Math.max(0, e - 6);
  const backPt = arr[backIdx];
  let dx = tipPt[1] - backPt[1], dy = tipPt[2] - backPt[2];
  const len = Math.hypot(dx, dy);
  if (len < 1.5) return;      // effectively stationary: an arrow would point at noise
  dx /= len; dy /= len;

  const x = p.X(tipPt[1]), y = p.Y(tipPt[2]);
  const size = 7, spread = 0.42;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.moveTo(x + dx * size, y + dy * size);
  ctx.lineTo(x + (-dx * 0.55 - dy * spread) * size, y + (-dy * 0.55 + dx * spread) * size);
  ctx.lineTo(x + (-dx * 0.55 + dy * spread) * size, y + (-dy * 0.55 - dx * spread) * size);
  ctx.closePath();
  ctx.fill();
}

function drawFlow() {
  const d = state.data;
  const { ctx, w, h } = fitCanvas($('canvasFlow'));
  ctx.clearRect(0, 0, w, h);
  const p = projector(d.frameWidth, d.frameHeight, w, h);

  // Frame border so the blank canvas reads as "the same field of view", not a floating plot.
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  ctx.strokeRect(p.ox + 0.5, p.oy + 0.5, d.frameWidth * p.scale - 1, d.frameHeight * p.scale - 1);

  const dets = detsAtFrame(d.frames, state.frame, state.frameStride);
  let shown = 0;

  if (state.show.flowTrails) {
    const older = new Map(), recent = new Map();
    if (state.show.flowAll) {
      // Full history: every filtered track's whole path - the "where does traffic actually
      // flow" view. On a 90s clip that is ~1,200 tracks, so batching matters most here.
      for (const tid of state.visibleTracks) {
        addTrailToBatch(older, recent, p, tid, state.frame, 0, trackColour(tid));
      }
    } else {
      for (const [tid] of dets) {
        if (!state.visibleTracks.has(tid)) continue;
        addTrailToBatch(older, recent, p, tid, state.frame,
                        state.frame - TRAIL_FRAMES, trackColour(tid));
      }
    }
    strokeBatched(ctx, older, 1.4, 0.25);
    strokeBatched(ctx, recent, 1.4, 0.9);
  }

  // Points and arrowheads are fills, batched by colour the same way as the strokes.
  const dots = new Map(), heads = new Map();
  for (const [tid, x1, , x2, y2] of dets) {
    if (!state.visibleTracks.has(tid)) continue;
    const colour = trackColour(tid);
    if (state.show.flowPoints) {
      const path = batchFor(dots, colour);
      const cx = p.X((x1 + x2) / 2), cy = p.Y(y2);
      path.moveTo(cx + 2.6, cy);            // moveTo before arc: avoids a connecting line
      path.arc(cx, cy, 2.6, 0, Math.PI * 2);
    }
    if (state.show.arrows) addArrowToBatch(heads, p, tid, state.frame, colour);
    shown++;
  }
  for (const [colour, path] of dots) { ctx.fillStyle = colour; ctx.fill(path); }
  for (const [colour, path] of heads) { ctx.fillStyle = colour; ctx.fill(path); }

  $('flowStats').textContent =
    `frame ${state.frame} · ${shown} on screen · ${state.visibleTracks.size} of ${Object.keys(d.tracks).length} tracks pass filters`;

  if (state.hover) hoverProbe(ctx, p, dets);
}

function hoverProbe(ctx, p, dets) {
  const d = state.data;
  let best = null, bestDist = 14;
  for (const [tid, x1, , x2, y2] of dets) {
    if (!state.visibleTracks.has(tid)) continue;
    const dx = p.X((x1 + x2) / 2) - state.hover.x, dy = p.Y(y2) - state.hover.y;
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist) { bestDist = dist; best = tid; }
  }
  const tip = $('tip');
  if (best === null) { tip.style.display = 'none'; return; }
  const t = d.tracks[String(best)];
  const cls = d.classOrder[t.c];
  const speedLine = (t.kmh === null || t.kmh === undefined)
    ? `<br>avg ${val(t.sp)} px/s · max ${val(t.spx)} px/s`
    : `<br><strong>${t.kmh} km/h</strong> avg · ${val(t.kmhx)} km/h max`;
  const dirName = (t.dir === null || t.dir === undefined) ? null : (d.compass || [])[t.dir];
  const turnName = (t.turn === null || t.turn === undefined) ? null : (d.turnTypes || [])[t.turn];
  tip.innerHTML = `<strong>#${best}</strong> ${cls}`
    + `<br>colour <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${t.col};vertical-align:0"></span> ${t.col}`
    + speedLine
    + (dirName ? `<br>heading ${dirName}${turnName ? ` · ${turnName}` : ''}` : '')
    + `<br>${(t.dur || 0).toFixed(1)}s tracked`;
  tip.style.display = 'block';
  tip.style.left = (state.hover.cx + 14) + 'px';
  tip.style.top = (state.hover.cy + 12) + 'px';

  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const arr = state.trackPoints.get(best);
  const last = arr[arr.length - 1];
  ctx.arc(p.X(last[1]), p.Y(last[2]), 7, 0, Math.PI * 2);
  ctx.stroke();
}

/* ---------------------------------------------------------------- stats & charts */

function renderStats() {
  const d = state.data;
  const vis = state.visibleTracks;
  let speeds = [], kmhs = [], totalDur = 0;
  for (const tid of vis) {
    const t = d.tracks[String(tid)];
    if (t.sp !== null && t.sp !== undefined) speeds.push(t.sp);
    if (t.kmh !== null && t.kmh !== undefined) kmhs.push(t.kmh);
    totalDur += t.dur || 0;
  }
  // Median, not mean/max: per-track maxima are corrupted by ID-switch "teleports" (a car
  // measured at 1331 km/h in the real data), so a mean over them is not a real statistic.
  const medKmh = kmhs.length ? quantile(kmhs, 0.5) : null;
  const p85Kmh = kmhs.length ? quantile(kmhs, 0.85) : null;
  const avgPx = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0;
  const cards = [
    { k: 'Tracks (filtered)', v: vis.size, n: `of ${Object.keys(d.tracks).length} total` },
    medKmh === null
      ? { k: 'Avg speed', v: avgPx.toFixed(1), n: 'px/s · no telemetry for this combo' }
      : { k: 'Median speed', v: medKmh.toFixed(1), n: 'km/h · geo-referenced from telemetry' },
    p85Kmh === null
      ? { k: 'Tracks w/ speed', v: speeds.length, n: 'pixel-space only' }
      : { k: 'p85 speed', v: p85Kmh.toFixed(1), n: `km/h · from ${kmhs.length} tracks` },
    { k: 'Mean track life', v: vis.size ? (totalDur / vis.size).toFixed(1) + 's' : '—', n: 'higher = less fragmentation' },
    { k: 'Frames', v: val((d.meta || {}).frames_processed), n: `${(d.fps || 0).toFixed(1)} fps` },
  ];
  $('stats').innerHTML = cards.map((c) =>
    `<div class="stat"><div class="k">${c.k}</div><div class="v">${c.v}</div><div class="n">${c.n}</div></div>`).join('');
}

function axis(ctx, w, h, pad) {
  ctx.strokeStyle = GRID; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
}

function renderCharts() {
  renderClassChart();
  renderSpeedChart();
  renderTimeChart();
  renderTable();
  renderLevel4();
  renderLevel3();
}

/* ---------------------------------------------------------------- Level 4 map view
 *
 * Draws the REAL OpenStreetMap road geometry as vectors, with map-matched trajectories,
 * desire lines and queue extents on top. Deliberately not a tile-based map (Leaflet + raster
 * tiles): we already hold the road geometry as coordinates, so drawing it directly means no
 * tile server, no API key, no network at demo time - and the same numbers are exported as
 * standard WGS84 GeoJSON for QGIS/geojson.io, which is where a real GIS workflow consumes it.
 *
 * Coordinates are local ENU metres about the drone's ground point, so +N is straight up and
 * the scale bar is exact rather than a Mercator approximation.
 */
const LANE_COLORS = ['#3987e5', '#eda100', '#1baf7a', '#d55181'];   // lane 1..4
const TILE_PX = 256;
const tileCache = new Map();     // "src/z/x/y" -> HTMLImageElement

/* --- Web Mercator ---------------------------------------------------------------------
 * Tiles are in Web Mercator (EPSG:3857), our trajectories are in local ENU metres. Rather
 * than reprojecting every point every frame, we convert once per redraw: ENU -> lat/lon
 * (pipeline/geodesy.py's inverse) -> Mercator pixel at the chosen zoom. At this latitude and
 * over a 900 m span the Mercator scale factor is effectively constant, so the map stays
 * geometrically honest.
 */
function lonToMercX(lon, z) { return (lon + 180) / 360 * Math.pow(2, z) * TILE_PX; }
function latToMercY(lat, z) {
  const r = lat * Math.PI / 180;
  return (1 - Math.asinh(Math.tan(r)) / Math.PI) / 2 * Math.pow(2, z) * TILE_PX;
}

/** Metres-per-degree at a latitude — same ellipsoidal formula as pipeline/geodesy.py, so the
 *  browser and the Python pipeline agree on where a point is. */
function metresPerDegree(lat) {
  const A = 6378137.0, F = 1 / 298.257223563, E2 = F * (2 - F);
  const r = lat * Math.PI / 180, s2 = Math.sin(r) ** 2;
  const m = A * (1 - E2) / Math.pow(1 - E2 * s2, 1.5);
  const n = A / Math.sqrt(1 - E2 * s2);
  return [Math.PI / 180 * m, Math.PI / 180 * n * Math.cos(r)];
}

function enuToLatLon(e, n, oLat, oLon) {
  const [mLat, mLon] = metresPerDegree(oLat);
  return [oLat + n / mLat, oLon + e / mLon];
}

function getTile(src, z, x, y) {
  const key = `${src}/${z}/${x}/${y}`;
  if (tileCache.has(key)) return tileCache.get(key);
  const img = new Image();
  img.src = `tiles/${key}.png`;
  // Redraw when a tile lands, otherwise the basemap stays blank until the next frame tick.
  img.onload = () => { if (state.rightView === 'map') requestMapDraw(); };
  img.onerror = () => { img.failed = true; };
  tileCache.set(key, img);
  return img;
}

function renderLevel4() {
  const d = state.data;
  const ml = d && d.mapLayers;
  const root = $('l4');
  if (!ml) { root.hidden = true; return; }
  root.hidden = false;

  const L = ml.layers;
  const carried = L.roads.filter((r) => r.carried_traffic).length;
  $('l4note').innerHTML =
    `<span class="pill">${ml.road_data}</span>`
    + `<span class="pill">origin ${ml.origin.lat.toFixed(5)}, ${ml.origin.lon.toFixed(5)}</span>`
    + `<span class="pill">exports level4.geojson (WGS84)</span>`;

  const laned = L.trajectories.filter((t) => t.lane !== null && t.lane !== undefined);
  const tiles = [
    { k: 'Trajectories bound', v: L.trajectories.length, n: `to ${carried} road links` },
    { k: 'Lane assigned', v: laned.length, n: `${Math.round(laned.length / Math.max(1, L.trajectories.length) * 100)}% of bound tracks` },
    { k: 'Desire lines', v: L.desire_lines.length, n: 'link → link movements' },
    { k: 'Queue extents', v: L.queues.length, n: 'drawn along the carriageway' },
    { k: 'Lane metric rows', v: (ml.lane_metrics || []).length, n: 'per link × lane' },
  ];
  $('l4stats').innerHTML = tiles.map((c) =>
    `<div class="stat"><div class="k">${c.k}</div><div class="v">${c.v}</div><div class="n">${c.n}</div></div>`).join('');

  requestMapDraw();
  fillLaneTable(ml);
}

/* View model for the map: centre in local ENU metres plus a ground resolution. Keeping the
 * view in METRES rather than in a tile zoom level is what lets the map zoom continuously
 * (any scale, not just powers of two) while still choosing the sharpest cached tile level
 * for whatever scale it happens to be at - the standard slippy-map decoupling. */
const EARTH_CIRC = 40075016.686;

function groundResolution(lat, z) {
  return EARTH_CIRC * Math.cos(lat * Math.PI / 180) / (TILE_PX * Math.pow(2, z));
}

function autoFitView(canvasW, canvasH, ml) {
  let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
  // Fit to the ROADS, not the trajectories: the network is the frame of reference, and a
  // stray trajectory shouldn't rescale the whole view.
  for (const r of ml.layers.roads) {
    for (const [e, n] of r.points) {
      if (e < minE) minE = e; if (e > maxE) maxE = e;
      if (n < minN) minN = n; if (n > maxN) maxN = n;
    }
  }
  if (!isFinite(minE)) return { cE: 0, cN: 0, mPerPx: 1 };
  const R = 430;      // cached tiles cover ~450 m; don't frame past the imagery
  minE = Math.max(minE, -R); maxE = Math.min(maxE, R);
  minN = Math.max(minN, -R); maxN = Math.min(maxN, R);
  const mPerPx = Math.max((maxE - minE) / (canvasW - 40), (maxN - minN) / (canvasH - 40));
  return { cE: (minE + maxE) / 2, cN: (minN + maxN) / 2, mPerPx };
}

function mapProjector(canvasW, canvasH, ml) {
  if (!ml.layers.roads.length) return null;
  if (!state.mapView) state.mapView = autoFitView(canvasW, canvasH, ml);
  const v = state.mapView;
  const o = ml.origin;

  // ENU metres -> screen pixels. North is up, scale is uniform, so geometry stays true.
  const P = (e, n) => [canvasW / 2 + (e - v.cE) / v.mPerPx,
                       canvasH / 2 - (n - v.cN) / v.mPerPx];
  const invP = (x, y) => [v.cE + (x - canvasW / 2) * v.mPerPx,
                          v.cN - (y - canvasH / 2) * v.mPerPx];

  // Ideal (continuous) zoom for the current ground resolution. The actual tile level is
  // chosen per basemap in drawTiles, because the two sources differ: OSM's standard style
  // stops at z19, Esri imagery goes to z20, and requesting a level a source doesn't have
  // just yields 404s and a blank map.
  const idealZ = Math.log2(groundResolution(o.lat, 0) / v.mPerPx);
  return { P, invP, idealZ, z: Math.round(idealZ), mPerPx: v.mPerPx, origin: o };
}

/** Deepest cached level for this source that isn't finer than what we're drawing at. */
function tileZoomFor(src, idealZ) {
  const idx = state.tileIndex;
  const zooms = (idx && idx.sources[src] && idx.sources[src].zooms) || [17, 18, 19];
  let z = Math.min(...zooms);
  for (const cand of zooms) if (cand <= Math.ceil(idealZ)) z = Math.max(z, cand);
  return z;
}

function drawTiles(ctx, w, h, p, src) {
  if (src === 'none' || !p) return;
  const z = tileZoomFor(src, p.idealZ);
  state._tileZ = z;
  const n = Math.pow(2, z), o = p.origin;
  // Work out which tiles the current view covers by projecting the view corners into
  // Mercator tile space.
  const [wE, sN] = p.invP(0, h);
  const [eE, nN] = p.invP(w, 0);
  const [sLat, wLon] = enuToLatLon(wE, sN, o.lat, o.lon);
  const [nLat, eLon] = enuToLatLon(eE, nN, o.lat, o.lon);
  const tx0 = Math.floor(lonToMercX(wLon, z) / TILE_PX);
  const tx1 = Math.floor(lonToMercX(eLon, z) / TILE_PX);
  const ty0 = Math.floor(latToMercY(nLat, z) / TILE_PX);
  const ty1 = Math.floor(latToMercY(sLat, z) / TILE_PX);

  let drawn = 0;
  for (let X = tx0; X <= tx1; X++) {
    for (let Y = ty0; Y <= ty1; Y++) {
      if (X < 0 || Y < 0 || X >= n || Y >= n) continue;
      const img = getTile(src, z, X, Y);
      if (!img.complete || img.failed || !img.naturalWidth) continue;
      // Project the tile's own NW and SE corners, so the imagery lands in exactly the same
      // frame as the data instead of relying on an assumed pixel scale.
      const nw = tile2LatLon(X, Y, z), se = tile2LatLon(X + 1, Y + 1, z);
      const [ax, ay] = p.P(...latLonToEnu(nw[0], nw[1], o.lat, o.lon));
      const [bx, by] = p.P(...latLonToEnu(se[0], se[1], o.lat, o.lon));
      ctx.drawImage(img, ax, ay, (bx - ax) + 1, (by - ay) + 1);  // +1 hides hairline seams
      drawn++;
    }
  }
  state._tilesDrawn = drawn;
  // Dim the basemap so the data layers read clearly on top of it.
  ctx.fillStyle = 'rgba(10,10,10,0.30)';
  ctx.fillRect(0, 0, w, h);
}

/** Batched form of drawMapArrow - appends to a per-colour Path2D instead of filling. */
// Map marker geometry, in one place so the whole map layer scales together. Deliberately much
// smaller than the flow panel's markers: the flow panel shows one clip's worth of tracks on a
// blank canvas, whereas the map draws the same tracks over satellite imagery at ~1.25 m/px, where
// a 3px dot is ~8 m of real ground - wider than the lane it sits in, so vehicles smear into a
// blob. These are ~1/3 the previous area, small enough that individual vehicles stay separable.
const MAP_DOT_R = 1.8;      // was 3.1
const MAP_RING_R = 3.1;     // was 5.4 - lane-binding ring, must stay clear of the dot
const MAP_ARROW = 4.6;      // was 8
const MAP_DRONE_R = 2.6;    // was 3.5

function addMapArrowToBatch(batches, p, tid, ge, gn, colour) {
  const arr = state.mapTrackPoints.get(tid);
  if (!arr) return;
  const [, e] = pointsUpTo(arr, state.frame, state.frame - TRAIL_FRAMES);
  if (e < 2) return;
  const backPt = arr[Math.max(0, e - 6)];
  if (Math.hypot(ge - backPt[1], gn - backPt[2]) < 0.6) return;   // stationary in metres

  const [x, y] = p.P(ge, gn);
  const [bx, by] = p.P(backPt[1], backPt[2]);
  let sx = x - bx, sy = y - by;
  const slen = Math.hypot(sx, sy) || 1;
  sx /= slen; sy /= slen;

  const size = MAP_ARROW, spread = 0.42;
  const path = batchFor(batches, colour);
  path.moveTo(x + sx * size, y + sy * size);
  path.lineTo(x + (-sx * 0.55 - sy * spread) * size, y + (-sy * 0.55 + sx * spread) * size);
  path.lineTo(x + (-sx * 0.55 + sy * spread) * size, y + (-sy * 0.55 - sx * spread) * size);
  path.closePath();
}

/** Arrowhead on the map, pointing along a track's recent real-world heading. Same shape and
 *  jitter-avoidance as drawArrow() for the flow panel, but reads from mapTrackPoints (ENU
 *  metres) and projects through the map's pan/zoom transform instead of the fixed pixel one. */
function drawMapArrow(ctx, p, tid, ge, gn, colour) {
  const arr = state.mapTrackPoints.get(tid);
  if (!arr) return;
  const [, e] = pointsUpTo(arr, state.frame, state.frame - TRAIL_FRAMES);
  if (e < 2) return;
  const backIdx = Math.max(0, e - 6);
  const backPt = arr[backIdx];
  const groundLen = Math.hypot(ge - backPt[1], gn - backPt[2]);
  if (groundLen < 0.6) return;   // effectively stationary in ground metres: no arrow on noise

  const [x, y] = p.P(ge, gn);
  const [bx, by] = p.P(backPt[1], backPt[2]);
  // Direction from the two already-projected screen points, not from the ENU delta directly:
  // p.P() flips Y (screen-down vs ENU-north-up), so deriving direction post-projection avoids
  // having to re-encode that flip here.
  let sx = x - bx, sy = y - by;
  const slen = Math.hypot(sx, sy) || 1;
  sx /= slen; sy /= slen;

  const size = MAP_ARROW, spread = 0.42;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.moveTo(x + sx * size, y + sy * size);
  ctx.lineTo(x + (-sx * 0.55 - sy * spread) * size, y + (-sy * 0.55 + sx * spread) * size);
  ctx.lineTo(x + (-sx * 0.55 + sy * spread) * size, y + (-sy * 0.55 - sx * spread) * size);
  ctx.closePath();
  ctx.fill();
}

/** Cached ENU bounding box for a polyline, so viewport culling is O(1) per track after the
 *  first frame. Cached on the object itself; the data is immutable once loaded. */
function trackBounds(points) {
  let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
  for (const [e, n] of points) {
    if (e < minE) minE = e; if (e > maxE) maxE = e;
    if (n < minN) minN = n; if (n > maxN) maxN = n;
  }
  return { minE, maxE, minN, maxN };
}

function tile2LatLon(x, y, z) {
  const n = Math.pow(2, z);
  const lon = x / n * 360 - 180;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
  return [lat, lon];
}

function latLonToEnu(lat, lon, oLat, oLon) {
  const [mLat, mLon] = metresPerDegree(oLat);
  return [(lon - oLon) * mLon, (lat - oLat) * mLat];
}

/** Wheel-zoom, drag-pan, double-click to reset. Zoom is anchored at the cursor so the point
 *  under the pointer stays put, which is what makes zooming feel controllable. */
function wireMapInteraction() {
  const c = $('mapCanvas');

  c.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const ml = state.data && state.data.mapLayers;
    if (!ml || !state.mapView) return;
    const r = c.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const v = state.mapView;
    // Ground point under the cursor before the zoom...
    const gE = v.cE + (mx - r.width / 2) * v.mPerPx;
    const gN = v.cN - (my - r.height / 2) * v.mPerPx;
    const factor = Math.exp(ev.deltaY * 0.0016);
    v.mPerPx = Math.min(MAX_M_PER_PX, Math.max(MIN_M_PER_PX, v.mPerPx * factor));
    // ...must remain under the cursor after it.
    v.cE = gE - (mx - r.width / 2) * v.mPerPx;
    v.cN = gN + (my - r.height / 2) * v.mPerPx;
    requestMapDraw();
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  c.addEventListener('mousedown', (ev) => {
    dragging = true; lastX = ev.clientX; lastY = ev.clientY;
    c.style.cursor = 'grabbing';
  });
  window.addEventListener('mouseup', () => { dragging = false; c.style.cursor = 'grab'; });
  window.addEventListener('mousemove', (ev) => {
    if (!dragging || !state.mapView) return;
    const v = state.mapView;
    v.cE -= (ev.clientX - lastX) * v.mPerPx;
    v.cN += (ev.clientY - lastY) * v.mPerPx;
    lastX = ev.clientX; lastY = ev.clientY;
    requestMapDraw();
  });
  c.addEventListener('dblclick', (ev) => {
    // Double-click zooms IN at the cursor, like every real map; the Fit button resets.
    ev.preventDefault();
    const r = c.getBoundingClientRect();
    zoomAbout(ev.clientX - r.left, ev.clientY - r.top, 0.5, r);
  });

  // Touch: one finger pans, two fingers pinch-zoom.
  let touchStart = null;
  c.addEventListener('touchstart', (ev) => {
    const r = c.getBoundingClientRect();
    touchStart = { t: [...ev.touches].map((t) => [t.clientX - r.left, t.clientY - r.top]),
                   view: { ...state.mapView }, r };
  }, { passive: true });
  c.addEventListener('touchmove', (ev) => {
    if (!touchStart || !state.mapView) return;
    ev.preventDefault();
    const r = touchStart.r;
    const pts = [...ev.touches].map((t) => [t.clientX - r.left, t.clientY - r.top]);
    const v = state.mapView, v0 = touchStart.view;
    if (pts.length === 1 && touchStart.t.length === 1) {
      v.cE = v0.cE - (pts[0][0] - touchStart.t[0][0]) * v0.mPerPx;
      v.cN = v0.cN + (pts[0][1] - touchStart.t[0][1]) * v0.mPerPx;
    } else if (pts.length >= 2 && touchStart.t.length >= 2) {
      const d0 = Math.hypot(touchStart.t[0][0] - touchStart.t[1][0],
                            touchStart.t[0][1] - touchStart.t[1][1]);
      const d1 = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]);
      if (d0 > 4 && d1 > 4) {
        v.mPerPx = Math.min(MAX_M_PER_PX, Math.max(MIN_M_PER_PX, v0.mPerPx * (d0 / d1)));
      }
    }
    requestMapDraw();
  }, { passive: false });
  c.addEventListener('touchend', () => { touchStart = null; }, { passive: true });

  // Keyboard, once the map has focus - arrows pan, +/- zoom, 0 fits.
  c.tabIndex = 0;
  c.addEventListener('keydown', (ev) => {
    const v = state.mapView;
    if (!v) return;
    const step = 60 * v.mPerPx;
    const r = c.getBoundingClientRect();
    const keys = { ArrowLeft: [-step, 0], ArrowRight: [step, 0],
                   ArrowUp: [0, step], ArrowDown: [0, -step] };
    if (keys[ev.key]) { v.cE += keys[ev.key][0]; v.cN += keys[ev.key][1]; }
    else if (ev.key === '+' || ev.key === '=') zoomAbout(r.width / 2, r.height / 2, 0.6, r);
    else if (ev.key === '-' || ev.key === '_') zoomAbout(r.width / 2, r.height / 2, 1 / 0.6, r);
    else if (ev.key === '0') state.mapView = null;
    else return;
    ev.preventDefault();
    requestMapDraw();
  });
}

const MIN_M_PER_PX = 0.05, MAX_M_PER_PX = 3.0;

/** Mark the map as needing a redraw. NEVER call drawMap() straight from an event handler:
 *  mousemove alone fires ~120x/s, and combined with the render loop and the zoom animation
 *  that was driving ~180 full map redraws per second. Each redraw blits ~30 tiles and strokes
 *  hundreds of paths, so the main thread had no headroom left to present video frames - which
 *  is exactly why the video stuttered while panning or zooming. Setting a flag lets the single
 *  rAF loop collapse any number of events into one draw per frame. */
function requestMapDraw() {
  state.mapDirty = true;
}

/** Zoom by `factor` keeping the ground point under (mx, my) fixed on screen. The animation is
 *  advanced by the main loop rather than its own requestAnimationFrame, so zooming never adds
 *  a second redraw per frame. */
function zoomAbout(mx, my, factor, rect, animate = true) {
  const v = state.mapView;
  if (!v) return;
  const target = Math.min(MAX_M_PER_PX, Math.max(MIN_M_PER_PX, v.mPerPx * factor));
  const anchor = {
    gE: v.cE + (mx - rect.width / 2) * v.mPerPx,
    gN: v.cN - (my - rect.height / 2) * v.mPerPx,
    mx, my, w: rect.width, h: rect.height,
  };
  if (!animate) {
    applyZoom(target, anchor);
    return;
  }
  state.mapAnim = { from: v.mPerPx, target, anchor, t0: performance.now(), dur: 180 };
  requestMapDraw();
}

function applyZoom(mPerPx, a) {
  const v = state.mapView;
  v.mPerPx = mPerPx;
  v.cE = a.gE - (a.mx - a.w / 2) * mPerPx;
  v.cN = a.gN + (a.my - a.h / 2) * mPerPx;
  requestMapDraw();
}

/** Advances an in-flight zoom animation. Called once per frame from tick(). */
function stepMapAnim(now) {
  const an = state.mapAnim;
  if (!an || !state.mapView) return;
  const k = Math.min(1, (now - an.t0) / an.dur);
  const ease = 1 - Math.pow(1 - k, 3);
  applyZoom(an.from + (an.target - an.from) * ease, an.anchor);
  if (k >= 1) state.mapAnim = null;
}

/** Live map view: basemap + optional network + trails + current-frame vehicles, all synced
 *  to the video clock exactly like the flow panel. */
function drawMap() {
  const d = state.data;
  const ml = d && d.mapLayers;
  if (!ml) return;
  const { ctx, w, h } = fitCanvas($('mapCanvas'));
  ctx.clearRect(0, 0, w, h);
  const p = mapProjector(w, h, ml);
  if (!p) { emptyMsg(ctx, w, h); return; }
  const show = state.mapShow;
  const L = ml.layers;

  drawTiles(ctx, w, h, p, show.base);

  if (show.roads) {
    // Batched by colour+width: ~89 links collapse to a handful of stroke calls. Keyed on
    // both because line width is a canvas state change too, not just colour.
    const roadBatches = new Map();
    for (const r of L.roads) {
      const colour = r.carried_traffic ? 'rgba(120,200,255,0.55)' : 'rgba(255,255,255,0.13)';
      const width = r.highway === 'primary' ? 2.6 : r.highway === 'tertiary' ? 1.8 : 1.0;
      const path = batchFor(roadBatches, `${colour}|${width}`);
      r.points.forEach(([e, n], i) => {
        const [x, y] = p.P(e, n);
        i ? path.lineTo(x, y) : path.moveTo(x, y);
      });
    }
    for (const [key, path] of roadBatches) {
      const [colour, width] = key.split('|');
      ctx.strokeStyle = colour;
      ctx.lineWidth = Number(width);
      ctx.stroke(path);
    }
  }

  if (show.desire) {
    for (const dl of L.desire_lines) {
      if (dl.same_link) continue;     // link-to-itself is "went straight", not a desire line
      ctx.strokeStyle = 'rgba(233,150,45,0.6)';
      ctx.lineWidth = Math.min(9, 1 + Math.sqrt(dl.vehicles) * 1.6);
      const [ax, ay] = p.P(dl.points[0][0], dl.points[0][1]);
      const [bx, by] = p.P(dl.points[1][0], dl.points[1][1]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
  }

  // Queue extents: only those active at the current playback time, so the map shows the
  // queue forming and clearing rather than every queue that ever existed.
  if (show.queues) {
    const t = video.currentTime || 0;
    ctx.lineCap = 'round';
    for (const q of L.queues) {
      const active = show.history || (t >= q.t_start && t < q.t_start + 2.0);
      if (!active) continue;
      ctx.strokeStyle = show.history ? 'rgba(249,96,80,0.22)' : 'rgba(249,96,80,0.75)';
      ctx.lineWidth = 8;
      ctx.beginPath();
      q.points.forEach(([e, n], i) => {
        const [x, y] = p.P(e, n);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
    ctx.lineCap = 'butt';
  }

  // Trails: the path each currently-visible track has taken up to now (or the whole clip in
  // history mode). Uses the same per-track colouring rule as the rest of the dashboard.
  const mf = d.mapFrames;
  const cur = detsAtFrame(mf, state.frame, state.frameStride);
  const visibleNow = new Set(cur.map((r) => r[0]));

  if (show.traj) {
    // Viewport bounds in ENU metres, so off-screen tracks can be rejected with two compares
    // instead of projecting and stroking every point. Matters most when zoomed in, which is
    // exactly when the tile layer is also costing the most.
    const [vw, vs] = p.invP(0, h);
    const [ve, vn] = p.invP(w, 0);
    const padM = 20 * p.mPerPx;
    const trails = new Map();
    for (const t of L.trajectories) {
      if (!state.visibleTracks.has(t.track_id)) continue;
      if (!show.history && !visibleNow.has(t.track_id)) continue;
      const bb = t._bb || (t._bb = trackBounds(t.points));
      if (bb.maxE < vw - padM || bb.minE > ve + padM
          || bb.maxN < vs - padM || bb.minN > vn + padM) continue;
      const path = batchFor(trails, trackColour(t.track_id));
      const pts = t.points;
      for (let i = 0; i < pts.length; i++) {
        const [x, y] = p.P(pts[i][0], pts[i][1]);
        i ? path.lineTo(x, y) : path.moveTo(x, y);
      }
    }
    strokeBatched(ctx, trails, 1.0, show.history ? 0.35 : 0.8);
  }

  // Current-frame vehicle positions - the live part, in lockstep with the video. Batched by
  // colour like everything else: on the 90s clips this is ~90 vehicles per frame.
  let shown = 0;
  const dots = new Map(), rings = new Map(), heads = new Map();
  for (const [tid, e10, n10, lane] of cur) {
    if (!state.visibleTracks.has(tid)) continue;
    const ge = e10 / 10, gn = n10 / 10;
    const [x, y] = p.P(ge, gn);
    const colour = trackColour(tid);

    if (show.arrows) addMapArrowToBatch(heads, p, tid, ge, gn, colour);

    if (show.points) {
      const dot = batchFor(dots, colour);
      dot.moveTo(x + MAP_DOT_R, y);
      dot.arc(x, y, MAP_DOT_R, 0, Math.PI * 2);
      if (lane) {                     // ring marks a vehicle with a confident lane binding
        const ring = batchFor(rings, LANE_COLORS[(lane - 1) % LANE_COLORS.length]);
        ring.moveTo(x + MAP_RING_R, y);
        ring.arc(x, y, MAP_RING_R, 0, Math.PI * 2);
      }
    }
    shown++;
  }
  for (const [colour, path] of heads) { ctx.fillStyle = colour; ctx.fill(path); }
  for (const [colour, path] of dots) { ctx.fillStyle = colour; ctx.fill(path); }
  ctx.lineWidth = 1.0;   // thinner too, so the ring stays a ring and not a second filled blob
  for (const [colour, path] of rings) { ctx.strokeStyle = colour; ctx.stroke(path); }

  // Drone position: origin of the local frame, i.e. where the camera actually was.
  const [dx, dy] = p.P(0, 0);
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(dx, dy, MAP_DRONE_R, 0, Math.PI * 2); ctx.fill();
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('drone', dx + 6, dy - 5);

  drawScaleBar(ctx, w, h, 1 / p.mPerPx);

  // North arrow - meaningful because the frame is geographic, not an image axis.
  ctx.strokeStyle = ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.lineWidth = 1.4;
  const ax = w - 22, ay = 34;
  ctx.beginPath(); ctx.moveTo(ax, ay + 16); ctx.lineTo(ax, ay - 10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ax - 4, ay - 4); ctx.lineTo(ax, ay - 11); ctx.lineTo(ax + 4, ay - 4);
  ctx.closePath(); ctx.fill();
  ctx.fillText('N', ax - 3, ay + 28);

  $('flowStats').textContent =
    `frame ${state.frame} · ${shown} on map · ${p.mPerPx.toFixed(2)} m/px`
    + (state.mapShow.base === 'none' ? '' : ` · tiles z${state._tileZ || '?'}`)
    + ' · scroll to zoom, drag to pan, double-click to fit';
}

function drawScaleBar(ctx, w, h, mPerPxScale) {
  // Pick a round distance that lands near 110 px on screen.
  const target = 110 / mPerPxScale;
  const nice = [10, 20, 25, 50, 100, 200, 250, 500, 1000];
  const m = nice.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
  const px = m * mPerPxScale;
  const x0 = 16, y0 = h - 18;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x0 + px, y0);
  ctx.moveTo(x0, y0 - 4); ctx.lineTo(x0, y0 + 4);
  ctx.moveTo(x0 + px, y0 - 4); ctx.lineTo(x0 + px, y0 + 4);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.8)';
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText(`${m} m`, x0 + px / 2 - 10, y0 - 7);
}

function fillLaneTable(ml) {
  const rows = ml.lane_metrics || [];
  $('laneHint').textContent =
    'Two speeds are reported because they answer different questions: "all" includes stopped '
    + 'vehicles (the lane\'s actual throughput), "running" counts only vehicles that moved '
    + '(excludes parked/stalled vehicles occupying the lane). Quoting only the first '
    + 'overstates lane differences.';
  $('tblLanes').querySelector('tbody').innerHTML = rows.map((r) => `<tr>
      <td>${r.link_name || '(unnamed)'}</td>
      <td>${r.link_id}</td>
      <td><span class="pill" style="background:${LANE_COLORS[(r.lane - 1) % LANE_COLORS.length]};color:#111">lane ${r.lane}</span></td>
      <td>${r.vehicles}</td>
      <td>${val(r.median_kmh)}</td>
      <td>${val(r.running_kmh)}</td>
      <td>${r.stationary_vehicles}</td>
      <td>${splitPills(r.modal_split)}</td>
    </tr>`).join('') || '<tr><td colspan="8" style="color:#898781">No lane-bound vehicles.</td></tr>';
}

/* ---------------------------------------------------------------- Level 3 aggregate */

/* Level 3 summarises across ALL road users in the clip, so it is deliberately NOT filtered by
 * the Level 2 chips: a lane volume or flow-density point computed from a filtered subset would
 * be a misleading traffic statistic. The panel says so explicitly rather than silently
 * ignoring the filter state. */
function renderLevel3() {
  const agg = state.data && state.data.aggregate;
  const root = $('l3');
  if (!agg) { root.hidden = true; return; }
  root.hidden = false;

  const q = agg.quality || {};
  const ax = agg.travel_axis || {};
  $('l3note').innerHTML =
    `<span class="pill">whole clip — not affected by the filters above</span>`
    + `<span class="pill">road axis ${ax.heading || '?'} · ${val(ax.observed_extent_m)} m observed</span>`
    + `<span class="pill">${val(q.rejected_pct, 0)}% speed samples rejected as implausible</span>`;

  const t = agg.totals || {};
  const tiles = [
    { k: 'Vehicles', v: val(t.vehicles), n: `${val(t.with_movement)} with a resolved movement` },
    { k: 'Median speed', v: t.median_kmh === null ? '—' : `${t.median_kmh}`, n: 'km/h · all vehicles, all frames' },
    { k: 'p85 speed', v: t.p85_kmh === null ? '—' : `${t.p85_kmh}`, n: 'km/h · prevailing speed' },
    { k: 'Corridors', v: (agg.corridors || []).length, n: `${(agg.corridors || []).reduce((s, c) => s + c.lanes_observed, 0)} lanes observed` },
    { k: 'Through / turning', v: `${(t.turn_counts || {}).through || 0} / ${((t.turn_counts || {}).left || 0) + ((t.turn_counts || {}).right || 0)}`, n: 'movement split' },
  ];
  $('l3stats').innerHTML = tiles.map((c) =>
    `<div class="stat"><div class="k">${c.k}</div><div class="v">${c.v}</div><div class="n">${c.n}</div></div>`).join('');

  drawSpeedProfile(agg);
  drawFundamental(agg);
  drawTurns(agg);
  drawQueues(agg);
  fillCorridors(agg);
  fillIntervals(agg);
}

function drawSpeedProfile(agg) {
  const { ctx, w, h } = fitCanvas($('chartProfile'));
  ctx.clearRect(0, 0, w, h);
  const bins = agg.speed_profile || [];
  if (!bins.length) { emptyMsg(ctx, w, h); $('hotspotNote').textContent = ''; return; }

  const pad = { l: 30, r: 10, t: 10, b: 24 };
  const max = Math.max(...bins.map((b) => b.p85_kmh)) * 1.1;
  const bw = (w - pad.l - pad.r) / bins.length;
  axis(ctx, w, h, pad);
  const Y = (v) => h - pad.b - (v / max) * (h - pad.t - pad.b);

  // Each bar is tinted by which speed band its value falls in, using the same green->red
  // ramp as the Speed category - so "where is it fast" reads identically here and on the map.
  bins.forEach((b, i) => {
    const x = pad.l + i * bw;
    const bandColour = (kmh) => {
      const k = bandOf(SPEED_BANDS, kmh);
      const idx = SPEED_BANDS.findIndex((s) => s.key === k);
      return SPEED_RAMP[idx < 0 ? 0 : idx];
    };
    ctx.fillStyle = bandColour(b.p85_kmh);
    roundRect(ctx, x + bw * 0.12, Y(b.p85_kmh), bw * 0.34, h - pad.b - Y(b.p85_kmh), 3);
    ctx.globalAlpha = 0.55;   // median sits behind p85 tonally, same hue family
    ctx.fillStyle = bandColour(b.median_kmh);
    roundRect(ctx, x + bw * 0.54, Y(b.median_kmh), bw * 0.34, h - pad.b - Y(b.median_kmh), 3);
    ctx.globalAlpha = 1;
  });

  ctx.fillStyle = MUTED;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText(`${max.toFixed(0)} km/h`, 2, pad.t + 8);
  ctx.fillText(`${bins[0].along_from_m} m`, pad.l, h - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`${bins[bins.length - 1].along_to_m} m`, w - pad.r, h - 8);
  ctx.textAlign = 'left';

  const hs = (agg.hotspots || [])[0];
  $('hotspotNote').textContent = hs
    ? `Highest prevailing speed: ${hs.p85_kmh} km/h between ${hs.along_from_m} m and ${hs.along_to_m} m along the road axis.`
    : '';
}

function drawFundamental(agg) {
  const { ctx, w, h } = fitCanvas($('chartFundamental'));
  ctx.clearRect(0, 0, w, h);
  const pts = (agg.fundamental || []).filter((p) => p.median_kmh !== null);
  if (!pts.length) { emptyMsg(ctx, w, h); return; }

  const pad = { l: 34, r: 12, t: 12, b: 26 };
  const maxD = Math.max(...pts.map((p) => p.density_veh_per_100m)) * 1.15 || 1;
  const maxF = Math.max(...pts.map((p) => p.flow_veh_per_h)) * 1.15 || 1;
  const maxS = Math.max(...pts.map((p) => p.median_kmh)) || 1;
  axis(ctx, w, h, pad);
  ctx.strokeStyle = GRID;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.stroke();

  const X = (d) => pad.l + (d / maxD) * (w - pad.l - pad.r);
  const Y = (f) => h - pad.b - (f / maxF) * (h - pad.t - pad.b);
  for (const p of pts) {
    // Lightness carries median speed - the third variable of the fundamental diagram.
    const lum = 0.30 + 0.55 * (p.median_kmh / maxS);
    ctx.fillStyle = `rgba(57,135,229,${lum.toFixed(2)})`;
    ctx.beginPath();
    ctx.arc(X(p.density_veh_per_100m), Y(p.flow_veh_per_h), 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.fillStyle = MUTED;
  ctx.font = '10px system-ui, sans-serif';
  ctx.save();
  ctx.translate(10, h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('flow (veh/h)', 0, 0);
  ctx.restore();
  ctx.textAlign = 'center';
  ctx.fillText('density (veh / 100 m)', (w + pad.l) / 2, h - 8);
  ctx.textAlign = 'left';
}

function drawTurns(agg) {
  const { ctx, w, h } = fitCanvas($('chartTurns'));
  ctx.clearRect(0, 0, w, h);
  const counts = (agg.totals || {}).turn_counts || {};
  const order = ['through', 'left', 'right', 'u-turn'];
  const colours = { through: '#3987e5', left: '#199e70', right: '#c98500', 'u-turn': '#d55181' };
  const items = order.filter((k) => counts[k]).map((k) => ({ k, v: counts[k] }));
  if (!items.length) { emptyMsg(ctx, w, h); return; }

  const total = items.reduce((s, i) => s + i.v, 0);
  let x = 0;
  const barH = 20, y = 14;
  ctx.font = '10px system-ui, sans-serif';
  for (const it of items) {
    const seg = (it.v / total) * w;
    ctx.fillStyle = colours[it.k];
    // 2px gap between segments so adjacent fills stay separable.
    roundRect(ctx, x, y, Math.max(2, seg - 2), barH, 4);
    if (seg > 46) {
      ctx.fillStyle = '#fff';
      ctx.fillText(`${it.v}`, x + 6, y + 14);
    }
    x += seg;
  }
  let lx = 0;
  for (const it of items) {
    ctx.fillStyle = colours[it.k];
    roundRect(ctx, lx, y + barH + 12, 9, 9, 2);
    ctx.fillStyle = INK2;
    const label = `${it.k} ${(it.v / total * 100).toFixed(0)}%`;
    ctx.fillText(label, lx + 14, y + barH + 20);
    lx += ctx.measureText(label).width + 30;
  }

  const od = agg.od_matrix || {};
  const totalOD = Object.values(od).reduce((s, v) => s + v, 0) || 1;
  $('tblOD').querySelector('tbody').innerHTML = Object.entries(od).slice(0, 12).map(([k, v]) =>
    `<tr><td>${k.replace('->', ' → ')}</td><td>${v}</td><td>${(v / totalOD * 100).toFixed(1)}%</td></tr>`
  ).join('') || '<tr><td colspan="3" style="color:#898781">No resolved movements.</td></tr>';
}

function drawQueues(agg) {
  const { ctx, w, h } = fitCanvas($('chartQueue'));
  ctx.clearRect(0, 0, w, h);
  const qs = agg.queues || [];
  if (!qs.length) {
    emptyMsg(ctx, w, h);
    $('queueNote').textContent = 'No interval had 2+ simultaneously stopped vehicles.';
    return;
  }
  const pad = { l: 34, r: 10, t: 12, b: 24 };
  const max = Math.max(...qs.map((q) => q.queue_length_m)) * 1.15 || 1;
  const bw = (w - pad.l - pad.r) / qs.length;
  axis(ctx, w, h, pad);
  const Y = (v) => h - pad.b - (v / max) * (h - pad.t - pad.b);
  qs.forEach((q, i) => {
    ctx.fillStyle = '#c98500';
    roundRect(ctx, pad.l + i * bw + 2, Y(q.queue_length_m), Math.max(2, bw - 4), h - pad.b - Y(q.queue_length_m), 3);
  });
  ctx.fillStyle = MUTED;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText(`${max.toFixed(0)} m`, 2, pad.t + 8);
  ctx.fillText(`${qs[0].t_start}s`, pad.l, h - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`${qs[qs.length - 1].t_start}s`, w - pad.r, h - 8);
  ctx.textAlign = 'left';
  const peak = qs.reduce((a, b) => (b.queue_length_m > a.queue_length_m ? b : a));
  $('queueNote').textContent =
    `Peak queue ${peak.queue_length_m} m (${peak.stopped_vehicles} stopped vehicles) at t=${peak.t_start}s.`;
}

function splitPills(obj) {
  const e = Object.entries(obj || {});
  if (!e.length) return '—';
  return e.sort((a, b) => b[1] - a[1]).slice(0, 4)
    .map(([k, v]) => `<span class="pill">${k.replace('_candidate', '')} ${v}</span>`).join('');
}

function fillCorridors(agg) {
  const cs = agg.corridors || [];
  $('corridorHint').textContent = agg.lanes_note || '';
  $('tblCorridors').querySelector('tbody').innerHTML = cs.map((c) => {
    const laneCells = c.lanes.length
      ? c.lanes.map((l) => `<span class="pill">${l.vehicles} veh @ ${l.centre_offset_m}m${l.median_kmh === null ? '' : ` · ${l.median_kmh} km/h`}</span>`).join('')
      : '<span style="color:#898781">too sparse to resolve</span>';
    return `<tr>
      <td>${c.approach_from}</td><td>${c.heading_of_travel}</td>
      <td>${c.vehicles}</td><td>${c.share_pct}%</td>
      <td>${val(c.median_kmh)}</td><td>${val(c.p85_kmh)}</td>
      <td>${laneCells}</td>
      <td>${splitPills(c.modal_split)}</td>
      <td>${splitPills(c.turn_counts)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="9" style="color:#898781">No corridor met the minimum traffic share.</td></tr>';
}

function fillIntervals(agg) {
  const rows = agg.intervals || [];
  $('tblIntervals').querySelector('tbody').innerHTML = rows.map((iv) => `<tr>
      <td>${iv.t_start}–${iv.t_end}s</td>
      <td>${iv.unique_vehicles}</td><td>${iv.arrivals}</td>
      <td>${iv.flow_veh_per_h}</td><td>${iv.mean_concurrent}</td>
      <td>${val(iv.median_kmh)}</td><td>${iv.stopped_vehicles}</td>
      <td>${splitPills(iv.by_class)}</td>
    </tr>`).join('') || '<tr><td colspan="8" style="color:#898781">No intervals.</td></tr>';
}

/** Bar chart of visible tracks grouped by the ACTIVE category (not always class) - so the
 *  chart answers the same question the colouring does. */
function renderClassChart() {
  const d = state.data;
  const { ctx, w, h } = fitCanvas($('chartClass'));
  ctx.clearRect(0, 0, w, h);
  $('chartClassTitle').textContent = `Tracked objects by ${CATEGORIES[state.activeCat].label.toLowerCase()}`;

  const counts = new Map();
  for (const tid of state.visibleTracks) {
    const b = state.bucketOf.get(tid);
    if (b !== undefined) counts.set(b, (counts.get(b) || 0) + 1);
  }
  const items = CATEGORIES[state.activeCat].buckets(d)
    .filter((b) => counts.get(b.key))
    .map((b) => ({ label: b.label, colour: b.colour, v: counts.get(b.key) }));
  const pad = { l: 4, r: 34, t: 4, b: 4 };
  if (!items.length) { emptyMsg(ctx, w, h); return; }

  const max = Math.max(...items.map((x) => x.v));
  const rowH = Math.min(26, (h - pad.t - pad.b) / items.length);
  const labelW = 96;
  const barMax = w - pad.l - pad.r - labelW;
  ctx.font = '11px system-ui, sans-serif';
  items.forEach((it, idx) => {
    const y = pad.t + idx * rowH;
    ctx.fillStyle = MUTED;
    ctx.textAlign = 'left';
    // Truncate rather than overlap the bars - band labels can be long ("Moderate 25–80").
    let label = it.label;
    while (label.length > 4 && ctx.measureText(label).width > labelW - 10) {
      label = label.slice(0, -1);
    }
    ctx.fillText(label, pad.l, y + rowH / 2 + 3);
    const bw = Math.max(2, (it.v / max) * barMax);
    ctx.fillStyle = it.colour;
    roundRect(ctx, pad.l + labelW, y + rowH / 2 - 5, bw, 10, 4);
    ctx.fillStyle = INK2;
    ctx.textAlign = 'right';
    ctx.fillText(String(it.v), w - pad.r + 26, y + rowH / 2 + 3);
  });
  ctx.textAlign = 'left';
}

function renderSpeedChart() {
  const d = state.data;
  const { ctx, w, h } = fitCanvas($('chartSpeed'));
  ctx.clearRect(0, 0, w, h);
  const speeds = [];
  for (const tid of state.visibleTracks) {
    const t = d.tracks[String(tid)];
    if (t.sp !== null && t.sp !== undefined) speeds.push(t.sp);
  }
  if (!speeds.length) { emptyMsg(ctx, w, h); return; }

  const pad = { l: 30, r: 8, t: 8, b: 22 };
  const hiCut = quantile(speeds, 0.98) || 1;   // clip the long tail so the bulk stays legible
  const nBins = 26;
  const bins = new Array(nBins).fill(0);
  for (const s of speeds) bins[Math.min(nBins - 1, Math.floor((s / hiCut) * nBins))]++;
  const max = Math.max(...bins);
  const bw = (w - pad.l - pad.r) / nBins;

  axis(ctx, w, h, pad);
  bins.forEach((v, i) => {
    if (!v) return;
    const bh = (v / max) * (h - pad.t - pad.b);
    ctx.fillStyle = '#3987e5';
    roundRect(ctx, pad.l + i * bw + 1, h - pad.b - bh, Math.max(1, bw - 2), bh, 3);
  });
  ctx.fillStyle = MUTED;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('0', pad.l, h - 7);
  ctx.textAlign = 'right';
  ctx.fillText(`${hiCut.toFixed(0)} px/s`, w - pad.r, h - 7);
  ctx.textAlign = 'left';
  ctx.fillText(String(max), 2, pad.t + 8);
}

/** Renders the time series ONCE into an offscreen buffer (per filter change / resize).
 *  The per-frame playback marker then just blits this buffer and draws one line - redrawing
 *  the whole series every animation frame, and rebuilding the legend DOM with it, was a real
 *  per-frame cost for no benefit (the series doesn't change as the video plays). */
function renderTimeChart() {
  const d = state.data;
  const live = $('chartTime');
  const dpr = window.devicePixelRatio || 1;
  const r = live.getBoundingClientRect();
  const w = r.width, h = r.height;
  if (!w || !h) return;

  const buf = state._timeBuf || (state._timeBuf = document.createElement('canvas'));
  buf.width = Math.round(w * dpr);
  buf.height = Math.round(h * dpr);
  const ctx = buf.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const counts = state.perFrameCounts;
  if (!counts || !counts.length) {
    const l = fitCanvas(live);
    l.ctx.clearRect(0, 0, l.w, l.h);
    emptyMsg(l.ctx, l.w, l.h);
    state._timeChartGeom = null;
    return;
  }

  const pad = { l: 28, r: 8, t: 8, b: 20 };
  // Series are per taxonomy class regardless of the active category: this chart is about
  // what's on screen over time, and class is the stable identity for that. It reflects the
  // filter by only showing classes that survive it.
  const present = new Set();
  for (const tid of state.visibleTracks) {
    const t = d.tracks[String(tid)];
    if (t) present.add(t.c);
  }
  const activeClasses = [...present].sort((a, b) => a - b);
  let max = 1;
  for (const row of counts) for (const c of activeClasses) max = Math.max(max, row[c]);

  ctx.strokeStyle = GRID; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();

  const X = (f) => pad.l + (f / Math.max(1, counts.length - 1)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v / max) * (h - pad.t - pad.b);

  ctx.lineWidth = 1.5;
  for (const c of activeClasses) {
    ctx.strokeStyle = CLASS_COLORS[c % CLASS_COLORS.length];
    ctx.beginPath();
    counts.forEach((row, f) => (f ? ctx.lineTo(X(f), Y(row[c])) : ctx.moveTo(X(f), Y(row[c]))));
    ctx.stroke();
  }
  ctx.fillStyle = MUTED;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText(String(max), 2, pad.t + 8);
  ctx.fillText('0s', pad.l, h - 6);
  ctx.textAlign = 'right';
  ctx.fillText(`${((counts.length - 1) / (d.fps || 30)).toFixed(0)}s`, w - pad.r, h - 6);
  ctx.textAlign = 'left';

  $('legendTime').innerHTML = activeClasses.map((c) =>
    `<span><i style="background:${CLASS_COLORS[c % CLASS_COLORS.length]}"></i>${d.classOrder[c].replace('_candidate', '')}</span>`).join('');
  state._timeChartGeom = { X, pad, w, h };
  state._lastMarkerFrame = undefined;   // force one blit so the buffer reaches the screen
}

/** Playback marker on the time chart: blit the cached series, then one vertical line. */
function drawTimeMarker() {
  const g = state._timeChartGeom;
  const buf = state._timeBuf;
  if (!g || !buf) return;
  if (state._lastMarkerFrame === state.frame) return;
  state._lastMarkerFrame = state.frame;

  const { ctx, w, h } = fitCanvas($('chartTime'));
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(buf, 0, 0, w, h);
  const x = g.X(state.frame);
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, g.pad.t); ctx.lineTo(x, h - g.pad.b);
  ctx.stroke();
}

function renderTable() {
  const d = state.data;
  const rows = [...state.visibleTracks]
    .map((tid) => ({ tid, ...d.tracks[String(tid)] }))
    .sort((a, b) => (b.dur || 0) - (a.dur || 0))
    .slice(0, 60);
  const body = rows.map((r) => `<tr>
      <td>#${r.tid}</td>
      <td>${d.classOrder[r.c].replace('_candidate', '')}</td>
      <td><span class="swatch" style="background:${r.col}"></span>${r.col}</td>
      <td>${val(r.sp)}</td><td>${val(r.spx)}</td><td>${(r.dur || 0).toFixed(1)}s</td>
    </tr>`).join('');
  $('tblTracks').querySelector('tbody').innerHTML = body
    || '<tr><td colspan="6" style="color:#898781">No tracks match the current filters.</td></tr>';
}

/* ---------------------------------------------------------------- small helpers */

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.closePath();
  ctx.fill();
}

function quantile(arr, q) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

function emptyMsg(ctx, w, h) {
  ctx.fillStyle = MUTED;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('No data for the current filters', w / 2, h / 2);
  ctx.textAlign = 'left';
}

boot();
