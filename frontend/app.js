/* Glympse app: plain-JS views over the Glyimpse API (see api.js). No demo data, no local model. */
(function () {
  'use strict';

  const api = window.glyimpseAPI;
  const root = document.getElementById('app');
  const LOW = 70, HIGH = 180;            // target range, mg/dL
  const SPIKE_RISE = 50;                 // fallback rise when the server has no personal threshold
  const MIN = 60000, HOUR = 3600000, DAY = 86400000;

  // ---------------------------------------------------------------- helpers
  const IC = {
    today: 'M22 12h-4l-3 9L9 3l-3 9H2',
    cal: 'M8 2v4M16 2v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z',
    bell: 'M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.94 1.94 0 0 0 3.4 0',
    user: 'M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
    drop: 'M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z',
    heart: 'M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z',
    pill: 'm10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7ZM8.5 8.5l7 7',
    pencil: 'M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z',
    list: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
    spark: 'M12 3l1.9 5.8L20 10l-6.1 1.2L12 17l-1.9-5.8L4 10l6.1-1.2z',
    upload: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12',
    server: 'M4 4h16v6H4zM4 14h16v6H4zM8 7h.01M8 17h.01',
    check: 'M20 6 9 17l-5-5',
    x: 'M18 6 6 18M6 6l12 12',
    signout: 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9',
    chevron: 'm9 18 6-6-6-6',
    plus: 'M12 5v14M5 12h14',
    eye: 'M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
    lock: 'M5 11h14a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2zM7 11V7a5 5 0 0 1 10 0v4',
    shield: 'M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1zM9 12l2 2 4-4',
    utensils: 'M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2M7 2v20M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7',
    camera: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2zM12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z'
  };
  const icon = (d, size = 20, sw = 2, color = 'currentColor') =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"></path></svg>`;
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f1 = n => n.toFixed(1);
  const round = n => Math.round(n);

  // Timestamps are the uploaded history's wall-clock (no timezone): parse and format them as UTC
  // so the browser's own timezone/DST never shifts them.
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const ms = ts => Date.parse(String(ts).slice(0, 19) + 'Z');
  const pad = n => String(n).padStart(2, '0');
  const clock = t => { const d = new Date(t), h = d.getUTCHours(); return ((h % 12) || 12) + ':' + pad(d.getUTCMinutes()) + ' ' + (h < 12 ? 'AM' : 'PM'); };
  const shortDate = t => { const d = new Date(t); return MONTHS[d.getUTCMonth()] + ' ' + d.getUTCDate(); };
  const longDate = t => DAYS[new Date(t).getUTCDay()] + ', ' + shortDate(t);
  const hourLabel = t => { const h = new Date(t).getUTCHours(); return ((h % 12) || 12) + (h < 12 ? 'a' : 'p'); };
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage unavailable */ } }
  };

  const NO_FOOD = new Set(['', 'no food', 'no food ingested', 'no intake', 'fasting', 'na', 'n/a', 'nan', 'none', 'null', 'data not available']);
  const FEATURE_NAMES = {
    baseline: 'Your glucose right before the meal', premeal_slope: 'Glucose trend before the meal',
    premeal_mean: 'Average glucose in the 30 min before', premeal_std: 'Glucose variability before the meal',
    hour_sin: 'Time of day', hour_cos: 'Time of day',
    food_sweet: 'Sweet items (sugar, cake, juice, honey…)', food_starch: 'Starchy foods (rice, bread, noodles…)',
    food_fruit: 'Fruit', food_protein: 'Protein foods', food_vegetable: 'Vegetables', food_dairy: 'Dairy',
    food_nuts: 'Nuts and seeds', food_other: 'Other foods', food_amount_grams: 'Portion size (grams of food)',
    food_amount_ml: 'Drink volume (ml)', food_quantity_unparsed: 'Portion size not stated',
    food_logged_now: 'Eating a meal now', minutes_since_logged_food: 'Time since your last meal',
    no_prior_logged_food: 'No earlier meal recorded', sugar_adjustment: 'Sugar in this meal (USDA, added on top of the model)'
  };
  function featureName(f) {
    if (FEATURE_NAMES[f]) return FEATURE_NAMES[f];
    if (f.startsWith('food_logs_')) return 'Meals eaten earlier (' + f.slice(10).replace('_', '–').replace('min', ' min') + ')';
    if (f.startsWith('insulin_')) return 'Insulin recorded in the past 24 h';
    if (f.startsWith('med_')) return 'Medication recorded in the past 24 h (' + f.split('_')[1] + ')';
    if (f.startsWith('csii_')) return 'Pump insulin recorded in the past 24 h';
    return f.replace(/_/g, ' ');
  }

  // ---------------------------------------------------------------- state
  const S = {
    tab: api.token ? 'today' : 'signin',
    user: store.get('glyimpse_user', ''),
    loading: false, health: null, healthError: null,
    status: null, data: null, loadError: null,
    live: null, raw: { readings: [], events: [] }, meals: [], catalog: null,
    builder: { items: [{ name: 'Rice', amount: 100 }], insulin: { name: '', dose: '' }, tablet: { name: '', dose: '' }, search: null },
    pred: null, predKey: null, predBusy: false, predError: null, confirmBusy: false,
    alts: null, altsBusy: false, altsError: null,
    admin: false,
    signinBusy: false, signinError: null, pwShown: false,
    sheetDay: null, toast: null, toastBad: false,
    reminders: store.get('glyimpse_reminders', [true, true, true])
  };
  let toastTimer = null, pollTimer = null;

  // Confirmed meals (server-side), shaped like the forecast summaries the views draw.
  const savedMeals = () => S.meals;
  function shapeMeal(m) {
    const F = summarize(m);
    return { meal_time: m.meal_time, food: m.dietary_intake, medication: m.medication || {}, day: m.day,
      base: F.base, h: F.h.map(z => ({ min: z.min, v: z.v })), verdict: F.verdict, level: F.level };
  }

  // ---------------------------------------------------------------- derived data
  // Everything comes from the replay: days 1-7, then day 8+ only up to the live clock's "now".
  function derive() {
    const L = S.live, R = S.raw;
    const readings = R.readings.map(r => ({ t: ms(r.timestamp), g: r.glucose, rec: r.recorded ?? r.glucose }));
    const first = ms(L.first_reading);
    const dayOf = t => Math.floor((t - first) / DAY) + 1;
    const meals = [], treatments = [];
    for (const e of R.events) {
      const t = ms(e.timestamp);
      if (e.dietary_intake && !NO_FOOD.has(e.dietary_intake.trim().toLowerCase())) meals.push({ t, food: e.dietary_intake, day: dayOf(t) });
      if (e.treatment) treatments.push({ t, text: e.treatment, day: dayOf(t) });
    }
    const last = readings.length ? readings[readings.length - 1] : null;
    const now = ms(L.now);
    return { readings, meals, treatments, first, dayOf, last, now, lastDay: dayOf(now), day8: ms(L.day8_start) };
  }

  // Binary search: readings with a <= t <= b.
  function between(a, b) {
    const r = S.data.readings;
    let lo = 0, hi = r.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (r[mid].t < a) lo = mid + 1; else hi = mid; }
    const out = [];
    for (let i = lo; i < r.length && r[i].t <= b; i++) out.push(r[i]);
    return out;
  }
  function rangeStats(list) {
    if (!list.length) return { avg: null, tir: null };
    const sum = list.reduce((s, r) => s + r.g, 0);
    return { avg: round(sum / list.length), tir: round(100 * list.filter(r => r.g >= LOW && r.g <= HIGH).length / list.length) };
  }
  function dayStats(d) {
    const start = S.data.first + (d - 1) * DAY;
    return { start, ...rangeStats(between(start, start + DAY - 1)) };
  }
  // Real (recorded) CGM only, never the simulated meal effect.
  function measuredPeak(t) {
    const after = between(t + 1, t + 120 * MIN);
    return after.length ? round(Math.max(...after.map(r => r.rec))) : null;
  }
  function measuredRise(t) {
    const before = between(t - 15 * MIN, t), peak = measuredPeak(t);
    return before.length && peak != null ? peak - before[before.length - 1].rec : null;
  }
  // This patient's spike lines from their own days 1-7 meals (server: personal.profile).
  // A forecast's own spike.thresholds wins; older servers send no severe_rise.
  function limits(spike) {
    const T = (spike && spike.thresholds && spike.thresholds.severe_rise != null) ? spike.thresholds
      : S.status && S.status.personal && S.status.personal.thresholds;
    return T || { rise: SPIKE_RISE, severe_rise: 80, personal: false };
  }

  // ---------------------------------------------------------------- forecast summary
  function summarize(r) {
    const base = r.baseline_glucose, t = ms(r.meal_time);
    const h = r.predictions.map(p => ({ min: p.minutes_after_meal, v: p.glucose_mg_dl, d: p.change_mg_dl, top: p.top_features }));
    const peak = h.reduce((a, b) => (b.v > a.v ? b : a));
    const rise = peak.v - base;
    const L = limits(r.spike);
    const level = r.spike ? r.spike.level : rise > L.severe_rise ? 'high' : rise > L.rise ? 'moderate' : 'none';
    return { t, base, h, peak, rise, level, limits: L, verdict: level === 'none' ? 'ok' : 'spike' };
  }
  const VERDICTS = { none: 'Unlikely to spike', moderate: 'Might spike', high: 'Likely to spike' };
  const hzClass = v => (v > HIGH ? 'high' : v < LOW ? 'low' : '');
  function peakText(F) {
    if (F.rise < 8) return `No clear peak expected. Glucose should stay near ${round(F.base)} mg/dL.`;
    return `Peaks at ~${round(F.peak.v)} mg/dL around ${clock(F.t + F.peak.min * MIN)}, ${F.peak.min} min after eating.`;
  }
  // Measured CGM within 5 minutes of a time, when the uploaded history covers it (e.g. dataset patients' day 8).
  function measuredAt(t) {
    if (!S.data) return null;
    const near = between(t - 5 * MIN, t + 5 * MIN);
    if (!near.length) return null;
    return near.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a)).g;
  }
  const horizonTiles = F => {
    const actual = F.h.map(z => measuredAt(F.t + z.min * MIN));
    return `<div class="horizons">${F.h.map((z, i) =>
      `<div class="hz"><div class="l">+${z.min} min</div><div class="v ${hzClass(z.v)}">${round(z.v)}</div><div class="c">${clock(F.t + z.min * MIN)}</div>${
        actual[i] == null ? '' : `<div class="c">actual <b class="${hzClass(actual[i])}">${round(actual[i])}</b></div>`}</div>`).join('')}</div>${
      actual.some(v => v != null) ? '<p class="meta">Top number: forecast. "Actual": the CGM reading recorded at that time in this history.</p>' : ''}`;
  };

  // ---------------------------------------------------------------- loading
  async function refreshHealth() {
    try { S.health = await api.health(); S.healthError = null; } catch (e) { S.health = null; S.healthError = e; }
  }

  function applyLive(state, append) {
    const { readings, events, ...clockState } = state;
    S.live = clockState;
    S.raw = append ? { readings: S.raw.readings.concat(readings), events: S.raw.events.concat(events) } : { readings, events };
    S.data = derive();
  }

  async function loadMeals() {
    try { S.meals = (await api.meals()).meals.map(shapeMeal); } catch (e) { S.meals = []; }
  }

  async function loadAll() {
    S.loading = true; S.loadError = null; render();
    await refreshHealth();
    try {
      const me = await api.me();
      S.admin = !!me.admin;
      S.user = me.username; store.set('glyimpse_user', me.username);
      S.status = await api.getHistory();
      applyLive(await api.live(), false);
      if (!S.catalog) S.catalog = await api.catalog();
      await loadMeals();
    } catch (e) {
      S.status = null; S.data = null; S.live = null;
      if (e.kind === 'auth') return;           // onUnauthorized already moved us to sign-in
      if (e.kind !== 'no_history') S.loadError = e;
    }
    S.loading = false; render();
    schedulePoll();
  }

  // The replay produces one CGM reading per `seconds_per_reading`; poll at that pace (1-5 s).
  function schedulePoll() {
    clearTimeout(pollTimer);
    if (!S.live || !api.token) return;
    const wait = Math.min(5, Math.max(1, S.live.seconds_per_reading)) * 1000;
    pollTimer = setTimeout(poll, wait);
  }

  async function poll() {
    try {
      const lastStamp = S.raw.readings.length ? S.raw.readings[S.raw.readings.length - 1].timestamp : null;
      const before = S.live.now;
      applyLive(await api.live(lastStamp), true);
      if (S.live.now !== before) liveUpdate();
    } catch (e) {
      if (e.kind === 'auth') return;
    }
    schedulePoll();
  }

  // Re-render views that show live data; on the meal builder only patch the clock text so dropdowns keep focus.
  function liveUpdate() {
    if (S.tab === 'log') {
      root.querySelectorAll('[data-live-clock]').forEach(el => { el.textContent = liveClockText(); });
      syncLogButtons();
    } else if (S.tab !== 'signin' && S.tab !== 'reminders') {
      render();
    }
  }

  async function control(action, extra) {
    try {
      applyLive(await api.control(action, extra), false);
      if (action === 'restart') { await loadMeals(); S.pred = null; S.alts = null; }
      render(); schedulePoll();
    } catch (e) { toast(e.message, true); }
  }

  // ---------------------------------------------------------------- actions
  function toast(message, bad = false) {
    S.toast = message; S.toastBad = bad; render();
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { S.toast = null; render(); }, 3200);
  }

  function go(tab) {
    S.tab = tab; S.sheetDay = null; render(); window.scrollTo(0, 0);
  }

  async function signIn(form) {
    const username = form.username.value.trim(), password = form.password.value;
    if (!/^[a-zA-Z0-9_.@-]{3,80}$/.test(username)) { S.signinError = 'Enter your username (letters, numbers, . _ @ -).'; return render(); }
    if (!password) { S.signinError = 'Enter your password.'; return render(); }
    S.signinBusy = true; S.signinError = null; render();
    try {
      await api.signIn(username, password);
      S.user = username; store.set('glyimpse_user', username);
      S.signinBusy = false; S.tab = 'today';
      await loadAll();
    } catch (e) {
      S.signinBusy = false; S.signinError = e.message; render();
    }
  }

  function resetSession() {
    clearTimeout(pollTimer);
    Object.assign(S, { status: null, data: null, live: null, raw: { readings: [], events: [] }, meals: [],
      pred: null, predKey: null, predError: null,
      alts: null, altsError: null, admin: false,
      loadError: null, sheetDay: null, tab: 'signin' });
  }

  async function signOut() {
    try { await api.signOut(); } catch (e) { /* token is cleared locally either way */ }
    resetSession(); S.signinError = null; render();
  }

  // ---------------------------------------------------------------- meal builder
  const foodInfo = name => (S.catalog ? S.catalog.foods.find(f => f.name === name) : null);
  const unitOf = name => (foodInfo(name) || { unit: 'g' }).unit;
  // Grams actually fed to the model/macros: USDA-resolved grams when a food was looked up,
  // otherwise the catalog portion amount. USDA weights are always grams.
  const itemGrams = i => (i.grams != null ? +i.grams : +i.amount || 0);
  const itemUnit = i => (i.usda ? 'g' : unitOf(i.name));
  function mealText() {
    return S.builder.items.filter(i => i.name && itemGrams(i))
      .map(i => `${i.name} ${itemGrams(i)} ${itemUnit(i)}`).join(', ');
  }
  function mealRequest(extra = {}) {
    const b = S.builder, req = { dietary_intake: mealText(), ...extra };
    const foods = resolvedFoods().map(f => ({ name: f.name, grams: f.grams, carbs_g: f.carbs_g, sugar_g: f.sugar_g }));
    if (foods.length) req.foods = foods;
    if (b.insulin.name && b.insulin.dose) req.insulin = { name: b.insulin.name, dose: +b.insulin.dose };
    if (b.tablet.name && b.tablet.dose) req.tablet = { name: b.tablet.name, dose: +b.tablet.dose };
    return req;
  }

  // ---- USDA nutrients: scale per-100 g to the selected portion. null stays "unknown", never 0.
  const NUTRIENT_FIELDS = ['kcal', 'carbs_g', 'protein_g', 'fat_g', 'fiber_g', 'sugar_g'];
  function scaleMacros(per100g, grams) {
    const factor = grams / 100, out = {};
    for (const key of NUTRIENT_FIELDS) {
      const v = per100g ? per100g[key] : null;
      out[key] = (v == null) ? null : (key === 'kcal' ? Math.round(v * factor) : Math.round(v * factor * 10) / 10);
    }
    return out;
  }
  const showNum = v => (v == null ? 'unknown' : v);
  // The clean confirmed-food object the rest of the app can consume.
  function resolvedFoods() {
    return S.builder.items.filter(i => i.name && itemGrams(i)).map(i => {
      const grams = itemGrams(i), m = i.usda ? scaleMacros(i.usda.per_100g, grams) : {};
      return { name: i.name, grams,
        kcal: m.kcal ?? null, carbs_g: m.carbs_g ?? null, protein_g: m.protein_g ?? null,
        fat_g: m.fat_g ?? null, fiber_g: m.fiber_g ?? null, sugar_g: m.sugar_g ?? null,
        fdcId: i.usda ? i.usda.fdcId : null, usda_description: i.usda ? i.usda.description : null };
    });
  }

  // ---- Look up USDA nutrients/portions for a row (by classifier label when known, else typed text).
  async function lookupFood(i) {
    const item = S.builder.items[i];
    if (!item) return;
    const query = item.label || item.name;
    if (!query || !query.trim()) { item.usda = null; item.grams = null; return render(); }
    item.lookupBusy = true; item.lookupError = null; render();
    try {
      const rec = item.fdcId ? await api.foodNutrientsById(item.fdcId) : await api.foodNutrients(query);
      item.usda = rec; item.per100 = false;
      if (rec.default_portion) { item.portionKey = '0'; item.grams = rec.default_portion.grams; }
      else { item.portionKey = 'per100'; item.grams = 100; }
    } catch (e) {
      item.usda = null; item.grams = null; item.lookupError = e;
    }
    item.lookupBusy = false; render();
  }

  function setPortion(i, key) {
    const item = S.builder.items[i];
    item.portionKey = key;
    if (key === 'per100') item.grams = 100;
    else if (key === 'custom') item.grams = +item.customGrams || item.grams || 100;
    else if (item.usda && item.usda.portions[+key]) item.grams = item.usda.portions[+key].grams;
    render();
  }

  // ---- Photo identification for a row.
  function pickPhoto(i) {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = 'image/*';
    input.setAttribute('capture', 'environment');
    input.onchange = () => { if (input.files && input.files[0]) identifyPhoto(i, input.files[0]); };
    input.click();
  }
  async function identifyPhoto(i, file) {
    const item = S.builder.items[i];
    if (!item) return;
    item.identifyBusy = true; item.identifyError = null; item.predictions = null; render();
    try {
      const form = new FormData(); form.append('image', file);
      const res = await api.identifyFood(form);
      item.predictions = res.predictions; item.needsConfirmation = res.needs_confirmation;
      // High confidence: preselect the top guess but keep the chips visible for correction.
      if (res.predictions && res.predictions[0]) await choosePrediction(i, res.predictions[0]);
    } catch (e) {
      item.identifyError = e;
    }
    item.identifyBusy = false; render();
  }
  async function choosePrediction(i, pred) {
    const item = S.builder.items[i];
    item.name = pred.display_name; item.label = pred.label; item.selectedLabel = pred.label; item.fdcId = null;
    await lookupFood(i);
  }

  // ---- "None of these, search" fallback: free-text USDA search, pick a result into the row.
  async function runFoodSearch(i, query) {
    S.builder.search = { row: i, query, busy: true, error: null, results: null }; render();
    try {
      const res = await api.foodSearch(query);
      if (S.builder.search && S.builder.search.query === query) S.builder.search.results = res.results;
    } catch (e) {
      if (S.builder.search) S.builder.search.error = e;
    }
    if (S.builder.search) S.builder.search.busy = false;
    render();
  }
  async function pickSearchResult(i, fdcId, description) {
    const item = S.builder.items[i];
    item.name = description; item.label = null; item.selectedLabel = null;
    item.fdcId = fdcId; item.predictions = null; item.needsConfirmation = false;
    S.builder.search = null;
    await lookupFood(i);
  }
  const formKey = () => JSON.stringify(mealRequest());
  function medicationText(med) {
    return Object.values(med || {}).join(' · ');
  }
  // Alternatives come back as text ("Rice 50 g, cake 40 g"): turn them into builder rows.
  function itemsFromText(text) {
    return text.split(',').map(s => s.trim().match(/^(.*?)\s+(\d+(?:\.\d+)?)\s*(g|ml)$/i)).filter(Boolean).map(m => {
      const known = S.catalog.foods.find(f => f.name.toLowerCase() === m[1].toLowerCase());
      return { name: known ? known.name : m[1], amount: +m[2] };
    });
  }

  async function predict() {
    const req = mealRequest();
    if (!req.dietary_intake) { S.predError = { message: 'Add at least one food with a portion.' }; return render(); }
    S.predBusy = true; S.predError = null; S.pred = null; S.alts = null; S.altsError = null; S.altsOpen = false; render();
    try {
      S.pred = await api.predict(req);
      S.predKey = formKey();
    } catch (e) {
      S.predError = e;
    }
    S.predBusy = false; render();
    if (S.pred && S.pred.spike && S.pred.spike.level !== 'none') loadAlternatives(S.pred.meal_time, S.predKey);
  }

  async function loadAlternatives(mealTime, key) {
    S.altsBusy = true; render();
    try {
      const result = await api.alternatives(mealRequest({ meal_time: mealTime }));
      if (S.predKey === key) { S.alts = result; S.altsOpen = result.alternatives.length > 0; }
    } catch (e) {
      if (S.predKey === key) S.altsError = e;
    }
    S.altsBusy = false; render();
  }

  function tryAlternative(text) {
    S.altsOpen = false;
    if (S.tab !== 'log') go('log');
    const items = itemsFromText(text);
    if (items.length) S.builder.items = items;
    predict();
  }

  // "I ate this": same meal at the same live time as the what-if, now saved to compare with what happens.
  async function confirmMeal() {
    if (!S.pred || S.predKey !== formKey()) return;
    S.confirmBusy = true; render();
    try {
      await api.predict(mealRequest({ meal_time: S.pred.meal_time, confirm: true }));
      // Expose the confirmed meal as one clean object for the rest of the app to plug into.
      const foods = resolvedFoods(), sum = k => foods.reduce((a, f) => a + (f[k] ?? 0), 0);
      window.glympseConfirmedMeal = { meal_time: S.pred.meal_time, dietary_intake: mealText(), foods,
        total: { kcal: sum('kcal'), carbs_g: sum('carbs_g'), protein_g: sum('protein_g'),
                 fat_g: sum('fat_g'), fiber_g: sum('fiber_g'), sugar_g: sum('sugar_g') } };
      await loadMeals();
      applyLive(await api.live(), false);
      S.pred = null; S.predKey = null; S.alts = null; S.altsError = null;
      S.confirmBusy = false;
      go('today');
      toast('Meal confirmed. Watch the forecast against your real glucose.');
    } catch (e) {
      S.confirmBusy = false; S.predError = e; render();
    }
  }

  function syncLogButtons() {
    const save = root.querySelector('[data-action="save-meal"]');
    if (save) save.disabled = !S.pred || S.predKey !== formKey() || S.confirmBusy;
    const stale = root.querySelector('[data-stale]');
    if (stale) stale.hidden = !S.pred || S.predKey === formKey();
  }

  // ---------------------------------------------------------------- views
  const brand = (size) => `<div class="brand"${size ? ` style="font-size:${size}px"` : ''}><span class="g">gly</span><span class="m">mpse</span></div>`;
  const roleChip = () => `<span class="outline-chip">${S.admin ? 'ADMIN' : 'PATIENT'}</span>`;
  const greeting = () => { const h = new Date().getHours(); return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening'; };

  function errorBox(e, actions = '') {
    if (!e) return '';
    const titles = { offline: 'SERVER OFFLINE', auth: 'SIGN IN AGAIN', no_history: 'NO HISTORY YET', invalid: "CAN'T USE THAT INPUT",
      model_missing: 'MODEL MISSING', too_large: 'FILE TOO LARGE', forbidden: 'NOT ALLOWED' };
    let action = actions;
    if (e.kind === 'offline') action = `<button class="link-btn" data-action="reload">Try again</button>`;
    return `<div class="alert" role="alert"><strong>${titles[e.kind] || 'SOMETHING WENT WRONG'}</strong>${esc(e.message)}${action ? `<div>${action}</div>` : ''}</div>`;
  }

  const SPEED_LABELS = { 300: 'Real time (5 min)', 10: '1 reading / 10 s', 3: '1 reading / 3 s', 1: '1 reading / 1 s' };
  function liveClockText() {
    const L = S.live;
    if (!L) return '';
    const t = ms(L.now);
    const state = L.finished ? ' · recording ended' : L.paused ? ' · paused' : '';
    return `Day ${L.day} · ${longDate(t)} · ${clock(t)}${state}`;
  }
  function liveBar() {
    const L = S.live;
    if (!L) return '';
    return `<section class="live-bar" aria-label="Replay clock">
  <div class="live-now"><span class="dot${L.paused || L.finished ? ' off' : ''}"></span><span data-live-clock>${esc(liveClockText())}</span></div>
  <div class="live-controls">
    <button class="link-btn" data-action="live-toggle" ${L.finished ? 'disabled' : ''}>${L.paused ? 'Play' : 'Pause'}</button>
    <select class="input sm" data-action="live-speed" aria-label="Replay speed">${L.speeds.map(s =>
      `<option value="${s}" ${s === L.seconds_per_reading ? 'selected' : ''}>${SPEED_LABELS[s] || s + ' s'}</option>`).join('')}</select>
    <button class="link-btn ghost" data-action="live-restart">Restart day 8</button>
  </div>
  <p class="hint">Days 1–7 are this patient's history. From day 8 the patient's real CGM plays back one reading per 5 minutes. A confirmed meal adds its predicted effect on top (shown as simulated); what-if checks never change it.</p>
</section>`;
  }

  function viewSignin() {
    return `<main class="signin">
  <div style="position:relative;width:300px;max-width:100%;aspect-ratio:1;margin:0 auto">
    <svg viewBox="0 0 300 300" style="position:absolute;inset:0;width:100%;height:100%" aria-hidden="true">
      <circle cx="150" cy="150" r="140" style="fill:#121416;stroke:#2a2d31;stroke-width:3"></circle>
      <circle cx="150" cy="150" r="140" transform="rotate(-80 150 150)" style="fill:none;stroke:#4fdfdc;stroke-width:3;stroke-dasharray:250 630"></circle>
      <circle cx="150" cy="150" r="140" transform="rotate(-125 150 150)" style="fill:none;stroke:#ff8b85;stroke-width:3;stroke-dasharray:50 830"></circle>
      <circle cx="125" cy="288" r="7" style="fill:#c4a5f5"></circle>
    </svg>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center">${brand(62)}</div>
  </div>
  <div class="eyebrow" style="text-align:center;margin-top:28px">YOUR HEALTH, IN THE PICTURE</div>
  <h1 class="title" style="text-align:center;margin:8px 0 0">Hey, welcome back.</h1>
  <p style="text-align:center;font-size:17px;color:var(--muted-2);margin:10px 0 0">A little glimpse into a healthier you.</p>
  <form id="signin-form" novalidate>
    <label for="gl-user" class="signin-label" style="margin-top:30px">Username</label>
    <div class="field-box">${icon(IC.user)}
      <input id="gl-user" name="username" autocomplete="username" autocapitalize="off" spellcheck="false" placeholder="Your username" value="${esc(S.user)}" required>
    </div>
    <label for="gl-pw" class="signin-label">Password</label>
    <div class="field-box">${icon(IC.lock)}
      <input id="gl-pw" name="password" type="${S.pwShown ? 'text' : 'password'}" autocomplete="current-password" placeholder="Your password" required>
      <button type="button" data-action="toggle-pw" aria-label="${S.pwShown ? 'Hide password' : 'Show password'}">${icon(IC.eye)}</button>
    </div>
    <p style="font-size:14px;color:var(--muted-2);line-height:1.5;margin:16px 0 0">Admin access only. Accounts are managed by your Glyimpse administrator.</p>
    ${S.signinError ? `<div class="alert" role="alert" style="margin-top:16px">${esc(S.signinError)}</div>` : S.healthError ? errorBox(S.healthError) : ''}
    <button class="btn lg" type="submit" ${S.signinBusy ? 'disabled' : ''}>${S.signinBusy ? 'Signing in…' : 'Sign in ' + icon(IC.chevron, 18, 2.2)}</button>
  </form>
</main>`;
  }

  function phaseCard() {
    const st = S.status;
    if (!st) {
      return `<section class="card mt" aria-label="Prediction status"><div class="card-head"><span class="card-title">Upload your history to start</span><span class="muted" style="font-size:14px">Day 0</span></div>
  <div class="segs">${'<span></span>'.repeat(8)}</div><div class="seg-labels"><span>Days 1–7 · learning</span><span>Day 8 · predictions</span></div>
  <p>Glympse learns from 7 days of CGM readings and meals. Upload your glucose history CSV in Profile to begin.</p></section>`;
    }
    const counts = st.daily_cgm_counts;
    const segs = counts.map(c => `<span style="background:${c >= 230 ? 'var(--teal)' : c > 0 ? 'var(--teal-dash)' : 'var(--border)'}"></span>`).join('') +
      `<span style="background:${st.ready_for_day8 ? 'var(--lilac)' : 'var(--border)'}"></span>`;
    let body;
    if (st.ready_for_day8) {
      body = `The model learned from your first 7 days. From ${longDate(ms(st.day8_start))}, ${clock(ms(st.day8_start))} it predicts: check any meal before you eat it to see whether it might spike your glucose.`;
    } else {
      const short = counts.map((c, i) => [i + 1, c]).filter(([, c]) => c < 230);
      body = short.length
        ? `Not ready yet: ${short.map(([d, c]) => `day ${d} has ${c} of 230 CGM readings`).join(', ')}. Upload more history to unlock predictions.`
        : `Upload history through the end of day 7 (${shortDate(ms(st.day8_start))}, ${clock(ms(st.day8_start))}) to unlock predictions.`;
    }
    const day = S.data ? S.data.lastDay : 0;
    return `<section class="card mt" aria-label="Prediction status"><div class="card-head"><span class="card-title">${st.ready_for_day8 ? 'Predictions are on' : 'Learning your patterns'}</span>
  <span class="muted" style="font-size:14px">${st.ready_for_day8 ? 'Day ' + day : 'Day ' + Math.min(day, 7) + ' of 7'}</span></div>
  <div class="segs">${segs}</div><div class="seg-labels"><span>Days 1–7 · learning</span><span>Day 8 · predictions</span></div>
  <p>${esc(body)}</p></section>`;
  }

  function todayChart() {
    const D = S.data;
    if (!D || !D.last) return `<div class="empty-chart">No CGM readings on this account yet.</div>`;
    // 8 hours back and 2 hours ahead of now, so a confirmed meal's forecast shows before reality catches up.
    const end0 = Math.floor(D.last.t / HOUR) * HOUR + 2 * HOUR, start = end0 - 10 * HOUR;
    const pts = between(start, end0);
    if (pts.length < 2) return `<div class="empty-chart">Not enough recent readings to draw a curve.</div>`;
    const W = 360, H = 150;
    const lo = Math.min(60, ...pts.map(p => p.g - 5)), hi = Math.max(200, ...pts.map(p => p.g + 10));
    const x = t => (t - start) / (end0 - start) * W, y = v => H - 8 - (v - lo) / (hi - lo) * (H - 16);
    // Break the line where the CGM has gaps longer than 15 minutes.
    const segments = [[]];
    pts.forEach((p, i) => { if (i && p.t - pts[i - 1].t > 15 * MIN) segments.push([]); segments[segments.length - 1].push([x(p.t), y(p.g)]); });
    const paths = segments.filter(s => s.length > 1).map(s => {
      const line = smooth(s);
      return `<path d="${line}L${f1(s[s.length - 1][0])} ${H}L${f1(s[0][0])} ${H}Z" fill="url(#gl-fill)"></path><path d="${line}" style="fill:none;stroke:#5ee3a0;stroke-width:3;stroke-linecap:round"></path>`;
    }).join('');
    // Where a confirmed meal's effect was added, keep the recording visible underneath.
    const simulated = pts.filter(p => Math.abs(p.g - p.rec) >= 0.5);
    const recorded = simulated.length > 1 ? `<path d="${smooth(simulated.map(p => [x(p.t), y(p.rec)]))}" style="fill:none;stroke:#9aa3ab;stroke-width:1.5;stroke-dasharray:2 3"></path>` : '';
    // Confirmed meals: dashed forecast from the pre-meal glucose through +120 min, drawn over the real curve.
    const forecasts = savedMeals().map(m => [[ms(m.meal_time), m.base], ...m.h.map(z => [ms(m.meal_time) + z.min * MIN, z.v])])
      .filter(line => line[line.length - 1][0] >= start && line[0][0] <= end0)
      .map(line => `<polyline points="${line.map(([t, v]) => f1(x(t)) + ',' + f1(y(Math.min(hi, Math.max(lo, v))))).join(' ')}" style="fill:none;stroke:#4fdfdc;stroke-width:2;stroke-dasharray:5 4"></polyline>`).join('');
    const dots = [...D.meals.filter(m => m.t >= start && m.t <= end0).map(m => [m.t, '#ff8b85']),
      ...savedMeals().map(m => [ms(m.meal_time), '#4fdfdc']).filter(([t]) => t >= start && t <= end0)]
      .map(([t, color]) => { const near = between(t - 10 * MIN, t + 10 * MIN)[0]; return near ? `<circle cx="${f1(x(t))}" cy="${f1(y(near.g))}" r="5" style="fill:#0f1113;stroke:${color};stroke-width:2.5"></circle>` : ''; }).join('');
    const labels = Array.from({ length: 11 }, (_, i) => `<span>${hourLabel(start + i * HOUR)}</span>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Glucose from ${clock(start)} to ${clock(end0)}">
  <defs><linearGradient id="gl-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5ee3a0" stop-opacity="0.35"></stop><stop offset="1" stop-color="#5ee3a0" stop-opacity="0"></stop></linearGradient></defs>
  <rect x="0" y="${f1(y(HIGH))}" width="${W}" height="${f1(y(LOW) - y(HIGH))}" fill="#5ee3a0" fill-opacity="0.05"></rect>
  <line x1="0" x2="${W}" y1="${f1(y(HIGH))}" y2="${f1(y(HIGH))}" stroke="#ff8b85" stroke-opacity="0.45" stroke-dasharray="4 5"></line>
  <line x1="0" x2="${W}" y1="${f1(y(LOW))}" y2="${f1(y(LOW))}" stroke="#c4a5f5" stroke-opacity="0.45" stroke-dasharray="4 5"></line>
  <line x1="${f1(x(D.last.t))}" x2="${f1(x(D.last.t))}" y1="0" y2="${H}" stroke="#9aa3ab" stroke-opacity="0.5"></line>
  ${recorded}${paths}${forecasts}${dots}</svg><div class="chart-labels">${labels}</div>
  <div class="legend sm"><span><i style="background:#5ee3a0"></i>${simulated.length ? 'Your CGM, simulated after a confirmed meal' : 'Your real CGM'}</span>${simulated.length ? '<span><i style="background:#9aa3ab"></i>What was actually recorded</span>' : ''}<span><i style="background:#4fdfdc"></i>Forecast for a confirmed meal</span><span><i style="background:#ff8b85"></i>Meal in your record</span></div></div>`;
  }

  function smooth(pts) {
    let d = 'M' + f1(pts[0][0]) + ' ' + f1(pts[0][1]);
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      d += 'C' + f1(p1[0] + (p2[0] - p0[0]) / 6) + ' ' + f1(p1[1] + (p2[1] - p0[1]) / 6) + ' ' + f1(p2[0] - (p3[0] - p1[0]) / 6) + ' ' +
        f1(p2[1] - (p3[1] - p1[1]) / 6) + ' ' + f1(p2[0]) + ' ' + f1(p2[1]);
    }
    return d;
  }

  function mealRows(day) {
    const D = S.data;
    const L = limits();
    const logged = D.meals.filter(m => m.day === day).map(m => {
      const peak = measuredPeak(m.t), rise = measuredRise(m.t);
      return { t: m.t, name: m.food, sub: peak ? `measured peak ${peak} mg/dL${rise != null ? `, rise ${rise >= 0 ? '+' : '−'}${Math.abs(round(rise))}` : ''}` : 'peak not measured',
        chip: rise == null ? ['neutral', 'Logged'] : rise > L.severe_rise ? ['spike', 'Big spike'] : rise > L.rise ? ['warn', 'Spike'] : ['ok', 'Steady'] };
    });
    const planned = savedMeals().filter(m => m.day === day).map(m => {
      const peak = Math.max(...m.h.map(z => z.v)), actual = measuredPeak(ms(m.meal_time));
      const med = medicationText(m.medication);
      return { t: ms(m.meal_time), name: 'You confirmed: ' + m.food + (med ? ' · ' + med : ''),
        sub: `forecast peak ~${round(peak)} mg/dL (rise ${m.base != null ? '+' + round(peak - m.base) : '?'})${actual != null ? ` · recorded peak so far ${actual}` : ''}`,
        chip: [m.level === 'high' ? 'spike' : m.level === 'moderate' ? 'warn' : 'ok', VERDICTS[m.level]] };
    });
    return [...logged, ...planned].sort((a, b) => a.t - b.t);
  }

  function viewToday() {
    const D = S.data, st = S.status;
    const head = `<header class="top"><div><div class="greet">${greeting()}, ${esc(S.user || 'there')}</div>${brand()}</div>
  <button class="icon-btn" data-action="go" data-tab="reminders" aria-label="Reminders">${icon(IC.bell)}<span class="badge"></span></button></header>`;
    if (S.loading && !D) return `<main>${head}<div class="info pulse" role="status">Loading your history…</div></main>`;

    const online = !S.healthError;
    let ringPart = '', statusRow = '', stats = '', meals = '';
    if (D && D.last) {
      const recent = between(D.last.t - DAY, D.last.t), s24 = rangeStats(recent);
      const inR = s24.tir / 100, C = 2 * Math.PI * 156;
      const inLen = Math.max(0, C * inR - 14), hiLen = Math.max(0, C * (1 - inR) - 26);
      const cur = round(D.last.g);
      const last30 = between(D.last.t - 30 * MIN, D.last.t);
      const slope = last30.length > 1 ? (last30[last30.length - 1].g - last30[0].g) / ((last30[last30.length - 1].t - last30[0].t) / MIN) : 0;
      const [trend, cls] = cur > HIGH ? ['↑ Above your range.', 'up'] : cur < LOW ? ['↓ Below your range.', 'down']
        : slope >= 1 ? ['↗ Rising.', 'up'] : slope <= -1 ? ['↘ Falling.', 'down'] : ['→ Looking steady.', ''];
      statusRow = `<div class="status-row"><span class="live"><span class="dot${online ? '' : ' off'}"></span>${online ? 'Live CGM · every 5 min' : 'Server offline'}</span>
        <span class="pill${cur > HIGH ? ' high' : cur < LOW ? ' low' : ''}">${cur > HIGH ? 'Above range' : cur < LOW ? 'Below range' : 'In range'}</span></div>`;
      ringPart = `<div class="ring"><svg viewBox="0 0 360 360" aria-hidden="true">
      <circle cx="180" cy="180" r="156" style="fill:none;stroke:#2a2d31;stroke-width:22"></circle>
      <circle cx="180" cy="180" r="128" style="fill:none;stroke:#3a3f44;stroke-width:3;stroke-dasharray:0 22;stroke-linecap:round"></circle>
      ${inLen ? `<circle cx="180" cy="180" r="156" transform="rotate(-90 180 180)" style="fill:none;stroke:#4fdfdc;stroke-width:22;stroke-linecap:round;stroke-dasharray:${f1(inLen)} ${f1(C)}"></circle>` : ''}
      ${hiLen ? `<circle cx="180" cy="180" r="156" transform="rotate(${f1(-90 + 360 * inR + 4)} 180 180)" style="fill:none;stroke:#ff8b85;stroke-width:22;stroke-linecap:round;stroke-dasharray:${f1(hiLen)} ${f1(C)}"></circle>` : ''}
      <circle cx="180" cy="24" r="11" style="fill:#4fdfdc;stroke:#0f1113;stroke-width:4"></circle></svg>
    <div class="ring-center"><div style="font-size:15px;color:var(--muted)">Latest · ${longDate(D.last.t)}</div>
      <div class="ring-value">${cur}</div><div class="ring-unit">mg/dL</div>
      <div class="ring-trend ${cls}">${trend}</div><div style="font-size:13px;color:var(--muted);margin-top:6px">${Math.abs(D.last.g - D.last.rec) >= 0.5
        ? `Simulated from your confirmed meal · recorded ${round(D.last.rec)}` : `Last reading · ${clock(D.last.t)}`}</div></div></div>
  <div class="legend"><span><i style="background:var(--teal)"></i>In range (70–180), last 24 h</span><span><i style="background:var(--coral)"></i>Outside range</span></div>`;
      const mealCount = D.meals.filter(m => m.t > D.last.t - DAY).length;
      stats = `<div class="divider"></div><div class="stats3">
    <div class="stat"><div class="num">${s24.tir}%</div><div class="lbl">Time in range</div></div>
    <div class="stat"><div class="num">${s24.avg}</div><div class="lbl">Avg glucose</div></div>
    <div class="stat"><div class="num">${mealCount}</div><div class="lbl">Meals, 24 h</div></div></div>`;
      const rows = mealRows(D.lastDay);
      meals = `<h2 class="section">Meals on day ${D.lastDay}</h2>${rows.length ? rows.map(m => `<div class="row"><span class="thumb">${esc(m.name.trim()[0] || '?').toUpperCase()}</span>
        <div class="grow"><div class="name">${esc(m.name)}</div><div class="sub">${clock(m.t)} · ${esc(m.sub)}</div></div><span class="chip ${m.chip[0]}">${m.chip[1]}</span></div>`).join('')
        : '<p class="empty">No meals recorded on this day yet.</p>'}`;
    } else {
      statusRow = `<div class="status-row"><span class="live"><span class="dot${online ? '' : ' off'}"></span>${online ? 'Connected to server' : 'Server offline'}</span><span class="pill low">No history</span></div>`;
    }

    const sources = `<h2 class="section">Your data</h2>
  <div class="source"><span class="ic">${icon(IC.drop, 18)}</span><div class="grow"><div class="name">Your CGM record</div>
    <div class="sub">${st ? `Days 1–7 learned · ${shortDate(ms(st.first_reading))} – ${shortDate(ms(st.day8_start))}` : 'No record on this account'}</div></div>
    ${st ? `<span class="synced">${icon(IC.check, 15, 2.2)}Synced</span>` : ''}</div>
  <div class="source"><span class="ic">${icon(IC.server, 18)}</span><div class="grow"><div class="name">Glyimpse server</div>
    <div class="sub">${online ? (S.health && S.health.shared_model_available ? 'Prediction model ready' : 'Prediction model missing') : 'Unreachable'}</div></div>
    ${online && S.health && S.health.shared_model_available ? `<span class="synced">${icon(IC.check, 15, 2.2)}Online</span>` : `<span class="synced bad">${icon(IC.x, 15, 2.2)}${online ? 'No model' : 'Offline'}</span>`}</div>`;

    return `<main data-screen-label="Today" class="${ringPart ? 'has-glance' : ''}">${head}${statusRow}${liveBar()}
  <div class="today-ring">${ringPart}</div><div class="today-alert">${errorBox(S.loadError)}</div>
  <div class="today-phase">${phaseCard()}</div>
  <section class="today-chart" aria-label="Glucose around now">${todayChart()}</section>
  <div class="today-stats">${stats}</div>
  <div class="today-bottom">
    <div class="today-meals">${meals}</div>
    <div class="today-aside">${sources}
      <div class="divider" style="margin:22px 0"></div>
      <button class="cta" data-action="go" data-tab="log"><span class="ic">${icon(IC.utensils)}</span><span><b>What if I eat…?</b><small>Pick foods and medication, see the predicted spike before you eat</small></span></button>
      <p class="note">Wellness insight, not medical advice.</p>
    </div>
  </div></main>`;
  }

  function predictionCard() {
    if (S.predBusy) return `<div class="info pulse" role="status">Checking this meal against your first 7 days…</div>`;
    if (S.predError) return errorBox(S.predError);
    if (!S.pred) return '';
    const r = S.pred, F = summarize(r);
    const cls = F.level === 'high' ? ' spike' : F.level === 'moderate' ? ' moderate' : '';
    const recent = r.recent_cgm || [];
    const lastRecent = recent[recent.length - 1];
    const top = (F.peak.top || []).map(f => `<li><span>${esc(featureName(f.feature))}</span><b class="${f.contribution_mg_dl > 0 ? 'up' : 'down'}">${f.contribution_mg_dl > 0 ? '+' : '−'}${Math.abs(f.contribution_mg_dl).toFixed(0)} mg/dL</b></li>`).join('');
    const med = medicationText(r.medication);
    return `<div class="pred${cls}" role="status">
  <div class="eyebrow-sm">What if · ${r.personalized ? `personalized from ${r.personal_training_meals} of your meals` : 'population model (fewer than 5 usable meals in your last 7 days)'}</div>
  <div class="verdict">${VERDICTS[F.level]}</div>
  <p class="meta">${esc(r.dietary_intake)}${med ? ' · ' + esc(med) : ''} · at ${clock(F.t)}</p>
  ${hypothesisChart(F)}
  ${spikeWhy(r.spike, F)}${sugarNote(r)}
  <div class="facts"><span>Peak <strong>~${round(F.peak.v)} mg/dL</strong></span><span>Rise <strong>${F.rise >= 0 ? '+' : '−'}${Math.abs(round(F.rise))}</strong></span><span>Start <strong>${round(F.base)}</strong></span></div>
  ${horizonTiles(F)}
  <p class="peak">${esc(peakText(F))}</p>
  <p class="meta">Starting point: ${recent.length} CGM readings from your history${lastRecent ? `, latest ${round(lastRecent.glucose)} mg/dL at ${clock(ms(lastRecent.timestamp))}` : ''}. Day ${r.day} of your recording.</p>
  ${top ? `<div class="eyebrow-sm" style="margin-top:14px">What moved this forecast most at +${F.peak.min} min</div><ul class="drivers">${top}</ul>` : ''}
  <p class="meta" style="margin-top:10px">These show how the model reached its number, not proven causes.</p>
  <p data-stale class="meta" hidden style="color:var(--yellow)">You changed the meal; check the forecast again before confirming.</p>
</div>${alternativesCard(F)}`;
  }

  // Predicted (hypothesis) curve after the meal, next to the real readings of the hour before it.
  function hypothesisChart(F) {
    const W = 340, H = 150, a = F.t - 60 * MIN, b = F.t + 125 * MIN;
    const real = between(a, Math.min(b, S.data.now));
    const fc = [[F.t, F.base], ...F.h.map(z => [F.t + z.min * MIN, z.v])];
    const L = F.limits, line = F.base + L.rise, big = F.base + L.severe_rise;
    const values = [...real.map(r => r.g), ...fc.map(p => p[1]), big + 10];
    const lo = Math.min(60, ...values) - 5, hi = Math.max(...values) + 15;
    const x = t => (t - a) / (b - a) * W, y = v => H - 18 - (v - lo) / (hi - lo) * (H - 26);
    const realLine = real.length > 1 ? `<path d="${smooth(real.map(r => [x(r.t), y(r.g)]))}" style="fill:none;stroke:#5ee3a0;stroke-width:2.5"></path>` : '';
    const fcLine = `<path d="${smooth(fc.map(([t, v]) => [x(t), y(v)]))}" style="fill:none;stroke:${F.level === 'none' ? '#4fdfdc' : '#ff8b85'};stroke-width:3;stroke-dasharray:6 4"></path>`;
    const dots = fc.slice(1).map(([t, v]) => `<circle cx="${f1(x(t))}" cy="${f1(y(v))}" r="3.5" fill="${v > line ? '#ff8b85' : '#4fdfdc'}"></circle>`).join('');
    const ticks = [-60, 0, 30, 60, 90, 120].map(m => `<text x="${f1(x(F.t + m * MIN))}" y="${H - 3}" text-anchor="${m < 0 ? 'start' : m === 120 ? 'end' : 'middle'}">${m === 0 ? 'meal' : (m > 0 ? '+' : '') + m}</text>`).join('');
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted glucose after this meal" style="width:100%;margin-top:12px;font-size:10px;fill:var(--muted-2)">
  <line x1="0" x2="${W}" y1="${f1(y(big))}" y2="${f1(y(big))}" stroke="#ff8b85" stroke-opacity="0.75" stroke-dasharray="2 4"></line>
  <text x="3" y="${f1(y(big) - 4)}" style="fill:#ff8b85">Big spike${L.personal ? ' (90th pct)' : ''}: +${round(L.severe_rise)} → ${round(big)}</text>
  <line x1="0" x2="${W}" y1="${f1(y(line))}" y2="${f1(y(line))}" stroke="#ffc46b" stroke-opacity="0.6" stroke-dasharray="4 5"></line>
  <text x="3" y="${f1(y(line) + 11)}" style="fill:#ffc46b">Spike${L.personal ? ' (75th pct)' : ''}: +${round(L.rise)} → ${round(line)}</text>
  <line x1="${f1(x(F.t))}" x2="${f1(x(F.t))}" y1="0" y2="${H - 14}" stroke="#9aa3ab" stroke-opacity="0.4"></line>
  ${realLine}${fcLine}${dots}${ticks}</svg>
  <div class="legend sm"><span><i style="background:#5ee3a0"></i>Real CGM</span><span><i style="background:${F.level === 'none' ? '#4fdfdc' : '#ff8b85'}"></i>Predicted if you eat this</span></div>
  ${limitsNote(L)}`;
  }

  function limitsNote(L) {
    const who = esc(S.user || 'you');
    return L.personal
      ? `<p class="meta" style="margin-top:8px">Spike lines for ${who}, from ${L.meals} of your meals in days 1–7: <b>spike</b> when the rise is above <b>+${round(L.rise)} mg/dL</b> (75th percentile of your meal rises), <b>big spike</b> above <b>+${round(L.severe_rise)} mg/dL</b> (90th percentile).</p>`
      : `<p class="meta" style="margin-top:8px">Default spike lines: rise above +${round(L.rise)} mg/dL is a spike, above +${round(L.severe_rise)} a big spike. Restart the server to use your own (75th / 90th percentile of your meals).</p>`;
  }

  function spikeWhy(spike, F) {
    if (!spike) return '';
    const T = F.limits, rise = `+${Math.max(0, round(F.rise))}`;
    const whose = T.personal ? `your usual rise after meals (from ${T.meals} meals in days 1–7)` : 'the default limit (too few clean meals in days 1–7)';
    if (spike.level === 'none') return `<p class="meta" style="margin-top:10px">Predicted rise ${rise} mg/dL stays under your spike line of +${round(T.rise)}, based on ${whose}.</p>`;
    const sweet = spike.sweet_items || [];
    const drivers = spike.food_drivers.map(d => featureName(d.feature).replace(/ \(.*\)$/, '').toLowerCase());
    const cause = sweet.length && spike.food_drivers.some(d => d.feature === 'food_sweet' || d.feature === 'sugar_adjustment')
      ? `The sweet items (${sweet.map(esc).join(', ')}) push this forecast up the most.`
      : drivers.length ? `Biggest food effects in this forecast: ${esc(drivers.join(', '))}.` : '';
    const limit = spike.level === 'high' ? `your big-spike line of +${round(T.severe_rise)}` : `your spike line of +${round(T.rise)}`;
    return `<p class="tip">Predicted rise ${rise} mg/dL is above ${limit}, based on ${whose}. ${cause}</p>`;
  }

  // The model barely reacts to sugar, so the server adds grams of sugar × this patient's rise per gram of carbs.
  function sugarNote(r) {
    const s = r.sugar_adjustment;
    if (!s || !s.sugar_g) return '';
    const peak = Math.max(...s.mg_dl);
    return `<p class="meta" style="margin-top:10px">Includes a sugar adjustment: ${s.sugar_g.toFixed(0)} g sugar × ${s.mg_dl_per_g} mg/dL per g
    (${s.personal ? 'your own carb response' : 'default carb response'}) adds up to +${round(peak)} mg/dL on top of the model.
    Sugar from ${s.items.some(i => i.source === 'usda') ? 'USDA' : 'a built-in USDA-based table'}.</p>`;
  }

  const altChip = level => level === 'none' ? ['ok', 'Avoids spike'] : level === 'moderate' ? ['warn', 'Smaller spike'] : ['spike', 'Still spikes'];

  // Inline: only a button that reopens the "Try instead" pop-up.
  function alternativesCard(F) {
    if (F.level === 'none') return '';
    if (S.altsBusy) return `<div class="info pulse" role="status">Looking for changes that keep you in range…</div>`;
    if (S.altsError) return errorBox(S.altsError);
    if (!S.alts) return '';
    const rows = S.alts.alternatives;
    if (!rows.length) return `<div class="info">No simple change to try: add amounts (g or ml) to each food so the model can compare smaller portions.</div>`;
    return `<button class="btn secondary" style="margin-top:14px" data-action="open-alts">Try instead: ${rows.length} way${rows.length === 1 ? '' : 's'} to avoid this spike</button>`;
  }

  function viewAlts() {
    if (!S.altsOpen || !S.alts || !S.alts.alternatives.length) return '';
    const saved = new Set(diet().map(d => d.dietary_intake));
    return `<div class="scrim" data-action="close-alts"></div>
  <div class="sheet" role="dialog" aria-modal="true" aria-label="Try instead">
    <button class="grab" data-action="close-alts" aria-label="Close"><span></span></button>
    <h2>Try instead</h2>
    <p class="muted" style="margin:0 0 6px">Same time, same starting glucose as ${esc(S.pred ? S.pred.dietary_intake : 'your meal')}. Checked against your spike line of +${round(limits(S.pred && S.pred.spike).rise)} mg/dL.</p>
    ${S.alts.alternatives.map(a => {
      const peak = a.spike.peak_glucose_mg_dl, delta = a.peak_change_mg_dl, chip = altChip(a.spike.level);
      const isSaved = saved.has(a.dietary_intake);
      return `<div class="row" style="align-items:flex-start"><div class="grow"><div class="name">${esc(a.label)}</div>
      <div class="sub" style="overflow-wrap:anywhere">${esc(a.dietary_intake)}</div>
      <div class="sub">Peak ~${round(peak)} mg/dL, rise +${Math.max(0, round(a.spike.rise_mg_dl))} (${delta <= 0 ? '−' : '+'}${Math.abs(round(delta))} vs. planned)</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
        <button class="link-btn" data-action="try-alt" data-value="${esc(a.dietary_intake)}">Use this meal</button>
        <button class="link-btn ghost" data-action="save-diet" data-value="${esc(a.dietary_intake)}" ${isSaved ? 'disabled' : ''}>${isSaved ? 'In your recommended diet' : 'Add to my recommended diet'}</button></div></div>
      <span class="chip ${chip[0]}">${chip[1]}</span></div>`;
    }).join('')}
    <p class="note">Model forecasts only. Not medical advice; never change insulin or medication based on this.</p>
  </div>`;
  }

  // Recommended diet: alternatives the patient saved, kept per account in this browser.
  const dietKey = () => 'glyimpse_diet_' + (S.user || '');
  const diet = () => store.get(dietKey(), []);
  function saveDiet(text) {
    const a = S.alts && S.alts.alternatives.find(x => x.dietary_intake === text);
    if (!a || diet().some(d => d.dietary_intake === text)) return;
    store.set(dietKey(), [...diet(), { dietary_intake: text, label: a.label, instead_of: S.pred ? S.pred.dietary_intake : '',
      peak: round(a.spike.peak_glucose_mg_dl), rise: round(a.spike.rise_mg_dl), level: a.spike.level, saved_at: new Date().toISOString() }]);
    toast('Added to your recommended diet.');
  }
  function removeDiet(text) { store.set(dietKey(), diet().filter(d => d.dietary_intake !== text)); render(); }

  function dietCard() {
    const list = diet();
    return `<section class="card mt" aria-label="Your recommended diet"><div class="card-head"><span class="card-title">Your recommended diet</span>
    <span class="muted" style="font-size:14px">${list.length ? list.length + ' saved' : 'from "Try instead"'}</span></div>
  ${list.length ? list.map(d => {
      const chip = altChip(d.level);
      return `<div class="row" style="align-items:flex-start"><div class="grow"><div class="name" style="overflow-wrap:anywhere">${esc(d.dietary_intake)}</div>
      <div class="sub">${esc(d.label)}${d.instead_of ? ' · instead of ' + esc(d.instead_of) : ''}</div>
      <div class="sub">Forecast peak ~${d.peak} mg/dL, rise +${Math.max(0, d.rise)}</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
        <button class="link-btn" data-action="try-alt" data-value="${esc(d.dietary_intake)}">Check it now</button>
        <button class="link-btn ghost" data-action="remove-diet" data-value="${esc(d.dietary_intake)}">Remove</button></div></div>
      <span class="chip ${chip[0]}">${chip[1]}</span></div>`;
    }).join('') : '<p class="empty">When a meal might spike, pick a "Try instead" option and add it here.</p>'}
  </section>`;
  }

  const options = (values, selected, label) => values.map(v =>
    `<option value="${esc(v)}" ${String(v) === String(selected) ? 'selected' : ''}>${esc(label ? label(v) : v)}</option>`).join('');

  const photoErrorText = e => (e && e.status === 503)
    ? "Photo recognition isn't set up on this server yet — type the food name instead."
    : ((e && e.message) || 'Could not identify the photo.');
  const lookupErrorText = e => (e && e.status === 404)
    ? 'No USDA match for that food — try the search below.'
    : ((e && e.message) || 'USDA lookup failed.');

  function portionSelect(item, i) {
    if (item.usda) {
      const opts = item.usda.portions.map((p, idx) =>
        `<option value="${idx}" ${String(item.portionKey) === String(idx) ? 'selected' : ''}>${esc(p.label)} (${p.grams} g)</option>`).join('');
      const extra = `<option value="per100" ${item.portionKey === 'per100' ? 'selected' : ''}>Per 100 g</option>
        <option value="custom" ${item.portionKey === 'custom' ? 'selected' : ''}>Custom…</option>`;
      return `<select class="input portion" data-item="${i}" data-part="portion" aria-label="Portion ${i + 1}">${opts}${extra}</select>`;
    }
    const info = foodInfo(item.name);
    const portions = info ? [...new Set([...info.portions, item.amount].filter(Boolean))].sort((p, q) => p - q) : (item.amount ? [item.amount] : []);
    const opts = options(portions, item.portionKey === 'custom' ? '' : item.amount, p => `${p} ${unitOf(item.name)}`);
    return `<select class="input portion" data-item="${i}" data-part="amount" aria-label="Portion ${i + 1}">${opts}<option value="custom" ${item.portionKey === 'custom' ? 'selected' : ''}>Custom…</option></select>`;
  }

  function nutritionCard(item) {
    const u = item.usda, dp = u.default_portion;
    const grams = item.per100 ? 100 : itemGrams(item);
    const m = scaleMacros(u.per_100g, grams);
    const standard = dp
      ? `USDA standard portion: <b>${esc(dp.label)} (${dp.grams} g)</b>`
      : `USDA lists no standard portion for this food — showing <b>per 100 g</b>`;
    const macros = `${showNum(m.kcal)} kcal · Carbs ${showNum(m.carbs_g)} g · Protein ${showNum(m.protein_g)} g · Fat ${showNum(m.fat_g)} g`;
    const notes = [];
    if (u.kcal_estimated) notes.push('Calories estimated from macros.');
    if (u.missing && u.missing.length) notes.push('USDA is missing: ' + u.missing.join(', ') + '.');
    return `<div class="card" style="margin-top:8px;padding:14px 16px">
    <div class="small" style="color:var(--teal);font-weight:700">${standard}</div>
    <div style="margin-top:4px;font-size:13px;color:var(--muted-2)">Compare this to what's on your plate.</div>
    <div style="margin-top:10px;font-size:15px">${item.per100 ? '<span class="muted small">per 100 g · </span>' : `<span class="muted small">for ${grams} g · </span>`}${macros}</div>
    <div class="row" style="margin-top:10px;gap:12px;align-items:center">
      <button type="button" class="link-btn ghost" data-action="toggle-per100" data-index="${item._i}">${item.per100 ? 'Show selected portion' : 'Show per 100 g'}</button>
      <span class="small muted">Source: USDA ${esc(u.source || '')}</span></div>
    ${notes.length ? `<div class="hint">${esc(notes.join(' '))}</div>` : ''}
  </div>`;
  }

  function searchBox(i) {
    const s = S.builder.search;
    return `<div class="card" style="margin-top:8px;padding:14px 16px">
    <div class="field-box"><input type="text" data-search-input="${i}" placeholder="Search USDA foods…" value="${esc(s.query || '')}" aria-label="Search USDA foods">
      <button type="button" data-action="run-search" data-index="${i}" aria-label="Search">${icon(IC.list, 18)}</button></div>
    ${s.busy ? '<div class="hint" role="status">Searching USDA…</div>' : ''}
    ${s.error ? `<div class="hint" style="color:var(--coral)">${esc(s.error.message || 'Search failed')}</div>` : ''}
    ${s.results ? (s.results.length
      ? `<div style="margin-top:8px">${s.results.map(r => `<button type="button" class="link-btn ghost" style="display:block;width:100%;text-align:left;height:auto;margin-top:6px;padding:10px 12px" data-action="pick-search" data-index="${i}" data-fdcid="${r.fdcId}" data-desc="${esc(r.description)}">${esc(r.description)} <span class="small muted">· ${esc(r.source)}</span></button>`).join('')}</div>`
      : '<div class="hint">No matches. Try a different word.</div>') : ''}
  </div>`;
  }

  function foodDetail(item, i) {
    let html = '';
    if (item.identifyBusy) html += `<div class="hint" role="status">Identifying photo…</div>`;
    if (item.identifyError) html += `<div class="hint" style="color:var(--coral)">${esc(photoErrorText(item.identifyError))}</div>`;
    if (item.predictions && item.predictions.length) {
      const top3 = item.predictions.slice(0, 3);
      html += `<div class="chips" role="group" aria-label="Photo guesses">
      ${top3.map(p => `<button type="button" data-action="choose-pred" data-index="${i}" data-label="${esc(p.label)}" data-name="${esc(p.display_name)}" ${item.selectedLabel === p.label ? 'aria-pressed="true" style="border-color:var(--teal);color:var(--teal)"' : ''}>${esc(p.display_name)} · ${Math.round(p.confidence * 100)}%</button>`).join('')}
      <button type="button" class="link-btn ghost" data-action="search-toggle" data-index="${i}">None of these, search</button></div>
      ${item.needsConfirmation ? `<div class="hint">Not fully sure — tap the right food, or search.</div>` : ''}`;
    }
    if (item.portionKey === 'custom') {
      html += `<div class="food-row" style="margin-top:6px"><input class="input" type="number" min="1" max="5000" step="1" inputmode="numeric" data-item="${i}" data-part="custom-grams" placeholder="Grams on your plate" value="${esc(item.customGrams ?? item.grams ?? '')}" aria-label="Custom grams ${i + 1}"></div>`;
    }
    if (item.lookupBusy) html += `<div class="hint" role="status">Looking up USDA nutrition…</div>`;
    else if (item.lookupError) html += `<div class="hint" style="color:var(--coral)">${esc(lookupErrorText(item.lookupError))} You can still set grams manually.</div>`;
    else if (item.usda) { item._i = i; html += nutritionCard(item); }
    if (S.builder.search && S.builder.search.row === i) html += searchBox(i);
    return html;
  }

  function foodRow(item, i) {
    const cam = `Identify food ${i + 1} from a photo`;
    return `<div class="food-row">
    <input class="input" list="glympse-foods" data-item="${i}" data-part="name" autocomplete="off" placeholder="Type a food, or use the camera" value="${esc(item.name || '')}" aria-label="Food ${i + 1}">
    <button class="icon-btn sm" data-action="identify-photo" data-index="${i}" aria-label="${cam}" title="${cam}" ${item.identifyBusy ? 'disabled' : ''}>${icon(IC.camera, 18)}</button>
    ${portionSelect(item, i)}
    <button class="icon-btn sm" data-action="remove-food" data-index="${i}" aria-label="Remove food ${i + 1}" ${S.builder.items.length < 2 ? 'disabled' : ''}>${icon(IC.x, 16)}</button>
  </div>${foodDetail(item, i)}`;
  }

  function medRow(kind) {
    const b = S.builder[kind], C = S.catalog;
    const list = kind === 'insulin' ? C.insulins : C.tablets;
    const chosen = list.find(m => m.name === b.name);
    const doses = chosen ? (kind === 'insulin' ? chosen.doses_iu : chosen.doses_mg) : [];
    const unit = kind === 'insulin' ? 'IU' : 'mg';
    return `<div class="food-row">
    <select class="input" data-med="${kind}" data-part="name" aria-label="${kind === 'insulin' ? 'Insulin' : 'Tablet'}">
      <option value="">${kind === 'insulin' ? 'No insulin' : 'No tablet'}</option>${options(list.map(m => m.name), b.name)}</select>
    <select class="input portion" data-med="${kind}" data-part="dose" aria-label="Dose" ${chosen ? '' : 'disabled'}>
      ${chosen ? options(doses, b.dose, d => `${d} ${unit}`) : '<option>Dose</option>'}</select>
  </div>`;
  }

  function viewLog() {
    const st = S.status, D = S.data, L = S.live;
    if (!st || !D || !S.catalog) return `<main data-screen-label="What if">${S.loading ? '<div class="info pulse" role="status">Loading…</div>' : errorBox(S.loadError) || '<p class="empty">No CGM record on this account.</p>'}</main>`;
    const before8 = L && D.now < D.day8;
    const note = !st.ready_for_day8 ? `<div class="banner"><div class="eyebrow">NOT ENOUGH DATA</div><p>Forecasts need each of the first 7 days to have at least 230 CGM readings.</p></div>`
      : before8 ? `<div class="banner"><div class="eyebrow">DAY 8 HASN'T STARTED</div><p>Forecasts start at ${clock(D.day8)} on day 8.</p></div>` : '';
    const canConfirm = S.pred && S.predKey === formKey() && !S.confirmBusy;
    const forecast = predictionCard();
    return `<main data-screen-label="What if">
  <div class="log-intro">
    <div class="eyebrow">WHAT IF I EAT…</div><h1 class="title">Predict before you eat.</h1>${note}
    <div class="banner"><div class="eyebrow">MEAL TIME = NOW</div><p data-live-clock>${esc(liveClockText())}</p></div>
  </div>
  <div class="log-form">
  <div class="field"><label>Foods and portions</label>
    <datalist id="glympse-foods">${S.catalog.foods.map(f => `<option value="${esc(f.name)}"></option>`).join('')}</datalist>
    ${S.builder.items.map(foodRow).join('')}
    <button class="link-btn" style="margin-top:10px" data-action="add-food">${icon(IC.plus, 15)} Add food</button>
    <div class="hint">Type a food or tap ${icon(IC.camera, 13)} to identify it from a photo. We pull USDA's standard portion so you can compare it to your plate.</div></div>
  <div class="field"><label>Insulin taken with this meal</label>${medRow('insulin')}</div>
  <div class="field"><label>Tablet taken with this meal</label>${medRow('tablet')}
    <div class="hint">Medication is fed to the model exactly like doses in your record. It's for exploring the forecast, not dosing advice.</div></div>
  <button class="btn secondary" data-action="predict" ${S.predBusy || before8 ? 'disabled' : ''}>${S.predBusy ? 'Predicting…' : 'See what happens (what if)'}</button>
  ${dietCard()}
  </div>
  <div class="log-result">${forecast || `<div class="log-placeholder"><div class="eyebrow">FORECAST</div><p>Check a meal and the predicted glucose curve shows up here, beside the foods you picked.</p></div>`}</div>
  <div class="log-foot">
  <button class="btn" data-action="save-meal" ${canConfirm ? '' : 'disabled'}>${S.confirmBusy ? 'Saving…' : 'I ate this: confirm meal'}</button>
  <p class="note">A what-if is only a prediction and is not saved. Confirming records the meal with its forecast so you can watch it against your real glucose. Photo recognition knows 101 common dishes — type the food or search if yours isn't found. A photo can't tell portion size, so we show USDA's standard portion to compare against your plate. Wellness insight, not medical advice.</p>
  </div>
</main>`;
  }

  function viewCalendar() {
    const D = S.data;
    const head = `<header class="top center">${brand()}${roleChip()}</header>
  <div class="eyebrow" style="margin-top:28px">YOUR RECORDING</div><h1 class="title lg">A little look back.</h1>`;
    if (!D) return `<main data-screen-label="Calendar">${head}<p class="lede">Upload your glucose history to see each day of your recording.</p>
      <button class="btn secondary" data-action="go" data-tab="profile">Upload history</button></main>`;
    const cells = [];
    for (let d = 1; d <= D.lastDay; d++) {
      const start = D.first + (d - 1) * DAY, hasMeals = D.meals.some(m => m.day === d) || savedMeals().some(m => m.day === d);
      const cls = d === D.lastDay ? ' today' : S.sheetDay === d ? ' sel' : '';
      cells.push(`<button data-action="open-day" data-day="${d}" aria-label="Day ${d}, ${shortDate(start)}, view day"><span class="cal-cell${cls}">${new Date(start).getUTCDate()}
        <span class="cal-dots"><i style="background:${d === D.lastDay ? 'var(--bg)' : d >= 8 ? 'var(--lilac)' : 'var(--teal)'}"></i><i style="background:var(--coral);opacity:${hasMeals ? 1 : 0}"></i></span></span></button>`);
    }
    return `<main data-screen-label="Calendar">${head}
  <p class="lede">Days 1–7 build your baseline. Predictions start on day 8, ${shortDate(D.day8)} at ${clock(D.day8)}. Each square is a 24-hour block from your first reading.</p>
  <div class="cal-head"><h2>${MONTHS[new Date(D.first).getUTCMonth()]} ${new Date(D.first).getUTCFullYear()}</h2><span class="muted" style="font-size:14px">${D.lastDay} day${D.lastDay === 1 ? '' : 's'}</span></div>
  <div class="cal-grid">${cells.join('')}</div>
  <div class="legend sm"><span><i style="background:var(--teal)"></i>Learning day</span><span><i style="background:var(--lilac)"></i>Prediction day</span><span><i style="background:var(--coral)"></i>Meals logged</span></div>
</main>`;
  }

  function forecastCards(list) {
    return list.map(m => {
      const F = { t: ms(m.meal_time), base: m.base, h: m.h, peak: m.h.reduce((a, b) => (b.v > a.v ? b : a)) };
      F.rise = F.peak.v - F.base;
      const actual = measuredPeak(F.t);
      return `<div class="card" style="margin-top:10px"><div class="card-head" style="align-items:flex-start"><div style="min-width:0"><div class="name" style="font-size:16px;font-weight:700;overflow-wrap:anywhere">${esc(m.food)}</div>
        <div class="sub small muted">${clock(F.t)} · day ${m.day}</div></div><span class="chip ${m.level === 'high' ? 'spike' : m.level === 'moderate' ? 'warn' : 'ok'}">${VERDICTS[m.level]}</span></div>
        ${medicationText(m.medication) ? `<div class="sub small muted">${esc(medicationText(m.medication))}</div>` : ''}
        ${horizonTiles(F)}<div style="margin-top:10px;font-size:14px;line-height:1.45;color:var(--text-2)">${esc(peakText(F))}</div>
        ${actual != null ? `<div style="margin-top:4px;font-size:13px;color:var(--muted-2)">Real peak so far: ${actual} mg/dL</div>` : ''}</div>`;
    }).join('');
  }

  function viewSheet() {
    const D = S.data, d = S.sheetDay;
    if (!D || !d) return '';
    const st = dayStats(d), rows = mealRows(d), forecasts = savedMeals().filter(m => m.day === d);
    const treat = D.treatments.filter(x => x.day === d).length;
    const cc = 2 * Math.PI * 20;
    return `<div class="scrim" data-action="close-sheet"></div>
  <div class="sheet" role="dialog" aria-label="Day ${d}">
    <button class="grab" data-action="close-sheet" aria-label="Close"><span></span></button>
    <h2>Day ${d} · ${longDate(st.start)}</h2>
    <p style="font-size:16px;color:var(--muted);margin:0">${d >= 8 ? 'Predictions on' : 'Learning'}${d === D.lastDay ? ' · latest day' : ''} · from ${clock(st.start)}</p>
    <div class="tiles2"><div class="tile teal">${icon(IC.drop, 20, 2, '#4fdfdc')}<div style="margin-top:12px"><span class="big">${st.avg ?? '–'}</span> <span style="font-size:14px;font-weight:600;color:var(--muted)">mg/dL</span></div><div style="font-size:15px;color:var(--muted)">Average glucose</div></div>
      <div class="tile lilac"><div style="display:flex;align-items:center;gap:12px"><svg width="50" height="50" viewBox="0 0 50 50" aria-hidden="true"><circle cx="25" cy="25" r="20" style="fill:none;stroke:#3a3346;stroke-width:5"></circle>
        <circle cx="25" cy="25" r="20" transform="rotate(-90 25 25)" style="fill:none;stroke:#c4a5f5;stroke-width:5;stroke-linecap:round;stroke-dasharray:${f1(cc * (st.tir || 0) / 100)} ${f1(cc)}"></circle></svg>
        <span class="big">${st.tir ?? '–'}%</span></div><div style="font-size:15px;color:var(--muted);margin-top:6px">Time in range</div></div></div>
    <div class="card-head" style="margin-top:26px;align-items:center"><span style="display:flex;align-items:center;gap:10px;font-size:19px;font-weight:700">${icon(IC.utensils, 20, 2, '#ff8b85')}Meals logged</span>
      <span class="muted" style="font-size:15px">${rows.length === 1 ? '1 meal' : rows.length + ' meals'}</span></div>
    ${rows.map(m => `<div class="row" style="padding:14px 0"><span class="thumb lg">${esc(m.name.trim()[0] || '?').toUpperCase()}</span>
      <div class="grow"><div class="name" style="font-size:17px">${esc(m.name)}</div><div class="sub" style="font-size:15px;margin-top:2px">${clock(m.t)}</div></div><span style="font-size:14px;color:var(--coral);text-align:right">${esc(m.sub)}</span></div>`).join('') || '<p class="empty">No meals recorded.</p>'}
    <div class="totals"><div><div class="num" style="color:var(--coral)">${rows.length}</div><div class="lbl">Meals</div></div>
      <div><div class="num" style="color:var(--teal)">${between(st.start, st.start + DAY - 1).length}</div><div class="lbl">CGM readings</div></div>
      <div><div class="num" style="color:var(--yellow)">${treat}</div><div class="lbl">Treatments</div></div></div>
    ${forecasts.length ? `<h3 style="font-size:19px;font-weight:700;margin:28px 0 4px">Forecast by meal</h3>${forecastCards(forecasts)}` : ''}
  </div>`;
  }

  function viewMeds() {
    const D = S.data;
    const recent = D ? D.treatments.slice(-12).reverse() : [];
    const forecasts = D ? savedMeals().filter(m => m.day === D.lastDay) : [];
    return `<main data-screen-label="Medication">
  <div class="meds-intro">
  <div class="eyebrow">MEDICATION</div><h1 class="title" style="margin-bottom:6px">What did you take?</h1>
  <p style="margin:0 0 18px;font-size:15px;line-height:1.5;color:var(--muted)">Forecasts include insulin and tablets from your record in the 24 hours before a meal, plus anything you pick in the What if screen.</p>
  </div>
  <div class="meds-cols">
  <section>
  <h2 class="section tight">Recorded so far</h2>
  ${recent.length ? recent.map(x => `<div class="row"><span class="thumb pill-ic">${icon(IC.pill)}</span><div class="grow"><div class="name" style="font-size:15px">${esc(x.text)}</div>
    <div class="sub">Day ${x.day} · ${shortDate(x.t)}, ${clock(x.t)}</div></div></div>`).join('') : `<p class="empty">${D ? 'No insulin or medication entries recorded yet.' : 'No CGM record on this account.'}</p>`}
  </section>
  <section>
  <h2 class="section tight">Confirmed meals today</h2>
  <p style="margin:0;font-size:14px;line-height:1.5;color:var(--muted-2)">${forecasts.length ? 'Forecast 30, 60, 90 and 120 minutes after each meal, with your real reading once the clock gets there.' : 'Confirm a meal in What if to see its forecast here.'}</p>
  ${forecastCards(forecasts)}
  </section>
  </div>
  <p class="note">Wellness insight, not medical advice. Don't change medication without your care team.</p>
</main>`;
  }

  function viewReminders() {
    const items = [['Breakfast', '8:00 AM'], ['Lunch', '12:30 PM'], ['Dinner', '7:00 PM']];
    return `<main data-screen-label="Reminders"><header class="top"><div><div class="greet">${greeting()}, ${esc(S.user || 'there')}</div>${brand()}</div></header>
  <div style="margin-top:28px;background:var(--teal-deep);border-radius:28px;padding:20px">
    <div style="display:flex;align-items:center;gap:8px;color:var(--teal);font-size:18px;font-weight:700">${icon(IC.bell, 18)}Smart nudges</div>
    <p style="margin:8px 0 0;font-size:15px;line-height:1.5;color:var(--muted)">Reminder preferences are saved on this device. Notifications aren't connected yet.</p></div>
  <div style="margin-top:12px">${items.map(([label, time], i) => `<div class="row" style="padding:22px 0;justify-content:space-between"><div><div style="font-size:18px;font-weight:700">${label}</div>
    <div style="font-size:15px;color:var(--muted-2);margin-top:2px">${time} · daily</div></div>
    <button class="switch" role="switch" aria-checked="${!!S.reminders[i]}" aria-label="${label} reminder" data-action="toggle-reminder" data-index="${i}"><span></span></button></div>`).join('')}</div>
</main>`;
  }

  function viewProfile() {
    const st = S.status, D = S.data;
    const learned = D ? D.readings.filter(r => r.t < D.day8) : [];
    const week = rangeStats(learned);
    const day = (iso) => `${shortDate(ms(iso))}, ${clock(ms(iso))}`;
    const id = (S.user || '').replace(/^patient/, '');
    return `<main data-screen-label="Profile">
  <header class="top center">${brand()}${roleChip()}</header>
  <div class="profile-intro"><div class="eyebrow" style="margin-top:28px">YOUR PROFILE</div><h1 class="title lg">Made for you.</h1></div>
  <div class="profile-hero" style="display:flex;flex-direction:column;align-items:center;margin-top:24px">
    <div class="avatar">${icon(IC.user, 44, 1.8)}<span class="heart">${icon(IC.heart, 15, 2.2)}</span></div>
    <div class="num" style="font-size:28px;margin-top:18px">${esc(S.user)}</div>
    <div style="font-size:16px;color:var(--muted-2);margin-top:4px">${S.admin ? 'Administrator account' : 'Patient account'}</div>
  </div>
  <div class="profile-body">
  <div class="profile-stats"><div><div class="num" style="color:var(--teal)">${D ? D.lastDay : 0}</div><div class="lbl">Current day</div></div>
    <div><div class="num" style="color:var(--coral)">${week.avg ?? '–'}</div><div class="lbl">Avg mg/dL</div></div>
    <div><div class="num" style="color:var(--lilac)">${week.tir ?? '–'}${week.tir == null ? '' : '%'}</div><div class="lbl">In range</div></div></div>
  <p class="hint profile-hint">Average and time in range (70–180) over your first 7 days.</p>

  <h2 class="display">Personal details</h2>
  ${st ? `<dl class="kv"><dt>Username</dt><dd>${esc(S.user)}</dd>
    ${/^\d+$/.test(id) ? `<dt>Patient ID</dt><dd>${esc(id)}</dd>` : ''}
    <dt>Your first 7 days</dt><dd>${shortDate(ms(st.first_reading))} – ${shortDate(ms(st.day8_start))}</dd>
    <dt>Day 8 starts</dt><dd>${day(st.day8_start)}</dd>
    <dt>Meals in your record</dt><dd>${D.meals.filter(m => m.t < D.day8).length}</dd>
    <dt>Insulin and medication entries</dt><dd>${D.treatments.filter(x => x.t < D.day8).length}</dd>
    <dt>Meals you confirmed</dt><dd>${savedMeals().length}</dd>
    ${st.personal ? `<dt>Your usual rise after a meal</dt><dd>+${st.personal.typical_rise_mg_dl ?? '–'} mg/dL</dd>
    <dt>Your spike line</dt><dd>+${st.personal.thresholds.rise} mg/dL${st.personal.thresholds.personal ? ` (from ${st.personal.thresholds.meals} meals)` : ' (default)'}</dd>
    <dt>Your big-spike line</dt><dd>+${st.personal.thresholds.severe_rise} mg/dL</dd>
    <dt>Rise per gram of carbs</dt><dd>${st.personal.carb_response.mg_dl_per_g} mg/dL${st.personal.carb_response.personal ? ` (from ${st.personal.carb_response.meals} meals)` : ' (default, too few meals)'}</dd>` : ''}</dl>`
    : '<p class="empty">No CGM record on this account.</p>'}
  ${st ? dietCard() : ''}

  <button class="btn ghost" style="margin-top:28px;height:60px;border-radius:20px;justify-content:flex-start;padding:0 20px;font-weight:600" data-action="signout">${icon(IC.signout)}<span style="flex:1;text-align:left">Sign out</span>${icon(IC.chevron, 18)}</button>
  </div>
</main>`;
  }

  function viewNav() {
    const item = (id, label, ic) => `<button class="tab" data-action="go" data-tab="${id}" ${S.tab === id ? 'aria-current="page"' : ''}><span>${icon(ic, 22, 1.9)}</span>${label}</button>`;
    return `<nav class="main" aria-label="Main"><div class="nav-brand">${brand(26)}</div>${item('today', 'Today', IC.today)}${item('calendar', 'Calendar', IC.cal)}
  <div class="fab-wrap"><button class="fab" data-action="go" data-tab="log" aria-label="Log a meal" ${S.tab === 'log' ? 'aria-current="page"' : ''}>${icon(IC.plus, 30, 2.4)}<span class="fab-label">What if</span></button></div>
  ${item('meds', 'Meds', IC.pill)}${item('profile', 'Profile', IC.user)}</nav>`;
  }

  const VIEWS = { signin: viewSignin, today: viewToday, log: viewLog, calendar: viewCalendar, meds: viewMeds, reminders: viewReminders, profile: viewProfile };

  function render() {
    const signedOut = S.tab === 'signin';
    root.className = 'app' + (signedOut ? ' no-nav' : '');
    root.innerHTML = VIEWS[S.tab]() + (signedOut ? '' : viewSheet() + (S.tab === 'log' ? viewAlts() : '') + viewNav()) +
      (S.toast ? `<div class="toast${S.toastBad ? ' bad' : ''}" role="status">${esc(S.toast)}</div>` : '');
    syncLogButtons();
  }

  // ---------------------------------------------------------------- events
  root.addEventListener('click', e => {
    const el = e.target.closest('[data-action]');
    if (!el || el.tagName === 'INPUT') return;
    const a = el.dataset.action;
    if (a === 'go') go(el.dataset.tab);
    else if (a === 'reload') loadAll();
    else if (a === 'toggle-pw') {
      const pw = root.querySelector('#gl-pw'), value = pw ? pw.value : '', user = root.querySelector('#gl-user');
      if (user) S.user = user.value;
      S.pwShown = !S.pwShown; render();
      const next = root.querySelector('#gl-pw'); next.value = value; next.focus();
    }
    else if (a === 'signout') signOut();
    else if (a === 'add-food') { S.builder.items.push({ name: '', amount: 100 }); render(); }
    else if (a === 'remove-food') { S.builder.items.splice(+el.dataset.index, 1); if (S.builder.search) S.builder.search = null; render(); }
    else if (a === 'identify-photo') { pickPhoto(+el.dataset.index); }
    else if (a === 'choose-pred') { choosePrediction(+el.dataset.index, { label: el.dataset.label, display_name: el.dataset.name }); }
    else if (a === 'toggle-per100') { const it = S.builder.items[+el.dataset.index]; it.per100 = !it.per100; render(); }
    else if (a === 'search-toggle') {
      const i = +el.dataset.index;
      S.builder.search = (S.builder.search && S.builder.search.row === i)
        ? null : { row: i, query: S.builder.items[i].name || '', busy: false, error: null, results: null };
      render();
    }
    else if (a === 'run-search') {
      const input = root.querySelector(`[data-search-input="${el.dataset.index}"]`);
      const q = input ? input.value.trim() : '';
      if (q) runFoodSearch(+el.dataset.index, q);
    }
    else if (a === 'pick-search') { pickSearchResult(+el.dataset.index, +el.dataset.fdcid, el.dataset.desc); }
    else if (a === 'predict') predict();
    else if (a === 'try-alt') tryAlternative(el.dataset.value);
    else if (a === 'open-alts') { S.altsOpen = true; render(); }
    else if (a === 'close-alts') { S.altsOpen = false; render(); }
    else if (a === 'save-diet') saveDiet(el.dataset.value);
    else if (a === 'remove-diet') removeDiet(el.dataset.value);
    else if (a === 'save-meal') confirmMeal();
    else if (a === 'live-toggle') control(S.live.paused ? 'play' : 'pause');
    else if (a === 'live-restart') { if (confirm('Restart the replay at the start of day 8? Confirmed meals are cleared.')) control('restart'); }
    else if (a === 'open-day') { S.sheetDay = +el.dataset.day; render(); }
    else if (a === 'close-sheet') { S.sheetDay = null; render(); }
    else if (a === 'toggle-reminder') { const i = +el.dataset.index; S.reminders[i] = !S.reminders[i]; store.set('glyimpse_reminders', S.reminders); render(); }
    else if (a === 'toggle-all-reminders') { const on = !S.reminders.some(Boolean); S.reminders = [on, on, on]; store.set('glyimpse_reminders', S.reminders); render(); }
  });
  root.addEventListener('change', e => {
    const t = e.target, d = t.dataset;
    if (d.action === 'live-speed') control('speed', { seconds_per_reading: +t.value });
    else if (d.item !== undefined) {
      const item = S.builder.items[+d.item];
      if (d.part === 'name') {
        // A typed/chosen food: reset any photo/USDA state and look it up fresh in USDA.
        item.name = t.value; item.label = null; item.selectedLabel = null; item.fdcId = null;
        item.predictions = null; item.needsConfirmation = false; item.portionKey = null;
        lookupFood(+d.item);  // async; re-renders with USDA portions + macros
      } else if (d.part === 'portion') {
        setPortion(+d.item, t.value);
      } else if (d.part === 'custom-grams') {
        item.customGrams = t.value; item.grams = +t.value || null; render();
      } else if (d.part === 'amount') {
        if (t.value === 'custom') { item.portionKey = 'custom'; item.grams = itemGrams(item); item.customGrams = item.grams; }
        else { item.amount = +t.value; item.grams = null; item.portionKey = null; }
        render();
      }
    } else if (d.med) {
      const med = S.builder[d.med];
      if (d.part === 'name') {
        const list = d.med === 'insulin' ? S.catalog.insulins : S.catalog.tablets;
        const chosen = list.find(m => m.name === t.value);
        med.name = t.value;
        med.dose = chosen ? (d.med === 'insulin' ? chosen.doses_iu : chosen.doses_mg)[0] : '';
      } else med.dose = t.value;
      render();
    }
  });
  root.addEventListener('submit', e => {
    if (e.target.id === 'signin-form') { e.preventDefault(); signIn(e.target); }
  });
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (S.altsOpen) { S.altsOpen = false; render(); } else if (S.sheetDay) { S.sheetDay = null; render(); }
  });

  api.onUnauthorized = () => { resetSession(); S.signinError = 'Your sign-in expired. Please sign in again.'; render(); };

  // The confirmed meal as a clean object the rest of the app can plug into:
  // { name, grams, kcal, carbs_g, protein_g, fat_g, fiber_g, sugar_g } per food, plus totals.
  window.glympse = {
    getConfirmedMeal: () => window.glympseConfirmedMeal || null,
    getCurrentFoods: () => resolvedFoods(),
  };

  // ---------------------------------------------------------------- boot
  render();
  if (api.token) loadAll(); else refreshHealth().then(render);
})();
