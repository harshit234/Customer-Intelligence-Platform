/* =============================================
   Meridian Customer Intelligence Platform
   Application Logic
   ============================================= */

'use strict';

/* ---- State ---- */
const state = {
  apiBase: localStorage.getItem('meridian_api_base') || 'http://localhost:8000',
  activeTab: 'dashboard',
  batchData: null,
  batchResults: null,
  charts: {},
};

/* =============================================
   UTILITIES
   ============================================= */

function $(id) { return document.getElementById(id); }

function showToast(message, type = 'info') {
  const icons = {
    success: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    error:   `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
    info:    `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `${icons[type] || icons.info}<span class="toast-message">${message}</span>`;
  $('toast-container').appendChild(toast);
  setTimeout(() => {
    toast.classList.add('toast-exit');
    toast.addEventListener('animationend', () => toast.remove());
  }, 4000);
}

function setLoading(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.classList.add('btn-loading');
  } else {
    btn.disabled = false;
    btn.classList.remove('btn-loading');
  }
}

async function apiCall(endpoint, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${state.apiBase}${endpoint}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/* =============================================
   SIDEBAR & TABS
   ============================================= */

function switchTab(tabId) {
  // Hide all
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

  // Show target
  const tabEl = $(`tab-${tabId}`);
  if (tabEl) tabEl.classList.remove('hidden');

  const navEl = $(`nav-${tabId}`);
  if (navEl) navEl.classList.add('active');

  state.activeTab = tabId;

  // Update topbar
  const titles = {
    dashboard: ['Dashboard', 'Platform overview & health'],
    predict:   ['Predict Conversion', 'Run ML prediction for a customer'],
    batch:     ['Batch Score', 'Score multiple customer records'],
    complaints:['Complaint Q&A', 'RAG-powered CFPB complaint intelligence'],
    intel:     ['Customer Intel', 'Combined ML + RAG customer insights'],
    metrics:   ['API Metrics', 'Real-time diagnostics & performance'],
  };
  const [title, subtitle] = titles[tabId] || ['Dashboard', ''];
  $('page-title').textContent = title;
  $('page-subtitle').textContent = subtitle;
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab(item.dataset.tab);
  });
});

document.querySelectorAll('.quick-action-card').forEach(card => {
  card.addEventListener('click', () => switchTab(card.dataset.tabTarget));
});

$('sidebar-toggle').addEventListener('click', () => {
  document.body.classList.toggle('sidebar-collapsed');
});

/* =============================================
   API CONFIG MODAL
   ============================================= */

function openApiModal() {
  $('api-base-url-input').value = state.apiBase;
  $('api-modal').classList.add('open');
}

function closeApiModal() {
  $('api-modal').classList.remove('open');
}

$('api-config-btn').addEventListener('click', openApiModal);
$('api-modal-close').addEventListener('click', closeApiModal);
$('api-modal-cancel').addEventListener('click', closeApiModal);
$('api-modal').addEventListener('click', (e) => { if (e.target === $('api-modal')) closeApiModal(); });

$('api-modal-save').addEventListener('click', () => {
  const url = $('api-base-url-input').value.trim().replace(/\/$/, '');
  if (!url) { showToast('Please enter a valid API URL.', 'error'); return; }
  state.apiBase = url;
  localStorage.setItem('meridian_api_base', url);
  $('api-base-url-display').textContent = url;
  closeApiModal();
  showToast('API URL saved. Checking connection...', 'info');
  checkHealth();
});

/* =============================================
   HEALTH CHECK
   ============================================= */

async function checkHealth() {
  const dot  = $('status-dot');
  const text = $('status-text');
  dot.className = 'status-dot checking';
  text.textContent = 'Checking...';

  try {
    const data = await apiCall('/health');
    dot.className = 'status-dot online';
    text.textContent = 'Online';

    // Dashboard stats
    $('dash-uptime').textContent = formatUptime(data.uptime_seconds);
    $('dash-model').textContent  = data.model_version || 'loaded';
    $('dash-index').textContent  = data.index_version || 'loaded';
    $('dash-status').textContent = data.status || 'healthy';

    showToast('Connected to API successfully!', 'success');
    // Auto-refresh metrics on dashboard
    if (state.activeTab === 'dashboard') refreshDashboardMetrics();
  } catch (err) {
    dot.className = 'status-dot offline';
    text.textContent = 'Offline';
    $('dash-uptime').textContent  = '—';
    $('dash-model').textContent   = '—';
    $('dash-index').textContent   = '—';
    $('dash-status').textContent  = 'Offline';
    showToast(`Connection failed: ${err.message}`, 'error');
  }
}

$('api-base-url-display').textContent = state.apiBase;

// Check on load
window.addEventListener('load', () => {
  checkHealth();
});

/* =============================================
   DASHBOARD — METRICS
   ============================================= */

async function refreshDashboardMetrics() {
  try {
    const data = await apiCall('/metrics');
    renderDashboardCharts(data);
    renderDashboardLatency(data);
    updateSummaryStats(data);
  } catch (err) {
    // Silently fail on dashboard — it will show empty states
  }
}

$('refresh-metrics-btn').addEventListener('click', refreshDashboardMetrics);

function renderDashboardCharts(data) {
  const endpoints = Object.keys(data.request_counts);
  const counts    = Object.values(data.request_counts);

  $('traffic-empty').classList.add('hidden');

  const canvas = $('traffic-chart');
  if (state.charts.traffic) state.charts.traffic.destroy();

  state.charts.traffic = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: endpoints,
      datasets: [{
        label: 'Requests',
        data: counts,
        backgroundColor: [
          'rgba(129,140,248,0.7)',
          'rgba(56,189,248,0.7)',
          'rgba(34,211,238,0.7)',
          'rgba(74,222,128,0.7)',
          'rgba(251,191,36,0.7)',
          'rgba(248,113,113,0.7)',
        ],
        borderColor: [
          'rgba(129,140,248,1)',
          'rgba(56,189,248,1)',
          'rgba(34,211,238,1)',
          'rgba(74,222,128,1)',
          'rgba(251,191,36,1)',
          'rgba(248,113,113,1)',
        ],
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        label: ctx => ` ${ctx.raw} requests`
      }}},
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8a94b8', font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8a94b8', font: { size: 11 }, precision: 0 }, beginAtZero: true },
      }
    }
  });
}

function renderDashboardLatency(data) {
  const container = $('latency-list');
  const latencies = data.latencies_ms;
  const maxVal = Math.max(...Object.values(latencies), 1);

  if (!latencies || Object.keys(latencies).length === 0) {
    container.innerHTML = '<div class="latency-empty">No latency data yet.</div>';
    return;
  }

  container.innerHTML = Object.entries(latencies).map(([endpoint, ms]) => `
    <div class="latency-item">
      <span class="latency-endpoint">/${endpoint}</span>
      <div class="latency-bar-wrap">
        <div class="latency-bar" style="width:${Math.max(4, (ms / maxVal) * 100)}%"></div>
      </div>
      <span class="latency-value">${ms.toFixed(1)} ms</span>
    </div>
  `).join('');
}

function updateSummaryStats(data) {
  // No-op for now; health stats already set
}

/* =============================================
   PREDICT FORM
   ============================================= */

const DEMO_CUSTOMER = {
  age: 32, job: 'management', marital: 'single', education: 'tertiary',
  default: 'no', balance: 2343, housing: 'yes', loan: 'no',
  contact: 'cellular', day: 5, month: 'may', campaign: 1,
  pdays: -1, previous: 0, poutcome: 'unknown',
};

function fillPredictForm(prefix, data) {
  Object.entries(data).forEach(([key, val]) => {
    const el = $(`${prefix}-${key}`);
    if (el) el.value = val;
  });
}

$('predict-fill-demo').addEventListener('click', () => fillPredictForm('p', DEMO_CUSTOMER));
$('intel-fill-demo').addEventListener('click', () => fillPredictForm('i', DEMO_CUSTOMER));

function getPredictPayload(prefix) {
  return {
    age:       parseInt($(`${prefix}-age`).value),
    job:       $(`${prefix}-job`).value,
    marital:   $(`${prefix}-marital`).value,
    education: $(`${prefix}-education`).value,
    default:   $(`${prefix}-default`).value,
    balance:   parseInt($(`${prefix}-balance`).value) || 0,
    housing:   $(`${prefix}-housing`).value,
    loan:      $(`${prefix}-loan`).value,
    contact:   $(`${prefix}-contact`).value,
    day:       parseInt($(`${prefix}-day`).value) || 15,
    month:     $(`${prefix}-month`).value,
    campaign:  parseInt($(`${prefix}-campaign`).value) || 1,
    pdays:     parseInt($(`${prefix}-pdays`).value),
    previous:  parseInt($(`${prefix}-previous`).value) || 0,
    poutcome:  $(`${prefix}-poutcome`).value,
  };
}

$('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('predict-submit-btn');
  setLoading(btn, true);
  try {
    const payload = getPredictPayload('p');
    const data = await apiCall('/predict', 'POST', payload);
    renderPredictResult(data);
  } catch (err) {
    showToast(`Prediction failed: ${err.message}`, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function getBandClass(prob) {
  if (prob >= 0.60) return 'high';
  if (prob >= 0.40) return 'medium';
  return 'low';
}

function getBandLabel(prob) {
  if (prob >= 0.60) return 'HIGH';
  if (prob >= 0.40) return 'MEDIUM';
  return 'LOW';
}

const BAND_EMOJI = { HIGH: '🟢', MEDIUM: '🟡', LOW: '🔴' };

function renderPredictResult(data) {
  const band = getBandLabel(data.probability);
  const bandClass = getBandClass(data.probability);

  $('predict-result-panel').querySelector('.result-placeholder').classList.add('hidden');
  $('predict-result-content').classList.remove('hidden');

  // Band display
  const bandDisplay = $('predict-band-display');
  bandDisplay.className = `result-band-display band-${bandClass}`;
  $('predict-band-icon').textContent = BAND_EMOJI[band];
  $('predict-band-value').textContent = band;

  // Gauge
  drawGauge('prob-gauge', data.probability);
  $('predict-prob-pct').textContent = `${(data.probability * 100).toFixed(1)}%`;

  // Details
  $('predict-pred-value').textContent = data.prediction === 1 ? '✅ Subscribed' : '❌ Not Subscribed';
  $('predict-decision-value').textContent = data.threshold_decision ? '✅ Yes' : '❌ No';
  $('predict-model-version').textContent = data.model_version || '—';

  showToast('Prediction complete!', 'success');
}

function drawGauge(canvasId, probability) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 8;
  const r  = Math.min(W, H * 2) / 2 - 12;
  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(30,38,64,1)';
  ctx.lineWidth   = 12;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Fill
  const pAngle = startAngle + probability * Math.PI;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  if (probability < 0.40) {
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(1, '#fca5a5');
  } else if (probability < 0.60) {
    grad.addColorStop(0, '#d97706');
    grad.addColorStop(1, '#fbbf24');
  } else {
    grad.addColorStop(0, '#16a34a');
    grad.addColorStop(1, '#4ade80');
  }

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, pAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 12;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Needle
  const needleAngle = startAngle + probability * Math.PI;
  const nx = cx + (r - 2) * Math.cos(needleAngle);
  const ny = cy + (r - 2) * Math.sin(needleAngle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#e8ecf7';
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
  ctx.fillStyle = '#e8ecf7';
  ctx.fill();
}

/* =============================================
   BATCH SCORE
   ============================================= */

// File drop
const uploadArea = $('batch-upload-area');
uploadArea.addEventListener('click', () => $('batch-file-input').click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) readJsonFile(file);
});

$('batch-file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) readJsonFile(file);
});

function readJsonFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    $('batch-json-input').value = e.target.result;
    showToast(`Loaded ${file.name}`, 'info');
  };
  reader.readAsText(file);
}

$('batch-submit-btn').addEventListener('click', async () => {
  const btn = $('batch-submit-btn');
  const raw = $('batch-json-input').value.trim();
  if (!raw) { showToast('Please enter JSON records.', 'error'); return; }

  let records;
  try {
    records = JSON.parse(raw);
    if (!Array.isArray(records)) throw new Error('Expected an array.');
  } catch (err) {
    showToast(`Invalid JSON: ${err.message}`, 'error');
    return;
  }

  setLoading(btn, true);
  try {
    const data = await apiCall('/batch-score', 'POST', { records });
    state.batchResults = data;
    renderBatchResults(data, records.length);
    showToast(`Scored ${data.scores.length} records!`, 'success');
  } catch (err) {
    showToast(`Batch scoring failed: ${err.message}`, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function renderBatchResults(data, total) {
  $('batch-result-placeholder').classList.add('hidden');
  $('batch-result-content').classList.remove('hidden');

  // Model version badge
  $('batch-model-version').textContent = data.model_version || '';

  // Summary bars
  const { summary } = data;
  const maxCount = Math.max(...Object.values(summary), 1);
  $('batch-summary-bars').innerHTML = ['HIGH', 'MEDIUM', 'LOW'].map(band => `
    <div class="summary-bar-row bar-${band.toLowerCase()}">
      <span class="summary-bar-label">${band}</span>
      <div class="summary-bar-track">
        <div class="summary-bar-fill" style="width:${(summary[band] / maxCount) * 100}%"></div>
      </div>
      <span class="summary-bar-count">${summary[band]}</span>
    </div>
  `).join('');

  // Table
  $('batch-results-body').innerHTML = data.scores.map(row => `
    <tr>
      <td>${row.record_index + 1}</td>
      <td>${row.prediction === 1 ? '✅ Yes' : '❌ No'}</td>
      <td>${(row.probability * 100).toFixed(1)}%</td>
      <td><span class="badge badge-${row.conversion_band.toLowerCase()}">${row.conversion_band}</span></td>
    </tr>
  `).join('');
}

$('batch-export-btn').addEventListener('click', () => {
  if (!state.batchResults) return;
  const rows = [['Record', 'Prediction', 'Probability', 'Band']];
  state.batchResults.scores.forEach(r => {
    rows.push([r.record_index + 1, r.prediction, r.probability.toFixed(4), r.conversion_band]);
  });
  const csv = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'batch_scores.csv';
  a.click();
});

/* =============================================
   COMPLAINTS Q&A
   ============================================= */

$('filters-toggle').addEventListener('click', () => {
  const body = $('filters-body');
  const expanded = $('filters-toggle').getAttribute('aria-expanded') === 'true';
  $('filters-toggle').setAttribute('aria-expanded', !expanded);
  body.classList.toggle('hidden', expanded);
});

document.querySelectorAll('.sample-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    $('complaints-query').value = chip.dataset.query;
    $('complaints-query').focus();
  });
});

$('complaints-submit-btn').addEventListener('click', async () => {
  const btn = $('complaints-submit-btn');
  const query = $('complaints-query').value.trim();
  if (!query) { showToast('Please enter a question.', 'error'); return; }

  const filters = {};
  const prod = $('filter-product').value.trim();
  const comp = $('filter-company').value.trim();
  const issue = $('filter-issue').value.trim();
  if (prod)  filters.product = prod;
  if (comp)  filters.company = comp;
  if (issue) filters.issue   = issue;

  setLoading(btn, true);
  try {
    const data = await apiCall('/ask-complaints', 'POST', {
      query,
      filters: Object.keys(filters).length > 0 ? filters : null
    });
    renderComplaintsResult(data);
    showToast('Got a response from the AI!', 'success');
  } catch (err) {
    showToast(`RAG query failed: ${err.message}`, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function renderComplaintsResult(data) {
  $('complaints-result-placeholder').classList.add('hidden');
  $('complaints-result-content').classList.remove('hidden');

  // Sufficiency badge
  const suf = data.evidence_sufficiency;
  const sufBadge = $('complaints-sufficiency-badge');
  sufBadge.className = `sufficiency-badge suf-${suf}`;
  sufBadge.textContent = suf;

  // Answer
  $('complaints-answer').textContent = data.answer;

  // Evidence
  const evidenceList = $('complaints-evidence-list');
  if (data.evidence_ids.length === 0) {
    evidenceList.innerHTML = '<span style="color:var(--text-muted);font-size:12px;">No evidence retrieved.</span>';
  } else {
    $('complaints-evidence-count').textContent = `${data.evidence_ids.length} sources`;
    $('complaints-evidence-count').className = 'badge badge-ghost';
    evidenceList.innerHTML = data.evidence_ids.map(id => `
      <div class="evidence-chip">
        <svg style="width:12px;height:12px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        ${id}
      </div>
    `).join('');
  }

  $('complaints-prompt-version').textContent = `Prompt version: ${data.prompt_version}`;
}

/* =============================================
   CUSTOMER INTEL
   ============================================= */

$('intel-submit-btn').addEventListener('click', async () => {
  const btn = $('intel-submit-btn');

  const customerFeatures = getPredictPayload('i');

  const filters = {};
  const prod = $('i-filter-product').value.trim();
  const comp = $('i-filter-company').value.trim();
  if (prod) filters.product = prod;
  if (comp) filters.company = comp;

  setLoading(btn, true);
  try {
    const data = await apiCall('/customer-intel', 'POST', {
      customer_features: customerFeatures,
      complaint_filters: Object.keys(filters).length > 0 ? filters : null
    });
    renderIntelResult(data);
    showToast('Customer intel retrieved!', 'success');
  } catch (err) {
    showToast(`Customer intel failed: ${err.message}`, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function renderIntelResult(data) {
  $('intel-result-placeholder').classList.add('hidden');
  $('intel-result-content').classList.remove('hidden');

  // Hero card
  drawGauge('intel-gauge', data.probability);
  $('intel-prob-pct').textContent = `${(data.probability * 100).toFixed(1)}%`;

  const bandChip = $('intel-band-chip');
  bandChip.textContent = data.conversion_band;
  bandChip.className = `intel-band-chip chip-${data.conversion_band}`;

  $('intel-versions').innerHTML = `
    <span>Model: <code>${data.model_version}</code></span>
    <span>Index: <code>${data.index_version}</code></span>
  `;

  // Themes
  const themesList = $('intel-themes-list');
  if (!data.top_complaint_themes || data.top_complaint_themes.length === 0) {
    themesList.innerHTML = '<div class="theme-card" style="color:var(--text-muted);font-size:13px;">No complaint themes retrieved.</div>';
  } else {
    themesList.innerHTML = data.top_complaint_themes.map(theme => `
      <div class="theme-card">
        <div class="theme-header">
          <span class="theme-product">${theme.product || '—'}</span>
          <span class="badge badge-ghost">${(theme.score * 100).toFixed(0)}% match</span>
        </div>
        <div class="theme-issue">⚠️ ${theme.issue || '—'}</div>
        <div class="theme-snippet">${theme.snippet || '—'}</div>
        <div class="theme-id" style="margin-top:6px;">ID: ${theme.complaint_id}</div>
      </div>
    `).join('');
  }
}

/* =============================================
   METRICS TAB
   ============================================= */

$('fetch-metrics-btn').addEventListener('click', async () => {
  const btn = $('fetch-metrics-btn');
  setLoading(btn, true);
  try {
    const data = await apiCall('/metrics');
    renderMetricsTab(data);
    showToast('Metrics updated!', 'success');
  } catch (err) {
    showToast(`Failed to fetch metrics: ${err.message}`, 'error');
  } finally {
    setLoading(btn, false);
  }
});

function renderMetricsTab(data) {
  // Top stats
  const totalReqs = Object.values(data.request_counts).reduce((a, b) => a + b, 0);
  const mlPreds = (data.predictions_count['0'] || 0) + (data.predictions_count['1'] || 0);
  $('m-total-requests').textContent = totalReqs;
  $('m-ml-predictions').textContent = mlPreds;
  $('m-rag-retrievals').textContent = data.rag_retrievals_count;
  $('m-avg-evidence').textContent   = data.rag_avg_evidence_count.toFixed(2);

  // Request counts table
  $('metrics-requests-table-wrap').innerHTML = `
    <table class="metrics-table">
      <thead><tr><th>Endpoint</th><th>Requests</th><th>Errors</th></tr></thead>
      <tbody>
        ${Object.entries(data.request_counts).map(([ep, cnt]) => `
          <tr>
            <td>/${ep}</td>
            <td>${cnt}</td>
            <td style="color:${(data.error_counts[ep] || 0) > 0 ? 'var(--accent-red)' : 'var(--text-muted)'}">${data.error_counts[ep] || 0}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;

  // Latency bars
  const maxLat = Math.max(...Object.values(data.latencies_ms), 1);
  $('metrics-latency-bars-wrap').innerHTML = `
    <div class="latency-bar-list">
      ${Object.entries(data.latencies_ms).map(([ep, ms]) => `
        <div class="latency-item">
          <span class="latency-endpoint">/${ep}</span>
          <div class="latency-bar-wrap">
            <div class="latency-bar" style="width:${Math.max(4, (ms / maxLat) * 100)}%"></div>
          </div>
          <span class="latency-value">${ms.toFixed(1)} ms</span>
        </div>
      `).join('')}
    </div>
  `;

  // Predictions distribution
  const pos = data.predictions_count['1'] || 0;
  const neg = data.predictions_count['0'] || 0;
  const total = pos + neg || 1;

  $('predictions-dist').innerHTML = `
    <div class="pred-pie-wrap">
      <canvas id="pred-pie" width="140" height="140"></canvas>
    </div>
    <div class="pred-legend">
      <div class="legend-item">
        <div class="legend-dot" style="background:var(--accent-green)"></div>
        <div>Subscribed (1): <strong>${pos}</strong> &mdash; ${((pos/total)*100).toFixed(1)}%</div>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:var(--accent-red)"></div>
        <div>Not Subscribed (0): <strong>${neg}</strong> &mdash; ${((neg/total)*100).toFixed(1)}%</div>
      </div>
    </div>
  `;

  // Donut chart for predictions
  if (state.charts.predPie) state.charts.predPie.destroy();
  state.charts.predPie = new Chart($('pred-pie'), {
    type: 'doughnut',
    data: {
      labels: ['Subscribed', 'Not Subscribed'],
      datasets: [{
        data: [pos, neg],
        backgroundColor: ['rgba(74,222,128,0.8)', 'rgba(248,113,113,0.8)'],
        borderColor:      ['rgba(74,222,128,1)',   'rgba(248,113,113,1)'],
        borderWidth: 2,
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw} predictions` } }
      }
    }
  });
}
