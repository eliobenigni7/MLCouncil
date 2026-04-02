const API_BASE = '/api';
let refreshInterval;

async function fetchAPI(endpoint) {
    const resp = await fetch(`${API_BASE}${endpoint}`);
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}

async function postAPI(endpoint, data) {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}

function statusBadge(status) {
    const map = {
        'SUCCESS': 'badge-ok',
        'STARTED': 'badge-warning',
        'RUNNING': 'badge-warning',
        'FAILED': 'badge-error',
        'ok': 'badge-ok',
        'degraded': 'badge-warning',
        'unhealthy': 'badge-error',
        'fresh': 'badge-ok',
        'stale': 'badge-warning',
        'no_data': 'badge-error',
        'unreachable': 'badge-error',
        'no_runs': 'badge-warning'
    };
    return `<span class="badge ${map[status] || ''}">${status}</span>`;
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

document.querySelectorAll('.sidebar nav a').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        document.querySelectorAll('.sidebar nav a').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        link.classList.add('active');
        document.getElementById(link.dataset.page).classList.add('active');
    });
});

async function refreshOverview() {
    try {
        const health = await fetchAPI('/health');
        const kpis = [];
        
        kpis.push(`
            <div class="kpi-card">
                <div class="kpi-label">API Status</div>
                <div class="kpi-value ${health.status === 'ok' ? 'ok' : 'error'}">${health.status.toUpperCase()}</div>
            </div>
        `);
        
        if (health.components) {
            kpis.push(`
                <div class="kpi-card">
                    <div class="kpi-label">Data Freshness</div>
                    <div class="kpi-value ${health.components.data_freshness === 'fresh' ? 'ok' : 'warning'}">${health.components.data_freshness || '--'}</div>
                </div>
            `);
            kpis.push(`
                <div class="kpi-card">
                    <div class="kpi-label">Arctic Store</div>
                    <div class="kpi-value ${health.components.arctic_store === 'ok' ? 'ok' : 'warning'}">${health.components.arctic_store || '--'}</div>
                </div>
            `);
        }
        
        document.getElementById('status-kpis').innerHTML = kpis.join('');
    } catch (e) {
        console.error('Failed to refresh overview:', e);
        document.getElementById('status-kpis').innerHTML = '<div class="kpi-card"><div class="kpi-value error">Connection Error</div></div>';
    }
}

async function refreshPipelineStatus() {
    try {
        const status = await fetchAPI('/pipeline/status');
        document.getElementById('last-run-id').textContent = status.run_id || '--';
        document.getElementById('pipeline-status').innerHTML = statusBadge(status.status);
        document.getElementById('pipeline-partition').textContent = status.partition || '--';
    } catch (e) {
        console.error('Failed to refresh pipeline:', e);
    }
}

document.getElementById('trigger-btn').addEventListener('click', async () => {
    const partition = document.getElementById('run-partition').value || null;
    const btn = document.getElementById('trigger-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';
    
    try {
        const result = await postAPI('/pipeline/run', {partition});
        showToast(`Pipeline started: ${result.run_id}`);
        setTimeout(refreshPipelineStatus, 2000);
    } catch (e) {
        showToast('Failed to trigger pipeline: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Pipeline';
    }
});

async function refreshPortfolio() {
    try {
        const weights = await fetchAPI('/portfolio/weights');
        const tbody = document.querySelector('#weights-table tbody');
        const entries = Object.entries(weights);
        
        if (entries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="2" style="color: var(--text-secondary);">No weights data</td></tr>';
        } else {
            tbody.innerHTML = entries
                .map(([ticker, weight]) => `<tr><td>${ticker}</td><td>${(weight * 100).toFixed(1)}%</td></tr>`)
                .join('');
        }
        
        const dates = await fetchAPI('/portfolio/orders/dates');
        const select = document.getElementById('order-date-select');
        
        if (dates.length === 0) {
            select.innerHTML = '<option value="">No orders</option>';
        } else {
            select.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join('');
            if (dates.length > 0) {
                loadOrdersForDate(dates[dates.length - 1]);
            }
        }
    } catch (e) {
        console.error('Failed to refresh portfolio:', e);
    }
}

async function loadOrdersForDate(date) {
    try {
        const orders = await fetchAPI(`/portfolio/orders/${date}`);
        const tbody = document.querySelector('#orders-table tbody');
        
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="color: var(--text-secondary);">No orders</td></tr>';
        } else {
            tbody.innerHTML = orders.map(o => `
                <tr>
                    <td>${o.ticker}</td>
                    <td>${o.direction}</td>
                    <td>$${(o.quantity || 0).toFixed(0)}</td>
                    <td>${((o.target_weight || 0) * 100).toFixed(1)}%</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error('Failed to load orders:', e);
    }
}

document.getElementById('order-date-select').addEventListener('change', (e) => {
    if (e.target.value) {
        loadOrdersForDate(e.target.value);
    }
});

async function refreshMonitoring() {
    try {
        const alerts = await fetchAPI('/monitoring/alerts');
        const tbody = document.querySelector('#alerts-table tbody');
        
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="color: var(--text-secondary);">No active alerts</td></tr>';
        } else {
            tbody.innerHTML = alerts.map(a => `
                <tr>
                    <td>${statusBadge(a.severity)}</td>
                    <td>${a.model_name}</td>
                    <td>${a.check_type}</td>
                    <td>${a.message}</td>
                    <td>${a.timestamp || '--'}</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error('Failed to refresh monitoring:', e);
    }
}

async function refreshConfig() {
    try {
        const universe = await fetchAPI('/config/universe');
        document.getElementById('universe-list').textContent = universe.universe?.tickers?.join(', ') || '--';
        
        const weights = await fetchAPI('/config/regime-weights');
        document.getElementById('regime-weights-display').textContent = JSON.stringify(weights.regime_weights, null, 2);
        
        const models = await fetchAPI('/config/models');
        document.getElementById('models-config-display').textContent = JSON.stringify(models, null, 2);
    } catch (e) {
        console.error('Failed to refresh config:', e);
    }
}

async function refreshAll() {
    await Promise.all([
        refreshOverview(),
        refreshPipelineStatus(),
        refreshPortfolio(),
        refreshMonitoring(),
        refreshConfig()
    ]);
}

document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('run-partition').value = today;
    
    refreshAll();
    refreshInterval = setInterval(refreshAll, 60000);
});
