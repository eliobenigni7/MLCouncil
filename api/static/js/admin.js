const API_BASE = '/api';
let refreshInterval;
let latestHealth = null;

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
    if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `API error: ${resp.status}`);
    }
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

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
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
        latestHealth = health;
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
            kpis.push(`
                <div class="kpi-card">
                    <div class="kpi-label">Runtime Profile</div>
                    <div class="kpi-value ${health.runtime_env?.status === 'valid' ? 'ok' : 'error'}">${(health.runtime_env?.profile || '--').toUpperCase()}</div>
                </div>
            `);
            kpis.push(`
                <div class="kpi-card">
                    <div class="kpi-label">Trading Ops</div>
                    <div class="kpi-value ${health.trading_operations?.status === 'success' ? 'ok' : 'warning'}">${health.trading_operations?.status || '--'}</div>
                </div>
            `);
        }

        document.getElementById('status-kpis').innerHTML = kpis.join('');
        renderOperatorBanner(health);
    } catch (e) {
        console.error('Failed to refresh overview:', e);
        document.getElementById('status-kpis').innerHTML = '<div class="kpi-card"><div class="kpi-value error">Connection Error</div></div>';
    }
}

function renderOperatorBanner(health) {
    const banner = document.getElementById('operator-banner');
    if (!banner) return;

    const runtimeInvalid = health.runtime_env?.status === 'invalid';
    const opsStatus = String(health.trading_operations?.status || '').toLowerCase();
    const opsRisk = ['blocked', 'degraded', 'error'].includes(opsStatus);
    const paused = Boolean(health.trading_operations?.paused);

    if (runtimeInvalid) {
        banner.className = 'operator-banner error';
        banner.textContent = 'Runtime profile is invalid. Fix config/runtime.env mismatches before running pipeline or trading actions.';
        return;
    }

    if (opsRisk) {
        banner.className = 'operator-banner warning';
        banner.textContent = `Last trading operation status is ${opsStatus || 'unknown'}. Review Trading and Monitoring panels before new executions.`;
        return;
    }

    if (paused) {
        banner.className = 'operator-banner warning';
        banner.textContent = 'Kill switch is active: trading automation is paused.';
        return;
    }

    banner.className = 'operator-banner hidden';
    banner.textContent = '';
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

async function refreshPipelinePartitionDefault() {
    try {
        const payload = await fetchAPI('/pipeline/latest-partition');
        if (payload.partition) {
            document.getElementById('run-partition').value = payload.partition;
        }
    } catch (e) {
        console.error('Failed to load latest partition:', e);
    }
}

document.getElementById('trigger-btn').addEventListener('click', async () => {
    const partition = document.getElementById('run-partition').value || null;
    const profile = latestHealth?.runtime_env?.profile || 'unknown';
    const partitionLabel = partition || 'latest available partition';
    const confirmMessage = `Confirm pipeline run for ${partitionLabel} (runtime profile: ${profile})?`;
    if (!window.confirm(confirmMessage)) {
        return;
    }
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
        const [alerts, settingsPayload] = await Promise.all([
            fetchAPI('/monitoring/alerts'),
            fetchAPI('/monitoring/settings')
        ]);
        const tbody = document.querySelector('#alerts-table tbody');
        const settingsForm = document.getElementById('settings-form');
        
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

        settingsForm.innerHTML = settingsPayload.settings.map(setting => `
            <div class="settings-field">
                <label for="setting-${escapeHtml(setting.key)}">${escapeHtml(setting.label)}</label>
                <input
                    id="setting-${escapeHtml(setting.key)}"
                    class="api-setting-input"
                    data-key="${escapeHtml(setting.key)}"
                    type="${setting.secret ? 'password' : 'text'}"
                    value="${escapeHtml(setting.value || '')}"
                    placeholder="${escapeHtml(setting.placeholder || '')}"
                    autocomplete="off"
                >
                <div class="settings-help">${escapeHtml(setting.description || '')}</div>
            </div>
        `).join('');

        const autoExecuteSetting = settingsPayload.settings.find(
            setting => setting.key === 'MLCOUNCIL_AUTO_EXECUTE'
        );
        const autoExecuteCheckbox = document.getElementById('auto-execute');
        if (autoExecuteCheckbox && autoExecuteSetting) {
            autoExecuteCheckbox.checked = String(autoExecuteSetting.value || '').toLowerCase() === 'true';
        }

        document.getElementById('settings-status').textContent = settingsPayload.path
            ? `Shared file: ${settingsPayload.path}`
            : '';
    } catch (e) {
        console.error('Failed to refresh monitoring:', e);
    }
}

async function updateSingleSetting(key, value) {
    const response = await fetch(`${API_BASE}/monitoring/settings`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({values: {[key]: value}})
    });
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    return response.json();
}

async function saveMonitoringSettings(event) {
    if (event) {
        event.preventDefault();
    }
    const inputs = document.querySelectorAll('.api-setting-input');
    const btn = document.getElementById('save-settings-btn');
    const statusEl = document.getElementById('settings-status');
    const values = {};

    inputs.forEach(input => {
        values[input.dataset.key] = input.value;
    });

    btn.disabled = true;
    statusEl.textContent = 'Saving...';

    try {
        const response = await fetch(`${API_BASE}/monitoring/settings`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({values})
        });
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        showToast('API settings saved');
        await refreshMonitoring();
    } catch (e) {
        console.error('Failed to save monitoring settings:', e);
        statusEl.textContent = 'Failed to save settings';
        showToast('Failed to save settings: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
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

async function refreshTrading() {
    try {
        const status = await fetchAPI('/trading/status');
        const statusKpis = document.getElementById('trading-status-kpis');
        const executeBtn = document.getElementById('execute-btn');
        let executeDisabledReason = '';
        
        if (!status.connected) {
            statusKpis.innerHTML = `<div class="kpi-card"><div class="kpi-value error">Disconnected</div><div class="kpi-label">${status.error || 'Check Alpaca config'}</div></div>`;
            executeDisabledReason = status.error || 'Trading connection unavailable';
        } else {
            statusKpis.innerHTML = `
                <div class="kpi-card">
                    <div class="kpi-label">Status</div>
                    <div class="kpi-value ok">${status.paper ? 'Paper Trading' : 'Live'}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Buying Power</div>
                    <div class="kpi-value">$${parseFloat(status.account?.buying_power || 0).toLocaleString()}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Portfolio Value</div>
                    <div class="kpi-value">$${parseFloat(status.account?.portfolio_value || 0).toLocaleString()}</div>
                </div>
            `;
        }

        executeBtn.disabled = Boolean(executeDisabledReason);
        executeBtn.title = executeDisabledReason;
        
        const posBody = document.querySelector('#positions-table tbody');
        if (status.positions && status.positions.length > 0) {
            posBody.innerHTML = status.positions.map(p => `
                <tr>
                    <td>${p.symbol}</td>
                    <td>${p.qty}</td>
                    <td>$${parseFloat(p.avg_price || 0).toFixed(2)}</td>
                    <td>$${parseFloat(p.current_price || 0).toFixed(2)}</td>
                    <td style="color: ${parseFloat(p.unrealized_pl || 0) >= 0 ? 'var(--ok)' : 'var(--error)'}">$${parseFloat(p.unrealized_pl || 0).toFixed(2)} (${parseFloat(p.unrealized_pl_pc || 0).toFixed(1)}%)</td>
                </tr>
            `).join('');
        } else {
            posBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-secondary);">No open positions</td></tr>';
        }
        
        try {
            const latest = await fetchAPI('/trading/orders/latest');
            const select = document.getElementById('trading-order-date');
            select.innerHTML = `<option value="${latest.date}">${latest.date}</option>`;
            if (!executeDisabledReason) {
                executeBtn.disabled = false;
                executeBtn.title = '';
            }
            await loadPendingOrders(latest.date);
        } catch (e) {
            executeBtn.disabled = true;
            executeBtn.title = 'No pending orders available';
            document.querySelector('#pending-orders-table tbody').innerHTML = '<tr><td colspan="4" style="color: var(--text-secondary);">No orders found</td></tr>';
        }
        
        const history = await fetchAPI('/trading/history?days=7');
        const histBody = document.querySelector('#trade-history-table tbody');
        if (history.trades && history.trades.length > 0) {
            histBody.innerHTML = history.trades.map(t => `
                <tr>
                    <td>${t.symbol || '--'}</td>
                    <td>${t.side || '--'}</td>
                    <td>${t.qty || '--'}</td>
                    <td>${t.status || '--'}</td>
                    <td>${t.submitted_at || '--'}</td>
                </tr>
            `).join('');
        } else {
            histBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-secondary);">No trade history</td></tr>';
        }
    } catch (e) {
        console.error('Failed to refresh trading:', e);
    }
}

async function loadPendingOrders(date) {
    try {
        const resp = await fetchAPI(`/trading/orders/pending/${date}`);
        const tbody = document.querySelector('#pending-orders-table tbody');
        
        if (resp.orders && resp.orders.length > 0) {
            tbody.innerHTML = resp.orders.map(o => `
                <tr>
                    <td>${o.ticker}</td>
                    <td>${(o.direction || 'buy').toUpperCase()}</td>
                    <td>${((o.target_weight || 0) * 100).toFixed(1)}%</td>
                    <td>$${((o.quantity || 0) * (o.price || 0)).toFixed(0)}</td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="4" style="color: var(--text-secondary);">No pending orders</td></tr>';
        }
    } catch (e) {
        console.error('Failed to load pending orders:', e);
    }
}

document.getElementById('execute-btn').addEventListener('click', async () => {
    const date = document.getElementById('trading-order-date').value;
    if (!date) return;
    const profile = latestHealth?.runtime_env?.profile || 'unknown';
    const confirmMessage = `Execute pending orders for ${date} (runtime profile: ${profile})?`;
    if (!window.confirm(confirmMessage)) {
        return;
    }
    
    const btn = document.getElementById('execute-btn');
    if (btn.disabled) return;
    btn.disabled = true;
    btn.textContent = 'Executing...';
    
    try {
        const result = await postAPI('/trading/execute', {date});
        const div = document.getElementById('execution-result');
        div.innerHTML = `
            <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 4px;">
                <strong>Execution complete:</strong><br>
                Orders submitted: ${result.orders_submitted}<br>
                Orders rejected: ${result.orders_rejected}<br>
                Liquidations: ${result.liquidations}
            </div>
        `;
        showToast('Orders executed successfully');
        await refreshTrading();
    } catch (e) {
        showToast('Execution failed: ' + e.message, 'error');
    } finally {
        btn.disabled = Boolean(btn.title);
        btn.textContent = 'Execute Orders';
    }
});

document.getElementById('trading-order-date').addEventListener('change', async (e) => {
    if (e.target.value) {
        await loadPendingOrders(e.target.value);
    }
});

document.getElementById('auto-execute').addEventListener('change', async (e) => {
    const checkbox = e.target;
    checkbox.disabled = true;
    try {
        await updateSingleSetting(
            'MLCOUNCIL_AUTO_EXECUTE',
            checkbox.checked ? 'true' : 'false'
        );
        showToast(`Auto-execute ${checkbox.checked ? 'enabled' : 'disabled'}`);
        await refreshMonitoring();
    } catch (err) {
        checkbox.checked = !checkbox.checked;
        showToast('Failed to update auto-execute: ' + err.message, 'error');
    } finally {
        checkbox.disabled = false;
    }
});

async function refreshAll() {
    await Promise.all([
        refreshOverview(),
        refreshPipelineStatus(),
        refreshPortfolio(),
        refreshMonitoring(),
        refreshConfig(),
        refreshTrading()
    ]);
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('settings-form-shell').addEventListener('submit', saveMonitoringSettings);
    
    refreshAll();
    refreshPipelinePartitionDefault();
    refreshInterval = setInterval(refreshAll, 60000);
});
