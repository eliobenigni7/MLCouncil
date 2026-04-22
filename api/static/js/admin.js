const API_BASE = '/api';
let refreshInterval;
let latestHealth = null;

function getApiKey() {
    return localStorage.getItem('mlcouncil_api_key') || '';
}

function setApiKey(value) {
    const apiKey = (value || '').trim();
    if (apiKey) {
        localStorage.setItem('mlcouncil_api_key', apiKey);
    } else {
        localStorage.removeItem('mlcouncil_api_key');
    }
}

function buildApiHeaders(extraHeaders = {}) {
    const headers = {...extraHeaders};
    const apiKey = getApiKey();
    if (apiKey) {
        headers['X-API-Key'] = apiKey;
    }
    return headers;
}

async function fetchAPI(endpoint) {
    const resp = await fetch(`${API_BASE}${endpoint}`, {headers: buildApiHeaders()});
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}

async function postAPI(endpoint, data) {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: buildApiHeaders({'Content-Type': 'application/json'}),
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
        'consistent': 'badge-ok',
        'inconsistent': 'badge-warning',
        'error': 'badge-error',
        'degraded': 'badge-warning',
        'unhealthy': 'badge-error',
        'fresh': 'badge-ok',
        'stale': 'badge-warning',
        'no_data': 'badge-error',
        'unreachable': 'badge-error',
        'no_runs': 'badge-warning'
    };
    return `<span class="badge ${map[status] || ''}">${escapeHtml(status)}</span>`;
}

function setTableRows(tbody, rows) {
    tbody.replaceChildren(...rows);
}

function makeEmptyRow(colspan, message) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = colspan;
    td.style.color = 'var(--text-secondary)';
    td.textContent = message;
    tr.appendChild(td);
    return tr;
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

function renderValidationSummary(summary) {
    const el = document.getElementById('validation-backtest-summary');
    if (!el) return;
    const latest = summary?.latest;
    if (!latest) {
        el.textContent = 'No validation/backtest artifacts found.';
        return;
    }
    el.innerHTML = `Status: ${statusBadge(summary.status || 'no_data')}<br>
        Latest: ${escapeHtml(latest.date || '--')}<br>
        Windows: ${escapeHtml(latest.walk_forward_window_count ?? '--')}<br>
        OOS Sharpe: ${escapeHtml(latest.oos_sharpe ?? '--')}<br>
        PBO: ${escapeHtml(latest.pbo ?? '--')}`;
}

function renderRuntimeConsistency(summary) {
    const el = document.getElementById('runtime-consistency-summary');
    if (!el) return;
    if (!summary) {
        el.textContent = '--';
        return;
    }
    const issues = (summary.issues || []).slice(0, 3).join('; ') || 'None';
    el.innerHTML = `Status: ${statusBadge(summary.status || 'inconsistent')}<br>
        Profile: ${escapeHtml(summary.profile || '--')}<br>
        Config hash: ${escapeHtml(summary.config_hash || '--')}<br>
        Issues: ${escapeHtml(issues)}`;
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

const apiKeyInput = document.getElementById('api-key');
if (apiKeyInput) {
    apiKeyInput.value = getApiKey();
    apiKeyInput.addEventListener('input', (e) => setApiKey(e.target.value));
}

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
        renderValidationSummary(health.validation_backtest_summary);
        renderRuntimeConsistency(health.config_runtime_summary);
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
            setTableRows(tbody, [makeEmptyRow(2, 'No weights data')]);
        } else {
            const rows = entries.map(([ticker, weight]) => {
                const tr = document.createElement('tr');
                const tdTicker = document.createElement('td');
                tdTicker.textContent = ticker;
                const tdWeight = document.createElement('td');
                tdWeight.textContent = `${(weight * 100).toFixed(1)}%`;
                tr.append(tdTicker, tdWeight);
                return tr;
            });
            setTableRows(tbody, rows);
        }
        
        const dates = await fetchAPI('/portfolio/orders/dates');
        const select = document.getElementById('order-date-select');
        
        if (dates.length === 0) {
            select.replaceChildren(Object.assign(document.createElement('option'), {value: '', textContent: 'No orders'}));
        } else {
            select.replaceChildren(...dates.map(d => Object.assign(document.createElement('option'), {value: d, textContent: d})));
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
            setTableRows(tbody, [makeEmptyRow(4, 'No orders')]);
        } else {
            setTableRows(tbody, orders.map(o => {
                const tr = document.createElement('tr');
                [o.ticker, o.direction, `$${(o.quantity || 0).toFixed(0)}`, `${((o.target_weight || 0) * 100).toFixed(1)}%`].forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value;
                    tr.appendChild(td);
                });
                return tr;
            }));
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
            setTableRows(tbody, [makeEmptyRow(5, 'No active alerts')]);
        } else {
            setTableRows(tbody, alerts.map(a => {
                const tr = document.createElement('tr');
                const cells = [null, a.model_name, a.check_type, a.message, a.timestamp || '--'];
                const severityCell = document.createElement('td');
                severityCell.innerHTML = statusBadge(a.severity);
                tr.appendChild(severityCell);
                cells.slice(1).forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value;
                    tr.appendChild(td);
                });
                return tr;
            }));
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
    const values = {};
    values[key] = value;
    const response = await fetch(`${API_BASE}/monitoring/settings`, {
        method: 'PUT',
        headers: buildApiHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({values})
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
            headers: buildApiHeaders({'Content-Type': 'application/json'}),
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
        const runtimeStatus = String(status.runtime_profile || '').toLowerCase();
        
        if (!status.connected) {
            statusKpis.innerHTML = '';
            const card = document.createElement('div');
            card.className = 'kpi-card';
            card.innerHTML = '<div class="kpi-value error">Disconnected</div>';
            const label = document.createElement('div');
            label.className = 'kpi-label';
            label.textContent = status.error || 'Check Alpaca config';
            card.appendChild(label);
            statusKpis.appendChild(card);
            executeDisabledReason = status.error || 'Trading connection unavailable';
        } else {
            statusKpis.innerHTML = '';
            const values = [
                ['Status', status.paper ? 'Paper Trading' : 'Live'],
                ['Buying Power', `$${parseFloat(status.account?.buying_power || 0).toLocaleString()}`],
                ['Portfolio Value', `$${parseFloat(status.account?.portfolio_value || 0).toLocaleString()}`],
            ];
            values.forEach(([labelText, valueText], idx) => {
                const card = document.createElement('div');
                card.className = 'kpi-card';
                const label = document.createElement('div');
                label.className = 'kpi-label';
                label.textContent = labelText;
                const value = document.createElement('div');
                value.className = idx === 0 ? 'kpi-value ok' : 'kpi-value';
                value.textContent = valueText;
                card.append(label, value);
                statusKpis.appendChild(card);
            });
        }

        if (status.paused || status.kill_switch_active || (runtimeStatus && runtimeStatus !== 'paper')) {
            executeDisabledReason = executeDisabledReason || (status.paused || status.kill_switch_active
                ? 'Trading is paused'
                : 'Execute Orders is only available in paper runtime profile');
        }

        executeBtn.disabled = Boolean(executeDisabledReason);
        executeBtn.title = executeDisabledReason;
        
        const posBody = document.querySelector('#positions-table tbody');
        if (status.positions && status.positions.length > 0) {
            setTableRows(posBody, status.positions.map(p => {
                const tr = document.createElement('tr');
                const values = [p.symbol, p.qty, `$${parseFloat(p.avg_price || 0).toFixed(2)}`, `$${parseFloat(p.current_price || 0).toFixed(2)}`];
                values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); });
                const td = document.createElement('td');
                td.style.color = parseFloat(p.unrealized_pl || 0) >= 0 ? 'var(--ok)' : 'var(--error)';
                td.textContent = `$${parseFloat(p.unrealized_pl || 0).toFixed(2)} (${parseFloat(p.unrealized_pl_pc || 0).toFixed(1)}%)`;
                tr.appendChild(td);
                return tr;
            }));
        } else {
            setTableRows(posBody, [Object.assign(document.createElement('tr'), {innerHTML: '<td colspan="5" style="color: var(--text-secondary);">No open positions</td>'})]);
        }
        
        try {
            const latest = await fetchAPI('/trading/orders/latest');
            const select = document.getElementById('trading-order-date');
            select.replaceChildren(Object.assign(document.createElement('option'), {value: latest.date, textContent: latest.date}));
            if (!executeDisabledReason) {
                executeBtn.disabled = false;
                executeBtn.title = '';
            }
            await loadPendingOrders(latest.date);
        } catch (e) {
            executeBtn.disabled = true;
            executeBtn.title = 'No pending orders available';
            setTableRows(document.querySelector('#pending-orders-table tbody'), [makeEmptyRow(4, 'No orders found')]);
        }
        
        const history = await fetchAPI('/trading/history?days=7');
        const histBody = document.querySelector('#trade-history-table tbody');
        if (history.trades && history.trades.length > 0) {
            setTableRows(histBody, history.trades.map(t => {
                const tr = document.createElement('tr');
                [t.symbol || '--', t.side || '--', t.qty || '--', t.status || '--', t.submitted_at || '--'].forEach(value => {
                    const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
                });
                return tr;
            }));
        } else {
            setTableRows(histBody, [makeEmptyRow(5, 'No trade history')]);
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
            setTableRows(tbody, resp.orders.map(o => {
                const tr = document.createElement('tr');
                [o.ticker, (o.direction || 'buy').toUpperCase(), `${((o.target_weight || 0) * 100).toFixed(1)}%`, `$${((o.quantity || 0) * (o.price || 0)).toFixed(0)}`].forEach(value => {
                    const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
                });
                return tr;
            }));
        } else {
            setTableRows(tbody, [makeEmptyRow(4, 'No pending orders')]);
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
        div.replaceChildren();
        const box = document.createElement('div');
        box.style.padding = '1rem';
        box.style.background = 'var(--bg-secondary)';
        box.style.borderRadius = '4px';
        const title = document.createElement('strong');
        title.textContent = 'Execution complete:';
        box.appendChild(title);
        box.appendChild(document.createElement('br'));
        [['Orders submitted', result.orders_submitted], ['Orders rejected', result.orders_rejected], ['Liquidations', result.liquidations]].forEach(([label, value]) => {
            const line = document.createElement('div');
            line.textContent = `${label}: ${value}`;
            box.appendChild(line);
        });
        div.appendChild(box);
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
