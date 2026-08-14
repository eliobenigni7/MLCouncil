import json, sqlite3, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path('/mnt/data/MLCouncil')
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'data' / 'results'
RISK_DIR = ROOT / 'data' / 'risk'
PDF_PATH = ROOT / 'mlcouncil_report_2026-04-28.pdf'


def safe_load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x:.1%}"


def pct2(x):
    return f"{x:.2%}"


def text_page(ax, title, lines):
    ax.axis('off')
    ax.text(0, 1.0, title, fontsize=15, fontweight='bold', va='top')
    y = 0.88
    for line in lines:
        ax.text(0, y, line, fontsize=10.5, va='top')
        y -= 0.08


# --- Alpaca/risk snapshots ---
risk_rows = []
for path in sorted(RISK_DIR.glob('risk_report_*.json')):
    dt = pd.to_datetime(path.stem.replace('risk_report_', ''), errors='coerce')
    if pd.isna(dt):
        continue
    payload = safe_load_json(path)
    if not isinstance(payload, dict):
        continue
    exposure = payload.get('exposure') if isinstance(payload.get('exposure'), dict) else {}
    risk_rows.append({
        'date': dt.normalize(),
        'portfolio_value': float(payload.get('portfolio_value', np.nan)),
        'net_exposure': float(exposure.get('net_exposure', np.nan)),
        'gross_exposure': float(exposure.get('gross_exposure', np.nan)),
        'drawdown_current': float(payload.get('max_drawdown_current', np.nan)),
        'sharpe_estimate': float(payload.get('sharpe_estimate', np.nan)),
        'breaches': len(payload.get('breaches', []) or []),
    })
risk_df = pd.DataFrame(risk_rows).dropna(subset=['date']).sort_values('date')
if not risk_df.empty:
    risk_df = risk_df.drop_duplicates('date', keep='last').set_index('date')

# --- Backtest ---
backtest_stats = {}
equity = pd.Series(dtype=float)
gross_equity = pd.Series(dtype=float)
try:
    sys.path.insert(0, str(ROOT))
    with open(RESULTS / 'backtest_result.pkl', 'rb') as f:
        bt = pickle.load(f)
    backtest_stats = getattr(bt, 'stats', {}) or {}
    equity = getattr(bt, 'equity_curve', pd.Series(dtype=float)).copy()
    gross_equity = getattr(bt, 'gross_equity_curve', pd.Series(dtype=float)).copy()
except Exception:
    pass
if not equity.empty:
    equity.index = pd.to_datetime(equity.index)
if not gross_equity.empty:
    gross_equity.index = pd.to_datetime(gross_equity.index)

if equity.empty and (RESULTS / 'equity_curve.parquet').exists():
    eq_df = pd.read_parquet(RESULTS / 'equity_curve.parquet')
    equity = eq_df.iloc[:, 0].copy()
    equity.index = pd.to_datetime(equity.index)

benchmark = pd.Series(dtype=float)
sp500_path = ROOT / 'data' / 'raw' / 'macro' / 'sp500.parquet'
if sp500_path.exists() and not equity.empty:
    try:
        sp = pd.read_parquet(sp500_path)
        date_col = 'valid_time' if 'valid_time' in sp.columns else ('date' if 'date' in sp.columns else None)
        price_col = 'sp500_price' if 'sp500_price' in sp.columns else ('close' if 'close' in sp.columns else None)
        if date_col and price_col:
            benchmark = sp[[date_col, price_col]].dropna().copy()
            benchmark[date_col] = pd.to_datetime(benchmark[date_col])
            benchmark = benchmark.set_index(date_col)[price_col].sort_index()
            benchmark = benchmark[(benchmark.index >= equity.index.min()) & (benchmark.index <= equity.index.max())]
            if not benchmark.empty:
                benchmark = benchmark / benchmark.iloc[0] * 100.0
    except Exception:
        benchmark = pd.Series(dtype=float)

net_equity = pd.Series(dtype=float)
gross_equity_n = pd.Series(dtype=float)
if not equity.empty and equity.iloc[0] != 0:
    net_equity = equity / equity.iloc[0] * 100.0
if not gross_equity.empty and gross_equity.iloc[0] != 0:
    gross_equity_n = gross_equity / gross_equity.iloc[0] * 100.0
if not benchmark.empty and not net_equity.empty:
    benchmark = benchmark.reindex(net_equity.index, method='ffill').dropna()

drawdown = pd.Series(dtype=float)
if not net_equity.empty:
    drawdown = net_equity / net_equity.cummax() - 1.0

rolling_63 = pd.Series(dtype=float)
if not equity.empty:
    rolling_63 = equity.pct_change().rolling(63).apply(lambda x: np.sqrt(252) * x.mean() / (x.std(ddof=0) + 1e-12), raw=False)

wf_summary = safe_load_json(RESULTS / 'walk_forward_summary.json') or {}
window_metrics = pd.read_parquet(RESULTS / 'walk_forward_windows.parquet') if (RESULTS / 'walk_forward_windows.parquet').exists() else pd.DataFrame()
wf_regime = pd.read_parquet(RESULTS / 'walk_forward_regime.parquet') if (RESULTS / 'walk_forward_regime.parquet').exists() else pd.DataFrame()
wf_benchmark = pd.read_parquet(RESULTS / 'walk_forward_benchmark.parquet') if (RESULTS / 'walk_forward_benchmark.parquet').exists() else pd.DataFrame()

# --- MLflow model evolution ---
mlflow_df = pd.DataFrame()
mlflow_db = ROOT / 'mlflow.db'
if mlflow_db.exists():
    conn = sqlite3.connect(str(mlflow_db))
    q = """
    select r.run_uuid, r.name, r.start_time, r.end_time,
           max(case when m.key='ic_mean' then m.value end) as ic_mean,
           max(case when m.key='ic_std' then m.value end) as ic_std,
           max(case when m.key='icir' then m.value end) as icir
    from runs r
    join latest_metrics m on r.run_uuid = m.run_uuid
    where r.name like 'lgbm_%'
    group by r.run_uuid, r.name, r.start_time, r.end_time
    order by r.start_time
    """
    mlflow_df = pd.read_sql_query(q, conn)
    conn.close()
if not mlflow_df.empty:
    mlflow_df['started_at'] = pd.to_datetime(mlflow_df['start_time'], unit='ms')
    mlflow_df['date'] = mlflow_df['started_at'].dt.normalize()
    daily = mlflow_df.groupby('date').agg(
        runs=('run_uuid', 'count'),
        ic_mean_median=('ic_mean', 'median'),
        ic_mean_max=('ic_mean', 'max'),
        icir_median=('icir', 'median'),
        icir_max=('icir', 'max'),
    ).reset_index()
else:
    daily = pd.DataFrame()

hmm_pkl = ROOT / 'models' / 'checkpoints' / 'hmm_latest.pkl'
lgbm_pkl = ROOT / 'models' / 'checkpoints' / 'lgbm_latest.pkl'
checkpoints = []
for name, p in [('LGBM', lgbm_pkl), ('HMM', hmm_pkl)]:
    if p.exists():
        checkpoints.append({'model': name, 'mtime': pd.to_datetime(p.stat().st_mtime, unit='s'), 'size_kb': p.stat().st_size / 1024.0})
checkpoints_df = pd.DataFrame(checkpoints)
current_regime = safe_load_json(RESULTS / 'current_regime.json') or {}

latest_risk = risk_df.iloc[-1] if not risk_df.empty else None
first_risk = risk_df.iloc[0] if not risk_df.empty else None
risk_return = np.nan
if latest_risk is not None and first_risk is not None and first_risk['portfolio_value']:
    risk_return = latest_risk['portfolio_value'] / first_risk['portfolio_value'] - 1.0
latest_mlflow = mlflow_df.sort_values('start_time').iloc[-1] if not mlflow_df.empty else None
best_mlflow = mlflow_df.sort_values('ic_mean', ascending=False).iloc[0] if not mlflow_df.empty else None

with PdfPages(PDF_PATH) as pdf:
    # 1) Summary
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle('MLCouncil — report operativo', fontsize=20, fontweight='bold', y=0.98)
    lines = []
    if latest_risk is not None:
        lines += [
            f"Portfolio Alpaca: {money(latest_risk['portfolio_value'])}",
            f"Return vs primo snapshot: {pct(risk_return)}",
            f"Net/Gross exposure: {pct2(latest_risk['net_exposure'])} / {pct2(latest_risk['gross_exposure'])}",
            f"Drawdown corrente: {pct2(latest_risk['drawdown_current'])}",
            f"Sharpe stimato: {latest_risk['sharpe_estimate']:.2f}",
            f"Breaches: {int(latest_risk['breaches'])}",
        ]
    if backtest_stats:
        lines += [
            '',
            f"Backtest: Sharpe {backtest_stats.get('sharpe', float('nan')):.2f} | MaxDD {pct2(backtest_stats.get('max_drawdown', float('nan')))} | CAGR {pct(backtest_stats.get('cagr', float('nan')))}",
            f"Final equity: {money(backtest_stats.get('final_equity', float('nan')))} | Costi stimati: {money(backtest_stats.get('estimated_costs_usd', float('nan')))} | Turnover: {pct(backtest_stats.get('turnover', float('nan')))}",
        ]
    if wf_summary:
        wf = wf_summary.get('walk_forward', {})
        lines += [
            '',
            f"Walk-forward OOS Sharpe: {wf.get('oos_sharpe', float('nan')):.2f} | OOS MaxDD: {pct2(wf.get('oos_max_drawdown', float('nan')))} | PBO: {wf.get('pbo', float('nan')):.2f}",
            f"Window count: {wf.get('walk_forward_window_count', 'n/d')} | OOS turnover: {pct(wf.get('oos_turnover', float('nan')))}",
            f"Equal-weight delta Sharpe: {wf.get('equal_weight_sharpe_delta', float('nan')):.2f}",
        ]
    if latest_mlflow is not None:
        lines += [
            '',
            f"LGBM latest run: {pd.to_datetime(latest_mlflow['start_time'], unit='ms').date()} | IC mean {latest_mlflow['ic_mean']:.4f} | ICIR {latest_mlflow['icir']:.3f}",
            f"LGBM best run: {pd.to_datetime(best_mlflow['start_time'], unit='ms').date()} | IC mean {best_mlflow['ic_mean']:.4f} | ICIR {best_mlflow['icir']:.3f}",
        ]
    if current_regime:
        lines += ["", f"Regime corrente: {current_regime.get('regime', 'n/d')} | bull {current_regime.get('bull', 0.0):.2%} | bear {current_regime.get('bear', 0.0):.2%} | transition {current_regime.get('transition', 0.0):.2%}"]
    text_page(axes[0, 0], 'Sintesi', lines)

    axes[0, 1].axis('off')
    axes[0, 1].text(0, 1.0, 'Indicatori chiave', fontsize=15, fontweight='bold', va='top')
    rows = []
    if backtest_stats:
        rows += [
            ['Backtest Sharpe', f"{backtest_stats.get('sharpe', float('nan')):.2f}"],
            ['Backtest MaxDD', pct2(backtest_stats.get('max_drawdown', float('nan')))],
            ['Backtest CAGR', pct(backtest_stats.get('cagr', float('nan')))],
            ['Backtest Final Equity', money(backtest_stats.get('final_equity', float('nan')))],
        ]
    if wf_summary:
        wf = wf_summary.get('walk_forward', {})
        rows += [
            ['WF OOS Sharpe', f"{wf.get('oos_sharpe', float('nan')):.2f}"],
            ['WF OOS MaxDD', pct2(wf.get('oos_max_drawdown', float('nan')))],
            ['WF PBO', f"{wf.get('pbo', float('nan')):.2f}"],
            ['WF Windows', str(wf.get('walk_forward_window_count', 'n/d'))],
        ]
    if latest_risk is not None:
        rows += [
            ['Alpaca Value', money(latest_risk['portfolio_value'])],
            ['Alpaca Net Exposure', pct2(latest_risk['net_exposure'])],
            ['Alpaca DD Corrente', pct2(latest_risk['drawdown_current'])],
        ]
    if latest_mlflow is not None:
        rows += [
            ['LGBM Latest IC Mean', f"{latest_mlflow['ic_mean']:.4f}"],
            ['LGBM Best IC Mean', f"{best_mlflow['ic_mean']:.4f}"],
            ['HMM checkpoint mtime', str(pd.to_datetime(hmm_pkl.stat().st_mtime, unit='s').date()) if hmm_pkl.exists() else 'n/d'],
        ]
    tbl = axes[0, 1].table(cellText=rows[:10], colLabels=['Metrica', 'Valore'], loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.35)

    axes[1, 0].axis('off')
    extra = []
    if not checkpoints_df.empty:
        for _, row in checkpoints_df.sort_values('model').iterrows():
            extra.append(f"{row['model']}: {row['mtime'].date()} | {row['size_kb']:.1f} KB")
    if current_regime:
        extra.append(f"Regime corrente: {current_regime.get('regime', 'n/d')}")
    text_page(axes[1, 0], 'Checkpoint / stato corrente', extra or ['Nessun metadato disponibile'])
    axes[1, 1].axis('off')
    axes[1, 1].text(0, 0.95, 'Note di lettura', fontsize=15, fontweight='bold', va='top')
    note_lines = [
        '• Il report usa snapshot rischio reali dal repository locale.',
        '• Il backtest è quello strategico end-to-end già persistito in data/results/.',
        '• L evoluzione modelli deriva dai run MLflow LGBM registrati nel DB locale.',
        '• Se un grafico non ha dati, la pagina resta vuota invece di inventare valori.',
    ]
    y = 0.82
    for line in note_lines:
        axes[1, 1].text(0, y, line, fontsize=10.5, va='top')
        y -= 0.12
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 2) Alpaca evolution
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Andamento portafoglio Alpaca', fontsize=18, fontweight='bold', y=0.98)
    if not risk_df.empty:
        ax = axes[0]
        ax.plot(risk_df.index, risk_df['portfolio_value'], color='#1f77b4', linewidth=2.2, label='Portfolio value')
        ax.set_ylabel('USD')
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(risk_df.index, risk_df['net_exposure'], color='#ff7f0e', linewidth=1.7, linestyle='--', label='Net exposure')
        ax2.set_ylabel('Exposure')
        ax2.set_ylim(0, max(0.1, float(np.nanmax(risk_df['net_exposure'].fillna(0))) * 1.4))
        l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc='upper left')
        ax.set_title('Valore portafoglio e esposizione netta')
        dd = risk_df['portfolio_value'] / risk_df['portfolio_value'].cummax() - 1.0
        axes[1].fill_between(risk_df.index, dd * 100, 0, color='#d62728', alpha=0.25)
        axes[1].plot(risk_df.index, dd * 100, color='#d62728', linewidth=1.8)
        axes[1].set_ylabel('Drawdown %')
        axes[1].set_xlabel('Data')
        axes[1].grid(True, alpha=0.25)
        axes[1].set_title('Drawdown del portafoglio')
    else:
        text_page(axes[0], 'Portafoglio Alpaca', ['Nessuno snapshot rischio disponibile'])
        axes[1].axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 3) Backtest results
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle('Risultati backtest', fontsize=18, fontweight='bold', y=0.98)
    ax = axes[0, 0]
    if not net_equity.empty:
        ax.plot(net_equity.index, net_equity.values, label='Strategy (net)', color='#2ca02c', linewidth=2.0)
    if not gross_equity_n.empty:
        ax.plot(gross_equity_n.index, gross_equity_n.values, label='Strategy (gross)', color='#17becf', linewidth=1.5, linestyle='--')
    if not benchmark.empty:
        ax.plot(benchmark.index, benchmark.values, label='SP500 benchmark', color='#9467bd', linewidth=1.5)
    ax.set_title('Equity curve normalizzata')
    ax.set_ylabel('Base 100')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if not drawdown.empty:
        ax.fill_between(drawdown.index, drawdown.values * 100, 0, color='#d62728', alpha=0.25)
        ax.plot(drawdown.index, drawdown.values * 100, color='#d62728', linewidth=1.8)
    ax.set_title('Drawdown netto')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if not rolling_63.empty:
        ax.plot(rolling_63.index, rolling_63.values, color='#1f77b4', linewidth=1.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Rolling Sharpe 63 giorni')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    if not wf_benchmark.empty:
        comp = wf_benchmark.copy().sort_values('benchmark_sharpe', ascending=True)
        y = np.arange(len(comp))
        ax.barh(y - 0.17, comp['strategy_sharpe'], height=0.32, label='Strategy Sharpe', color='#2ca02c')
        ax.barh(y + 0.17, comp['benchmark_sharpe'], height=0.32, label='Benchmark Sharpe', color='#7f7f7f')
        ax.set_yticks(y)
        ax.set_yticklabels(comp['benchmark'])
        ax.set_title('Sharpe vs benchmark')
        ax.legend(fontsize=8)
        ax.grid(True, axis='x', alpha=0.25)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 4) Walk-forward detail
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle('Walk-forward e regime', fontsize=18, fontweight='bold', y=0.98)
    ax = axes[0, :].ravel()[0]
    if not window_metrics.empty:
        x = np.arange(len(window_metrics))
        ax.plot(x, window_metrics['test_sharpe'], marker='o', color='#1f77b4', linewidth=1.8, label='Test Sharpe')
        ax2 = ax.twinx()
        ax2.bar(x, window_metrics['test_turnover'], alpha=0.18, color='#ff7f0e', label='Turnover')
        ax.set_title('Metriche per finestra walk-forward')
        ax.set_xlabel('Window')
        ax.set_ylabel('Sharpe')
        ax2.set_ylabel('Turnover')
        ax.grid(True, alpha=0.25)
        l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=8)
    else:
        text_page(ax, 'Walk-forward', ['Nessun dato window_metrics disponibile'])
    ax = axes[1, 0]
    if not wf_regime.empty:
        ax.bar(wf_regime['regime'], wf_regime['sharpe'], color=['#2ca02c', '#d62728', '#ff7f0e'])
        ax.set_title('Sharpe per regime')
        ax.set_ylabel('Sharpe')
        ax.grid(True, axis='y', alpha=0.25)
    ax = axes[1, 1]
    if not wf_regime.empty:
        ax.bar(wf_regime['regime'], wf_regime['n_obs'], color='#9467bd')
        ax.set_title('Osservazioni per regime')
        ax.set_ylabel('n obs')
        ax.grid(True, axis='y', alpha=0.25)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 5) Model evolution
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=False)
    fig.suptitle('Evoluzione modelli', fontsize=18, fontweight='bold', y=0.98)
    ax = axes[0]
    if not mlflow_df.empty:
        ax.scatter(mlflow_df['started_at'], mlflow_df['ic_mean'], s=14, alpha=0.28, color='#7f7f7f', label='All runs')
        if not daily.empty:
            ax.plot(daily['date'], daily['ic_mean_median'], color='#1f77b4', linewidth=2.2, label='Daily median IC mean')
            ax.plot(daily['date'], daily['ic_mean_max'], color='#ff7f0e', linewidth=2.0, linestyle='--', label='Daily best IC mean')
        if latest_mlflow is not None:
            latest_date = pd.to_datetime(latest_mlflow['start_time'], unit='ms')
            ax.scatter([latest_date], [latest_mlflow['ic_mean']], s=90, color='#d62728', zorder=5, label='Latest run')
            ax.annotate(f"latest\n{latest_mlflow['ic_mean']:.4f}", (latest_date, latest_mlflow['ic_mean']), textcoords='offset points', xytext=(8, 8), fontsize=9)
        if best_mlflow is not None:
            best_date = pd.to_datetime(best_mlflow['start_time'], unit='ms')
            ax.scatter([best_date], [best_mlflow['ic_mean']], s=90, color='#2ca02c', zorder=5, label='Best run')
            ax.annotate(f"best\n{best_mlflow['ic_mean']:.4f}", (best_date, best_mlflow['ic_mean']), textcoords='offset points', xytext=(8, -18), fontsize=9)
        ax.set_title('LGBM IC mean per run')
        ax.set_ylabel('IC mean')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
    else:
        text_page(ax, 'LGBM', ['Nessun run MLflow trovato'])

    ax = axes[1]
    if not mlflow_df.empty:
        ax.scatter(mlflow_df['started_at'], mlflow_df['icir'], s=14, alpha=0.25, color='#8c564b')
        if not daily.empty:
            ax.plot(daily['date'], daily['icir_median'], color='#9467bd', linewidth=2.0, label='Daily median ICIR')
            ax.plot(daily['date'], daily['icir_max'], color='#17becf', linewidth=1.8, linestyle='--', label='Daily best ICIR')
        ax.set_title('ICIR per run')
        ax.set_ylabel('ICIR')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 6) Top runs and notes
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={'height_ratios': [1.6, 1]})
    ax = axes[0]
    ax.axis('off')
    top_runs = mlflow_df.sort_values('ic_mean', ascending=False).head(10).copy() if not mlflow_df.empty else pd.DataFrame()
    if not top_runs.empty:
        top_runs['date'] = top_runs['started_at'].dt.date.astype(str)
        show = top_runs[['date', 'name', 'ic_mean', 'icir']].copy()
        show['ic_mean'] = show['ic_mean'].map(lambda x: f"{x:.4f}")
        show['icir'] = show['icir'].map(lambda x: f"{x:.3f}")
        table = ax.table(cellText=show.values.tolist(), colLabels=['Data', 'Run', 'IC mean', 'ICIR'], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8.8)
        table.scale(1, 1.2)
        ax.set_title('Top 10 run LGBM per IC mean', fontsize=13, fontweight='bold')
    else:
        text_page(ax, 'Top run', ['Nessun dato'])

    ax = axes[1]
    ax.axis('off')
    notes = []
    if not top_runs.empty:
        notes.append(f"Miglior IC mean osservato: {top_runs.iloc[0]['ic_mean']:.4f} il {top_runs.iloc[0]['date']}")
        if latest_mlflow is not None:
            notes.append(f"Ultimo retrain: {pd.to_datetime(latest_mlflow['start_time'], unit='ms').date()} con IC mean {latest_mlflow['ic_mean']:.4f}")
    if hmm_pkl.exists():
        notes.append(f"HMM checkpoint: {pd.to_datetime(hmm_pkl.stat().st_mtime, unit='s').date()}")
    notes.append('Ablation WF: non disponibile nel risultato corrente (parquet vuoto).')
    text_page(ax, 'Osservazioni', notes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

summary = {
    'pdf_path': str(PDF_PATH),
    'risk_rows': int(len(risk_df)),
    'mlflow_runs': int(len(mlflow_df)),
    'backtest_final_equity': backtest_stats.get('final_equity'),
    'backtest_sharpe': backtest_stats.get('sharpe'),
    'wf_oos_sharpe': wf_summary.get('walk_forward', {}).get('oos_sharpe') if wf_summary else None,
}
(ROOT / 'mlcouncil_report_2026-04-28_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
print(json.dumps(summary, indent=2, default=str))
