"""CouncilStrategy — NautilusTrader strategy che consuma i segnali MLCouncil.

Parità backtest-to-live
-----------------------
Lo stesso codice gira identico in backtest e paper/live trading.
L'unica differenza è il VenueAdapter usato dal runner:
  - Backtest  : SimulatedExchange (BacktestEngine)
  - Paper/Live: AlpacaAdapter     (LiveNode)

Flusso giornaliero (EOD)
------------------------
1. ``on_bar()`` riceve la barra di chiusura giornaliera (DAY bar).
2. Legge il segnale aggregato da ``data/orders/{today}.parquet``
   (prodotto dalla pipeline Dagster).
3. Calcola il delta pesi vs posizioni correnti.
4. Invia MarketOrder da eseguire all'apertura T+1.

Slippage simulato
-----------------
In backtest il FillModel è configurato con ``prob_slippage`` nel runner;
qui applichiamo anche un aggiustamento esplicito del prezzo atteso
via ``slippage_bps`` per scopi di analisi (slippage attribution).
"""

from __future__ import annotations

import sys
from datetime import date, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# NautilusTrader imports — fallback a shim se non disponibile
try:
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.enums import (
        BarAggregation,
        PriceType,
        AggregationSource,
        OrderSide,
        TimeInForce,
    )
    from nautilus_trader.model.events import OrderFilled
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Quantity, Price

    _NT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NT_AVAILABLE = False
    # Shims minimali per permettere l'importazione senza NautilusTrader
    StrategyConfig = object
    Strategy = object
    Bar = object
    OrderFilled = object

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ORDERS_DIR = _ROOT / "data" / "orders"


# ===========================================================================
# Config
# ===========================================================================

if _NT_AVAILABLE:
    class CouncilStrategyConfig(StrategyConfig, frozen=True):
        """Configurazione immutabile per CouncilStrategy."""

        universe: list[str]
        """Lista dei ticker dell'universo (es. ['AAPL', 'MSFT', ...])."""

        venue_name: str = "SIM"
        """Nome del venue Nautilus (SimulatedExchange in backtest)."""

        rebalance_freq: str = "daily"
        """Frequenza di ribilanciamento ('daily')."""

        max_positions: int = 20
        """Numero massimo di posizioni simultanee."""

        slippage_bps: float = 3.0
        """Slippage atteso in basis-point (usato per analisi e fill price
        attribution; il FillModel del runner applica lo slippage effettivo)."""

        commission_bps: float = 1.0
        """Commissioni in basis-point per analisi costi."""

        min_order_usd: float = 100.0
        """Ordini di valore assoluto inferiore a questo threshold vengono ignorati."""

        portfolio_value: float = 100_000.0
        """Valore di portafoglio iniziale in USD (per sizing degli ordini)."""
else:  # pragma: no cover
    class CouncilStrategyConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# ===========================================================================
# Strategy
# ===========================================================================

if _NT_AVAILABLE:
    class CouncilStrategy(Strategy):
        """Strategy NautilusTrader che esegue il council ML.

        Gira identica in backtest e paper/live trading.

        Architettura
        ------------
        - ``on_start``        : subscribe ai DAY bar, inizializza tracker
        - ``on_bar``          : a fine giornata legge gli ordini Dagster e li
                                converte in MarketOrder da eseguire su T+1 apertura
        - ``on_order_filled`` : logga il fill, aggiorna slippage tracker

        I segnali pre-calcolati vengono letti da ``data/orders/{date}.parquet``.
        Questo file è prodotto dalla pipeline Dagster e contiene:
            ticker, direction, quantity, target_weight
        """

        def __init__(self, config: CouncilStrategyConfig) -> None:
            super().__init__(config)
            self.universe: list[str] = list(config.universe)
            self.venue_name: str = config.venue_name
            self.rebalance_freq: str = config.rebalance_freq
            self.max_positions: int = config.max_positions
            self.slippage_bps: float = config.slippage_bps
            self.commission_bps: float = config.commission_bps
            self.min_order_usd: float = config.min_order_usd
            self.portfolio_value: float = config.portfolio_value

            # Tracking interno
            self._instrument_ids: dict[str, InstrumentId] = {}
            self._bar_types: dict[str, BarType] = {}
            self._last_bar_date: Optional[date] = None

            # Fill tracker per slippage analysis
            # {client_order_id: {ticker, expected_px, fill_px, qty, side}}
            self._fill_tracker: dict[str, dict] = {}

            # Portfolio snapshot: ticker → weight
            self._current_weights: dict[str, float] = {}

        # ── Lifecycle ────────────────────────────────────────────────────────

        def on_start(self) -> None:
            """Subscribe ai DAY bar di tutti i ticker dell'universo."""
            from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

            venue = Venue(self.venue_name)

            for ticker in self.universe:
                instrument_id = InstrumentId(Symbol(ticker), venue)
                self._instrument_ids[ticker] = instrument_id

                bar_spec = BarSpecification(
                    step=1,
                    aggregation=BarAggregation.DAY,
                    price_type=PriceType.LAST,
                )
                bar_type = BarType(
                    instrument_id=instrument_id,
                    spec=bar_spec,
                    aggregation_source=AggregationSource.EXTERNAL,
                )
                self._bar_types[ticker] = bar_type
                self.subscribe_bars(bar_type)

            self.log.info(
                f"CouncilStrategy avviata — universo: {len(self.universe)} ticker, "
                f"venue={self.venue_name}"
            )

        def on_stop(self) -> None:
            self.cancel_all_orders()
            self.log.info(
                f"CouncilStrategy fermata. "
                f"Fill analizzati: {len(self._fill_tracker)}"
            )

        # ── Bar handler ──────────────────────────────────────────────────────

        def on_bar(self, bar: Bar) -> None:
            """Elabora ogni barra giornaliera.

            Alla ricezione dell'ultimo bar giornaliero, legge le
            istruzioni pre-calcolate da Dagster e invia MarketOrder.
            """
            # Estrai la data dalla barra
            bar_date = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").date()

            # Evita elaborazioni duplicate per lo stesso giorno
            if bar_date == self._last_bar_date:
                return
            self._last_bar_date = bar_date

            # Prova a leggere gli ordini Dagster per questa data
            orders_df = self._load_dagster_orders(bar_date)
            if orders_df is None or orders_df.empty:
                self.log.debug(
                    f"[{bar_date}] Nessun ordine trovato in {_ORDERS_DIR}"
                )
                return

            # Aggiorna il portfolio value stimato
            portfolio_value = self._estimate_portfolio_value()

            # Genera e invia MarketOrder
            orders = self._generate_orders(orders_df, portfolio_value)
            for order in orders:
                self.submit_order(order)

            self.log.info(
                f"[{bar_date}] Inviati {len(orders)} ordini "
                f"(portfolio_value≈${portfolio_value:,.0f})"
            )

        def on_order_filled(self, event: OrderFilled) -> None:
            """Aggiorna il fill tracker e logga il confronto expected vs actual."""
            coid = str(event.client_order_id)

            if coid in self._fill_tracker:
                entry = self._fill_tracker[coid]
                fill_px = float(event.last_px)
                expected_px = entry.get("expected_px", fill_px)
                side_sign = 1.0 if event.is_buy else -1.0

                # Slippage in bps: (fill - expected) / expected * 10000 * sign
                # Positivo = slippage avverso
                if expected_px > 0:
                    slippage_actual_bps = (
                        (fill_px - expected_px) / expected_px * 10_000 * side_sign
                    )
                else:
                    slippage_actual_bps = 0.0

                entry["fill_px"] = fill_px
                entry["slippage_actual_bps"] = slippage_actual_bps

                self.log.debug(
                    f"Fill {coid}: ticker={entry['ticker']} "
                    f"expected_px={expected_px:.4f} fill_px={fill_px:.4f} "
                    f"slippage={slippage_actual_bps:.2f} bps"
                )

            # Aggiorna i pesi correnti stimati
            ticker = self._ticker_from_instrument_id(event.instrument_id)
            if ticker:
                qty = float(event.last_qty)
                if event.is_buy:
                    self._current_weights[ticker] = (
                        self._current_weights.get(ticker, 0.0) + qty * float(event.last_px)
                    )
                else:
                    self._current_weights[ticker] = max(
                        0.0,
                        self._current_weights.get(ticker, 0.0) - qty * float(event.last_px),
                    )

        # ── Order generation ─────────────────────────────────────────────────

        def _generate_orders(
            self,
            orders_df: pd.DataFrame,
            portfolio_value: float,
        ) -> list:
            """Converte il DataFrame ordini Dagster in MarketOrder NautilusTrader.

            Il fill avviene sull'apertura del giorno T+1 (bar_execution=True
            nel BacktestEngine, che usa il prezzo di apertura della barra
            successiva come fill price).

            Parameters
            ----------
            orders_df : pd.DataFrame
                Colonne attese: ticker, direction, quantity, target_weight.
                ``quantity`` è in USD.
            portfolio_value : float
                Valore corrente del portafoglio in USD.

            Returns
            -------
            list[MarketOrder]
            """
            market_orders = []

            for _, row in orders_df.iterrows():
                ticker = str(row["ticker"])
                direction = str(row["direction"]).lower()
                usd_qty = float(row.get("quantity", 0.0))

                if ticker not in self._instrument_ids:
                    self.log.warning(f"Ticker {ticker!r} non nell'universo — skip")
                    continue

                if abs(usd_qty) < self.min_order_usd:
                    continue

                instrument_id = self._instrument_ids[ticker]

                # Ottieni l'ultimo prezzo per convertire USD → shares
                last_price = self._get_last_price(instrument_id)
                if last_price is None or last_price <= 0:
                    self.log.warning(f"Nessun prezzo per {ticker} — skip")
                    continue

                # Converti USD in numero di azioni (intero)
                n_shares = max(1, int(usd_qty / last_price))

                order_side = (
                    OrderSide.BUY if direction == "buy" else OrderSide.SELL
                )

                # Expected fill price con slippage simulato
                expected_px = last_price * (
                    1 + (self.slippage_bps / 10_000) * (1 if order_side == OrderSide.BUY else -1)
                )

                order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=order_side,
                    quantity=Quantity.from_int(n_shares),
                    time_in_force=TimeInForce.DAY,
                )

                # Registra nel tracker per analisi slippage successiva
                self._fill_tracker[str(order.client_order_id)] = {
                    "ticker":      ticker,
                    "expected_px": expected_px,
                    "side":        direction,
                    "qty_shares":  n_shares,
                    "qty_usd":     usd_qty,
                }

                market_orders.append(order)

            return market_orders

        # ── Helpers ──────────────────────────────────────────────────────────

        def _load_dagster_orders(self, bar_date: date) -> Optional[pd.DataFrame]:
            """Legge il file ordini generato dalla pipeline Dagster."""
            path = _ORDERS_DIR / f"{bar_date}.parquet"
            if not path.exists():
                return None
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                self.log.error(f"Errore lettura ordini {path}: {exc}")
                return None

        def _estimate_portfolio_value(self) -> float:
            """Stima il valore del portafoglio sommando le posizioni aperte."""
            total = 0.0
            for ticker, instrument_id in self._instrument_ids.items():
                net_pos = self.portfolio.net_position(instrument_id)
                if net_pos != 0:
                    price = self._get_last_price(instrument_id) or 0.0
                    total += abs(float(net_pos)) * price
            # Fallback al valore iniziale configurato
            return total if total > 0 else self.portfolio_value

        def _get_last_price(self, instrument_id: InstrumentId) -> Optional[float]:
            """Recupera l'ultimo prezzo (close) per un strumento dalla cache."""
            ticker = self._ticker_from_instrument_id(instrument_id)
            if ticker and ticker in self._bar_types:
                bar_type = self._bar_types[ticker]
                bar = self.cache.bar(bar_type)
                if bar is not None:
                    return float(bar.close)
            return None

        def _ticker_from_instrument_id(self, instrument_id: InstrumentId) -> Optional[str]:
            """Reverse lookup: InstrumentId → ticker string."""
            for ticker, iid in self._instrument_ids.items():
                if iid == instrument_id:
                    return ticker
            return None

        def get_fill_report(self) -> pd.DataFrame:
            """Ritorna un DataFrame con l'analisi dei fill e dello slippage."""
            rows = list(self._fill_tracker.values())
            if not rows:
                return pd.DataFrame(
                    columns=[
                        "ticker", "side", "qty_shares", "qty_usd",
                        "expected_px", "fill_px", "slippage_actual_bps",
                    ]
                )
            df = pd.DataFrame(rows)
            for col in ["fill_px", "slippage_actual_bps"]:
                if col not in df.columns:
                    df[col] = float("nan")
            return df

else:  # pragma: no cover — shim quando NT non è installato

    class CouncilStrategy:  # type: ignore[no-redef]
        """Shim per ambienti senza NautilusTrader."""

        def __init__(self, config):
            self.config = config
