"""Trading service - executes orders to Alpaca Paper with safety guards."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from council.artifacts import write_artifact_manifest
from data.contracts import validate_asset_contract
from runtime_env import get_runtime_profile, get_trading_settings

if TYPE_CHECKING:
    from execution.alpaca_adapter import AlpacaConfig, AlpacaLiveNode

_ROOT = Path(__file__).parents[2]
TRADE_LOG_DIR = _ROOT / "data" / "paper_trades"
OPERATIONS_DIR = _ROOT / "data" / "operations"
INTRADAY_OPERATIONS_DIR = OPERATIONS_DIR / "intraday"
ORDERS_DIR = _ROOT / "data" / "orders"
_TRADE_LOG_LOCK = threading.Lock()


class TradingService:
    def __init__(self):
        self._node: Optional["AlpacaLiveNode"] = None
        self._config: Optional["AlpacaConfig"] = None
        self._risk_engine = None
        self._alert_dispatcher = None

    def _get_config(self) -> "AlpacaConfig":
        if self._config is None:
            from execution.alpaca_adapter import AlpacaConfig

            self._config = AlpacaConfig.from_env()
        return self._config

    @property
    def node(self) -> "AlpacaLiveNode":
        if self._node is None:
            from execution.alpaca_adapter import AlpacaLiveNode

            self._node = AlpacaLiveNode(self._get_config())
        return self._node

    @property
    def risk_engine(self):
        if getattr(self, "_risk_engine", None) is None:
            from council.risk_engine import RiskEngine

            self._risk_engine = RiskEngine()
        return self._risk_engine

    @property
    def trading_settings(self):
        return get_trading_settings()

    @property
    def alert_dispatcher(self):
        if getattr(self, "_alert_dispatcher", None) is None:
            from council.alerts import AlertDispatcher

            self._alert_dispatcher = AlertDispatcher()
        return self._alert_dispatcher

    @property
    def is_paper(self) -> bool:
        config = self._get_config()
        return config.mode.value == "paper"

    def get_status(self) -> dict:
        """Return Alpaca connection status, runtime state, and account info."""
        paused = self._is_automation_paused()
        guard_error = self._paper_guard_error()
        base = {
            "connected": False,
            "paper": self.is_paper,
            "runtime_profile": get_runtime_profile(),
            "paused": paused,
            "kill_switch_active": paused,
            "paper_guard_ok": guard_error is None,
            "account": {},
            "positions": [],
        }
        if guard_error:
            base["error"] = guard_error
            return base

        try:
            account = self.node.get_account_info()
            positions = self.node.get_all_positions()
            base.update(
                {
                    "connected": True,
                    "account": account,
                    "positions": (
                        positions.to_dict(orient="records") if not positions.empty else []
                    ),
                }
            )
            return base
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)
            return base

    def get_pending_orders(self, date: str) -> list[dict]:
        """Load orders from parquet for given date."""
        path = ORDERS_DIR / f"{date}.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def get_latest_order_date(self) -> Optional[str]:
        """Return most recent order file date."""
        if not ORDERS_DIR.exists():
            return None
        files = sorted(ORDERS_DIR.glob("*.parquet"))
        if not files:
            return None
        return files[-1].stem

    def get_reconciliation(self, date: str) -> dict[str, Any]:
        """Return current-vs-target reconciliation for a trade date."""
        return self.build_pretrade_snapshot(date)["reconciliation"]

    def build_pretrade_snapshot(self, date: str) -> dict[str, Any]:
        """Build the pre-trade operational snapshot for a trade date."""
        context = self._prepare_execution_context(date)
        return self._public_context(context)

    def _validate_order(
        self,
        order: dict,
        account: dict,
        positions: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Validate single order against safety limits."""
        del positions  # reserved for richer checks

        max_position = self.trading_settings.max_position_size
        portfolio_value = float(account.get("portfolio_value", 0))
        if portfolio_value <= 0:
            return False, "No buying power"

        symbol = order.get("ticker")
        is_crypto = self.node._is_crypto(str(symbol))
        direction = str(order.get("direction", "buy")).lower()
        quantity = float(order.get("share_quantity") or order.get("quantity") or 0)
        if not is_crypto:
            quantity = int(quantity)
        # Allow fractional-share equity orders: they'll use notional-based
        # submission when rounded qty == 0 but notional is >= $1.
        notional_value = float(order.get("requested_notional", 0) or 0)
        if quantity <= 0 and notional_value < 1.0:
            return False, f"Invalid quantity for {symbol}"

        if direction == "buy":
            target_weight = order.get("target_weight")
            if target_weight is not None:
                weight = float(target_weight)
            else:
                price = float(order.get("estimated_price") or order.get("price") or 0.0)
                requested_notional = float(
                    order.get("requested_notional") or quantity * max(price, 0.0)
                )
                weight = requested_notional / portfolio_value if requested_notional > 0 else 0.0

            if weight > max_position:
                return False, f"{symbol} exceeds max position {max_position:.0%}"

        return True, "OK"

    def execute_orders(self, date: str) -> dict:
        """Execute pending orders for date to Alpaca Paper."""
        existing_execution = self._load_execution_record(date)
        if existing_execution is not None:
            return {
                "error": f"Orders for {date} have already been executed",
                "date": date,
                "pretrade": existing_execution.get("pretrade", {}),
                "reconciliation": existing_execution.get("reconciliation", {}),
                "lineage": existing_execution.get("lineage", {}),
                "operations_path": str(self._execution_record_path(date)),
            }

        context = self._prepare_execution_context(date)
        snapshot = self._public_context(context)

        if snapshot["pretrade"]["blocked"]:
            operations_path = self._write_operations(
                date,
                self._operation_payload(
                    snapshot=snapshot,
                    trade_status="blocked",
                    orders_submitted=0,
                    orders_rejected=0,
                    liquidations=0,
                    warnings=snapshot["pretrade"].get("warnings", []),
                    errors=[snapshot["pretrade"]["reason"]],
                ),
            )
            return {
                "error": snapshot["pretrade"]["reason"],
                "lineage": snapshot["lineage"],
                "pretrade": snapshot["pretrade"],
                "reconciliation": snapshot["reconciliation"],
                "operations_path": str(operations_path),
            }

        account = context["account"]
        positions_df = context["positions_df"]
        orders = context["orders"]
        normalized_orders = context["normalized_orders"]

        target_tickers = {order["ticker"] for order in orders}
        liquidate_results = []
        if not positions_df.empty and "symbol" in positions_df.columns:
            for _, pos in positions_df.iterrows():
                if pos["symbol"] not in target_tickers:
                    liquidation_qty = float(pos["qty"])
                    if not self.node._is_crypto(str(pos["symbol"])):
                        liquidation_qty = int(liquidation_qty)
                    result = self.node.submit_order(pos["symbol"], liquidation_qty, "sell")
                    liquidate_results.append(result)

        order_results = []
        for order in normalized_orders:
            symbol = order["ticker"]
            direction = order["direction"]
            quantity = float(order["share_quantity"])
            if not self.node._is_crypto(symbol):
                quantity = int(quantity)

            valid, msg = self._validate_order(order, account, positions_df)
            if not valid:
                order_results.append(
                    {"symbol": symbol, "status": "rejected", "reason": msg}
                )
                continue

            result = self.node.submit_order(symbol, quantity, direction)
            order_results.append(
                {
                    **result,
                    "symbol": symbol,
                    "requested_notional": float(order["requested_notional"]),
                    "share_quantity": quantity,
                    "estimated_price": float(order["estimated_price"]),
                }
            )

        reconciliation = self._build_reconciliation(
            date=date,
            orders=orders,
            normalized_orders=normalized_orders,
            positions_df=positions_df,
            account=account,
            order_results=order_results,
        )
        snapshot["reconciliation"] = reconciliation

        rejected = [result for result in order_results if result.get("status") == "rejected"]
        if rejected:
            self._dispatch_alert(
                severity="warning",
                check_type="paper_execution_degraded",
                message=f"{len(rejected)} orders rejected during execution for {date}",
                recommendation="Review the rejected symbols before rerunning execute.",
                metric_value=float(len(rejected)),
                threshold=0.0,
            )

        self._log_trade(
            date,
            {
                "orders": order_results,
                "liquidations": liquidate_results,
                "account": account,
                "lineage": snapshot["lineage"],
                "pretrade": snapshot["pretrade"],
                "reconciliation": reconciliation,
            },
        )

        operations_path = self._write_operations(
            date,
            self._operation_payload(
                snapshot=snapshot,
                trade_status="degraded" if rejected else "success",
                orders_submitted=len([r for r in order_results if r.get("status") != "rejected"])
                + len(liquidate_results),
                orders_rejected=len(rejected),
                liquidations=len(liquidate_results),
                warnings=snapshot["pretrade"].get("warnings", []),
                errors=[],
            ),
        )
        self._write_execution_record(date, self._operation_payload(
            snapshot=snapshot,
            trade_status="degraded" if rejected else "success",
            orders_submitted=len([r for r in order_results if r.get("status") != "rejected"])
            + len(liquidate_results),
            orders_rejected=len(rejected),
            liquidations=len(liquidate_results),
            warnings=snapshot["pretrade"].get("warnings", []),
            errors=[],
        ))

        return {
            "date": date,
            "orders_submitted": len(
                [result for result in order_results if result.get("status") != "rejected"]
            ),
            "orders_rejected": len(rejected),
            "liquidations": len(liquidate_results),
            "lineage": snapshot["lineage"],
            "pretrade": snapshot["pretrade"],
            "reconciliation": reconciliation,
            "operations_path": str(operations_path),
            "results": order_results,
        }

    def liquidate_all(self) -> dict:
        """Liquidate all current positions."""
        guard_error = self._paper_guard_error()
        if guard_error:
            return {"error": guard_error}
        try:
            results = self.node.liquidate_all()
            return {"liquidated": len(results), "results": results}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def get_trade_history(self, days: int = 7) -> list[dict]:
        """Load trade history from log files."""
        logs = self._load_trade_logs(days)
        if not logs:
            return []
        return self._sync_trade_history_with_broker(logs)

    def _prepare_execution_context(self, date: str) -> dict[str, Any]:
        orders = self.get_pending_orders(date)
        lineage = self._extract_lineage(orders)
        return self._prepare_execution_context_from_orders(
            execution_key=date,
            orders=orders,
            lineage=lineage,
        )

    def _prepare_execution_context_from_orders(
        self,
        *,
        execution_key: str,
        orders: list[dict[str, Any]],
        lineage: dict[str, str] | None = None,
        symbols_to_close: list[str] | None = None,
    ) -> dict[str, Any]:
        runtime_profile = get_runtime_profile()
        paused = self._is_automation_paused()
        guard_error = self._paper_guard_error()

        account: dict[str, Any] = {}
        positions_df = pd.DataFrame()
        connection_error: str | None = None
        if guard_error is None:
            try:
                account = self.node.get_account_info()
                positions_df = self.node.get_all_positions()
            except Exception as exc:  # noqa: BLE001
                connection_error = str(exc)
                guard_error = connection_error

        extracted_lineage = lineage or self._extract_lineage(orders)
        normalized_orders = [
            self._normalize_order(order, positions_df)
            for order in orders
        ]

        reconciliation = self._build_reconciliation(
            date=execution_key,
            orders=orders,
            normalized_orders=normalized_orders,
            positions_df=positions_df,
            account=account,
        )

        if symbols_to_close is None:
            symbols_to_close = reconciliation["symbols_to_close"]

        pretrade = {
            "blocked": False,
            "reason": None,
            "breaches": [],
            "warnings": [],
            "risk_report_path": None,
            "order_count": len(orders),
            "projected_turnover": reconciliation["projected_turnover"],
            "invalid_orders": [],
        }

        if guard_error:
            pretrade["blocked"] = True
            pretrade["reason"] = guard_error
        elif not orders:
            pretrade["blocked"] = True
            pretrade["reason"] = f"No orders found for {execution_key}"
        elif paused:
            pretrade["blocked"] = True
            pretrade["reason"] = "Trading paused by kill switch"
        else:
            settings = self.trading_settings
            max_daily = settings.max_daily_orders
            max_turnover = settings.max_turnover

            if len(orders) > max_daily:
                pretrade["blocked"] = True
                pretrade["reason"] = f"Order count {len(orders)} exceeds max {max_daily}"
            else:
                invalid_orders = []
                for order in normalized_orders:
                    valid, msg = self._validate_order(order, account, positions_df)
                    if not valid:
                        invalid_orders.append({"symbol": order["ticker"], "reason": msg})

                pretrade["invalid_orders"] = invalid_orders
                if invalid_orders:
                    pretrade["warnings"].append(
                        f"{len(invalid_orders)} orders will be rejected by per-order limits"
                    )

                risk_report = self._compute_projected_risk(
                    date=execution_key,
                    normalized_orders=normalized_orders,
                    positions_df=positions_df,
                    account=account,
                    symbols_to_close=symbols_to_close,
                )
                pretrade["risk_report_path"] = risk_report["path"]
                pretrade["breaches"] = risk_report["breaches"]
                pretrade["warnings"].extend(risk_report["warnings"])
                if risk_report["blocked"]:
                    pretrade["blocked"] = True
                    pretrade["reason"] = risk_report["reason"]
                elif reconciliation["projected_turnover"] > max_turnover:
                    if reconciliation["current_symbols"]:
                        pretrade["blocked"] = True
                        pretrade["reason"] = (
                            f"Projected turnover {reconciliation['projected_turnover']:.2%} "
                            f"exceeds limit {max_turnover:.2%}"
                        )
                    else:
                        pretrade["warnings"].append(
                            "Initial portfolio bootstrap detected; turnover limit waived for first deployment."
                        )

        snapshot = {
            "date": execution_key,
            "paper": self.is_paper,
            "paused": paused,
            "runtime_profile": runtime_profile,
            "lineage": extracted_lineage,
            "pretrade": pretrade,
            "reconciliation": reconciliation,
            "paper_guard_ok": guard_error is None,
            "connection_error": connection_error,
            "account": account,
            "positions": positions_df.to_dict(orient="records") if not positions_df.empty else [],
        }

        if pretrade["blocked"] and pretrade["reason"]:
            severity = "critical" if pretrade["breaches"] else "warning"
            self._dispatch_alert(
                severity=severity,
                check_type="pretrade_risk",
                message=pretrade["reason"],
                recommendation="Resolve the hard stop before executing paper orders.",
                metric_value=float(len(pretrade["breaches"])),
                threshold=0.0,
            )

        snapshot["orders"] = orders
        snapshot["normalized_orders"] = normalized_orders
        snapshot["positions_df"] = positions_df
        snapshot["account"] = account
        return snapshot

    def execute_intraday_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        as_of = str(decision.get("as_of") or datetime.now(timezone.utc).isoformat())
        execution_key = as_of.split("T")[0]
        decision_id = str(decision.get("decision_id") or "").strip()
        if decision_id:
            existing_execution = self._load_intraday_execution_record(decision_id)
            if existing_execution is not None:
                return {
                    **existing_execution,
                    "error": f"Decision {decision_id} has already been executed",
                }
        orders = self._orders_from_intraday_decision(decision)
        orders, sizing_warnings = self._risk_adjust_intraday_orders(orders)
        symbols_to_close = decision.get("symbols_to_close", [])
        lineage = {
            "decision_id": str(decision.get("decision_id", "unknown")),
            "strategy_version": str(decision.get("strategy_version", "intraday-v1")),
            "market_session": str(decision.get("market_session", "unknown")),
        }
        context = self._prepare_execution_context_from_orders(
            execution_key=execution_key,
            orders=orders,
            lineage=lineage,
            symbols_to_close=symbols_to_close,
        )
        snapshot = self._public_context(context)
        if sizing_warnings:
            snapshot["pretrade"].setdefault("warnings", [])
            snapshot["pretrade"]["warnings"].extend(sizing_warnings)

        if snapshot["pretrade"]["blocked"]:
            blocked_payload = {
                "error": snapshot["pretrade"]["reason"],
                "lineage": snapshot["lineage"],
                "pretrade": snapshot["pretrade"],
                "reconciliation": snapshot["reconciliation"],
            }
            operations = self._operation_payload(
                snapshot=snapshot,
                trade_status="blocked",
                orders_submitted=0,
                orders_rejected=0,
                liquidations=0,
                warnings=snapshot["pretrade"].get("warnings", []),
                errors=[snapshot["pretrade"]["reason"]],
            )
            operations_path = self._write_intraday_operations(
                decision_id or execution_key,
                operations,
            )
            blocked_payload["operations_path"] = str(operations_path)
            if decision_id:
                self._write_intraday_execution_record(
                    decision_id,
                    {
                        **blocked_payload,
                        "execution_status": "blocked",
                    },
                )
            return blocked_payload

        account = context["account"]
        positions_df = context["positions_df"]
        normalized_orders = context["normalized_orders"]

        order_results = []
        for order in normalized_orders:
            valid, msg = self._validate_order(order, account, positions_df)
            if not valid:
                order_results.append(
                    {"symbol": order["ticker"], "status": "rejected", "reason": msg}
                )
                continue

            is_crypto = self.node._is_crypto(order["ticker"])
            qty = (
                float(order["share_quantity"])
                if is_crypto
                else int(order["share_quantity"])
            )
            requested_notional = float(order.get("requested_notional", 0.0))

            # For intraday equity orders, always use notional (fractional shares)
            # Alpaca supports fractional shares for equities via notional orders
            if not is_crypto and requested_notional >= 1.0:
                result = self.node.submit_order_notional(
                    order["ticker"],
                    requested_notional,
                    order["direction"],
                )
            elif qty <= 0:
                order_results.append(
                    {"symbol": order["ticker"], "status": "rejected",
                     "reason": f"Rounded qty is 0 (notional ${requested_notional:.0f} < 1 share @ ${order.get('estimated_price', 0):.0f})"}
                )
                continue
            else:
                result = self.node.submit_order(
                    order["ticker"],
                    qty,
                    order["direction"],
                )
            order_results.append(
                {
                    **result,
                    "symbol": order["ticker"],
                    "requested_notional": float(order["requested_notional"]),
                    "share_quantity": (
                        float(order["share_quantity"])
                        if self.node._is_crypto(order["ticker"])
                        else int(order["share_quantity"])
                    ),
                    "estimated_price": float(order["estimated_price"]),
                }
            )

        reconciliation = self._build_reconciliation(
            date=execution_key,
            orders=orders,
            normalized_orders=normalized_orders,
            positions_df=positions_df,
            account=account,
            order_results=order_results,
        )
        snapshot["reconciliation"] = reconciliation
        trade_status = (
            "degraded"
            if any(result.get("status") == "rejected" for result in order_results)
            else "success"
        )
        operations = self._operation_payload(
            snapshot=snapshot,
            trade_status=trade_status,
            orders_submitted=len(
                [result for result in order_results if result.get("status") != "rejected"]
            ),
            orders_rejected=len(
                [result for result in order_results if result.get("status") == "rejected"]
            ),
            liquidations=0,
            warnings=snapshot["pretrade"].get("warnings", []),
            errors=[],
        )
        operations_path = self._write_intraday_operations(
            decision_id or execution_key,
            operations,
        )
        self._log_trade(
            execution_key,
            {
                "orders": order_results,
                "account": account,
                "lineage": snapshot["lineage"],
                "pretrade": snapshot["pretrade"],
                "reconciliation": reconciliation,
            },
        )
        result_payload = {
            "date": execution_key,
            "orders_submitted": len(
                [result for result in order_results if result.get("status") != "rejected"]
            ),
            "orders_rejected": len(
                [result for result in order_results if result.get("status") == "rejected"]
            ),
            "liquidations": 0,
            "lineage": snapshot["lineage"],
            "pretrade": snapshot["pretrade"],
            "reconciliation": reconciliation,
            "results": order_results,
            "operations_path": str(operations_path),
        }
        if decision_id:
            self._write_intraday_execution_record(
                decision_id,
                {
                    **result_payload,
                    "execution_status": trade_status,
                },
            )
        return result_payload

    def _paper_guard_error(self) -> str | None:
        if not self.is_paper:
            return "Live trading blocked - paper mode only"

        base_url = self.trading_settings.alpaca_base_url
        if base_url and "paper-api.alpaca.markets" not in base_url:
            return "Trading blocked - ALPACA_BASE_URL must point to Alpaca paper"

        return None

    def _load_trade_logs(self, days: int) -> list[dict[str, Any]]:
        if not TRADE_LOG_DIR.exists():
            return []

        logs: list[dict[str, Any]] = []
        for payload in sorted(TRADE_LOG_DIR.glob("*.json"))[-days:]:
            try:
                entries = json.loads(payload.read_text())
                if isinstance(entries, list):
                    logs.extend(
                        entry
                        for entry in entries
                        if isinstance(entry, dict) and self._is_trade_log_entry(entry)
                    )
            except Exception:  # noqa: BLE001
                continue
        return logs

    @staticmethod
    def _is_trade_log_entry(entry: dict[str, Any]) -> bool:
        required = {"order_id", "symbol", "side", "status"}
        return required.issubset(entry)

    def _sync_trade_history_with_broker(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            live_orders = self.node.list_orders(status="all", limit=max(len(logs) * 2, 50))
        except Exception:  # noqa: BLE001
            return logs

        if not live_orders:
            return logs

        live_by_id = {
            order.get("order_id"): order for order in live_orders if order.get("order_id")
        }
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in logs:
            order_id = entry.get("order_id")
            if order_id and order_id in live_by_id:
                seen_ids.add(order_id)
                merged.append({**entry, **live_by_id[order_id]})
            else:
                merged.append(entry)

        for order_id, order in live_by_id.items():
            if order_id not in seen_ids:
                merged.append(order)

        merged.sort(
            key=lambda item: str(item.get("submitted_at") or item.get("filled_at") or ""),
            reverse=True,
        )
        return merged

    def _is_automation_paused(self) -> bool:
        raw = os.getenv("MLCOUNCIL_AUTOMATION_PAUSED")
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return self.trading_settings.automation_paused

    def _normalize_order(self, order: dict, positions_df: pd.DataFrame) -> dict[str, Any]:
        symbol = str(order.get("ticker"))
        raw_quantity = float(order.get("quantity", 0) or 0)
        target_weight = order.get("target_weight")

        current_position_qty = 0.0
        current_price = 0.0
        if not positions_df.empty and "symbol" in positions_df.columns:
            matches = positions_df.loc[positions_df["symbol"] == symbol]
            if not matches.empty:
                row = matches.iloc[0]
                current_position_qty = float(row.get("qty", 0) or 0)
                current_price = float(
                    row.get("current_price", row.get("avg_price", 0.0)) or 0.0
                )

        estimated_price = float(order.get("price") or 0.0)
        if estimated_price <= 0:
            estimated_price = current_price
        if estimated_price <= 0:
            try:
                estimated_price = float(self.node.get_latest_price(symbol))
            except Exception:  # noqa: BLE001
                estimated_price = 0.0

        if target_weight is not None:
            requested_notional = abs(raw_quantity)
            share_quantity = (
                requested_notional / estimated_price if estimated_price > 0 else 0.0
            )
        else:
            share_quantity = raw_quantity
            requested_notional = (
                abs(share_quantity * estimated_price) if estimated_price > 0 else 0.0
            )

        if not self.node._is_crypto(symbol):
            current_position_qty = int(current_position_qty)
            share_quantity = int(share_quantity)

        return {
            **order,
            "direction": str(order.get("direction", "buy")).lower(),
            "estimated_price": estimated_price,
            "requested_notional": requested_notional,
            "share_quantity": share_quantity,
            "current_position_qty": current_position_qty,
        }

    def _build_reconciliation(
        self,
        *,
        date: str,
        orders: list[dict],
        normalized_orders: list[dict],
        positions_df: pd.DataFrame,
        account: dict,
        order_results: list[dict] | None = None,
    ) -> dict[str, Any]:
        target_symbols = sorted({order["ticker"] for order in orders})
        current_symbols = []
        if not positions_df.empty and "symbol" in positions_df.columns:
            current_symbols = sorted(positions_df["symbol"].astype(str).tolist())

        target_set = set(target_symbols)
        current_set = set(current_symbols)
        portfolio_value = float(account.get("portfolio_value", 0) or 0)
        projected_turnover = 0.0
        if portfolio_value > 0:
            projected_turnover = sum(
                float(order.get("requested_notional", 0.0)) for order in normalized_orders
            ) / portfolio_value

        reconciliation = {
            "date": date,
            "target_symbols": target_symbols,
            "current_symbols": current_symbols,
            "symbols_to_open": sorted(target_set - current_set),
            "symbols_to_close": sorted(current_set - target_set),
            "symbols_to_resize": sorted(target_set & current_set),
            "projected_turnover": projected_turnover,
        }

        if order_results is not None:
            reconciliation["accepted_symbols"] = sorted(
                result["symbol"]
                for result in order_results
                if result.get("status") != "rejected" and result.get("symbol")
            )
            reconciliation["rejected_symbols"] = sorted(
                result["symbol"]
                for result in order_results
                if result.get("status") == "rejected" and result.get("symbol")
            )

        return reconciliation

    def _compute_projected_risk(
        self,
        *,
        date: str,
        normalized_orders: list[dict],
        positions_df: pd.DataFrame,
        account: dict,
        symbols_to_close: list[str] | None = None,
    ) -> dict[str, Any]:
        from council import risk_engine as risk_mod
        from council.risk_engine import Position
        from data.features.sector_exposure import compute_effective_sector_cap

        portfolio_value = float(account.get("portfolio_value", 0) or 0)
        target_symbols = sorted({order["ticker"] for order in normalized_orders})

        projected_positions: list[Position] = []
        for order in normalized_orders:
            target_weight = order.get("target_weight")
            if target_weight is None:
                delta_qty = order["share_quantity"]
                if order["direction"] == "sell":
                    delta_qty *= -1
                projected_qty = max(order["current_position_qty"] + delta_qty, 0)
                if projected_qty <= 0:
                    continue
                price = float(order["estimated_price"] or 0.0) or 1.0
                current_value = projected_qty * price
            else:
                target_weight = float(target_weight)
                if target_weight <= 0:
                    continue
                current_value = portfolio_value * target_weight
                price = float(order["estimated_price"] or 0.0) or 1.0
                projected_qty = max(current_value / price, 0.0)

            projected_positions.append(
                Position(
                    symbol=order["ticker"],
                    quantity=projected_qty,
                    avg_price=price,
                    current_price=price,
                )
            )

        # Exclude positions being closed from projected risk
        if symbols_to_close:
            symbols_to_close_set = set(symbols_to_close)
            projected_positions = [
                p for p in projected_positions if p.symbol not in symbols_to_close_set
            ]

        returns = self._load_historical_returns(target_symbols)
        settings = self.trading_settings
        base_sector_limit = settings.max_sector_exposure
        self.risk_engine.limits.max_sector_exposure = compute_effective_sector_cap(
            target_symbols,
            base_sector_cap=base_sector_limit,
            max_position=settings.max_position_size,
        )
        report = self.risk_engine.compute_full_risk(
            positions=projected_positions,
            returns=returns,
            portfolio_value=portfolio_value,
        )
        risk_payload = report.to_dict()
        validate_asset_contract("risk_report", risk_payload)
        risk_mod.RISK_DIR.mkdir(parents=True, exist_ok=True)
        path = self.risk_engine.save_report(report, date=date)
        write_artifact_manifest(
            path,
            artifact_type="risk_report",
            lineage={"execution_date": date},
            metadata={"breach_count": len(report.breaches)},
        )
        breaches = [
            {
                "limit_name": breach.limit_name,
                "current_value": breach.current_value,
                "limit_value": breach.limit_value,
                "severity": breach.severity,
                "message": breach.message,
            }
            for breach in report.breaches
        ]
        high_breaches = [
            breach
            for breach in breaches
            if str(breach["severity"]).upper() == "HIGH"
        ]
        warnings = [
            breach["message"]
            for breach in breaches
            if str(breach["severity"]).upper() != "HIGH"
        ]

        return {
            "path": str(path),
            "breaches": breaches,
            "warnings": warnings,
            "blocked": bool(high_breaches),
            "reason": high_breaches[0]["message"] if high_breaches else None,
        }

    def _load_historical_returns(
        self,
        symbols: list[str],
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        ohlcv_dir = _ROOT / "data" / "ohlcv"
        frames = []
        for symbol in symbols:
            ticker_dir = ohlcv_dir / symbol
            if not ticker_dir.exists():
                continue
            for payload in sorted(ticker_dir.glob("*.parquet")):
                try:
                    frame = pd.read_parquet(payload, columns=["valid_time", "adj_close"])
                except Exception:  # noqa: BLE001
                    continue
                if frame.empty:
                    continue
                frame = frame.copy()
                frame["ticker"] = symbol
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=symbols)

        history = pd.concat(frames, ignore_index=True)
        history = history.sort_values(["ticker", "valid_time"])
        history["ret_1d"] = history.groupby("ticker")["adj_close"].pct_change()
        wide = history.pivot(index="valid_time", columns="ticker", values="ret_1d")
        return wide.dropna(how="all").tail(lookback_days)

    def _operation_payload(
        self,
        *,
        snapshot: dict[str, Any],
        trade_status: str,
        orders_submitted: int,
        orders_rejected: int,
        liquidations: int,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "date": snapshot["date"],
            "trade_status": trade_status,
            "runtime_profile": snapshot["runtime_profile"],
            "paused": snapshot["paused"],
            "paper": snapshot["paper"],
            "paper_guard_ok": snapshot["paper_guard_ok"],
            "orders_generated": snapshot["pretrade"]["order_count"],
            "orders_submitted": orders_submitted,
            "orders_rejected": orders_rejected,
            "liquidations": liquidations,
            "lineage": snapshot["lineage"],
            "pretrade": snapshot["pretrade"],
            "reconciliation": snapshot["reconciliation"],
            "warnings": warnings,
            "errors": errors,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write_operations(self, date: str, data: dict[str, Any]) -> Path:
        validate_asset_contract("operations_report", data)
        OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = OPERATIONS_DIR / f"{date}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        write_artifact_manifest(
            path,
            artifact_type="operations_report",
            lineage=data.get("lineage") if isinstance(data.get("lineage"), dict) else {},
            metadata={
                "trade_status": data.get("trade_status", "unknown"),
                "date": data.get("date", date),
            },
        )
        return path

    def _execution_record_path(self, date: str) -> Path:
        return OPERATIONS_DIR / f"{date}_execution.json"

    def _load_execution_record(self, date: str) -> dict[str, Any] | None:
        path = self._execution_record_path(date)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None
        return payload if isinstance(payload, dict) else None

    def _write_execution_record(self, date: str, data: dict[str, Any]) -> Path:
        validate_asset_contract("operations_report", data)
        OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = self._execution_record_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        write_artifact_manifest(
            path,
            artifact_type="execution_record",
            lineage=data.get("lineage") if isinstance(data.get("lineage"), dict) else {},
            metadata={
                "trade_status": data.get("trade_status", "unknown"),
                "date": data.get("date", date),
            },
        )
        return path

    def _intraday_execution_record_path(self, decision_id: str) -> Path:
        return INTRADAY_OPERATIONS_DIR / f"{decision_id}_execution.json"

    def _load_intraday_execution_record(self, decision_id: str) -> dict[str, Any] | None:
        path = self._intraday_execution_record_path(decision_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None
        return payload if isinstance(payload, dict) else None

    def _write_intraday_execution_record(
        self,
        decision_id: str,
        data: dict[str, Any],
    ) -> Path:
        INTRADAY_OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = self._intraday_execution_record_path(decision_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        write_artifact_manifest(
            path,
            artifact_type="intraday_execution_record",
            lineage=data.get("lineage") if isinstance(data.get("lineage"), dict) else {},
            metadata={
                "decision_id": decision_id,
                "execution_status": data.get("execution_status", "unknown"),
            },
        )
        return path

    def _write_intraday_operations(self, execution_id: str, data: dict[str, Any]) -> Path:
        validate_asset_contract("operations_report", data)
        INTRADAY_OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = INTRADAY_OPERATIONS_DIR / f"{execution_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        write_artifact_manifest(
            path,
            artifact_type="intraday_operations_report",
            lineage=data.get("lineage") if isinstance(data.get("lineage"), dict) else {},
            metadata={
                "trade_status": data.get("trade_status", "unknown"),
                "date": data.get("date", execution_id),
            },
        )
        return path

    def _log_trade(self, date: str, data: dict) -> None:
        TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = TRADE_LOG_DIR / f"{date}.json"
        with _TRADE_LOG_LOCK:
            existing = []
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except Exception:  # noqa: BLE001
                    existing = []
            existing.append(data)
            path.write_text(json.dumps(existing, indent=2, default=str))

    def _dispatch_alert(
        self,
        *,
        severity: str,
        check_type: str,
        message: str,
        recommendation: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        try:
            from council.alerts import AlertResult, Severity

            level = Severity.CRITICAL if severity == "critical" else Severity.WARNING
            self.alert_dispatcher.dispatch(
                [
                    AlertResult(
                        is_alert=True,
                        severity=level,
                        model_name="trading",
                        check_type=check_type,
                        message=message,
                        recommendation=recommendation,
                        metric_value=metric_value,
                        threshold=threshold,
                    )
                ]
            )
        except Exception:  # noqa: BLE001
            return

    def _extract_lineage(self, orders: list[dict]) -> dict[str, str]:
        if not orders:
            return {}

        keys = (
            "pipeline_run_id",
            "data_version",
            "feature_version",
            "model_version",
        )
        first = orders[0]
        return {
            key: str(first[key])
            for key in keys
            if key in first and first[key] is not None
        }

    def _orders_from_intraday_decision(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        # Extract prices from feature snapshot if available
        feature_prices: dict[str, float] = {}
        feature_snapshot = decision.get("feature_snapshot", {})
        for ticker, feats in feature_snapshot.get("features", {}).items():
            if isinstance(feats, dict) and "close" in feats:
                try:
                    feature_prices[ticker] = float(feats["close"])
                except (TypeError, ValueError):
                    continue

        for intent in decision.get("execution_intents", []):
            ticker = str(intent.get("ticker", ""))
            side = str(intent.get("side", "buy")).lower()
            quantity_notional = float(intent.get("quantity_notional", 0.0) or 0.0)
            estimated_price = float(intent.get("estimated_price", 0.0) or 0.0)

            # Prefer the intent price, then the feature snapshot, then a live lookup.
            if estimated_price <= 0 and ticker in feature_prices:
                estimated_price = feature_prices[ticker]
            if estimated_price <= 0:
                try:
                    latest_price = self.node.get_latest_price(ticker)
                    if latest_price:
                        estimated_price = float(latest_price)
                except Exception:  # noqa: BLE001
                    estimated_price = 0.0

            # Convert notional to an actual trade quantity when we know the price.
            # For crypto this is mandatory because Alpaca expects coin units.
            # For equities this keeps the existing share-based execution path while
            # preserving the fractional quantity in the order payload.
            if quantity_notional > 0 and estimated_price > 0:
                share_quantity = quantity_notional / estimated_price
            else:
                share_quantity = 0.0

            orders.append(
                {
                    "ticker": ticker,
                    "direction": side,
                    "quantity": quantity_notional,
                    "share_quantity": share_quantity,
                    "target_weight": float(intent.get("target_weight", 0.0) or 0.0),
                    "price": estimated_price,
                    "decision_id": decision.get("decision_id"),
                    "strategy_version": decision.get("strategy_version", "intraday-v1"),
                }
            )
        return orders

    def _risk_adjust_intraday_orders(
        self,
        orders: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not orders:
            return [], []

        warnings: list[str] = []
        try:
            account = self.node.get_account_info()
            positions_df = self.node.get_all_positions()
            portfolio_value = float(account.get("portfolio_value", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001
            return orders, [f"intraday risk-adjust skipped: {exc}"]

        if portfolio_value <= 0:
            return orders, ["intraday risk-adjust skipped: portfolio_value<=0"]

        from execution.alpaca_adapter import AlpacaLiveNode

        settings = self.trading_settings
        max_equity_position = float(settings.max_position_size)
        max_crypto_position = float(
            os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20")
        )
        max_crypto_turnover = float(
            os.getenv("MLCOUNCIL_MAX_CRYPTO_TURNOVER", str(settings.max_turnover))
        )

        current_weight_by_symbol: dict[str, float] = {}
        if not positions_df.empty and {"symbol", "current_value"}.issubset(positions_df.columns):
            grouped = (
                positions_df.assign(
                    current_value=pd.to_numeric(positions_df["current_value"], errors="coerce").fillna(0.0)
                )
                .groupby("symbol", as_index=False)["current_value"]
                .sum()
            )
            for _, row in grouped.iterrows():
                symbol = str(row["symbol"])
                current_weight_by_symbol[symbol] = float(row["current_value"]) / portfolio_value

        adjusted_orders: list[dict[str, Any]] = []
        for order in orders:
            ticker = str(order.get("ticker", ""))
            if not ticker:
                continue
            direction = str(order.get("direction", "buy")).lower()
            target_weight = float(order.get("target_weight", 0.0) or 0.0)
            requested_notional = float(order.get("quantity", 0.0) or 0.0)

            if direction != "buy":
                adjusted_orders.append(order)
                continue
            if target_weight <= 0.0 or requested_notional <= 0.0:
                continue

            current_weight = float(current_weight_by_symbol.get(ticker, 0.0))
            max_allowed = (
                max_crypto_position
                if AlpacaLiveNode._is_crypto(ticker)
                else max_equity_position
            )
            available_weight = max(max_allowed - current_weight, 0.0)
            if available_weight <= 0.0:
                warnings.append(
                    f"{ticker}: dropped buy intent (current {current_weight:.2%} >= limit {max_allowed:.2%})"
                )
                continue
            if target_weight > available_weight:
                scale = available_weight / target_weight
                scaled = dict(order)
                scaled["target_weight"] = available_weight
                scaled["quantity"] = requested_notional * scale
                adjusted_orders.append(scaled)
                warnings.append(
                    f"{ticker}: target_weight scaled {target_weight:.2%}->{available_weight:.2%} to fit position cap"
                )
            else:
                adjusted_orders.append(order)

        if not adjusted_orders:
            return adjusted_orders, warnings

        only_crypto = all(
            AlpacaLiveNode._is_crypto(str(order.get("ticker", "")))
            for order in adjusted_orders
        )
        turnover_cap = max_crypto_turnover if only_crypto else float(settings.max_turnover)
        max_notional = turnover_cap * portfolio_value
        total_notional = sum(abs(float(order.get("quantity", 0.0) or 0.0)) for order in adjusted_orders)
        if total_notional > max_notional > 0:
            scale = max_notional / total_notional
            for order in adjusted_orders:
                order["quantity"] = float(order.get("quantity", 0.0) or 0.0) * scale
                order["target_weight"] = float(order.get("target_weight", 0.0) or 0.0) * scale
            warnings.append(
                f"intraday intents scaled by {scale:.2f} to respect turnover cap {turnover_cap:.2%}"
            )

        # No post-filter needed: fractional shares are supported via notional orders
        return adjusted_orders, warnings

    def _public_context(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": context["date"],
            "paper": context["paper"],
            "paused": context["paused"],
            "runtime_profile": context["runtime_profile"],
            "lineage": context["lineage"],
            "pretrade": context["pretrade"],
            "reconciliation": context["reconciliation"],
            "paper_guard_ok": context["paper_guard_ok"],
            "connection_error": context["connection_error"],
            "account": context["account"],
            "positions": context["positions"],
        }


service = TradingService()
