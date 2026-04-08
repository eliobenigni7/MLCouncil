from __future__ import annotations

import importlib


def test_validate_runtime_profile_reports_missing_paper_keys(monkeypatch, tmp_path):
    env_path = tmp_path / "runtime.paper.env"
    env_path.write_text("ALPACA_BASE_URL=https://paper-api.alpaca.markets\n")

    for key in (
        "ALPACA_PAPER_KEY",
        "ALPACA_PAPER_SECRET",
        "MLCOUNCIL_MAX_DAILY_ORDERS",
        "MLCOUNCIL_MAX_TURNOVER",
        "MLCOUNCIL_MAX_POSITION_SIZE",
        "MLCOUNCIL_AUTOMATION_PAUSED",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(env_path))

    import runtime_env

    importlib.reload(runtime_env)
    result = runtime_env.validate_runtime_profile()

    assert result["profile"] == "paper"
    assert result["valid"] is False
    assert result["missing"]
    assert "MLCOUNCIL_AUTOMATION_PAUSED" in result["missing"]
    assert result["paper_guard_ok"] is True


def test_validate_runtime_profile_accepts_complete_paper_env(monkeypatch, tmp_path):
    env_path = tmp_path / "runtime.paper.env"
    env_path.write_text(
        "\n".join(
            [
                "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
                "ALPACA_PAPER_KEY=test-paper-key",
                "ALPACA_PAPER_SECRET=test-paper-secret",
                "MLCOUNCIL_MAX_DAILY_ORDERS=20",
                "MLCOUNCIL_MAX_TURNOVER=0.30",
                "MLCOUNCIL_MAX_POSITION_SIZE=0.10",
                "MLCOUNCIL_AUTOMATION_PAUSED=false",
            ]
        )
        + "\n"
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(env_path))

    import runtime_env

    importlib.reload(runtime_env)
    runtime_env.load_runtime_env(override=True)
    result = runtime_env.validate_runtime_profile()

    assert result["valid"] is True
    assert result["missing"] == []
    assert result["paper_guard_ok"] is True


def test_validate_runtime_profile_flags_live_base_url_in_paper(monkeypatch, tmp_path):
    env_path = tmp_path / "runtime.paper.env"
    env_path.write_text(
        "\n".join(
            [
                "ALPACA_BASE_URL=https://api.alpaca.markets",
                "ALPACA_PAPER_KEY=test-paper-key",
                "ALPACA_PAPER_SECRET=test-paper-secret",
                "MLCOUNCIL_MAX_DAILY_ORDERS=20",
                "MLCOUNCIL_MAX_TURNOVER=0.30",
                "MLCOUNCIL_MAX_POSITION_SIZE=0.10",
                "MLCOUNCIL_AUTOMATION_PAUSED=false",
            ]
        )
        + "\n"
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(env_path))

    import runtime_env

    importlib.reload(runtime_env)
    runtime_env.load_runtime_env(override=True)
    result = runtime_env.validate_runtime_profile()

    assert result["valid"] is False
    assert result["paper_guard_ok"] is False
    assert any("paper endpoint" in error.lower() for error in result["errors"])


def test_load_runtime_env_prefers_real_legacy_keys_over_placeholder_canonical_values(
    monkeypatch, tmp_path
):
    env_path = tmp_path / "runtime.paper.env"
    env_path.write_text(
        "\n".join(
            [
                "ALPACA_PAPER_KEY=replace-me",
                "ALPACA_PAPER_SECRET=replace-me",
                "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
            ]
        )
        + "\n"
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(env_path))
    monkeypatch.setenv("ALPACA_API_KEY", "live-legacy-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "live-legacy-secret")
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)

    import runtime_env

    importlib.reload(runtime_env)
    runtime_env.load_runtime_env(override=True)

    assert runtime_env.validate_required_env("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET") == []
    assert runtime_env.os.getenv("ALPACA_PAPER_KEY") == "live-legacy-key"
    assert runtime_env.os.getenv("ALPACA_PAPER_SECRET") == "live-legacy-secret"


def test_load_runtime_env_applies_real_file_values_over_existing_env_defaults(
    monkeypatch, tmp_path
):
    env_path = tmp_path / "runtime.local.env"
    env_path.write_text("MLCOUNCIL_AUTO_EXECUTE=true\n")

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(env_path))
    monkeypatch.setenv("MLCOUNCIL_AUTO_EXECUTE", "false")

    import runtime_env

    importlib.reload(runtime_env)
    runtime_env.load_runtime_env()

    assert runtime_env.os.getenv("MLCOUNCIL_AUTO_EXECUTE") == "true"
