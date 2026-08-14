# Security Policy

MLCouncil handles broker credentials and market-data API keys. Treat this
project's security posture accordingly.

## Supported versions

| Version | Status |
|---|---|
| master (latest) | Supported |

There are no tagged releases yet; security fixes land on `master`.

## Reporting a vulnerability

- **Preferred**: use GitHub **private vulnerability reporting** on this
  repository (Security → Report a vulnerability).
- **Fallback**: email `eliobenigni7@gmail.com` with subject
  `[MLCouncil security] ...`. Do not open public issues for vulnerabilities.

Please include: affected component, minimal reproduction, impact, and (if
known) a proposed fix. You will receive an acknowledgement within 72 hours;
we aim for a fix or mitigation within 14 days of triage.

## What to never commit

- `.env` files (gitignored) — real `ALPACA_*`, `POLYGON_*`, `MLCOUNCIL_API_KEY`,
  `ALERT_EMAIL`, `SMTP_PASSWORD` values belong there, never in tracked files.
- `secrets/` directory contents (gitignored; used by Docker secrets).
- Any token, key or password in code, docs, tests, or commit messages.

Before every push: `python scripts/check_no_plaintext_secrets.py` and a manual
`git diff` review. Placeholder values (`YOUR_PAPER_SECRET`, `your_alpaca_key`)
are acceptable in `.env.example` and adapter defaults — real values are not.

## Runtime hardening

- Runtime profiles fail closed on missing/placeholder values
  (`runtime_env.is_placeholder_env_value`).
- Pickle artifacts are loaded via `council.pickle_security.trusted_pickle_load`
  which fails closed without a `.hash` sidecar (M11).
- Admin/API surface requires `MLCOUNCIL_API_KEY` in paper/prod profiles;
  rate limiting applies via `slowapi`.
- Order execution respects hard caps (max orders/day, turnover, position
  size) and a kill switch (`MLCOUNCIL_AUTOMATION_PAUSED`).

## Incident response

1. Rotate the affected credential immediately.
2. Pause automation (`MLCOUNCIL_AUTOMATION_PAUSED=true`) if execution is
   affected.
3. `git log` + `git reflog` to identify any leaked material; force-push/rotate
   as needed.
4. Open a private advisory once the issue is contained.
