from __future__ import annotations

import signal
import sys
import time

from api.services import intraday_runtime_service


def main() -> int:
    intraday_runtime_service.start()

    def _shutdown(signum, frame):  # noqa: ARG001
        intraday_runtime_service.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        intraday_runtime_service.stop()
        sys.exit(0)
