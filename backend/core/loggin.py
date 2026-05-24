"""
app.core.loggin
───────────────
Structured JSON logging configuration.

Call configure_logging() once in the FastAPI lifespan before any other
module emits log records. After that, every module just uses:

    import logging
    log = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import json_log_formatter  # type: ignore[import]
    _HAS_JSON_FORMATTER = True
except ImportError:
    _HAS_JSON_FORMATTER = False


class _JSONFormatter(logging.Formatter):
    """
    Minimal fallback JSON formatter that doesn't require an extra dependency.
    If json-log-formatter is installed it is used instead (richer output).
    """

    def format(self, record: logging.LogRecord) -> str:
        import json, traceback

        payload: dict[str, Any] = {
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            "time":    self.formatTime(record, self.datefmt),
        }

        if record.exc_info:
            payload["exception"] = traceback.format_exception(*record.exc_info)

        # Merge any extra fields passed via logger.info("msg", extra={...})
        _standard_attrs = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "args", "msg",
        }
        for key, val in record.__dict__.items():
            if key not in _standard_attrs and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger to emit structured JSON to stdout.

    This must be called before uvicorn configures its own loggers so that
    the access log and error log adopt the same format.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if _HAS_JSON_FORMATTER:
        formatter: logging.Formatter = json_log_formatter.JSONFormatter()
    else:
        formatter = _JSONFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers uvicorn or pytest may have installed already.
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.WARNING)
