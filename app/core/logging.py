"""
Centralised logging configuration for the NLP server.

- Emits structured JSON to stdout (one log line per record).
- Attaches PIIRedactFilter to every handler so that PII that accidentally
  leaks into a log message is masked before it reaches any log sink.
- Call setup_logging() exactly once at application startup in main.py.
"""
import json
import logging
import re
from typing import Any


# ── JSON formatter ─────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    Renders a LogRecord as a single-line JSON object.

    Standard fields always present:
        ts      – timestamp (YYYY-MM-DDTHH:MM:SS)
        level   – DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger  – logger name (module path)
        msg     – the formatted message string
        exc     – "ExcType: message" if exc_info was captured, else null

    Any keys passed via logger.xxx(..., extra={...}) are merged in.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "exc": None,
        }

        # Include exc_info as "ExcType: message" only — never the full traceback,
        # which may contain local variable reprs with PII.
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type is not None:
                payload["exc"] = f"{exc_type.__name__}: {exc_value}"

        # Merge any extra fields (method, path, status, ms, etc.)
        _BUILTIN_ATTRS = frozenset({
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        })
        for key, value in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


# ── PII redaction filter ───────────────────────────────────────────────────────

class PIIRedactFilter(logging.Filter):
    """
    Logging filter that scrubs PII from every LogRecord message.

    Reuses the pre-compiled regex constants from app.services.chat_intent.
    Runs on every LogRecord before it is written to any handler.
    The filter mutates record.msg in-place so it is safe with all formatters.

    Applied substitutions (same order as sanitize_message):
        phone  → [TELEFON]
        TC ID  → [TC_KİMLİK]
        email  → [E_POSTA]
        money  → [TUTAR]
        names  → [KİŞİ]
    """

    def __init__(self) -> None:
        super().__init__()
        # Lazy import: PIIRedactFilter.__init__ is called the first time a
        # LogRecord is created — by which point all modules are fully loaded,
        # so there is no circular-import risk.
        try:
            from app.services.chat_intent import (
                EMAIL_PATTERN,
                MONEY_PATTERN,
                NAME_PATTERN,
                NAME_STOP_WORDS,
                PHONE_PATTERN,
                TC_PATTERN,
            )
            self._phone = PHONE_PATTERN
            self._tc = TC_PATTERN
            self._email = EMAIL_PATTERN
            self._money = MONEY_PATTERN
            self._name = NAME_PATTERN
            self._name_stop = NAME_STOP_WORDS
            self._enabled = True
        except ImportError:
            self._enabled = False

    def _redact(self, text: str) -> str:
        text = self._phone.sub("[TELEFON]", text)
        text = self._tc.sub("[TC_KİMLİK]", text)
        text = self._email.sub("[E_POSTA]", text)
        text = self._money.sub("[TUTAR]", text)

        def _replace_name(m: re.Match) -> str:
            name = m.group(0)
            return name if name in self._name_stop else "[KİŞİ]"

        text = self._name.sub(_replace_name, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._enabled:
            return True
        # Redact the message string
        record.msg = self._redact(str(record.msg))
        # Redact %-style format args — only redact string values, leave
        # non-string args (int, float, etc.) untouched to avoid breaking
        # third-party formatters that use %d / %f specifiers.
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    self._redact(a) if isinstance(a, str) else a
                    for a in record.args
                )
        # Redact string values in extra fields (set via logger.xxx(..., extra={...}))
        _BUILTIN_ATTRS = frozenset({
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        })
        for key, value in list(record.__dict__.items()):
            if key not in _BUILTIN_ATTRS and not key.startswith("_") and isinstance(value, str):
                setattr(record, key, self._redact(value))
        return True


# ── Public entry point ─────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once at startup.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
               Read from LOG_LEVEL env-var in main.py and passed here.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any handlers uvicorn may have already attached to avoid duplicate output.
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(PIIRedactFilter())
    root.addHandler(handler)

    # Suppress uvicorn's built-in access log — we emit our own in the HTTP middleware.
    logging.getLogger("uvicorn.access").propagate = False
