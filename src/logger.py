import json
import logging
import time


class StructuredLogger:
    """
    Emits newline-delimited JSON logs.
    Every line carries request_id so you can grep all lines for one request.
    This is the Dapper/production pattern: one ID traces a full call chain.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _emit(self, level: str, request_id: str, event: str, **fields):
        record = {
            "ts": round(time.time(), 3),
            "level": level,
            "request_id": request_id,
            "event": event,
            **fields,
        }
        self._logger.info(json.dumps(record))

    def info(self, request_id: str, event: str, **fields):
        self._emit("INFO", request_id, event, **fields)

    def error(self, request_id: str, event: str, **fields):
        self._emit("ERROR", request_id, event, **fields)
