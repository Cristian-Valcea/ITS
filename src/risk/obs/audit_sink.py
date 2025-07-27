import json
import datetime
import pathlib
from typing import Union
from ..event_types import RiskEvent            # relative import back to parent pkg

class JsonAuditSink:
    """
    Very thin wrapper that writes each RiskEvent as a JSON line.
    Safe for moderate event rates (< 50k events/s) with sync writes.
    Switch to an async file writer or Kafka producer later if needed.
    """
    def __init__(self, path: Union[str, pathlib.Path] = "logs/risk_audit.jsonl"):
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered text IO (buffering=1)
        self._fh = self._path.open("a", buffering=1, encoding="utf-8")

    def write(self, event: RiskEvent) -> None:
        payload = {
            "ts"        : datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "event_type": event.event_type.name,
            "priority"  : event.priority.name,
            "source"    : event.source,
            "data"      : event.data,
        }
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

