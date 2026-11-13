"""Content-addressed cache for generated outputs.

A key is the SHA-256 of the canonical JSON of every input that affects the output
(model, prompt version, decoding parameters, source text). Writes are atomic.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


class OutputCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(**fields: Any) -> str:
        """Stable key for a set of inputs; argument order does not matter."""
        payload = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return str(json.loads(path.read_text(encoding="utf-8"))["output"])

    def put(self, key: str, output: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"output": output}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        output = compute()
        self.put(key, output)
        return output
