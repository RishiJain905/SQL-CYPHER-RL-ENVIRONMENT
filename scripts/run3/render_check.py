"""Serve step 40 on the PC and run `sqlrl.run3.render check` (or `add`), then stop the server.

    python scripts/run3/render_check.py check --episodes <file> [--n 60]
    python scripts/run3/render_check.py add --episodes <teacher episodes> --out <file>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run2 import server  # noqa: E402
from sqlrl.run2.model import models_dir  # noqa: E402
from sqlrl.run3 import render, settings  # noqa: E402

PORT = 8098


def main(argv: list[str]) -> int:
    if server.running():
        print("another llama-server is running: one GPU job at a time")
        return 2
    log = settings.get().data / "render-server"
    log.mkdir(parents=True, exist_ok=True)
    model = models_dir() / "run2-step40" / "Spark-X2.6-4B-SQL-Cypher-Agent-RL2-Q8_0.gguf"
    server.start(model, log, port=PORT, slots=4, ctx=4 * 12288)
    try:
        return render.main([*argv, "--url", f"http://127.0.0.1:{PORT}"])
    finally:
        if not server.stop(log):
            print(json.dumps({"warning": "the server did not stop"}))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
