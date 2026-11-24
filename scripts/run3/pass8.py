"""Run 3, phase 0.2 on the PC: serve a model, run `sqlrl.run3.measure run`, stop the server.

    python scripts/run3/pass8.py --dir <measure dir> [--model <gguf>] [--limit 50]

Windows (llama-server is a Windows build). One GPU job at a time: refuses if another
llama-server runs. 16 slots of 12,288 tokens each (run 3's episodes carry whole schemas and up
to 8 results; Spark's KV cache is about 36 KB a token, so this fits the 16 GB card beside the
4.4 GB model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.run2 import server  # noqa: E402
from sqlrl.run2.model import models_dir  # noqa: E402
from sqlrl.run3 import measure  # noqa: E402

PORT = 8098  # run 3's own port (run 2: 8097)
SLOTS, SLOT_CTX = 16, 12288


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kept = models_dir() / "run2-step40" / "Spark-X2.6-4B-SQL-Cypher-Agent-RL2-Q8_0.gguf"
    ap.add_argument("--model", default=str(kept))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--greedy", action="store_true", help="temperature 0, for the dev set")
    a = ap.parse_args(argv)
    if server.running():
        print("another llama-server is running: one GPU job at a time")
        return 2
    d = Path(a.dir)
    log = d / "server"
    log.mkdir(parents=True, exist_ok=True)
    server.start(Path(a.model), log, port=PORT, slots=SLOTS, ctx=SLOTS * SLOT_CTX)
    try:
        temp = 0.0 if a.greedy else measure.SAMPLING["temperature"]
        res = measure.run(d, f"http://127.0.0.1:{PORT}", a.samples, SLOTS, a.limit, temp)
    finally:
        if not server.stop(log):
            raise RuntimeError("the server did not stop")
    print(json.dumps(res))
    print(json.dumps(measure.summary(d)["overall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
