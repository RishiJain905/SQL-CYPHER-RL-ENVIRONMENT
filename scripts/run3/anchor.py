"""Run 3, phase 0.1: the anchor (documentation/run3/phase0.md 0.1, design.md 5.1).

    python scripts/run3/anchor.py [--model <gguf>] [--name step40-pc]

Step 40 as the PC serves it answers run 2's evaluation set greedy, with turns, through run 2's
own evaluation path (the same prompt, call limit, single-turn sets and result format as run 2's
evaluation after step 40), then is scored in WSL against that evaluation question by question.
Passes at 57.0 or more with at least 95% of questions agreeing. Writes to
`<data>/run3/phase0/anchor-<name>/`. Windows (it starts and stops its own llama-server).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.paths import get_paths, to_wsl  # noqa: E402
from sqlrl.run2 import config as rcfg  # noqa: E402
from sqlrl.run2 import env, episodes, server  # noqa: E402
from sqlrl.run2.model import models_dir  # noqa: E402
from sqlrl.run2.orchestrate import eval_pubs  # noqa: E402
from sqlrl.run2.phase0 import wsl  # noqa: E402

# run 2's evaluation after step 40 (moved with the finished run to the history drive)
REF_PROTOCOL = {"prompt_variant": "turns", "max_queries": 3,
                "single_turn_sets": ["sqale", "text2cypher"],
                "obs": {"rows": 10, "style": "table"}, "history": "full", "look": 1}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    kept = models_dir() / "run2-step40" / "Spark-X2.6-4B-SQL-Cypher-Agent-RL2-Q8_0.gguf"
    ap.add_argument("--model", default=str(kept))
    ap.add_argument("--name", default="step40-pc")
    ap.add_argument("--ref", required=True, help="run 2's eval-0040 folder")
    a = ap.parse_args(argv)
    if server.running():
        print("another llama-server is running: one GPU job at a time")
        return 2
    cfg = rcfg.new_config()
    out = get_paths().data / "run3" / "phase0" / f"anchor-{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    proto = episodes.claim(out / "episodes.jsonl", cfg)
    if proto != REF_PROTOCOL:
        raise SystemExit(f"protocol differs from run 2's evaluation: {proto}")
    server.start(Path(a.model), out, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"])
    try:
        url = f"http://127.0.0.1:{server.PORT}"
        res: dict = {}
        for _ in range(3):  # a failed episode is retried
            res = episodes.generate(url, eval_pubs(), cfg, out / "episodes.jsonl",
                                    episodes.eval_spec(cfg), 1, env.GraphHost())  # fmt: skip
            if not res["errors"]:
                break
    finally:
        if not server.stop(out):
            raise RuntimeError("the server did not stop")
    (out / "scored.jsonl").unlink(missing_ok=True)
    code = wsl("sqlrl.run2.eval_score",
               f"--dir {to_wsl(out)} --rows evalset --ref run2_step40={to_wsl(Path(a.ref))}",
               out / "score.log")  # fmt: skip
    if code != 0:
        raise RuntimeError(f"scoring failed (see {out}/score.log)")
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"engine": res, "final": s["greedy"]["overall"]["final"],
                      "vs": s.get("vs")}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
