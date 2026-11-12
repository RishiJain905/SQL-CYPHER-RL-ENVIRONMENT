"""One real step of rollouts, timed (Windows, GPU, a few minutes):

    C:\\Python314\\python.exe -I scripts\\run1\\bench_rollouts.py --label new

The pilots' step-1 questions (32 x 8 rollouts, the same seeds, the SFT model with no LoRA), so
the time compares with the pilots' own step 1 (`rollout_s` in their `train_log.jsonl`). Writes
the rollouts and a `result.json` under `<data>/run1/bench/rollouts-<label>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sqlrl.paths import get_paths  # noqa: E402
from sqlrl.run1 import config, rollouts, server  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", default="new")
    ap.add_argument("--model", default=None, help="a GGUF (default: the merged SFT Q8_0)")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--logprob-rollouts", type=int, default=1)
    ap.add_argument("--kv-unified", action="store_true")
    ap.add_argument("--questions", type=int, default=32)
    a = ap.parse_args(argv)
    p = get_paths()
    cfg = config.new_config({"slots": a.slots, "logprob_rollouts": a.logprob_rollouts})
    pool = [json.loads(x) for x in (p.data / "pilots" / "pool-v1.jsonl").open(encoding="utf-8")]
    rows = pool[: a.questions]
    out_dir = p.data / "run1" / "bench" / f"rollouts-{a.label}"
    out = out_dir / "rollouts.jsonl"
    out.unlink(missing_ok=True)
    ctx = 65536 if a.kv_unified else cfg["slot_ctx"] * a.slots
    info = server.start(Path(a.model) if a.model else p.model_gguf, out_dir, slots=a.slots,
                        ctx=ctx, kv_unified=a.kv_unified)  # fmt: skip
    try:
        res = rollouts.generate(f"http://127.0.0.1:{server.PORT}", rows, cfg, 1, out)
    finally:
        server.stop(out_dir)
    res.update(
        label=a.label,
        slots=a.slots,
        kv_unified=a.kv_unified,
        server_start_s=info["seconds"],
        logprob_rollouts=a.logprob_rollouts,
    )
    (out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
