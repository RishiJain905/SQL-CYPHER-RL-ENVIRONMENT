"""The dashboard's data for run 2: everything a page needs to show the run, as small JSON files
(Windows, standard library only, read-only on the run):

    C:\\Python314\\python.exe -I scripts\\run2\\feed.py --run run2

Writes `<run dir>/feed/`:

- `state.json`: the run's status and phase, the step in progress, the time left, the stop rules
  that would fire now, the bounds the relative stop rules use (the mean of the first five
  steps), the settings, `BEST.json`, the last events. Changes every tick.
- `steps-<n>.json` (10 steps each): one flat row of numbers per trained step. Everything
  `train_step` logged, with the nested values flattened (`solve_<set>`, `reward_<set>`,
  `calls_<n>`, `ends_<how>`, `retry_<after>_n` and `retry_<after>_again`).
- `evals.json`: every evaluation of the run (greedy with turns overall, per dataset and per
  kind of question; the sampled block after steps 20 and 40; the sign tests; the returned dev
  slice). Changes only after an evaluation.
- `refs.json`: what the run is read against, measured before it: the start adapter with turns
  (the run's start reference), the start adapter on one query (run 1b's step 30), and the
  start adapter on the returned dev slice. Never changes: push it once.

The files are rewritten whole each time, so pushing them to a page's store is idempotent.
`documentation/run2/dashboard-brief.md` describes every field. A value the run hasn't produced
yet is absent or null: nothing here raises on a missing key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlrl.run2 import canaries, pool
from sqlrl.run2 import config as rcfg

CHUNK = 10
RETURNED_START = "phase0/wording-step30/turns-c-returned"  # the start adapter, returned slice
QUICK_S, FULL_S = 330.0, 1230.0  # an evaluation's time until the run has measured its own

CONFIG_KEYS = (
    "steps", "quick_eval_steps", "eval_steps", "mix", "group_size", "questions_per_step",
    "pool", "pool_offset", "returned_per_step", "dev_returned", "returned_guard", "init_from",
    "max_queries", "prompt_variant", "single_turn_sets", "temperature", "think_budget",
    "max_tokens", "slots", "turn_cost", "reward_malformed", "adv_std_floor", "dup_probe",
    "hardcode_rule", "lr", "warmup_steps", "minibatches", "max_seq", "method", "method_cfg",
    "stop", "refs", "gate_p", "turn_p", "first_floor", "lora_r", "lora_alpha", "smoke",
)  # fmt: skip


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lines(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _num(v: Any) -> Any:
    return float(f"{v:.5g}") if isinstance(v, float) else v


def _retry(prefix: str, d: dict[str, Any] | None, row: dict[str, Any]) -> None:
    """`retry_after` flattened: after an empty result, an error, or rows, how many first calls
    there were and the share that was followed by another call."""
    for after, v in (d or {}).items():
        if isinstance(v, dict):
            row[f"{prefix}{after}_n"] = v.get("n")
            row[f"{prefix}{after}_again"] = _num(v.get("again"))


def step_row(m: dict[str, Any]) -> dict[str, Any]:
    """One step's metrics as a flat row of numbers."""
    row: dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, bool):
            row[k] = v
        elif isinstance(v, (int, float)) or v is None:
            row[k] = _num(v)
    for name, key in (("reward", "reward_by_set"), ("solve", "solve_by_set")):
        for s, v in (m.get(key) or {}).items():
            row[f"{name}_{s}"] = _num(v)
    for name, key in (("calls", "calls"), ("ends", "ends")):
        for s, v in (m.get(key) or {}).items():
            row[f"{name}_{s}"] = v
    _retry("retry_", m.get("retry_after"), row)
    return row


def history(run_dir: Path) -> list[dict[str, Any]]:
    by: dict[int, dict[str, Any]] = {}
    for m in _lines(run_dir / "train_log.jsonl"):
        if isinstance(m.get("step"), int):
            by[m["step"]] = m
    return [by[k] for k in sorted(by)]


def greedy_block(b: dict[str, Any] | None) -> dict[str, Any] | None:
    """A block of greedy episodes (an evaluation overall, a dataset, a kind of question)."""
    if not b:
        return None
    final, first = b.get("final") or {}, b.get("first") or {}
    ch, pr, hk = b.get("changes") or {}, b.get("protocol") or {}, b.get("hacking") or {}
    out = {
        "n": b.get("n"), "with_turns": b.get("with_turns"),
        "right": final.get("right"), "wrong": final.get("wrong"),
        "undecided": final.get("undecided"), "no_query": final.get("no_query"),
        "decided": _num(final.get("decided")), "strict": _num(final.get("strict")),
        "first_right": first.get("right"), "first_decided": _num(first.get("decided")),
        "fixed": ch.get("fixed"), "broken": ch.get("broken"), "net": ch.get("net"),
        "kept_right": ch.get("kept_right"), "kept_wrong": ch.get("kept_wrong"),
        "broken_frac": _num(ch.get("broken_frac")), "fixed_frac": _num(ch.get("fixed_frac")),
        "calls_mean": _num(pr.get("calls_mean")), "calls": pr.get("calls") or {},
        "limit_frac": _num(pr.get("limit_frac")), "loop_frac": _num(hk.get("loop_frac")),
        "copied_frac": _num(hk.get("copied_frac")),
        "hardcoded_frac": _num(hk.get("hardcoded_frac")),
    }  # fmt: skip
    _retry("retry_", pr.get("retry_after"), out)
    return out


def sampled_block(b: dict[str, Any] | None) -> dict[str, Any] | None:
    """A block of 8 sampled answers a question (run 1's shape, kept as it is)."""
    if not b:
        return None
    return {"questions": b.get("questions"), "greedy_pass1": _num(b.get("greedy_pass1")),
            "mean_at8": _num(b.get("mean_at8")),
            "pass_at_k": [_num(x) for x in b.get("pass_at_k") or []],
            "always": b.get("always"), "sometimes": b.get("sometimes"),
            "never": b.get("never")}  # fmt: skip


def _blocks(d: dict[str, Any] | None, fn) -> dict[str, Any]:
    return {k: fn(v) for k, v in (d or {}).items() if isinstance(v, dict)}


def returned_brief(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """The returned dev slice in the form an evaluation carries it (`orchestrate`)."""
    b = ((summary or {}).get("greedy") or {}).get("overall")
    if not b:
        return None
    g = greedy_block(b) or {}
    return {"n": g["n"], "first_right": g["first_right"], "right": g["right"],
            "fixed": g["fixed"], "broken": g["broken"], "broken_frac": g["broken_frac"],
            "calls_mean": g["calls_mean"]}  # fmt: skip


def eval_block(summary: dict[str, Any]) -> dict[str, Any]:
    """An evaluation with turns: the greedy answers (always), the sampled ones (a full
    evaluation), the sign tests, the returned dev slice."""
    greedy, sampled = summary.get("greedy") or {}, summary.get("sampled") or {}
    out: dict[str, Any] = {
        "step": summary.get("step"), "kind": summary.get("kind"),
        "greedy": greedy_block(greedy.get("overall")),
        "greedy_by_set": _blocks(greedy.get("by_set"), greedy_block),
        "by_look": _blocks(summary.get("by_look"), greedy_block),
        "vs": summary.get("vs") or {},
        "returned_dev": summary.get("returned_dev"),
    }  # fmt: skip
    if sampled.get("overall"):
        out["sampled"] = sampled_block(sampled["overall"])
        out["sampled_by_set"] = _blocks(sampled.get("by_set"), sampled_block)
        out["sampled_calls_mean"] = _num(summary.get("sampled_calls_mean"))
    return out


def refs_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """What the run is read against (all measured before it started, on the start adapter)."""
    data = pool.data_dir()
    refs = cfg.get("refs") or {}
    out: dict[str, Any] = {"init_from": cfg.get("init_from")}
    start = _read_json(data / refs["start_with_turns"] / "summary.json") if refs.get(
        "start_with_turns") else None  # fmt: skip
    if start:
        out["start"] = {**eval_block(start), "step": 0}
    one = _read_json(data / refs["single_turn"] / "summary.json") if refs.get(
        "single_turn") else None  # fmt: skip
    if one and one.get("overall"):
        out["single_turn"] = {"overall": sampled_block(one["overall"]),
                              "by_set": _blocks(one.get("by_set"), sampled_block)}  # fmt: skip
    back = returned_brief(_read_json(data / RETURNED_START / "summary.json"))
    if back:
        out["returned_start"] = back
    return out


def status_of(run_dir: Path, events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if (run_dir / "DONE").exists():
        return "done", []
    if (run_dir / "FAILED").exists():
        last = next((e.get("text", "") for e in reversed(events) if e.get("kind") == "error"), "")
        return "failed", [last]
    if (run_dir / "PAUSED").exists():
        paused = _read_json(run_dir / "PAUSED") or {}
        return "paused", list(paused.get("why") or [])
    if (run_dir / "STOPPED").exists():
        return "stopped", ["stopped before a step (`pause.py --after-step`)"]
    if not events:
        return "not started", []
    if not alive(run_dir):
        return "dead", ["the run's process is gone with no DONE, PAUSED or FAILED file: a "
                        "crash or a reboot. The start command resumes it"]  # fmt: skip
    return "running", []


def alive(run_dir: Path) -> bool:
    """Whether the process in `run.pid` still exists (True when that can't be told)."""
    try:
        pid = int((run_dir / "run.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() == 5  # access denied: it exists
    try:
        code = wintypes.DWORD()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
    finally:
        k32.CloseHandle(h)


def bounds(hist: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
    """The lines the relative stop rules compare with: each is the mean of the run's first
    five steps moved by the rule's margin, so they exist from step 6 on (`canaries.py`)."""
    if len(hist) <= canaries.BASE_STEPS or not rules:
        return {}
    base = hist[: canaries.BASE_STEPS]

    def bm(key: str) -> float | None:
        vals = [h[key] for h in base if isinstance(h.get(key), (int, float))]
        return mean(vals) if vals else None

    out: dict[str, Any] = {}

    def add(key: str, how: str, rule: str, side: str) -> None:
        b, r = bm(key), rules.get(rule)
        if b is None or r is None:
            return
        line = b * r if how == "times" else b + r if how == "plus" else b - r
        out[key] = {"start": _num(b), "bound": _num(line), "side": side}

    add("entropy_t1", "times", "entropy_min_frac", "below")
    add("distinct_final_mean", "times", "diversity_min_frac", "below")
    add("first_solve_rate", "minus", "first_right_down", "below")
    add("limit_frac", "plus", "limit_up", "above")
    add("loop_frac", "plus", "loops_up", "above")
    add("broken_frac", "plus", "broken_up", "above")
    add("copied_frac", "plus", "copied_up", "above")
    add("distinct_frac", "plus", "distinct_up", "above")
    add("dup_differ_frac", "plus", "dup_up", "above")
    t1, ratio = bm("t1_tokens_median"), rules.get("length_ratio")
    if t1 and ratio:
        out["t1_tokens_median"] = {"start": _num(t1), "low": _num(t1 / ratio),
                                   "high": _num(t1 * ratio), "side": "outside"}  # fmt: skip
    return out


def _secs(a: str | None, b: str | None) -> float | None:
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return (datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds()
    except (TypeError, ValueError):
        return None


def eval_seconds(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    """How long this run's evaluations took: from an `evaluating` event to the next event."""
    out: dict[str, list[float]] = {"quick": [], "full": []}
    for e, nxt in zip(events, events[1:], strict=False):
        if e.get("phase") == "evaluating" and nxt.get("phase") != "failed":
            s = _secs(e.get("at"), nxt.get("at"))
            if s and s > 0:
                out["full" if "sampled" in e.get("text", "") else "quick"].append(s)
    return out


def hours_left(cfg: dict[str, Any], hist: list[dict[str, Any]], events: list[dict[str, Any]],
               evals_done: set[int]) -> dict[str, Any]:  # fmt: skip
    secs = [h["step_s"] for h in hist[-10:] if isinstance(h.get("step_s"), (int, float))]
    mean_s = mean(secs) if secs else None
    step_now = hist[-1]["step"] if hist else 0
    took = eval_seconds(events)
    quick_s = mean(took["quick"]) if took["quick"] else QUICK_S
    full_s = mean(took["full"]) if took["full"] else FULL_S
    full = set(cfg.get("eval_steps") or [])
    todo = [k for k in sorted(set(cfg.get("quick_eval_steps") or []) | full)
            if k not in evals_done and k >= step_now]  # fmt: skip
    eval_s = sum(full_s if k in full else quick_s for k in todo)
    left = max(int(cfg.get("steps") or 0) - step_now, 0)
    return {"mean_step_s": _num(mean_s) if mean_s else None,
            "quick_eval_s": _num(float(quick_s)), "full_eval_s": _num(float(full_s)),
            "evals_left": todo,
            "hours_left": _num((left * mean_s + eval_s) / 3600) if mean_s else None}  # fmt: skip


def build(name: str) -> dict[str, Any]:
    run_dir = pool.data_dir() / name
    cfg = _read_json(run_dir / "config.json") or {}
    events = _lines(run_dir / "events.jsonl")
    hist = history(run_dir)
    status, why = status_of(run_dir, events)
    last_phase = next((e for e in reversed(events) if e.get("phase")), {})
    step_now = hist[-1]["step"] if hist else 0
    cur = rcfg.step_dir(run_dir, step_now + 1) / "rollouts.jsonl"
    try:
        done = sum(1 for _ in cur.open(encoding="utf-8")) if cur.exists() else 0
    except OSError:
        done = 0
    total = int(cfg.get("questions_per_step") or 0) * int(cfg.get("group_size") or 0)
    evals = []
    for d in sorted(run_dir.glob("eval-*")):
        s = _read_json(d / "summary.json")
        if s and s.get("greedy"):
            evals.append(eval_block(s))
    meta = _read_json(run_dir / "run_meta.json") or {}
    rules = cfg.get("stop") or {}
    try:
        stop_now = canaries.stop_reasons(hist, rules) if rules else []
    except (KeyError, TypeError, ValueError) as e:
        stop_now = [f"the stop rules could not be read: {type(e).__name__}: {e}"]
    mix, back = cfg.get("mix") or {}, cfg.get("returned_per_step") or {}
    one = set(cfg.get("single_turn_sets") or [])
    with_turns = sum(n for s, n in mix.items() if s not in one)
    state = {
        "run": name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "why": why,
        "phase": last_phase.get("phase"),
        "phase_text": last_phase.get("text"),
        "phase_at": last_phase.get("at"),
        "step_now": step_now,
        "steps_total": cfg.get("steps"),
        "started_at": meta.get("started_at"),
        "launched_by": meta.get("launched_by"),
        "in_progress": {"step": step_now + 1, "episodes_done": done, "episodes_total": total},
        "timing": hours_left(cfg, hist, events, {e["step"] for e in evals}),
        "stop_rules_now": stop_now,
        "bounds": bounds(hist, rules),
        # a step in plain counts: how many of its questions have turns, and how many of the
        # gretel ones are returned questions (decisions R17)
        "shape": {
            "questions": sum(mix.values()),
            "with_turns": with_turns,
            "one_query": sum(mix.values()) - with_turns,
            "returned": sum(back.values()),
            "group_size": cfg.get("group_size"),
        },  # fmt: skip
        "config": {k: cfg.get(k) for k in CONFIG_KEYS if k in cfg},
        "best": _read_json(run_dir / "BEST.json"),
        "evals_done": [e["step"] for e in evals],
        "events": events[-40:],
        "step_files": (len(hist) + CHUNK - 1) // CHUNK,
    }
    return {"state": state, "steps": [step_row(h) for h in hist], "evals": evals,
            "refs": refs_block(cfg) if cfg else {}}  # fmt: skip


def write(name: str) -> dict[str, Any]:
    out = pool.data_dir() / name / "feed"
    if not out.parent.exists():  # never create a run's folder: the run does that when it starts
        return {"feed": None, "files": [], "changed": [], "status": "not started", "step_now": 0,
                "phase": None, "why": [f"no run named {name}"], "stop_rules_now": [],
                "hours_left": None}  # fmt: skip
    data = build(name)
    out.mkdir(parents=True, exist_ok=True)
    changed = []

    def put(fname: str, doc: Any) -> None:
        text = json.dumps(doc)
        f = out / fname
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            changed.append(fname)

    put("state.json", data["state"])
    files = ["state.json"]
    rows = data["steps"]
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        fname = f"steps-{i // CHUNK + 1}.json"
        put(fname, {"from": chunk[0]["step"], "to": chunk[-1]["step"], "steps": chunk})
        files.append(fname)
    put("evals.json", {"evals": data["evals"]})
    put("refs.json", data["refs"])
    files += ["evals.json", "refs.json"]
    s = data["state"]
    # `changed`: the files whose content differs from the last tick, i.e. the ones to push
    return {"feed": str(out), "files": files, "changed": changed, "status": s["status"],
            "step_now": s["step_now"], "phase": s["phase"], "why": s["why"],
            "stop_rules_now": s["stop_rules_now"],
            "hours_left": s["timing"]["hours_left"], "watch": watch(rows)}  # fmt: skip


WATCH = (
    "solve_rate", "first_solve_rate", "calls_mean", "fixed", "broken", "broken_frac",
    "limit_frac", "loop_frac", "retry_empty_again", "retry_error_again", "retry_rows_again",
    "informative_frac", "distinct_final_mean", "entropy_t1", "t1_tokens_median", "kl_start",
    "kl_start_raw", "kl_step1", "mismatch", "hardcoded_frac", "copied_frac", "look_leak_frac",
    "returned_solve", "returned_first_solve", "returned_broken_frac", "returned_calls_mean",
    "returned_weak_only_frac", "gretel_other_fixed", "gretel_other_broken", "grad_norm",
    "look_error_frac", "malformed_frac", "template_match", "template_equal_share", "step_s",
    "peak_gb",
)  # fmt: skip
MEAN5 = (
    "solve_rate", "first_solve_rate", "calls_mean", "broken_frac", "returned_solve",
    "returned_first_solve", "returned_broken_frac", "gretel_other_fixed", "gretel_other_broken",
    "kl_start", "entropy_t1", "informative_frac",
)  # fmt: skip


def watch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the watching agent reads after a step (the runbook's table), instead of the whole
    row: the latest step's numbers and the mean of the last five steps for the rates that
    swing with the batch."""
    if not rows:
        return {}
    last, tail = rows[-1], rows[-5:]

    def m5(key: str) -> float | None:
        vals = [r[key] for r in tail if isinstance(r.get(key), (int, float))]
        return _num(mean(vals)) if vals else None

    return {"step": last.get("step"), "last": {k: last[k] for k in WATCH if k in last},
            "mean5": {k: m5(k) for k in MEAN5}, "mean5_steps": len(tail)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="run2")
    a = ap.parse_args(argv)
    print(json.dumps(write(a.run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
