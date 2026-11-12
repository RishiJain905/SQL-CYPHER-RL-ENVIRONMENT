"""Run 1, step by step (Windows; start it hidden and detached, it runs for hours):

    C:\\Python314\\python.exe -I scripts\\run1\\run.py --run run1

Each step: llama-server with the rollout model (the SFT model with the LoRA after the step
before merged in), the step's rollouts (`rollouts.py`), the server stopped, then the update
inside WSL (`train_step.py`: it scores, trains, saves and writes the next rollout model). The
GPU holds one of them at a time. After the steps in `eval_steps` the yardstick is run
(`evaluate`), and the gate decides whether the run goes on.

The run pauses by itself, for a look, when a canary crosses its stop rule (`canaries.py`) or the
gate fails; `pause.py` stops it at any moment. The same command resumes it: a step without
`TRAINED` starts again from its kept rollouts and scores.

Every child process gets CREATE_NO_WINDOW: a console window that pops up and is closed ends its
process (decisions R3).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import REPO, get_paths, to_wsl
from sqlrl.run1 import canaries, server
from sqlrl.run1 import config as rcfg
from sqlrl.run1.model import merged_step, rollout_model_path
from sqlrl.run1.pool import load_pool
from sqlrl.run1.rollouts import generate, read_done

NO_WINDOW = server.NO_WINDOW
EVAL_BUDGET = 1024  # the yardstick's thinking budget (decisions R1)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def event(run_dir: Path, kind: str, text: str, **extra: Any) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"at": now(), "kind": kind, "text": text, **extra}) + "\n")


def run_dir_of(name: str) -> Path:
    return get_paths().data / "run1" / name


def wsl(module: str, args: str, log: Path) -> int:
    p = get_paths()
    cmd = f"cd {to_wsl(REPO)} && PYTHONPATH=src {p.wsl_venv}/bin/python -m {module} {args}"
    with log.open("a", encoding="utf-8") as f:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)  # fmt: skip
    return r.returncode


def keep_wsl_warm() -> subprocess.Popen:
    """A process that stays inside WSL for the whole run. Without one WSL shuts down while the
    rollouts run on Windows, and every update starts cold (Python, torch and the model read
    from disk again: 3-4 minutes instead of 25 s). It also re-reads the model's copy on WSL's
    disk now and then, which keeps it in the file cache. `pause.py` and the run's end stop it."""
    p = get_paths()
    warm = f"cat {p.model_hf_wsl}/*.safetensors > /dev/null 2>&1; " if p.model_hf_wsl else ""
    cmd = f"while true; do {warm}sleep 90; done  # sqlrl.run1.keepalive"
    return subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)  # fmt: skip


def rollout_model(cfg: dict[str, Any], run_dir: Path, k: int) -> tuple[Path, Path | None]:
    """(the GGUF, the adapter or None) that samples step k: the policy after step k - 1."""
    p = get_paths()
    if cfg["rollout_model"] != "merged":
        lora = rcfg.step_dir(run_dir, k - 1) / "adapter.gguf" if k > 1 else None
        return p.model_gguf, lora
    out = rollout_model_path(cfg["run"])
    if not out.exists():  # a copy of the released model; the update writes into it
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".copying")
        shutil.copyfile(p.model_gguf, tmp)
        tmp.replace(out)
        out.with_suffix(".json").write_text(json.dumps({"step": 0}), encoding="utf-8")
    if merged_step(out) != k - 1:  # a merge that was cut: write it again from the saved LoRA
        event(run_dir, "warn", f"the rollout model isn't step {k - 1}'s: merging it again")
        code = wsl("sqlrl.run1.tools", f"merge --run-dir {to_wsl(run_dir)} --step {k - 1}",
                   run_dir / "merge.log")  # fmt: skip
        if code != 0 or merged_step(out) != k - 1:
            raise RuntimeError(f"the rollout model for step {k} could not be written "
                               "(see merge.log)")  # fmt: skip
    return out, None


def do_rollouts(cfg: dict[str, Any], run_dir: Path, k: int, rows: list[dict]) -> None:
    sdir = rcfg.step_dir(run_dir, k)
    out = sdir / "rollouts.jsonl"
    want = len(rows) * cfg["group_size"]
    if len(read_done(out)) >= want:
        return
    model, lora = rollout_model(cfg, run_dir, k)
    event(run_dir, "info", f"step {k}: rollouts", phase="rollouts", step=k)
    t0 = time.time()
    server.start(model, sdir, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"], lora=lora)
    try:
        serve_s = time.time() - t0
        res = generate(f"http://127.0.0.1:{server.PORT}", rows, cfg, k, out)
    finally:
        if not server.stop(sdir):
            raise RuntimeError("the rollout server did not stop")
    rm = sdir / "rollout_meta.json"
    prev = json.loads(rm.read_text(encoding="utf-8")) if rm.exists() else {}
    res["seconds"] = round(res["seconds"] + prev.get("seconds", 0.0), 1)
    res["serve_s"] = round(serve_s + prev.get("serve_s", 0.0), 1)
    rm.write_text(json.dumps(res), encoding="utf-8")
    if len(read_done(out)) < want:
        raise RuntimeError(f"step {k}: {res['errors']} rollouts failed; run again to retry them")


def do_update(run_dir: Path, k: int, launched_by: str) -> dict[str, Any]:
    sdir = rcfg.step_dir(run_dir, k)
    code = wsl("sqlrl.run1.train_step", f"--run-dir {to_wsl(run_dir)} --step {k} "
               f"--launched-by {launched_by}", sdir / "train.log")  # fmt: skip
    if code != 0:
        raise RuntimeError(f"step {k}: the update failed (exit {code}, see train.log)")
    return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))


def history(run_dir: Path) -> list[dict[str, Any]]:
    """The trained steps' metrics in order (the last line of a step wins)."""
    log = run_dir / "train_log.jsonl"
    by: dict[int, dict[str, Any]] = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                by[m["step"]] = m
    return [by[k] for k in sorted(by)]


def evaluate(cfg: dict[str, Any], run_dir: Path, k: int, launched_by: str) -> dict[str, Any]:
    """Step 1's yardstick on the policy after step k: the evalset answered greedily and 8 times
    sampled (the baseline's settings and seeds), scored inside WSL. Returns the summary."""
    p = get_paths()
    out = run_dir / f"eval-{k:04d}"
    done = out / "summary.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import llamaeval

    from sqlrl.step1_baseline import sample  # the one yardstick: the baseline's own sampler

    model, lora = rollout_model(cfg, run_dir, k + 1)  # the policy after step k
    event(run_dir, "info", f"evaluation after step {k}: greedy + 8 sampled answers per evalset "
          "question", phase="evaluating", step=k)  # fmt: skip
    server.start(model, out, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"], lora=lora,
                 reasoning_budget=EVAL_BUDGET)  # fmt: skip
    try:
        url = f"http://127.0.0.1:{server.PORT}"
        g = llamaeval.generate(p.evalset, out / "preds_greedy.jsonl", url)
        s = sample.sample(out, url)
    finally:
        if not server.stop(out):
            raise RuntimeError("the evaluation server did not stop")
    if g["errors"] or s["errors"]:
        raise RuntimeError(f"{g['errors']} greedy and {s['errors']} sampled answers failed")
    code = wsl("sqlrl.run1.eval_score", f"--run-dir {to_wsl(run_dir)} --step {k} "
               f"--launched-by {launched_by}", out / "score.log")  # fmt: skip
    if code != 0 or not done.exists():
        raise RuntimeError(f"scoring the evaluation failed (exit {code}, see {out.name}/score.log)")
    return json.loads(done.read_text(encoding="utf-8"))


def gate_failed(summary: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Why the run must stop after this evaluation (None: go on): pass@8 lower than the
    baseline's, question by question, by more than chance."""
    t = summary["vs_baseline"]["pass8"]
    if t["worse"] > t["better"] and t["p"] < cfg["gate_p"]:
        return (f"pass@8 is down: {t['worse']} questions lost against {t['better']} gained "
                f"(p = {t['p']:.3f})")  # fmt: skip
    return None


def run(name: str, launched_by: str = "agent", new: dict[str, Any] | None = None) -> int:
    run_dir = run_dir_of(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "driver.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    if not (run_dir / "config.json").exists():
        rcfg.save(rcfg.new_config({**(new or {}), "run": name}), run_dir)
    cfg = rcfg.load(run_dir)
    resumed = any((run_dir / m).exists() for m in ("PAUSED", "STOPPED", "FAILED"))
    for marker in ("FAILED", "STOPPED", "PAUSED", "STOP"):
        (run_dir / marker).unlink(missing_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    meta = run_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"name": name, "started_at": now(), "launched_by": launched_by},
                                   indent=1), encoding="utf-8")  # fmt: skip
    k = 0
    keeper = keep_wsl_warm()
    try:
        pool = load_pool(cfg["pool"])
        rcfg.step_questions(pool, cfg, cfg["steps"])  # the pool must reach the last step
        if server.running():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        event(run_dir, "info", f"{name} ({cfg['method']}): {cfg['steps']} steps"
              + (" (resumed)" if resumed else ""), phase="running")  # fmt: skip
        for k in range(1, cfg["steps"] + 1):
            sdir = rcfg.step_dir(run_dir, k)
            if not (sdir / "TRAINED").exists():
                if (run_dir / "STOP").exists():
                    event(run_dir, "warn", f"stopped before step {k} (STOP file)", phase="stopped")
                    (run_dir / "STOPPED").write_text(str(k), encoding="utf-8")
                    return 0
                sdir.mkdir(parents=True, exist_ok=True)
                do_rollouts(cfg, run_dir, k, rcfg.step_questions(pool, cfg, k))
                event(run_dir, "info", f"step {k}: update", phase="training", step=k)
                m = do_update(run_dir, k, launched_by)
                event(run_dir, "info", f"step {k} done: reward {m['reward_mean']:.3f}, solve "
                      f"{m['solve_rate']:.3f}, {m['trained_seqs']} sequences trained, "
                      f"{m['step_s'] / 60:.1f} min", phase="running", step=k)  # fmt: skip
                why = canaries.stop_reasons(history(run_dir), cfg["stop"])
                if why:
                    event(run_dir, "warn", f"paused after step {k} for a look: " + "; ".join(why),
                          phase="paused", step=k)  # fmt: skip
                    (run_dir / "PAUSED").write_text(json.dumps({"step": k, "why": why}), "utf-8")
                    return 0
            if k in cfg["eval_steps"] and not cfg.get("smoke"):
                summary = evaluate(cfg, run_dir, k, launched_by)
                o = summary["overall"]
                event(run_dir, "info", f"evaluation after step {k}: greedy "
                      f"{o['greedy_pass1']:.3f}, mean@8 {o['mean_at8']:.3f}, pass@8 "
                      f"{o['pass_at_k'][-1]:.3f}", phase="running", step=k)  # fmt: skip
                why_gate = gate_failed(summary, cfg)
                if why_gate and not (run_dir / f"eval-{k:04d}" / "GATE_SEEN").exists():
                    (run_dir / f"eval-{k:04d}" / "GATE_SEEN").write_text("", encoding="utf-8")
                    event(run_dir, "warn", f"stopped by the gate after step {k}: {why_gate}",
                          phase="paused", step=k)  # fmt: skip
                    (run_dir / "PAUSED").write_text(json.dumps({"step": k, "why": [why_gate]}),
                                                    "utf-8")  # fmt: skip
                    return 0
        event(run_dir, "info", "run finished", phase="done")
        (run_dir / "DONE").write_text("", encoding="utf-8")
        return 0
    except Exception as e:
        event(run_dir, "error", f"failed at step {k}: {type(e).__name__}: {str(e)[:300]}",
              phase="failed", step=k)  # fmt: skip
        (run_dir / "FAILED").write_text(str(k), encoding="utf-8")
        return 1
    finally:
        keeper.kill()


def evalcheck(name: str = "evalcheck", launched_by: str = "agent") -> int:
    """The yardstick on the SFT model itself, through run 1's own evaluation path (the copy
    of the released GGUF, the run's server settings, the scorer): no training. It proves the
    path end to end and shows that it reproduces the baseline. Writes `<name>/eval-0000/`."""
    run_dir = run_dir_of(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = (run_dir / "driver.log").open("a", encoding="utf-8", buffering=1)
    if not (run_dir / "config.json").exists():
        rcfg.save(rcfg.new_config({"run": name, "steps": 1, "eval_steps": []}), run_dir)
    cfg = rcfg.load(run_dir)
    (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    keeper = keep_wsl_warm()
    try:
        if server.running():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        o = evaluate(cfg, run_dir, 0, launched_by)["overall"]
        event(
            run_dir,
            "info",
            f"yardstick on the SFT model: greedy {o['greedy_pass1']:.3f}, "
            f"mean@8 {o['mean_at8']:.3f}, pass@8 {o['pass_at_k'][-1]:.3f}",
            phase="done",
        )
        return 0
    except Exception as e:
        event(run_dir, "error", f"evaluation check failed: {type(e).__name__}: {str(e)[:300]}",
              phase="failed")  # fmt: skip
        return 1
    finally:
        keeper.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="run1", help="the run's name (its folder under run1/)")
    ap.add_argument("--evalcheck", action="store_true",
                    help="only run the yardstick on the SFT model (no training)")  # fmt: skip
    ap.add_argument("--launched-by", default="agent")
    new = ap.add_argument_group("a new run's settings (ignored when the run exists)")
    new.add_argument("--mix", choices=sorted(rcfg.MIXES), default=None)
    new.add_argument("--lr", type=float, default=None)
    new.add_argument("--steps", type=int, default=None)
    new.add_argument("--pool", default=None)
    new.add_argument("--smoke", action="store_true",
                     help="a loop test: 2 steps of 8 SQL questions, no evaluation")  # fmt: skip
    a = ap.parse_args(argv)
    if a.evalcheck:
        return evalcheck("evalcheck" if a.run == "run1" else a.run, a.launched_by)
    over: dict[str, Any] = {"lr": a.lr, "steps": a.steps, "pool": a.pool}
    if a.mix:
        over["mix"] = rcfg.MIXES[a.mix]
    if a.smoke:
        over.update(smoke=True, steps=a.steps or 2, eval_steps=[],
                    mix={"gretel": 4, "sqale": 4})  # fmt: skip
    return run(a.run, a.launched_by, over)


if __name__ == "__main__":
    sys.exit(main())
