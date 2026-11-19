"""Run 2, step by step (Windows; start it hidden and detached, it runs for hours):

    C:\\Python314\\python.exe -I scripts\\run2\\run.py --run run2

Each step: llama-server with the rollout model (the SFT model with the adapter after the step
before merged in; before step 1, the start adapter), the step's episodes (`episodes.py`: the
model queries the look-around databases of `env.py`), the server stopped, then the update inside
WSL (`train_step.py`). The GPU holds one of them at a time. After the steps in
`quick_eval_steps` the evaluation set is answered greedily with turns; after those in
`eval_steps` also 8 times sampled (`evaluate`).

The run pauses by itself, for a look, when a canary crosses its stop rule (`canaries.py`), when
pass@8 with turns falls below the start's (the gate), or when an evaluation is worse than the
run's best earlier one. `pause.py` stops it at any moment. The same command resumes it: a step
without `TRAINED` starts again from its kept episodes and scores.

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
from sqlrl.run2 import canaries, env, episodes, integrity, pool, server
from sqlrl.run2 import config as rcfg
from sqlrl.run2.model import merged_step, rollout_model_path, start_state

NO_WINDOW = server.NO_WINDOW


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def event(run_dir: Path, kind: str, text: str, **extra: Any) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"at": now(), "kind": kind, "text": text, **extra}) + "\n")


def run_dir_of(name: str) -> Path:
    return pool.data_dir() / name


def wsl(module: str, args: str, log: Path) -> int:
    p = get_paths()
    cmd = f"cd {to_wsl(REPO)} && PYTHONPATH=src {p.wsl_venv}/bin/python -m {module} {args}"
    with log.open("a", encoding="utf-8") as f:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW)  # fmt: skip
    return r.returncode


def keep_wsl_warm() -> subprocess.Popen:
    """A process that stays inside WSL for the whole run, so every update starts warm (run 1's
    harness note); it also re-reads the model's copy on WSL's disk to keep it cached."""
    p = get_paths()
    warm = f"cat {p.model_hf_wsl}/*.safetensors > /dev/null 2>&1; " if p.model_hf_wsl else ""
    cmd = f"while true; do {warm}sleep 90; done  # sqlrl.run2.keepalive"
    return subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)  # fmt: skip


def rollout_model(cfg: dict[str, Any], run_dir: Path, k: int) -> Path:
    """The GGUF that samples step k: the policy after step k - 1 (step 0: the start adapter)."""
    p = get_paths()
    out = rollout_model_path(cfg["run"])
    if not out.exists():  # a copy of the released model; the merge writes into it
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".copying")
        shutil.copyfile(p.model_gguf, tmp)
        tmp.replace(out)
    if merged_step(out) != k - 1:  # the first merge, or one that was cut: from the saved LoRA
        if k > 1 or merged_step(out) is not None:
            event(run_dir, "warn", f"the rollout model isn't step {k - 1}'s: merging it again")
        code = wsl("sqlrl.run2.tools", f"merge --run-dir {to_wsl(run_dir)} --step {k - 1}",
                   run_dir / "merge.log")  # fmt: skip
        if code != 0 or merged_step(out) != k - 1:
            raise RuntimeError(f"the rollout model for step {k} could not be written "
                               "(see merge.log)")  # fmt: skip
    return out


def public_rows(rows: list[dict[str, Any]], scripts: dict[str, str]) -> list[dict[str, Any]]:
    """Questions as the rollout side may see them: the reference stops here (`env.public_row`)."""
    return [env.public_row(r, scripts.get(r["id"])) for r in rows]


def do_rollouts(cfg: dict[str, Any], run_dir: Path, k: int, pubs: list[dict]) -> None:
    sdir = rcfg.step_dir(run_dir, k)
    out = sdir / "rollouts.jsonl"
    want = len(pubs) * cfg["group_size"]
    if len(episodes.read_done(out)) >= want:
        return
    model = rollout_model(cfg, run_dir, k)
    event(run_dir, "info", f"step {k}: rollouts", phase="rollouts", step=k)
    t0 = time.time()
    server.start(model, sdir, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"])
    try:
        serve_s = time.time() - t0
        res = episodes.generate(f"http://127.0.0.1:{server.PORT}", pubs, cfg, out,
                                episodes.train_spec(cfg, k), cfg["group_size"],
                                env.GraphHost())  # fmt: skip
    finally:
        if not server.stop(sdir):
            raise RuntimeError("the rollout server did not stop")
    rm = sdir / "rollout_meta.json"
    prev = json.loads(rm.read_text(encoding="utf-8")) if rm.exists() else {}
    res["seconds"] = round(res["seconds"] + prev.get("seconds", 0.0), 1)
    res["serve_s"] = round(serve_s + prev.get("serve_s", 0.0), 1)
    rm.write_text(json.dumps(res), encoding="utf-8")
    if len(episodes.read_done(out)) < want:
        raise RuntimeError(f"step {k}: {res['errors']} episodes failed; run again to retry them")


def do_update(run_dir: Path, k: int, launched_by: str, tries: int = 3) -> dict[str, Any]:
    """Step k's update in WSL; then its state files are checked as they are on the disk. A
    damaged file means the step is trained again from step k - 1."""
    sdir = rcfg.step_dir(run_dir, k)
    for attempt in range(1, tries + 1):
        code = wsl("sqlrl.run2.train_step", f"--run-dir {to_wsl(run_dir)} --step {k} "
                   f"--launched-by {launched_by}", sdir / "train.log")  # fmt: skip
        if code != 0:
            raise RuntimeError(f"step {k}: the update failed (exit {code}, see train.log)")
        bad = integrity.mismatched(sdir)
        if not bad:
            return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
        event(run_dir, "warn", f"step {k}: {', '.join(bad)} on the disk differ from what was "
              f"written (try {attempt} of {tries}); training the step again",
              phase="training", step=k)  # fmt: skip
        for name in ("TRAINED", *integrity.STATE_FILES):
            (sdir / name).unlink(missing_ok=True)
            (sdir / (name + ".sha256")).unlink(missing_ok=True)
    raise RuntimeError(f"step {k}: its saved state was damaged on the disk {tries} times")


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


def eval_pubs() -> list[dict[str, Any]]:
    """The evaluation set as public rows. Read only here, by the evaluation."""
    return public_rows(pool.read_jsonl(get_paths().evalset), pool.load_scripts("evalset"))


def ref_args(cfg: dict[str, Any]) -> str:
    return " ".join(f"--ref {name}={to_wsl(pool.data_dir() / rel)}"
                    for name, rel in (cfg.get("refs") or {}).items())  # fmt: skip


def evaluate(cfg: dict[str, Any], run_dir: Path, k: int, launched_by: str,
             full: bool) -> dict[str, Any]:  # fmt: skip
    """The evaluation set with turns on the policy after step k: greedy (`full` False), or
    greedy and 8 sampled episodes a question. Scored inside WSL; returns the summary. A quick
    evaluation's greedy episodes are kept when the full one follows at the same step."""
    out = run_dir / f"eval-{k:04d}"
    done = out / "summary.json"
    if done.exists():
        s = json.loads(done.read_text(encoding="utf-8"))
        if s["kind"] == "full" or not full:
            return s
    out.mkdir(exist_ok=True)
    model = rollout_model(cfg, run_dir, k + 1)  # the policy after step k
    kind = "greedy and 8 sampled episodes" if full else "greedy"
    event(run_dir, "info", f"evaluation after step {k}: the evalset with turns, {kind}",
          phase="evaluating", step=k)  # fmt: skip
    episodes.claim(out / "episodes.jsonl", cfg)
    server.start(model, out, slots=cfg["slots"], ctx=cfg["slot_ctx"] * cfg["slots"])
    try:
        url = f"http://127.0.0.1:{server.PORT}"
        res: dict[str, Any] = {}
        for _ in range(3):  # a failed episode is retried
            res = episodes.generate(url, eval_pubs(), cfg, out / "episodes.jsonl",
                                    episodes.eval_spec(cfg), 9 if full else 1,
                                    env.GraphHost())  # fmt: skip
            if not res["errors"]:
                break
    finally:
        if not server.stop(out):
            raise RuntimeError("the evaluation server did not stop")
    if res.get("errors"):
        raise RuntimeError(f"{res['errors']} evaluation episodes failed after 3 tries")
    (out / "scored.jsonl").unlink(missing_ok=True)  # the episode file grew: score it again
    code = wsl(
        "sqlrl.run2.eval_score",
        f"--dir {to_wsl(out)} --run-dir {to_wsl(run_dir)} "
        f"--step {k} --launched-by {launched_by} {ref_args(cfg)}",
        out / "score.log",
    )
    if code != 0 or not done.exists():
        raise RuntimeError(f"scoring the evaluation failed (exit {code}, see {out.name}/score.log)")
    return json.loads(done.read_text(encoding="utf-8"))


def check_start_reference(cfg: dict[str, Any]) -> None:
    """The gate compares every full evaluation with the start adapter's own evaluation with
    turns (`refs.start_with_turns`, made by `phase0.py heldout`). It must exist, and it must
    have been made under this run's protocol: against a reference made with another prompt,
    call limit or set of single-turn datasets the gate would compare two different tasks."""
    rel = (cfg.get("refs") or {}).get("start_with_turns")
    if not rel or cfg.get("smoke") or not (cfg["eval_steps"] or cfg.get("quick_eval_steps")):
        return
    ref = pool.data_dir() / rel
    if not (ref / "summary.json").exists():
        raise RuntimeError(f"no start reference at {rel}: run `phase0.py heldout --samples 8`")
    if not episodes.same_protocol(ref, cfg):
        raise RuntimeError(f"the start reference at {rel} was made under another protocol "
                           "than this run's: make it again (`phase0.py heldout`)")  # fmt: skip


def gate_failed(summary: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Why the run must stop after a full evaluation (None: go on): pass@8 with turns lower
    than the start adapter's with turns, question by question, by more than chance."""
    t = ((summary.get("vs") or {}).get("start_with_turns") or {}).get("pass8")
    if t and t["worse"] > t["better"] and t["p"] < cfg["gate_p"]:
        return (f"pass@8 with turns is down: {t['worse']} questions lost against "
                f"{t['better']} gained (p = {t['p']:.3f})")  # fmt: skip
    return None


def turned(summary: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Why the run should pause after an evaluation (None: go on): greedy with turns is worse
    than the run's best earlier evaluation, question by question."""
    best = (summary.get("vs") or {}).get("best") or {}
    t = best.get("greedy")
    if cfg.get("turn_p") and t and t["worse"] > t["better"] and t["p"] < cfg["turn_p"]:
        return (f"worse than the evaluation after step {best.get('step')}: {t['worse']} "
                f"questions down against {t['better']} up on greedy with turns "
                f"(p = {t['p']:.3f})")  # fmt: skip
    return None


def _pause(run_dir: Path, k: int, why: list[str], label: str) -> None:
    event(run_dir, "warn", f"{label} after step {k}: " + "; ".join(why), phase="paused", step=k)
    (run_dir / "PAUSED").write_text(json.dumps({"step": k, "why": why}), encoding="utf-8")


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
        questions = pool.load_pool(cfg["pool"])
        scripts = pool.load_scripts(cfg["pool"])
        rcfg.step_questions(questions, cfg, cfg["steps"])  # the pool must reach the last step
        start_state(cfg)  # the start adapter must be there
        if server.running():
            raise RuntimeError("another llama-server is running: one GPU job at a time")
        if not env.GraphHost().ok():
            raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
        check_start_reference(cfg)
        event(
            run_dir,
            "info",
            f"{name} ({cfg['method']}, from {cfg.get('init_from')}): "
            f"{cfg['steps']} steps" + (" (resumed)" if resumed else ""),
            phase="running",
        )
        quick, full = set(cfg.get("quick_eval_steps") or []), set(cfg["eval_steps"])
        for k in range(1, cfg["steps"] + 1):
            sdir = rcfg.step_dir(run_dir, k)
            if not (sdir / "TRAINED").exists():
                if (run_dir / "STOP").exists():
                    event(run_dir, "warn", f"stopped before step {k} (STOP file)", phase="stopped")
                    (run_dir / "STOPPED").write_text(str(k), encoding="utf-8")
                    return 0
                sdir.mkdir(parents=True, exist_ok=True)
                pubs = public_rows(rcfg.step_questions(questions, cfg, k), scripts)
                do_rollouts(cfg, run_dir, k, pubs)
                event(run_dir, "info", f"step {k}: update", phase="training", step=k)
                m = do_update(run_dir, k, launched_by)
                event(run_dir, "info", f"step {k} done: solve {m['solve_rate']:.3f} (first query "
                      f"{m['first_solve_rate']:.3f}), {m['calls_mean']:.2f} calls an episode, "
                      f"{m['trained_seqs']} sequences trained, {m['step_s'] / 60:.1f} min",
                      phase="running", step=k)  # fmt: skip
                why = canaries.stop_reasons(history(run_dir), cfg["stop"])
                if why:
                    _pause(run_dir, k, why, "paused for a look")
                    return 0
            if (k in quick or k in full) and not cfg.get("smoke"):
                s = evaluate(cfg, run_dir, k, launched_by, full=k in full)
                o = s["greedy"]["overall"]
                text = (f"evaluation after step {k}: greedy with turns {o['final']['decided']:.3f}"
                        f" (strict {o['final']['strict']:.3f}), first query "
                        f"{o['first']['decided']:.3f}, fixed {o['changes']['fixed']}, broken "
                        f"{o['changes']['broken']}")  # fmt: skip
                if "sampled" in s:
                    so = s["sampled"]["overall"]
                    text += f", mean@8 {so['mean_at8']:.3f}, pass@8 {so['pass_at_k'][-1]:.3f}"
                event(run_dir, "info", text, phase="running", step=k)
                edir = run_dir / f"eval-{k:04d}"
                for check, seen, label in ((gate_failed, "GATE_SEEN", "stopped by the gate"),
                                           (turned, "TURN_SEEN", "paused for a look")):  # fmt: skip
                    reason = check(s, cfg)
                    if reason and not (edir / seen).exists():
                        (edir / seen).write_text("", encoding="utf-8")
                        _pause(run_dir, k, [reason], label)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="run2", help="the run's name (its folder under run2/)")
    ap.add_argument("--launched-by", default="agent")
    new = ap.add_argument_group("a new run's settings (ignored when the run exists)")
    new.add_argument("--mix", choices=sorted(rcfg.MIXES), default=None)
    new.add_argument("--lr", type=float, default=None)
    new.add_argument("--steps", type=int, default=None)
    new.add_argument("--init-from", default=None, help="an adapter folder under `paths.models`")
    new.add_argument("--prompt-variant", default=None)
    new.add_argument("--single-turn", default=None,
                     help="datasets kept on run 1's protocol, comma-separated (default: "
                          "sqale; `none`: every dataset gets turns)")  # fmt: skip
    new.add_argument("--smoke", action="store_true",
                     help="a loop test: 2 steps of 8 questions (2 per dataset) from the pool's "
                          "tail, no evaluation")  # fmt: skip
    a = ap.parse_args(argv)
    over: dict[str, Any] = {"lr": a.lr, "steps": a.steps, "init_from": a.init_from,
                            "prompt_variant": a.prompt_variant}  # fmt: skip
    if a.mix:
        over["mix"] = rcfg.MIXES[a.mix]
    if a.single_turn:  # "none": every dataset gets turns
        over["single_turn_sets"] = [s for s in a.single_turn.split(",") if s and s != "none"]
    if a.smoke:
        # 8 probe episodes are too few to check the step-1 drift reading: its rule is for real runs
        over.update(smoke=True, steps=a.steps or 2, eval_steps=[], quick_eval_steps=[],
                    pool_tail=True, mix=dict.fromkeys(rcfg.MIXES["even"], 2),
                    stop={"kl_step1_max": 1.0})  # fmt: skip
    return run(a.run, a.launched_by, over)


if __name__ == "__main__":
    sys.exit(main())
