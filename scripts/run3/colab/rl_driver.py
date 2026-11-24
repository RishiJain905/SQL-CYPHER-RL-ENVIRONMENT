"""Run 3, phase 2 on the Colab VM: the RL loop as one detached process (R26: never a long
`colab exec`; documentation/run3/phase2.md).

    nohup python rl_driver.py devcheck --run /content/rlrun > devcheck.log 2>&1 &
    nohup python rl_driver.py run --run /content/rlrun [--questions 16] > driver.log 2>&1 &

Both serve the rollout model with the CUDA llama-server (`/content/llama.cpp`), a copy of the
released Q8_0 into which the policy's adapter is merged in float32 (never bf16).

- **devcheck:** the start checkpoint, served, answers the run-3 dev set's 300 triples greedy;
  writes `devcheck.json` (the PC's reading was 46.4%; phase2.md wants it within 3 points).
- **run:** each step draws its groups (`rl.draw_step`), plays 8 episodes a group, writes them as
  they finish (`step-k/episodes.jsonl`, without the long prompts), computes the advantages, runs
  the update (`Learner`), saves the adapter and the optimiser (`step-k/`, then `TRAINED`), splits
  them into 40 MB parts for `colab download`, merges the adapter into the rollout model, restarts
  the server, appends `metrics.jsonl`, rewrites `state.json` and checks the stop rules. It
  resumes from the last `TRAINED` step. It stops, saved, at `max_hours`, on a stop rule, or when
  the file `PAUSE` appears in the run folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path("/content/rl")
sys.path.insert(0, str(REPO / "src"))

from sqlrl.run3 import episode, measure, protocol, rl  # noqa: E402
from sqlrl.run3.rl_learner import Learner, write_json  # noqa: E402

LLAMA = Path("/content/llama.cpp/build/bin/llama-server")
RELEASE = Path("/content/release/Spark-X2.6-4B-SQL-Cypher-Agent-Q8_0.gguf")
DATA = Path("/content/run3data")
PORT, SLOTS, SLOT_CTX = 8098, 16, 12288
PART = 40 * 1024 * 1024  # the files API refused 90 MB uploads (2026-11-24); 40 MB works


class Server:
    def __init__(self, gguf: Path, log: Path):
        self.gguf, self.log, self.p = gguf, log, None

    def start(self) -> float:
        t0 = time.time()
        self.p = subprocess.Popen([str(LLAMA), "-m", str(self.gguf), "-ngl", "99", "-np",
                                   str(SLOTS), "-c", str(SLOTS * SLOT_CTX), "--cache-ram", "0",
                                   "-ctxcp", "0", "--host", "127.0.0.1", "--port", str(PORT)],
                                  stdout=self.log.open("a"), stderr=subprocess.STDOUT)  # fmt: skip
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                return round(time.time() - t0, 1)
            except Exception:  # noqa: BLE001 - not up yet
                if self.p.poll() is not None:
                    raise RuntimeError("llama-server exited (see server.log)") from None
                time.sleep(1)
        raise RuntimeError("llama-server did not come up")

    def stop(self) -> None:
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.p = None


def event(run: Path, level: str, text: str) -> None:
    with (run / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "level": level, "text": text}) + "\n")  # fmt: skip
    print(f"[{level}] {text}", flush=True)


def compact(rec: dict[str, Any]) -> dict[str, Any]:
    """The episode without its long texts (the PC reads outcomes, not prompts)."""
    keep = ("qid", "kind", "ending", "outcome", "reward", "why", "n_calls", "seconds", "gid",
            "sample", "stratum", "engine", "db", "source", "error", "last_query_ok")  # fmt: skip
    out = {k: rec.get(k) for k in keep if k in rec}
    out["setting"] = {k: (rec.get("setting") or {}).get(k) for k in ("schema", "tools")}
    out["calls"] = [{"tokens": c["tokens"], "closed": c["closed"], "eos": c["eos"]}
                    for c in rec.get("calls") or []]  # fmt: skip
    out["final_query"] = (rec.get("final_query") or "")[:400]
    return out


def play(g: dict[str, Any], sample: int, step: int, tasks: measure.Tasks, vers: dict[str, str],
         temperature: float, cfg: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    try:
        task = tasks.task(g)
    except KeyError:  # its gold failed this VM's gold check: no task here (skipped, R30)
        return {"qid": g.get("qid"), "kind": g["kind"], "outcome": "skipped", "reward": None,
                "gid": g["gid"], "sample": sample, "stratum": g["stratum"],
                "engine": g["engine"], "db": g["db"], "source": g["source"],
                "calls": []}  # fmt: skip
    qid = task["q"]["id"]
    srcs = [(s["engine"], vers[s["engine"]]) for s in task["sources"]]
    setting = protocol.draw_setting(srcs, rl.stable(f"run3-rl:{qid}:{g['engine']}:{step}"))
    seed = rl.stable(f"run3-rl:{qid}:{g['engine']}:{step}:{sample}") % 2**31
    pol = rl.RecordingPolicy(f"http://127.0.0.1:{PORT}", temperature=temperature,
                             think_budget=cfg["think_budget"], max_tokens=cfg["max_tokens"],
                             seed=seed)  # fmt: skip
    try:
        rec = episode.run_episode(task, pol, setting)
        episode.grade(task, rec)
    except Exception as e:  # noqa: BLE001 - recorded, the step goes on
        rec = {"qid": qid, "kind": task["kind"], "ending": "crash", "outcome": "crash",
               "reward": None, "error": f"{type(e).__name__}: {e}"[:300]}  # fmt: skip
    rec.update(gid=g["gid"], sample=sample, stratum=g["stratum"], engine=g["engine"],
               db=g["db"], source=g["source"], calls=pol.calls)  # fmt: skip
    return rec


def rollouts(groups: list[dict[str, Any]], step: int, out: Path, n: int, temperature: float,
             cfg: dict[str, Any], tasks: measure.Tasks,
             vers: dict[str, str]) -> list[dict]:  # fmt: skip
    """Every group's `n` episodes; written as they finish. SQL groups first, then the graph."""
    recs: list[dict[str, Any]] = []
    lock = threading.Lock()
    with out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(SLOTS) as ex:
        for part in (
            [g for g in groups if g["engine"] != "neo4j"],
            [g for g in groups if g["engine"] == "neo4j"],
        ):
            futs = [ex.submit(play, g, s, step, tasks, vers, temperature, cfg)
                    for g in part for s in range(n)]  # fmt: skip
            for fu in as_completed(futs):
                r = fu.result()
                with lock:
                    recs.append(r)
                    f.write(json.dumps(compact(r), default=str) + "\n")
                    f.flush()
    return recs


def split(path: Path, into: Path) -> list[str]:
    into.mkdir(parents=True, exist_ok=True)
    names = []
    with path.open("rb") as f:
        i = 0
        while chunk := f.read(PART):
            p = into / f"{path.name}.part-{i:02d}"
            p.write_bytes(chunk)
            names.append(p.name)
            i += 1
    return names


def last_trained(run: Path) -> int:
    done = sorted(int(p.parent.name.split("-")[1]) for p in run.glob("step-*/TRAINED"))
    return done[-1] if done else 0


def cmd_devcheck(run: Path) -> int:
    run.mkdir(parents=True, exist_ok=True)
    gguf = run / "rollout.gguf"
    learner = Learner(DATA / "rl" / "start" / "lora_state.safetensors")
    if not gguf.exists():
        shutil.copyfile(RELEASE, gguf)
    tmpl = learner.check_template(gguf)
    merge = learner.merge(gguf, 0)
    event(run, "info", f"devcheck: template {tmpl}, merged the start checkpoint ({merge})")
    del learner
    import torch

    torch.cuda.empty_cache()
    srv = Server(gguf, run / "server.log")
    srv.start()
    try:
        groups = [json.loads(x) for x in (DATA / "rl" / "dev-groups.jsonl").read_text()
                  .splitlines()]  # fmt: skip
        d = run / "devcheck"
        d.mkdir(exist_ok=True)
        (d / "groups.jsonl").write_text("\n".join(json.dumps(g) for g in groups))
        res = measure.run(d, f"http://127.0.0.1:{PORT}", 1, SLOTS, None, 0.0)
        summ = measure.summary(d)
        eps = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines()]
        ans = [e for e in eps if e.get("kind") == "answer" and e.get("outcome") != "skipped"]
        rate = sum(e.get("outcome") == "right" for e in ans) / max(len(ans), 1)
        out = {"right_rate": round(rate, 4), "answerable": len(ans), "minutes": res["minutes"],
               "overall": summ["overall"], "template": tmpl,
               "versions": res["versions"]}  # fmt: skip
        write_json(run / "devcheck.json", out)
        event(run, "info", f"devcheck: {rate:.3f} right on answerable dev questions "
              f"(the PC: 0.464)")  # fmt: skip
    finally:
        srv.stop()
    return 0


def cmd_run(run: Path, questions: int, max_hours: float) -> int:
    cfg = {**rl.CFG, "questions": questions, "max_hours": max_hours}
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", cfg)
    t_start = time.time()
    k0 = last_trained(run)
    state_dir = run / f"step-{k0:04d}" if k0 else None
    learner = Learner(DATA / "rl" / "start" / "lora_state.safetensors", cfg, state_dir)
    gguf = run / "rollout.gguf"
    if not gguf.exists():
        shutil.copyfile(RELEASE, gguf)
    marker = gguf.with_suffix(".json")
    if not marker.exists() or json.loads(marker.read_text()).get("step") != k0:
        learner.merge(gguf, k0)  # the rollout model holds the policy after step k0
    pool = json.loads((DATA / "rl" / "pool.json").read_text())
    dbs = json.loads((DATA / "rl" / "bundle-dbs.json").read_text())
    tasks, vers = measure.Tasks(), measure.versions()
    srv = Server(gguf, run / "server.log")
    srv.start()
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()] if (
        run / "metrics.jsonl").exists() else []  # fmt: skip
    # hours used so far: the steps already recorded (a resumed run), plus this process's time.
    # (Before 2026-11-24 22:58 each row added the previous row's cumulative hours: counted twice.)
    base_h = sum(float(r.get("step_s") or 0.0) for r in rows) / 3600
    event(run, "info", f"run: resuming after step {k0} ({base_h:.2f} h used)" if k0
          else "run: step 1 starts")  # fmt: skip
    status, why = "running", ""
    k = k0
    try:
        while True:
            k += 1
            hours = base_h + (time.time() - t_start) / 3600
            if (run / "PAUSE").exists():
                status, why = "paused", "PAUSE file"
                break
            if hours >= max_hours:
                status, why = "done", f"max_hours {max_hours} reached"
                break
            sd = run / f"step-{k:04d}"
            sd.mkdir(exist_ok=True)
            write_json(run / "state.json", {"status": "rollouts", "step": k, "hours": hours,
                                            "updated": time.time()})  # fmt: skip
            t0 = time.time()
            groups = rl.draw_step(pool, k, questions, dbs, cfg)
            for i, g in enumerate(groups):
                g["gid"] = i
            recs = rollouts(groups, k, sd / "episodes.jsonl", cfg["group"], cfg["temperature"],
                            cfg, tasks, vers)  # fmt: skip
            t1 = time.time()
            by: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in recs:
                by[r["gid"]].append(r)
            exs, over = [], 0
            for rs in by.values():
                rews = [(r.get("reward") if r.get("reward") is not None else -0.1) for r in rs]
                for r, a in zip(rs, rl.advantages(rews, cfg["adv_floor"]), strict=True):
                    if r.get("outcome") in ("crash", "skipped"):  # not the model's doing
                        continue
                    e, o = rl.examples(r, learner.tok, a, cfg["max_seq"])
                    exs += e
                    over += o
            write_json(run / "state.json", {"status": "update", "step": k, "hours": hours,
                                            "updated": time.time()})  # fmt: skip
            upd = learner.update(exs, k)
            hashes = learner.save(sd)
            (sd / "TRAINED").write_text(json.dumps(hashes))
            parts = {f: split(sd / f, sd / "parts") for f in ("lora_state.safetensors",
                                                               "optim.pt")}  # fmt: skip
            srv.stop()
            merge = learner.merge(gguf, k)
            serve_s = srv.start()
            m = rl.step_metrics(recs)
            row = {"step": k, **m, **{f"upd_{a}": b for a, b in upd.items()},
                   "kl": upd.get("kl"), "examples_over": over,
                   "rollout_s": round(t1 - t0, 1), "step_s": round(time.time() - t0, 1),
                   "merge_s": merge["seconds"], "serve_s": serve_s,
                   "hours": round(base_h + (time.time() - t_start) / 3600, 3),
                   "groups": [g["stratum"] for g in groups], "parts": parts}  # fmt: skip
            rows.append(row)
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            event(run, "info", f"step {k}: reward {m['reward_mean']}, solve {m['solve_rate']}, "
                  f"malformed {m['malformed']}, kl {upd.get('kl')}, {row['step_s']} s")  # fmt: skip
            reasons = rl.stop_check(rows)
            if reasons:
                status, why = "paused", "stop rule: " + "; ".join(reasons)
                event(run, "warn", why)
                break
    except Exception as e:  # noqa: BLE001 - recorded for the PC, then the process ends
        status, why = "dead", f"{type(e).__name__}: {e}"[:400]
        event(run, "error", why + "\n" + traceback.format_exc()[-1500:])
    finally:
        srv.stop()
        write_json(run / "state.json", {"status": status, "why": why, "step": last_trained(run),
                                        "updated": time.time()})  # fmt: skip
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("devcheck")
    d.add_argument("--run", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--questions", type=int, default=rl.CFG["questions"])
    r.add_argument("--max-hours", type=float, default=rl.CFG["max_hours"])
    a = ap.parse_args()
    if a.cmd == "devcheck":
        return cmd_devcheck(Path(a.run))
    return cmd_run(Path(a.run), a.questions, a.max_hours)


if __name__ == "__main__":
    sys.exit(main())
