"""Run 3, phase 0.2: pass@8 of a model on the training pools, the useful band
(documentation/run3/phase0.md 0.2, design.md 8.1).

    python -m sqlrl.run3.measure plan --out <dir> --bird 120 --cypher 60 --nonanswer 40
    python -m sqlrl.run3.measure run --dir <dir> --url http://127.0.0.1:8098 [--limit 50]
    python -m sqlrl.run3.measure summary --dir <dir>

- **plan** fixes the groups (`groups.jsonl`) before anything is sampled: per SQL engine `--bird`
  training questions, per CypherBench training graph `--cypher`, per SQL engine `--nonanswer`
  write requests and as many wrong-database questions; training split only (`splits`).
- **run** plays `--samples` (8) episodes a group at temperature 1.0 (run 2's sampling). The
  setting is drawn once a group from the seed `run3-p0:<question id>:<engine>`, so the samples
  differ only by sampling; each sample has its own seed. Appends to `episodes.jsonl` (each
  record holds the whole conversation, for the warm start) and resumes where it stopped;
  `--limit` plays only the first N groups (the timed pilot).
- **summary** writes `summary.json`: per stratum and per harness variant (schema shown x tools),
  groups never / useful / always right, mean@8, pass@8, endings, calls, the explore share, and
  the episodes' context length against the 16 GB trainer's limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sqlrl.run3 import engines, episode, pools, protocol, settings, sources, splits
from sqlrl.run3.policy import LlamaPolicy

SAMPLING = {"temperature": 1.0, "think_budget": 1024, "max_tokens": 2048}
TRAINER_LIMIT = 3840  # run 2's longest sequence on the 16 GB card


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


# ---- plan -------------------------------------------------------------------------------------


def _databases(source: str) -> list[str]:
    return sorted(p.stem for p in (settings.get().pools / "goldcheck" / source).glob("*.jsonl"))


def plan(n_bird: int, n_cypher: int, n_non: int, seed: int = 0,
         split: str = "train") -> list[dict[str, Any]]:  # fmt: skip
    """The groups, as references a run rebuilds its tasks from (no gold in this file).
    `split` "train" (phase 0's band) or "dev" (the warm start's fixed dev set)."""
    rnd = random.Random(f"run3-p0-plan:{seed}:{split}")
    groups: list[dict[str, Any]] = []
    bird_dbs = [d for d in _databases("bird") if d not in splits.HELDOUT_DBS["bird"]]

    def ok(source: str, db: str, qid: str) -> bool:
        return splits.split_of(source, db, qid) == split

    for e in engines.SQL_ENGINES:
        cands = [(db, t["q"]["id"]) for db in bird_dbs for t in pools.answer_tasks("bird", db, e)
                 if ok("bird", db, t["q"]["id"])]  # fmt: skip
        rnd.shuffle(cands)
        groups += [{"stratum": f"bird/{e}", "kind": "answer", "source": "bird", "db": db,
                    "qid": q, "engine": e} for db, q in cands[:n_bird]]  # fmt: skip
    for g in splits.CYPHERBENCH_TRAIN_GRAPHS:
        cands = [t["q"]["id"] for t in pools.answer_tasks("cypherbench", g, "neo4j")
                 if ok("cypherbench", g, t["q"]["id"])]  # fmt: skip
        rnd.shuffle(cands)
        groups += [{"stratum": f"cypher/{g}", "kind": "answer", "source": "cypherbench", "db": g,
                    "qid": q, "engine": "neo4j"} for q in cands[:n_cypher]]  # fmt: skip
    for i, e in enumerate(engines.SQL_ENGINES):  # non-answers, spread over the databases
        dbs = bird_dbs[i::4] or bird_dbs
        for j in range(n_non):
            db = dbs[j % len(dbs)]
            n = j // len(dbs)  # the dev set's non-answers are drawn with their own seed
            groups.append({"stratum": f"write/{e}", "kind": "write", "source": "bird", "db": db,
                           "engine": e, "n": n, "tag": split})  # fmt: skip
            groups.append({"stratum": f"wrongdb/{e}", "kind": "unanswerable", "source": "bird",
                           "db": db, "engine": e, "n": n, "tag": split})  # fmt: skip
    for k, g in enumerate(groups):
        g["gid"] = k
    # SQL first, then the graphs one after another (Neo4j holds one graph at a time)
    return sorted(groups, key=lambda g: (g["engine"] == "neo4j", g["db"] if g["engine"] ==
                                         "neo4j" else "", g["gid"]))  # fmt: skip


class Tasks:
    """Rebuilds a group's task (with its gold) from the gold check, cached per database."""

    def __init__(self) -> None:
        self._answer: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        self._others = sources.load("bird")
        self._lock = threading.Lock()

    def task(self, g: dict[str, Any]) -> dict[str, Any]:
        if g["kind"] == "answer":
            key = (g["source"], g["db"], g["engine"])
            with self._lock:
                if key not in self._answer:
                    self._answer[key] = {t["q"]["id"]: t for t in
                                         pools.answer_tasks(*key)}  # fmt: skip
            return self._answer[key][g["qid"]]
        tag = "" if g.get("tag", "train") == "train" else f"{g['tag']}:"
        seed = stable_int(f"run3-p0:{tag}{g['db']}:{g['engine']}")
        if g["kind"] == "write":
            ts = pools.write_requests("bird", g["db"], g["engine"], g["n"] + 1, seed)
        else:
            ts = pools.wrong_database("bird", g["db"], g["engine"], self._others, g["n"] + 1, seed)
        return ts[min(g["n"], len(ts) - 1)]


# ---- run --------------------------------------------------------------------------------------


class CountingPolicy(LlamaPolicy):
    """LlamaPolicy that records each call's prompt and generated token counts."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.tokens: list[tuple[int, int]] = []
        self.rendered: list[str] = []  # the server's prompt text, one per turn

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        text = super().render(messages, tools)
        self.rendered.append(text)
        return text

    def _complete(self, prompt: str, n: int, stop: list[str] | None) -> dict[str, Any]:
        r = super()._complete(prompt, n, stop)
        self.tokens.append((int(r.get("tokens_evaluated") or 0),
                            int(r.get("tokens_predicted") or 0)))  # fmt: skip
        return r


def versions() -> dict[str, str]:
    out = {}
    db = _databases("bird")[0]
    for e in engines.SQL_ENGINES:
        c = engines.open_sql(e, "bird", db)
        try:
            out[e] = c.version()
        finally:
            c.close()
    out["neo4j"] = engines.Neo4j().version()
    return out


def setting_for(g: dict[str, Any], task: dict[str, Any], vers: dict[str, str]):
    qid = task["q"]["id"]
    srcs = [(s["engine"], vers[s["engine"]]) for s in task["sources"]]
    return protocol.draw_setting(srcs, stable_int(f"run3-p0:{qid}:{g['engine']}"))


def play(g: dict[str, Any], sample: int, tasks: Tasks, url: str, vers: dict[str, str],
         temperature: float = SAMPLING["temperature"]) -> dict[str, Any]:  # fmt: skip
    try:
        task = tasks.task(g)
    except KeyError:  # its gold failed this machine's gold check: no task here (skipped, R30)
        return {"qid": g.get("qid"), "kind": g["kind"], "outcome": "skipped", "ending": None,
                "reward": None, "gid": g["gid"], "sample": sample, "stratum": g["stratum"],
                "engine": g["engine"], "source": g["source"], "db": g["db"], "setting": {},
                "tokens": {"calls": 0, "max_context": 0, "generated": 0}}  # fmt: skip
    setting = setting_for(g, task, vers)
    seed = stable_int(f"run3-p0:{task['q']['id']}:{g['engine']}:{sample}") % 2**31
    pol = CountingPolicy(url, temperature=temperature,
                         think_budget=SAMPLING["think_budget"],
                         max_tokens=SAMPLING["max_tokens"], seed=seed)  # fmt: skip
    t0 = time.time()
    try:
        rec = episode.run_episode(task, pol, setting)
        episode.grade(task, rec)
    except Exception as e:  # one broken episode is a finding, not a stop
        rec = {"qid": task["q"]["id"], "kind": task["kind"], "ending": "crash",
               "outcome": "crash", "reward": None, "setting": setting.record(),
               "error": f"{type(e).__name__}: {e}"[:300],
               "seconds": round(time.time() - t0, 2)}  # fmt: skip
    ctx = [p + n for p, n in pol.tokens]
    rec.update(gid=g["gid"], sample=sample, stratum=g["stratum"], engine=g["engine"],
               source=g["source"], db=g["db"], seed=seed,
               tokens={"calls": len(pol.tokens), "max_context": max(ctx, default=0),
                       "generated": sum(n for _, n in pol.tokens)})  # fmt: skip
    if rec.get("outcome") == "right":  # the warm start trains on exactly what the model saw
        rec["rendered"] = pol.rendered
    return rec


def run(d: Path, url: str, samples: int = 8, workers: int = 16, limit: int | None = None,
        temperature: float = SAMPLING["temperature"]) -> dict[str, Any]:  # fmt: skip
    groups = [json.loads(x) for x in (d / "groups.jsonl").read_text("utf-8").splitlines()]
    if limit:  # the timed pilot: a spread over every stratum, in the plan's order
        step = max(len(groups) // limit, 1)
        groups = groups[::step][:limit]
    out = d / "episodes.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text("utf-8").splitlines():
            r = json.loads(line)
            if r.get("outcome") != "crash":
                done.add((r["gid"], r["sample"]))
    todo = [(g, s) for g in groups for s in range(samples) if (g["gid"], s) not in done]
    tasks, vers = Tasks(), versions()
    lock = threading.Lock()
    n = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(lambda gs: play(gs[0], gs[1], tasks, url, vers, temperature), todo):
            with lock:
                n += 1
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(json.dumps({"done": n, "of": len(todo),
                                      "minutes": round((time.time() - t0) / 60, 1)}),
                          flush=True)  # fmt: skip
    return {"episodes": n, "minutes": round((time.time() - t0) / 60, 1), "versions": vers}


# ---- summary ----------------------------------------------------------------------------------


def _band(k: int, n: int) -> str:
    return "never" if k == 0 else ("always" if k == n else "useful")


def _explored(r: dict[str, Any]) -> bool:
    """A look before the end: at least one tool call that was not the ending."""
    return r.get("n_calls", 0) > 0


def _stats(recs: list[dict[str, Any]], groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    bands = Counter()
    for rs in groups.values():
        bands[_band(sum(r.get("outcome") == "right" for r in rs), len(rs))] += 1
    ng = max(len(groups), 1)
    ctx = sorted(r.get("tokens", {}).get("max_context", 0) for r in recs)
    pct = (lambda p: ctx[min(int(p * len(ctx)), len(ctx) - 1)]) if ctx else (lambda p: 0)
    return {
        "groups": len(groups),
        "episodes": len(recs),
        "never": bands["never"], "useful": bands["useful"], "always": bands["always"],
        "useful_share": round(bands["useful"] / ng, 3),
        "mean_at8": round(sum(r.get("outcome") == "right" for r in recs) / max(len(recs), 1), 3),
        "pass_at8": round((bands["useful"] + bands["always"]) / ng, 3),
        "endings": dict(Counter(r.get("ending") for r in recs)),
        "calls_mean": round(sum(r.get("n_calls", 0) for r in recs) / max(len(recs), 1), 2),
        "explore_share": round(sum(_explored(r) for r in recs) / max(len(recs), 1), 3),
        "context_p50": pct(0.5), "context_p90": pct(0.9), "context_max": ctx[-1] if ctx else 0,
        "over_trainer_limit": round(sum(c > TRAINER_LIMIT for c in ctx) / max(len(ctx), 1), 3),
    }  # fmt: skip


def summary(d: Path) -> dict[str, Any]:
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for x in (d / "episodes.jsonl").read_text("utf-8").splitlines():
        r = json.loads(x)  # a retried crash is superseded by its retry (the file only appends)
        latest[(r["gid"], r["sample"])] = r
    recs = list(latest.values())
    by: dict[str, dict[str, Any]] = {}
    keys = {
        "stratum": lambda r: r["stratum"],
        "variant": lambda r: f"{r['setting']['schema']}/{r['setting']['tools']}",
        "pool": lambda r: r["stratum"].split("/")[0],
    }
    for name, key in keys.items():
        parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            parts[key(r)].append(r)
        by[name] = {}
        for k, rs in sorted(parts.items()):
            gs: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in rs:
                gs[r["gid"]].append(r)
            by[name][k] = _stats(rs, gs)
    allg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        allg[r["gid"]].append(r)
    out = {"overall": _stats(recs, allg), **by,
           "crashes": sum(r.get("outcome") == "crash" for r in recs),
           "why_wrong": dict(Counter(r.get("why") for r in recs
                                     if r.get("outcome") == "wrong").most_common(15))}  # fmt: skip
    (d / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--bird", type=int, default=120)
    p.add_argument("--cypher", type=int, default=60)
    p.add_argument("--nonanswer", type=int, default=40)
    p.add_argument("--split", choices=["train", "dev"], default="train")
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8098")
    r.add_argument("--samples", type=int, default=8)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--greedy", action="store_true", help="temperature 0 (the dev set)")
    s = sub.add_parser("summary")
    s.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        gs = plan(a.bird, a.cypher, a.nonanswer, split=a.split)
        with (d / "groups.jsonl").open("w", encoding="utf-8") as f:
            for g in gs:
                f.write(json.dumps(g) + "\n")
        print(json.dumps({"groups": len(gs), "strata": dict(Counter(g["stratum"] for g in gs))}))
    elif a.cmd == "run":
        temp = 0.0 if a.greedy else SAMPLING["temperature"]
        print(json.dumps(run(Path(a.dir), a.url, a.samples, a.workers, a.limit, temp)))
    else:
        print(json.dumps(summary(Path(a.dir)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
