"""Run 1's question pool, `pool-v2` (inside WSL, with the trainer's venv; CPU, resumable):

    python -m sqlrl.run1.pool [--name pool-v2] [--per-set cypherbench=2000,gretel=2000,...]

Training-split questions only (the SFT run's `gold-full` set), never the evalset or the test
set (excluded by id and by question text). Three filters the pilots' pool didn't have:

1. **The reference must be able to earn the reward.** Each question's own reference query is
   scored by checker 4 as if the model had written it. A reference the checker can't decide
   (it returns nothing, or only zeros and nulls, on the checker's databases; or no rule proves
   it) would give every rollout "undecided": the question is left out. For gretel, whose
   reference runs on the dataset's own rows, the rows must also say something.
2. **The prompt fits a rollout slot:** at most `--max-prompt-tokens` tokens, counted with the
   model's own chat template.
3. **The schema fits training memory** (`--max-schema-chars`, as in the pilots).

The file holds each dataset's accepted questions in a seeded order, tagged `set_index`; the run
picks a step's questions per dataset from its mix (`config.step_questions`), so changing the
mix never needs a new pool. Every question appears once: no question is seen twice in a run.
Verdicts are kept in `<name>.checked.jsonl`, so a re-run (or a larger `--per-set`) only checks
new candidates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from sqlrl.paths import get_paths

SETS = ["cypherbench", "gretel", "sqale", "text2cypher"]
DEFAULT_PER_SET = {"cypherbench": 2000, "gretel": 2000, "sqale": 1600, "text2cypher": 1000}


def question_text(ex: dict[str, Any]) -> str:
    users = [m for m in ex["messages"] if m["role"] == "user"]
    return " ".join((users[0].get("content") or "").split()).lower() if users else ""


def candidates(train: list[dict[str, Any]], held_out: list[dict[str, Any]], seed: int = 42,
               max_schema_chars: int = 12000) -> dict[str, list[dict[str, Any]]]:  # fmt: skip
    """Each dataset's usable training questions in a seeded order (held-out ids and question
    texts removed, duplicates of a question text kept once)."""
    banned_ids = {e["id"] for e in held_out}
    banned_q = {question_text(e) for e in held_out}
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    seen: set[tuple[str, str]] = set()
    for ex in train:
        if ex["origin"] not in by_set or ex.get("task") != "call":
            continue
        q = question_text(ex)
        if ex["id"] in banned_ids or q in banned_q:
            continue
        if (ex.get("meta") or {}).get("schema_chars", 0) > max_schema_chars:
            continue
        key = (ex["origin"], q + "\x00" + str((ex.get("meta") or {}).get("db_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        by_set[ex["origin"]].append(ex)
    for s in SETS:
        by_set[s].sort(key=lambda e: e["id"])
        random.Random(f"{seed}-{s}").shuffle(by_set[s])
    return by_set


def reference_call(ex: dict[str, Any]) -> tuple[str, str]:
    """(the reference query, the reply text a model would write to call the tool with it)."""
    fn = ex["messages"][-1]["tool_calls"][0]["function"]
    args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
    arg = next(iter(args))
    gold = (ex.get("meta") or {}).get("gold") or args[arg]
    text = (f"</think><tool_call>{fn['name']}<arg_key>{arg}</arg_key>"
            f"<arg_value>{gold}</arg_value></tool_call>")  # fmt: skip
    return gold, text


def self_check(ex: dict[str, Any], data, checker: int = 4) -> tuple[bool, str]:
    """Would a rollout that writes the reference query get the reward? (ok, why not)."""
    from asktrainer import publiceval
    from asktrainer.data.schema import Example
    from asktrainer.querycheck import _connect, _informative, _run

    e = Example.model_validate(ex)
    gold, text = reference_call(ex)
    item = publiceval.score(e, text, data, checker=checker)
    if item.get("exec_ok") is not True:
        return False, item.get("reason") or "undecided"
    if e.origin == "gretel":  # real rows: the reference's result must say something
        db = _connect(data.schema_script(e))
        try:
            rows = _run(db, gold)
        finally:
            db.close()
        if not _informative(rows):
            return False, "reference_says_nothing"
    return True, "ok"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool_path(name: str) -> Path:
    return get_paths().data / "run1" / f"{name}.jsonl"


def load_pool(name: str) -> dict[str, list[dict[str, Any]]]:
    """{dataset: its questions in pool order}."""
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    for row in read_jsonl(pool_path(name)):
        by_set[row["origin"]].append(row)
    for rows in by_set.values():
        rows.sort(key=lambda r: r["set_index"])
    return by_set


def build(name: str, per_set: dict[str, int], seed: int, max_prompt_tokens: int,
          log=print) -> dict[str, Any]:  # fmt: skip
    p = get_paths()
    sys.path.insert(0, str(p.trainer_repo / "src"))
    from asktrainer import publiceval
    from asktrainer.settings import get_paths as trainer_paths
    from transformers import AutoTokenizer

    from sqlrl.run1.srpo import render_ids

    public = trainer_paths().public
    train = read_jsonl(public / "gold-full" / "examples_train.jsonl")
    held_out = read_jsonl(p.evalset) + read_jsonl(public / "testset" / "examples_val.jsonl")
    cands = candidates(train, held_out, seed)
    data = publiceval.PublicData(public)
    if publiceval._graph() is None:
        raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
    tok = AutoTokenizer.from_pretrained(str(p.model_hf), trust_remote_code=True)
    out = pool_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".checked.jsonl")
    checked: dict[str, dict[str, Any]] = {}
    if log_path.exists():
        checked = {r["id"]: r for r in read_jsonl(log_path)}
    stats: dict[str, Any] = {"available": {s: len(cands[s]) for s in SETS}}
    accepted: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    with log_path.open("a", encoding="utf-8", newline="\n") as lf:
        for s in SETS:
            t0, want, pos = time.time(), per_set.get(s, 0), 0
            while len(accepted[s]) < want and pos < len(cands[s]):
                # a batch a little larger than what is missing, checked graph by graph (one
                # graph load per CypherBench graph), accepted in the pool's seeded order
                need = want - len(accepted[s])
                batch = cands[s][pos : pos + max(int(need * 1.25) + 8, 16)]
                pos += len(batch)
                todo = [e for e in batch if e["id"] not in checked]
                todo.sort(key=lambda e: (e.get("meta") or {}).get("db_id") or "")
                for e in todo:
                    n_tok = len(render_ids(tok, e["messages"][:2], e["tools"], "high"))
                    if n_tok > max_prompt_tokens:
                        v = {"id": e["id"], "origin": s, "ok": False, "reason": "prompt_too_long",
                             "prompt_tokens": n_tok}  # fmt: skip
                    else:
                        try:
                            ok, why = self_check(e, data)
                        except Exception as err:  # a reference the checker can't even run
                            ok, why = False, f"check_error:{type(err).__name__}"
                        v = {"id": e["id"], "origin": s, "ok": ok, "reason": why,
                             "prompt_tokens": n_tok}  # fmt: skip
                    checked[e["id"]] = v
                    lf.write(json.dumps(v) + "\n")
                    lf.flush()
                for e in batch:
                    if checked[e["id"]]["ok"] and len(accepted[s]) < want:
                        row = dict(e)
                        row["rl_thinking"] = "high"
                        row["set_index"] = len(accepted[s])
                        row["prompt_tokens"] = checked[e["id"]]["prompt_tokens"]
                        accepted[s].append(row)
                log(f"{s}: {len(accepted[s])}/{want} accepted, {pos} looked at, "
                    f"{time.time() - t0:.0f} s")  # fmt: skip
            seen = [checked[e["id"]] for e in cands[s][:pos]]
            why: dict[str, int] = {}
            for v in seen:
                if not v["ok"]:
                    why[v["reason"]] = why.get(v["reason"], 0) + 1
            stats[s] = {"accepted": len(accepted[s]), "looked_at": pos,
                        "rejected": dict(sorted(why.items(), key=lambda kv: -kv[1]))}  # fmt: skip
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for s in SETS:
            for row in accepted[s]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    stats.update(pool=name, questions=sum(len(v) for v in accepted.values()), seed=seed,
                 held_out=len(held_out), max_prompt_tokens=max_prompt_tokens)  # fmt: skip
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="pool-v2")
    ap.add_argument("--per-set", default="", help="e.g. cypherbench=2000,gretel=2000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-prompt-tokens", type=int, default=4000)
    a = ap.parse_args(argv)
    per_set = dict(DEFAULT_PER_SET)
    for part in filter(None, a.per_set.split(",")):
        k, v = part.split("=")
        per_set[k] = int(v)
    stats = build(a.name, per_set, a.seed, a.max_prompt_tokens,
                  log=lambda m: print(m, flush=True))  # fmt: skip
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
