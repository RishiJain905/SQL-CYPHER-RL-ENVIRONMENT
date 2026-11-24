"""Run 3's teacher (design 3.5): a hosted model playing run 3's episodes, graded like step 40.

    python -m sqlrl.run3.teacher pilot --out <dir> [--n 200] [--cap 2.0]     # costs money

**Spending needs Rishi's OK every time** (CLAUDE.md): `pilot` refuses to start without
`--approved`, and stops the batch once the spend reaches `--cap` US dollars (OpenRouter's own
cost accounting, call by call).

`TeacherPolicy` is a policy for `episode.run_episode`: it sends the conversation and the tools to
OpenRouter (native function calling, one call a turn), and turns the reply back into run 3's own
call format (`protocol.call_text`), so the same reader, environment and grader score it as they
score step 40. The model and effort are D162's: `deepseek/deepseek-v4.1-flash-1017` at
`reasoning.effort: max`. Its reasoning text is kept (not excluded, as D162 did) because a
warm-start example needs a thought before its call: the pilot measures how long it is.

The key comes only from `sqlrl.secrets` as a `SecretStr`, is used in the request header and
nowhere else, and never appears in a record, a log or an error message.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sqlrl.run3 import protocol

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash-1017"
EFFORT = "max"
MAX_TOKENS = 16000


class Budget:
    """The batch's spend, shared by every worker; `spent` from OpenRouter's usage accounting."""

    def __init__(self, cap: float):
        self.cap, self.spent, self.calls = cap, 0.0, 0
        self._lock = threading.Lock()

    def add(self, cost: float | None) -> None:
        with self._lock:
            self.spent += float(cost or 0.0)
            self.calls += 1

    def over(self) -> bool:
        with self._lock:
            return self.spent >= self.cap


class OverBudget(RuntimeError):
    pass


def to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run 3's messages as OpenAI chat messages: each tool call gets an id, each tool result
    the id of the call before it; arguments as JSON text; the reasoning field dropped."""
    out: list[dict[str, Any]] = []
    last_id = None
    for k, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for j, tc in enumerate(m["tool_calls"]):
                last_id = f"call_{k}_{j}"
                fn = tc["function"]
                calls.append({"id": last_id, "type": "function",
                              "function": {"name": fn["name"],
                                           "arguments": json.dumps(fn["arguments"])}})  # fmt: skip
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": calls})  # fmt: skip
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": last_id or "call_none",
                        "content": m["content"]})  # fmt: skip
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def to_text(message: dict[str, Any]) -> str:
    """The reply as run 3's turn text: the reasoning, then the call(s) in the model's format."""
    reasoning = (message.get("reasoning") or "").strip()
    tcs = message.get("tool_calls") or []
    if not tcs:
        return reasoning + protocol.THINK_END + (message.get("content") or "")
    parts = []
    for tc in tcs:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_unreadable": fn.get("arguments")}
        args = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in args.items()}
        parts.append(protocol.call_text(fn.get("name", ""), args)[len(protocol.THINK_END) :])
    return reasoning + protocol.THINK_END + "".join(parts)


class TeacherPolicy:
    def __init__(self, budget: Budget, model: str = MODEL, effort: str = EFFORT,
                 max_tokens: int = MAX_TOKENS, timeout: float = 180):  # fmt: skip
        self.budget, self.model, self.effort = budget, model, effort
        self.max_tokens, self.timeout = max_tokens, timeout
        self.usage: list[dict[str, Any]] = []

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        from sqlrl import secrets

        key = secrets.get().openrouter_api_key
        if key is None or not key.get_secret_value().strip():
            raise RuntimeError("OPENROUTER_API_KEY is not set (python -m sqlrl.doctor)")
        req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"), headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key.get_secret_value()})  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:  # the status and OpenRouter's message, no headers
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read()[:300]!r}") from None

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        if self.budget.over():
            raise OverBudget(f"spend reached the cap (${self.budget.cap})")
        body = {"model": self.model, "messages": to_openai(messages), "tools": tools,
                "tool_choice": "auto", "parallel_tool_calls": False,
                "max_tokens": self.max_tokens, "usage": {"include": True},
                "reasoning": {"effort": self.effort, "exclude": False}}  # fmt: skip
        t0 = time.time()
        data = self._post(body)
        u = data.get("usage") or {}
        self.budget.add(u.get("cost"))
        self.usage.append({"cost": u.get("cost"), "prompt": u.get("prompt_tokens"),
                           "completion": u.get("completion_tokens"),
                           "reasoning": (u.get("completion_tokens_details") or {}).get(
                               "reasoning_tokens"), "s": round(time.time() - t0, 1),
                           "provider": data.get("provider")})  # fmt: skip
        return to_text(data["choices"][0]["message"])


# ---- the pilot --------------------------------------------------------------------------------


def pilot_groups(p0: Path, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Phase 0's groups for the pilot: mostly the ones step 40 never got right (where examples
    are the only way in), some wrong-database questions, a few from the useful band to compare."""
    groups = {json.loads(x)["gid"]: json.loads(x)
              for x in (p0 / "groups.jsonl").read_text("utf-8").splitlines()}  # fmt: skip
    right: Counter = Counter()
    seen: Counter = Counter()
    for line in (p0 / "episodes.jsonl").read_text("utf-8").splitlines():
        r = json.loads(line)
        seen[r["gid"]] += 1
        right[r["gid"]] += r.get("outcome") == "right"
    full = [g for g in groups if seen[g] >= 8]
    never = [g for g in full if right[g] == 0 and groups[g]["kind"] == "answer"]
    wrongdb = [g for g in full if groups[g]["kind"] == "unanswerable" and right[g] < 8]
    useful = [g for g in full if 0 < right[g] < 8 and groups[g]["kind"] == "answer"]
    rnd = random.Random(f"run3-teacher-pilot:{seed}")
    for lst in (never, wrongdb, useful):
        rnd.shuffle(lst)
    k_w, k_u = n // 5, n // 10
    pick = never[: n - k_w - k_u] + wrongdb[:k_w] + useful[:k_u]
    return [{**groups[g], "band": ("never" if g in never else "wrongdb" if g in wrongdb
                                   else "useful")} for g in pick]  # fmt: skip


def run_pilot(out: Path, n: int, cap: float, workers: int = 8) -> dict[str, Any]:
    from sqlrl.run3 import settings

    p0 = settings.get().data / "phase0" / "pass8-step40"
    out.mkdir(parents=True, exist_ok=True)
    lock_file = settings.get().data / "neo4j.busy"  # the run-3 Neo4j holds one graph at a time:
    lock_file.write_text("teacher pilot", encoding="utf-8")  # phase 1's evaluations wait on it
    try:
        return _run_pilot(p0, out, n, cap, workers)
    finally:
        lock_file.unlink(missing_ok=True)


def _run_pilot(p0: Path, out: Path, n: int, cap: float, workers: int) -> dict[str, Any]:
    from sqlrl.run3 import episode, measure

    gs = pilot_groups(p0, n)
    (out / "groups.jsonl").write_text("\n".join(json.dumps(g) for g in gs), encoding="utf-8")
    tasks, vers, budget = measure.Tasks(), measure.versions(), Budget(cap)
    lock = threading.Lock()

    def play(g: dict[str, Any]) -> dict[str, Any]:
        task = tasks.task(g)
        setting = measure.setting_for(g, task, vers)
        pol = TeacherPolicy(budget)
        try:
            rec = episode.run_episode(task, pol, setting)
            episode.grade(task, rec)
        except OverBudget as e:
            rec = {"qid": task["q"]["id"], "outcome": "skipped", "why": str(e)}
        except Exception as e:  # noqa: BLE001 - a broken episode is recorded, never the key
            rec = {
                "qid": task["q"]["id"],
                "outcome": "crash",
                "error": f"{type(e).__name__}: {e}"[:300],
            }
        rec.update(gid=g["gid"], stratum=g["stratum"], band=g["band"], engine=g["engine"],
                   source=g["source"], db=g["db"], kind=g["kind"], usage=pol.usage,
                   cost=round(sum(u["cost"] or 0 for u in pol.usage), 6))  # fmt: skip
        return rec

    recs = []
    with (
        (out / "episodes.jsonl").open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(workers) as ex,
    ):
        for rec in ex.map(play, gs):
            with lock:
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
    played = [r for r in recs if r.get("outcome") not in ("skipped", "crash")]
    summ = {"episodes": len(recs), "played": len(played), "spent_usd": round(budget.spent, 4),
            "per_episode_usd": round(budget.spent / max(len(played), 1), 5),
            "right": sum(r.get("outcome") == "right" for r in played),
            "by_band": {b: {"n": sum(r["band"] == b for r in played),
                            "right": sum(r["band"] == b and r.get("outcome") == "right"
                                         for r in played)}
                        for b in ("never", "wrongdb", "useful")},
            "endings": dict(Counter(r.get("ending") for r in played)),
            "reasoning_tokens_p50": sorted(u["reasoning"] or 0 for r in played
                                           for u in r["usage"])[len(played) // 2] if played else 0,
            "crashes": sum(r.get("outcome") == "crash" for r in recs),
            "skipped_over_cap": sum(r.get("outcome") == "skipped" for r in recs)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--cap", type=float, default=2.0, help="US dollars; the batch stops there")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--approved", action="store_true", help="Rishi OK'd this spend")
    a = ap.parse_args(argv)
    if not a.approved:
        print("refused: spending needs Rishi's OK (pass --approved once he has given it)")
        return 2
    print(json.dumps(run_pilot(Path(a.out), a.n, a.cap, a.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
