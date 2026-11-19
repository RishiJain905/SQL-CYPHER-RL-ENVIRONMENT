"""Multi-turn episodes through llama-server's raw completion endpoint (Windows, standard library
only, next to the server). One engine for the training rollouts, the evaluations and the
no-training tests, so they all measure the same thing.

**An episode** (decisions R15, design.md section 2): the model thinks and calls the query tool;
the environment (`env.py`) runs the query on the question's look-around database and the result
comes back as a tool message; the model may fix the query and call again, at most `max_queries`
times, or stop with a plain reply. **Its answer is the last query that ran.**

**The context is built token by token and only ever grows.** A turn is sampled as in run 1 (the
thought up to `</think>` or the thinking budget, then the call, on the worker's own slot).
After a call the result is appended as the chat template renders a tool message:

    <start><|Tool|><tool_response> RESULT </tool_response><end><start><|Bot|><think>

The wrapper tokens come from the template; **the result's own text is tokenised as plain text**
(`parse_special` off), so a cell that holds `</tool_response>` or `<|Bot|>` stays data and can't
end the tool message or open a turn. Because nothing earlier is ever re-rendered, the slot's
cache holds the whole episode, and the trainer reads one sequence per episode: `gen_mask` is 1
on the tokens the model sampled and 0 on the result blocks and forced tokens, `gen_turn` says
which turn each token belongs to.

**Scheduling** is run 1's (a worker per slot, a question's episodes one after another on a
slot, idle workers help) with one addition: the checker's Neo4j holds one graph at a time, so
Cypher questions take turns on a lane (`Plan`). Episodes that need the graph that is loaded run
side by side on up to `cypher_slots` workers; the other workers take SQL questions; the graph
changes only when no Cypher episode is running.

An episode line: {"qid", "origin", "g", "seed", "slot", "prompt_ids", "gen_ids", "gen_mask",
"gen_turn", "behavior_lp", "turns", "queries", "first_query", "final_query", "n_calls", "end",
"obs", "text", "first_text", "look_error", "secs"}. `text` is the final query as a one-call
reply (`calls.as_call`): what the checker scores.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from sqlrl.run2 import calls, env, prompt

THINK_END = calls.THINK_END
BOS, EOS = "<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
TOOL_OPEN = BOS + "<|Tool|><tool_response>"
TOOL_CLOSE = "</tool_response>" + EOS + BOS + "<|Bot|><think>"
LONG_FIRST = ["cypherbench", "text2cypher", "sqale", "gretel"]
CTX_MARGIN = 16  # tokens kept free in a slot
MIN_ROOM = 96  # a turn isn't started with less room than this
WAIT = "wait"
MSG_UNREADABLE = {
    "cut": "the tool call was cut off",
    "several_calls": "one tool call at a time",
    "wrong_tool": "that tool is not available here",
    "bad_argument": "the tool call has no query",
    "markup": "the tool call could not be read",
}


def _post(url: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """A llama-server's endpoints, with the rendered prompts cached per question."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._lock = threading.Lock()
        self._prompts: dict[tuple[str, str, int], list[int]] = {}
        self.think_end_id = self._one(THINK_END)
        self.eos_id = self._one(EOS)
        self.tool_open = self.tokenize(TOOL_OPEN)
        self.tool_close = self.tokenize(TOOL_CLOSE)

    def _one(self, text: str) -> int:
        ids = self.tokenize(text)
        if len(ids) != 1:
            raise RuntimeError(f"{text} is not one token: {ids}")
        return ids[0]

    def tokenize(self, text: str, special: bool = True) -> list[int]:
        r = _post(self.url + "/tokenize", {"content": text, "add_special": False,
                                           "parse_special": special})  # fmt: skip
        return list(r["tokens"])

    def render(self, messages: list[dict[str, Any]], tools: list, level: str) -> str:
        """The chat template's text for `messages` with a generation prompt."""
        kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
        if level != "off":
            kwargs["reasoning_effort"] = level
        body = {"messages": messages, "tools": tools, "chat_template_kwargs": kwargs}
        return _post(self.url + "/apply-template", body)["prompt"]

    def prompt_ids(self, pub: dict[str, Any], cfg: dict[str, Any]) -> list[int]:
        variant, max_queries = prompt.protocol(cfg, pub["origin"])
        key = (pub["id"], variant, max_queries)
        with self._lock:
            if key in self._prompts:
                return self._prompts[key]
        msgs = prompt.episode_messages(pub, cfg)
        ids = self.tokenize(self.render(msgs, pub["tools"], pub.get("thinking") or "high"))
        with self._lock:
            self._prompts[key] = ids
        return ids

    def observation_ids(self, text: str) -> list[int]:
        """A tool result as the next tokens of the context: the template's wrapper around the
        result's text, which is tokenised as plain text (no special tokens inside it)."""
        return self.tool_open + self.tokenize(text, special=False) + self.tool_close

    def complete(
        self,
        ids: list[int],
        n: int,
        sampling: dict[str, Any],
        seed: int,
        slot: int,
        stop: list[str] | None = None,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        body = {"prompt": ids, "n_predict": n, "temperature": sampling["temperature"],
                "top_p": sampling.get("top_p", 1.0), "top_k": sampling.get("top_k", 0),
                "min_p": 0.0, "seed": seed, "return_tokens": True, "cache_prompt": True,
                "id_slot": slot}  # fmt: skip
        if logprobs:
            body["n_probs"] = 1
        if stop:
            body["stop"] = stop
        return _post(self.url + "/completion", body)


def _chosen_logprobs(r: dict[str, Any]) -> list[float | None] | None:
    """The sampled tokens' log-probabilities as the server computed them; None when the request
    didn't ask for them or the build doesn't report them."""
    probs = r.get("completion_probabilities")
    if not probs:
        return None
    out: list[float | None] = []
    for p in probs:
        if "logprob" not in p:
            return None
        out.append(float(p["logprob"]))
    return out


def _turn(
    server: Server,
    ctx: list[int],
    room: int,
    cfg: dict[str, Any],
    sampling: dict[str, Any],
    seed: int,
    slot: int,
    logprobs: bool,
) -> dict[str, Any]:
    """One turn's tokens: the thought (closed for the model at the budget, mask 0) and what
    follows it. Run 1's two requests, with the room left in the slot respected."""
    end = server.think_end_id
    r1 = server.complete(ctx, max(min(cfg["think_budget"], room - 2), 1), sampling, seed, slot,
                         [THINK_END], logprobs)  # fmt: skip
    think = list(r1.get("tokens") or [])
    lp1 = _chosen_logprobs(r1)
    if lp1 is not None and len(lp1) == len(think) - 1:  # the stop token has no probability
        lp1 = lp1 + [None]
    forced = not think or think[-1] != end
    mask = [1] * len(think)
    if forced:
        think.append(end)
        mask.append(0)
        lp1 = lp1 + [None] if lp1 is not None else None
    n2 = max(min(cfg["max_tokens"] - len(think), room - len(think)), 1)
    r2 = server.complete(ctx + think, n2, sampling, seed + 1, slot, None, logprobs)
    answer = list(r2.get("tokens") or [])
    lp2 = _chosen_logprobs(r2)
    lps = lp1 + lp2 if lp1 is not None and lp2 is not None else None
    if lps is not None and len(lps) != len(think) + len(answer):
        lps = None  # the server's lists don't line up with its tokens: don't guess
    return {
        "ids": think + answer,
        "mask": mask + [1] * len(answer),
        "lps": lps,
        "text": (r1.get("content") or "") + THINK_END + (r2.get("content") or ""),
        "think_tokens": len(think),
        "answer": answer,
        "forced": forced,
        "finish": r2.get("stop_type"),
        "closed": bool(answer) and answer[-1] == server.eos_id,
    }


def episode(server: Server, pub: dict[str, Any], look: env.Look, g: int, seed: int,
            cfg: dict[str, Any], slot: int, sampling: dict[str, Any], obs: dict[str, Any],
            logprobs: bool = False, history: str = "full") -> dict[str, Any]:  # fmt: skip
    """One episode of `pub` on `slot`. `history` "stripped" (tests only) shows earlier turns
    without their thoughts, as a client that drops `reasoning_content` would: its token lists
    then aren't one growing sequence and must not be trained on."""
    t0 = time.time()
    _, max_q = prompt.protocol(cfg, pub["origin"])
    single = pub["origin"] in (cfg.get("single_turn_sets") or [])
    first = server.prompt_ids(pub, cfg)
    ctx = list(first)
    limit = int(cfg["slot_ctx"]) - CTX_MARGIN
    gen: list[int] = []
    mask: list[int] = []
    turn_of: list[int] = []
    lps: list[float | None] | None = [] if logprobs else None
    turns: list[dict[str, Any]] = []
    ran: list[dict[str, Any]] = []
    calls_used, end = 0, "limit"
    for t in range(1, max_q + 2):
        room = limit - len(ctx)
        if room < MIN_ROOM:
            end = "context"
            break
        tr = _turn(server, ctx, room, cfg, sampling, seed + 2 * (t - 1), slot, logprobs)
        gen += tr["ids"]
        mask += tr["mask"]
        turn_of += [t] * len(tr["ids"])
        if lps is not None:
            lps = lps + tr["lps"] if tr["lps"] is not None else None
        kind, query, why = calls.read_call(tr["text"], pub["tool"], pub["arg"])
        turn: dict[str, Any] = {"t": t, "text": tr["text"], "tokens": len(tr["ids"]),
                                "think_tokens": tr["think_tokens"], "forced": tr["forced"],
                                "finish": tr["finish"], "call": kind, "why": why}  # fmt: skip
        turns.append(turn)
        if kind == "none":
            end = "reply" if ran else "no_query"
            break
        if single:  # run 1's protocol: the first call is the answer, nothing is shown
            if kind == "ok":
                calls_used = 1
                ran.append({"t": t, "query": query, "kind": "unseen", "rows": None, "shown": 0,
                            "tables": None, "ms": 0.0, "obs": ""})  # fmt: skip
            end = "single"
            break
        if not tr["closed"]:  # the turn ran out of tokens: nothing can be rendered after it
            end = "cut"
            break
        if calls_used >= max_q:  # one call too many: it is not run
            end = "limit"
            break
        calls_used += 1
        if kind == "ok":
            res = look.run(query)
            text = env.render(res, obs["rows"], obs["style"])
            shown = min(len(res.rows), obs["rows"])
            item = {"t": t, "query": query, "kind": res.kind, "rows": res.total, "shown": shown,
                    "tables": res.tables, "ms": res.ms, "obs": text}  # fmt: skip
            ran.append(item)
            turn["result"] = res.kind
        else:
            text = "Error: " + MSG_UNREADABLE.get(why, MSG_UNREADABLE["markup"])
            turn["result"] = "unreadable"
        turn["obs"] = text
        o = server.observation_ids(text)
        gen += o
        mask += [0] * len(o)
        turn_of += [t] * len(o)
        if lps is not None:
            lps = lps + [None] * len(o)
        if history == "stripped":  # the turn without its thought: `</think>`, the call, the result
            ctx = ctx[:-1] + [server.think_end_id] + tr["answer"] + o
        else:
            ctx = ctx + tr["ids"] + o
    first_q = ran[0]["query"] if ran else None
    final_q = ran[-1]["query"] if ran else None
    return {
        "qid": pub["id"],
        "origin": pub["origin"],
        "g": g,
        "seed": seed,
        "slot": slot,
        "prompt_ids": first,
        "gen_ids": gen,
        "gen_mask": mask,
        "gen_turn": turn_of,
        "behavior_lp": lps,
        "turns": turns,
        "queries": ran,
        "first_query": first_q,
        "final_query": final_q,
        "n_calls": calls_used,
        "end": end,
        "obs": obs,
        "history": history,
        "text": calls.as_call(final_q, pub["tool"], pub["arg"]),
        "first_text": calls.as_call(first_q, pub["tool"], pub["arg"]),
        "look_error": look.error,
        "secs": round(time.time() - t0, 2),
    }


def read_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a half-written last line: answered again
                continue
            if "error" not in r:
                done.add((r["qid"], r["g"]))
    return done


def seed_for(step: int, q: int, g: int) -> int:
    """Distinct per step, question and episode, with room for every turn's two requests (a turn
    uses seed + 2 (t - 1) and the next one)."""
    return 1_000_000 * step + 1_000 * q + 20 * g


class Plan:
    """Which episode a worker does next.

    As in run 1, a worker stays on its question (the prompt is in its slot's cache), then starts
    the next unstarted one, then takes an episode from the question with the most left. Cypher
    questions add the lane: `key_of[q]` is the graph a question needs (None for SQL). Only
    questions of the lane's graph may run; at most `cypher_slots` Cypher episodes at once; the
    lane moves to the next graph when no Cypher episode is running and none of its graph is
    left. `take` returns (question, episode, taken from another worker), `WAIT` (only Cypher
    work is left and the lane is busy), or None (nothing left)."""

    def __init__(self, groups: list[tuple[int, list[int]]],
                 key_of: dict[int, tuple[str, str] | None], cypher_slots: int = 8):  # fmt: skip
        self._lock = threading.Lock()
        self._pending = {q: list(gs) for q, gs in groups if gs}
        self._unstarted = [q for q, gs in groups if gs]
        self._key = key_of
        self._cap = max(cypher_slots, 1)
        self.lane: tuple[str, str] | None = None
        self._flying = 0  # Cypher episodes running now
        self.switches = 0

    def _pop(self, q: int, stolen: bool) -> tuple[int, int, bool]:
        g = self._pending[q].pop(0)
        if not self._pending[q]:
            del self._pending[q]
        if q in self._unstarted:
            self._unstarted.remove(q)
        if self._key.get(q) is not None:
            self._flying += 1
        return q, g, stolen

    def _cypher(self, current: int | None) -> tuple[int, int, bool] | None:
        if self._flying >= self._cap:
            return None
        mine = [q for q in self._pending if self._key.get(q) == self.lane] if self.lane else []
        if not mine and self._flying == 0:  # the lane is free: the next graph
            nxt = next((q for q in self._unstarted if self._key.get(q) is not None), None)
            if nxt is None:
                nxt = next((q for q in self._pending if self._key.get(q) is not None), None)
            if nxt is None:
                return None
            self.lane = self._key[nxt]
            self.switches += 1
            mine = [q for q in self._pending if self._key.get(q) == self.lane]
        if not mine:
            return None
        if current in mine:
            return self._pop(current, False)
        fresh = [q for q in self._unstarted if q in mine]
        if fresh:
            return self._pop(fresh[0], False)
        return self._pop(max(mine, key=lambda k: len(self._pending[k])), True)

    def take(self, current: int | None) -> tuple[int, int, bool] | str | None:
        with self._lock:
            if not self._pending:
                return None
            if current in self._pending and self._key.get(current) is None:
                return self._pop(current, False)
            job = self._cypher(current)
            if job is not None:
                return job
            sql = [q for q in self._unstarted if self._key.get(q) is None]
            if sql:
                return self._pop(sql[0], False)
            left = [q for q in self._pending if self._key.get(q) is None]
            if left:
                return self._pop(max(left, key=lambda k: len(self._pending[k])), True)
            return WAIT

    def done(self, q: int) -> None:
        with self._lock:
            if self._key.get(q) is not None:
                self._flying -= 1


def group_order(pubs: list[dict[str, Any]]) -> list[int]:
    """Question indices: the datasets with the longest answers first, CypherBench by graph (so
    each graph is loaded once), longer prompts first within that."""
    rank = {s: i for i, s in enumerate(LONG_FIRST)}
    return sorted(
        range(len(pubs)),
        key=lambda q: (
            rank.get(pubs[q]["origin"], len(rank)),
            str(pubs[q]["db"].get("graph") or ""),
            -int(pubs[q].get("prompt_tokens") or 0),
            q,
        ),
    )


def train_spec(cfg: dict[str, Any], step: int):
    """The training rollouts' settings per (question, episode): sampled, a result format drawn
    per episode, the server's log-probabilities for the first `logprob_rollouts` of a group."""
    n_lp = int(cfg.get("logprob_rollouts", cfg["group_size"]))
    sampling = {"temperature": cfg["temperature"], "top_p": cfg["top_p"], "top_k": 0}

    def spec(q: int, g: int) -> dict[str, Any]:
        seed = seed_for(step, q, g)
        return {"seed": seed, "sampling": sampling, "obs": env.obs_setting(cfg, seed),
                "logprobs": g < n_lp}  # fmt: skip

    return spec


def eval_spec(cfg: dict[str, Any], base: int = 7_000):
    """An evaluation's settings: episode 0 is greedy, episodes 1.. are sampled as the yardstick
    samples (temperature 1.0, no top-p / top-k cut), all with the evaluation's result format."""
    obs = env.obs_setting(cfg, None)

    def spec(q: int, g: int) -> dict[str, Any]:
        if g == 0:
            sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
        else:
            sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
        return {"seed": seed_for(base, q, g), "sampling": sampling, "obs": obs, "logprobs": False}

    return spec


def generate(url: str, pubs: list[dict[str, Any]], cfg: dict[str, Any], out: Path, spec,
             n_per_question: int, graphs: env.GraphHost | None = None,
             history: str = "full") -> dict[str, Any]:  # fmt: skip
    """Run `n_per_question` episodes of every question into `out` (append; a restart keeps
    what's done, and a failed episode is retried on the next run)."""
    server = Server(url)
    if out.exists():  # a crash mid-write leaves a last line without its newline: cut it
        data = out.read_bytes()
        if data and not data.endswith(b"\n"):
            out.write_bytes(data[: data.rfind(b"\n") + 1])
    done = read_done(out)
    order = group_order(pubs)
    groups = [(q, [g for g in range(n_per_question) if (pubs[q]["id"], g) not in done])
              for q in order]  # fmt: skip
    key_of = {q: (env.GraphHost.key(pubs[q]) if pubs[q]["dialect"] == "cypher" else None)
              for q in range(len(pubs))}  # fmt: skip
    if any(k is not None for q, k in key_of.items() if dict(groups).get(q)):
        if graphs is None or not graphs.ok():
            raise RuntimeError("the checker's Neo4j (asktrainer-neo4j) doesn't answer: start it")
    plan = Plan(groups, key_of, int(cfg.get("cypher_slots", 8)))
    n_jobs = sum(len(gs) for _, gs in groups)
    lock = threading.Lock()
    tot = {"errors": 0, "gen_tokens": 0, "stolen": 0, "done": 0, "waits": 0}
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as f:

        def worker(slot: int) -> None:
            current: int | None = None
            look: env.Look | None = None
            fails = 0
            try:
                while fails < 3:  # a slot that keeps failing stops; the others finish its work
                    job = plan.take(current)
                    if job is None:
                        return
                    if job == WAIT:
                        with lock:
                            tot["waits"] += 1
                        time.sleep(0.2)
                        continue
                    q, g, stolen = job
                    pub, sp = pubs[q], spec(q, g)
                    try:
                        if q != current or look is None:
                            if look is not None:
                                look.close()
                            look = None
                            if key_of[q] is not None:
                                graphs.ensure(pub)
                            look = env.Look(pub, graphs)
                        current = q
                        res = episode(server, pub, look, g, sp["seed"], cfg, slot, sp["sampling"],
                                      sp["obs"], sp["logprobs"], history)  # fmt: skip
                        fails = 0
                    except Exception as e:  # retried on the next run
                        fails += 1
                        res = {"qid": pub["id"], "g": g, "seed": sp["seed"],
                               "error": f"{type(e).__name__}: {str(e)[:160]}"}  # fmt: skip
                    finally:
                        plan.done(q)
                    with lock:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f.flush()
                        tot["done"] += 1
                        tot["stolen"] += stolen
                        if "error" in res:
                            tot["errors"] += 1
                        else:
                            tot["gen_tokens"] += sum(res["gen_mask"])
            finally:
                if look is not None:
                    look.close()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(cfg["slots"])]  # fmt: skip
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    secs = time.time() - t0
    loads = dict(graphs.load_s) if graphs is not None else {}
    return {"rollouts": n_jobs, "attempted": tot["done"], "errors": tot["errors"],
            "seconds": round(secs, 1), "gen_tokens": tot["gen_tokens"],
            "taken_over": tot["stolen"], "lane_waits": tot["waits"],
            "graph_loads": len(loads), "graph_load_s": round(sum(loads.values()), 1),
            "graph_failed": len(graphs.failed) if graphs is not None else 0,
            "tok_s": round(tot["gen_tokens"] / secs, 1) if secs > 0 else None}  # fmt: skip
