"""Rebuild an episode's conversation turn by turn and render it as step 40 sees it (Spark's chat
template through llama-server), for warm-start examples whose prompts were not recorded (the
teacher's episodes).

    python -m sqlrl.run3.render check --episodes <phase0 pass8>/episodes.jsonl --url ... [--n 40]
    python -m sqlrl.run3.render add --episodes <teacher>/episodes.jsonl --out <file> --url ...

`check` rebuilds step 40's own right episodes, whose prompts `measure` recorded from the server,
and compares byte for byte: the rebuild must reproduce what the model saw before it is trusted
with the teacher's. `add` writes the right episodes with a `rendered` list (one prompt a turn),
the shape `warmstart.collect` reads. Needs a llama-server serving a Spark GGUF (the template is
the model's own).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlrl.run3 import protocol
from sqlrl.run3.policy import LlamaPolicy


def setting_from_record(d: dict[str, Any]) -> protocol.Setting:
    srcs = [protocol.Source(**s) for s in d.get("sources") or []]
    return protocol.Setting(**{k: v for k, v in d.items() if k != "sources"}, sources=srcs)


def turn_contexts(rec: dict[str, Any]) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """(messages, tools) before each turn, as `episode.run_episode` built them."""
    s = setting_from_record(rec["setting"])
    tools = protocol.tools(s)
    # `rec["prompt"]` is the episode's live message list (run_episode keeps a reference), so it
    # holds the whole conversation: the opening is its first two messages (system, user)
    messages = [dict(m) for m in rec["prompt"][:2]]
    out = []
    for turn in rec["turns"]:
        out.append(([dict(m) for m in messages], tools))
        if "obs" not in turn:  # the ending: nothing follows
            break
        text = turn["text"]
        call = protocol.read_call(text, s)
        messages.append(protocol.tool_call_message(call, text) if call.kind == "ok" else
                        {"role": "assistant", "content": protocol.answer_part(text)})  # fmt: skip
        messages.append({"role": "tool", "content": turn["obs"]})
    return out


def render_episode(rec: dict[str, Any], pol: LlamaPolicy) -> list[str]:
    return [pol.render(m, t) for m, t in turn_contexts(rec)]


def cmd_check(episodes: Path, url: str, n: int) -> dict[str, Any]:
    pol = LlamaPolicy(url)
    seen = same = 0
    first_diff = None
    for line in episodes.read_text("utf-8").splitlines():
        r = json.loads(line)
        if not r.get("rendered"):
            continue
        got = render_episode(r, pol)
        seen += 1
        if got == r["rendered"]:
            same += 1
        elif first_diff is None:
            k = next((i for i, (a, b) in enumerate(zip(got, r["rendered"], strict=False))
                      if a != b), len(got))  # fmt: skip
            first_diff = {"qid": r["qid"], "turn": k, "turns": len(r["rendered"]),
                          "rebuilt_turns": len(got)}  # fmt: skip
        if seen >= n:
            break
    return {"episodes": seen, "identical": same, "first_difference": first_diff}


def cmd_add(episodes: Path, out: Path, url: str) -> dict[str, Any]:
    pol = LlamaPolicy(url)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for line in episodes.read_text("utf-8").splitlines():
            r = json.loads(line)
            if r.get("outcome") != "right" or not r.get("turns"):
                continue
            r["rendered"] = render_episode(r, pol)
            r["teacher"] = True
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return {"written": n, "out": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--episodes", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8098")
    c.add_argument("--n", type=int, default=40)
    a_ = sub.add_parser("add")
    a_.add_argument("--episodes", required=True)
    a_.add_argument("--out", required=True)
    a_.add_argument("--url", default="http://127.0.0.1:8098")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        print(json.dumps(cmd_check(Path(a.episodes), a.url, a.n), indent=1))
    else:
        print(json.dumps(cmd_add(Path(a.episodes), Path(a.out), a.url), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
