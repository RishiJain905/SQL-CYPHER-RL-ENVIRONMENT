import json

from sqlrl.baseline import sample


class FakeLlamaeval:
    @staticmethod
    def request_body(ex, max_tokens):
        return {"messages": ex["messages"], "temperature": 0.0, "top_k": 1,
                "max_tokens": max_tokens}  # fmt: skip


def test_request_body_switches_sampling_on():
    body = sample.request_body(FakeLlamaeval, {"messages": []}, k=2, i=7)
    assert body["temperature"] == 1.0 and body["top_p"] == 1.0
    assert body["top_k"] == 0 and body["min_p"] == 0.0
    assert body["seed"] == 3007
    assert body["max_tokens"] == 2048


def test_seeds_differ_across_samples_and_questions():
    seeds = {sample.seed(k, i) for k in range(8) for i in range(200)}
    assert len(seeds) == 1600


def test_read_preds_retries_failed_answers(tmp_path):
    f = tmp_path / "preds_s0.jsonl"
    rows = [
        {"id": "a", "output": "x"},
        {"id": "b", "output": "", "error": "timeout"},
        {"id": "c", "output": "y"},
        {"id": "c", "output": "", "error": "late failure"},  # never replaces a real answer
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert sample.read_done(f) == {"a", "c"}
    f.write_text(f.read_text() + json.dumps({"id": "b", "output": "z"}) + "\n", encoding="utf-8")
    preds = sample.read_preds(f)
    assert preds["b"]["output"] == "z" and preds["c"]["output"] == "y"
    assert sample.read_done(f) == {"a", "b", "c"}
