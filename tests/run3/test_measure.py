from sqlrl.run3 import measure


def _task(qid="bird-7"):
    return {"q": {"id": qid}, "sources": [{"engine": "duckdb", "source": "bird", "db_id": "x"}]}


def test_setting_is_one_per_group_and_seeds_differ():
    g = {"engine": "duckdb"}
    vers = {"duckdb": "1.3.2"}
    a = measure.setting_for(g, _task(), vers).record()
    b = measure.setting_for(g, _task(), vers).record()
    assert a == b  # the 8 samples of a group share it
    others = {str(measure.setting_for(g, _task(f"bird-{i}"), vers).record()) for i in range(40)}
    assert len(others) > 5  # groups differ
    seeds = {measure.stable_int(f"run3-p0:bird-7:duckdb:{s}") for s in range(8)}
    assert len(seeds) == 8


def test_bands_and_stats():
    assert measure._band(0, 8) == "never"
    assert measure._band(8, 8) == "always"
    assert measure._band(3, 8) == "useful"
    recs = [
        {
            "gid": 0,
            "outcome": "right" if i < 3 else "wrong",
            "ending": "submit",
            "n_calls": 1,
            "tokens": {"max_context": 1000 + 500 * i},
        }
        for i in range(8)
    ]
    st = measure._stats(recs, {0: recs})
    assert st["useful"] == 1 and st["pass_at8"] == 1.0 and st["mean_at8"] == 0.375
    assert st["over_trainer_limit"] == 0.25  # 4000 and 4500 are over 3840
