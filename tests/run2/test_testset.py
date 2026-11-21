"""The test-set harness (`sqlrl.run2.testset`): what it reads and the settings it answers under."""

from __future__ import annotations

from sqlrl.paths import get_paths
from sqlrl.run2 import config as rcfg
from sqlrl.run2 import episodes
from sqlrl.run2 import pool as rpool
from sqlrl.run2.testset import one_query


def test_the_test_set_sits_beside_the_evaluation_set():
    p = rpool.testset_path()
    assert p.name == "examples_val.jsonl"
    assert p.parent.name == "testset"
    assert p.parent.parent == get_paths().evalset.parent.parent


def test_one_query_is_the_anchor_setting_and_leaves_the_run_config_alone():
    cfg = rcfg.new_config({"prompt_variant": "turns", "max_queries": 3})
    one = one_query(cfg)
    assert one["prompt_variant"] == "sft" and one["max_queries"] == 1
    assert cfg["prompt_variant"] == "turns" and cfg["max_queries"] == 3
    assert {k: v for k, v in one.items() if k not in ("prompt_variant", "max_queries")} == {
        k: v for k, v in cfg.items() if k not in ("prompt_variant", "max_queries")
    }


def test_the_two_protocols_are_told_apart():
    """`episodes.claim` refuses to mix them in one folder: the protocols must differ."""
    cfg = rcfg.new_config({"prompt_variant": "turns", "max_queries": 3})
    assert episodes._protocol(cfg, "full") != episodes._protocol(one_query(cfg), "full")
