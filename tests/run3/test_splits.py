from sqlrl.run3 import splits


def test_heldout_and_never():
    assert splits.split_of("bird", "airline", "bird-1") == "heldout"
    assert splits.split_of("birddev", "superhero", "birddev-1") == "never"
    assert splits.split_of("cypherbench", "nba", "cb-1") == "never"  # a test-split graph
    assert splits.split_of("cypherbench", "soccer", "cb-2") in ("dev", "train")


def test_dev_share_and_stability():
    ids = [f"bird-{i}" for i in range(4000)]
    dev = [i for i in ids if splits.split_of("bird", "movie", i) == "dev"]
    assert 150 < len(dev) < 250  # about 1 in 20
    assert dev == [i for i in ids if splits.split_of("bird", "movie", i) == "dev"]  # stable
    assert not splits.trainable("bird", "movie", dev[0])
