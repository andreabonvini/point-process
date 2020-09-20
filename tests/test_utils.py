from pp.utils import load


def test_load():
    path = "tests/test_data/Y2.csv"
    r = load(path)
    assert r.shape == (733,)
