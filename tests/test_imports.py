import importlib


def test_imports():
    assert importlib.import_module("fieldbird_sim") is not None
    assert importlib.import_module("fieldbird_sim.cli") is not None
    assert importlib.import_module("fieldbird_sim.run") is not None
