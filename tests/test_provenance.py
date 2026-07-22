from importlib import metadata

import fieldbird_sim
from fieldbird_sim import provenance


def test_missing_distribution_metadata_uses_honest_version_fallback(monkeypatch):
    def _missing(_distribution: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(fieldbird_sim.metadata, "version", _missing)

    assert fieldbird_sim._read_version() == "0+unknown"


def test_runtime_metadata_records_environment(monkeypatch):
    monkeypatch.setattr(
        provenance.metadata,
        "version",
        lambda distribution: f"installed-{distribution}",
    )

    runtime = provenance.runtime_metadata()

    assert runtime["fieldbird_sim_version"] == fieldbird_sim.__version__
    assert runtime["python_version"]
    assert runtime["python_implementation"]
    assert runtime["platform"]
    assert runtime["dependencies"] == {
        label: f"installed-{distribution}"
        for label, distribution in provenance.RUNTIME_DISTRIBUTIONS.items()
    }


def test_missing_optional_runtime_metadata_is_explicit(monkeypatch):
    def _version(distribution: str) -> str:
        if distribution == "PyYAML":
            raise metadata.PackageNotFoundError
        return f"installed-{distribution}"

    monkeypatch.setattr(provenance.metadata, "version", _version)

    versions = provenance.installed_dependency_versions()

    assert versions["pyyaml"] is None
    assert versions["numpy"] == "installed-numpy"
