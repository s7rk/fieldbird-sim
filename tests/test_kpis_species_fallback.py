from fieldbird_sim.legacy.globals import SPECIES_LIST
from fieldbird_sim.legacy.kpis import _resolve_species_list


def test_resolve_species_list_uses_default_when_none():
    assert _resolve_species_list(None) == list(SPECIES_LIST)


def test_resolve_species_list_handles_empty_and_string():
    assert _resolve_species_list([]) == list(SPECIES_LIST)
    assert _resolve_species_list("Feldlerche") == ["Feldlerche"]
