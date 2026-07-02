from types import SimpleNamespace

from hypercloning.models.mixtral.layout import LEGACY_MODULELIST_EXPERTS, PACKED_EXPERTS, UNKNOWN, detect_mixtral_moe_layout


class _LegacyExperts(list):
    pass


def test_detect_packed_experts_layout():
    moe = SimpleNamespace(experts=SimpleNamespace(gate_up_proj=object(), down_proj=object()))
    assert detect_mixtral_moe_layout(moe) == PACKED_EXPERTS


def test_detect_legacy_modulelist_layout():
    experts = _LegacyExperts([SimpleNamespace(w1=object(), w2=object(), w3=object())])
    moe = SimpleNamespace(experts=experts)
    assert detect_mixtral_moe_layout(moe) == LEGACY_MODULELIST_EXPERTS


def test_detect_unknown_layout():
    assert detect_mixtral_moe_layout(SimpleNamespace()) == UNKNOWN
