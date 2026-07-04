from pathlib import Path
from types import SimpleNamespace

from hc_validate.random_models import apply_json_overrides, infer_auto_map_from_architecture_dir, normalize_families, safe_filename


def test_safe_filename_normalizes_model_labels():
    assert safe_filename("Org/Model Name@v1") == "org-model-name-v1"


def test_normalize_families_handles_aliases():
    assert normalize_families(["gpt-neox", "gemma-2"]) == ["pythia", "gemma2"]


def test_apply_json_overrides_sets_attributes():
    cfg = SimpleNamespace(hidden_size=64)
    apply_json_overrides(cfg, {"hidden_size": 128, "model_type": "demo"})
    assert cfg.hidden_size == 128
    assert cfg.model_type == "demo"


def test_infer_auto_map_from_architecture_dir(tmp_path: Path):
    (tmp_path / "configuration_demo.py").write_text("class DemoConfig:\n    pass\n", encoding="utf-8")
    (tmp_path / "modeling_demo.py").write_text("class DemoForCausalLM:\n    pass\n", encoding="utf-8")
    assert infer_auto_map_from_architecture_dir(tmp_path) == {
        "AutoConfig": "configuration_demo.DemoConfig",
        "AutoModelForCausalLM": "modeling_demo.DemoForCausalLM",
    }
