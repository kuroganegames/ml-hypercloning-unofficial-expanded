"""Built-in architecture cloner specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelClonerSpec:
    """Lazy registration metadata for an architecture-specific cloner."""

    key: str
    config_module: str
    config_name: str
    cloner_module: str
    cloner_name: str


BUILTIN_CLONERS: tuple[ModelClonerSpec, ...] = (
    ModelClonerSpec("llama", "transformers", "LlamaConfig", "hypercloning.llama_cloning", "clone_llama"),
    ModelClonerSpec("gemma", "transformers", "GemmaConfig", "hypercloning.gemma_cloning", "clone_gemma"),
    ModelClonerSpec("gemma2", "transformers", "Gemma2Config", "hypercloning.gemma_cloning", "clone_gemma2"),
    ModelClonerSpec("opt", "transformers", "OPTConfig", "hypercloning.opt_cloning", "clone_opt"),
    ModelClonerSpec("pythia", "transformers", "GPTNeoXConfig", "hypercloning.pythia_cloning", "clone_pythia"),
    ModelClonerSpec("qwen3", "transformers", "Qwen3Config", "hypercloning.qwen3_cloning", "clone_qwen3"),
    ModelClonerSpec("mixtral", "transformers", "MixtralConfig", "hypercloning.models.mixtral", "clone_mixtral"),
    ModelClonerSpec("olmo", "hf_olmo.configuration_olmo", "OLMoConfig", "hypercloning.olmo_cloning", "clone_olmo"),
)


__all__ = ["BUILTIN_CLONERS", "ModelClonerSpec"]
