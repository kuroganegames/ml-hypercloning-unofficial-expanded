from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from hc_validate import ValidationSpec
from hc_validate.inputs import InputMatrixSpec, make_input_matrix, tokenizer_fingerprint
from hc_validate.serialization import SaveReloadSpec, run_save_reload_check
from hc_validate.suite import P0ValidationSpec, run_p0_validation
from hypercloning.common import clone_matrix


class ToyLM(nn.Module):
    def __init__(self, vocab_size=19, hidden_size=4, intermediate_size=8, num_heads=2, base=None):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=1,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if base is None:
            torch.manual_seed(0)
            self.embed_tokens.weight.data.normal_()
            self.lm_head.weight.data.normal_()
        else:
            self.embed_tokens.weight.data = clone_matrix(self.embed_tokens.weight.shape, base.embed_tokens.weight.data, normalize=False)
            self.lm_head.weight.data = clone_matrix(self.lm_head.weight.shape, base.lm_head.weight.data, normalize=True)

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"config": dict(self.config.__dict__), "state_dict": self.state_dict()}, path / "toy_model.pt")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        data = torch.load(Path(path) / "toy_model.pt", map_location="cpu")
        cfg = data["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_heads=cfg["num_attention_heads"],
        )
        model.load_state_dict(data["state_dict"])
        return model

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        hidden = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            active_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            active_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=-100)
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states, attentions=None)


class ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 19
    padding_side = "right"
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}

    def __len__(self):
        return self.vocab_size

    def save_pretrained(self, path):
        Path(path, "toy_tokenizer.txt").write_text("toy", encoding="utf-8")

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None, add_special_tokens=True):
        rows = []
        for text in texts:
            ids = [self.bos_token_id] if add_special_tokens else []
            ids.extend([(ord(ch) % (self.vocab_size - 4)) + 4 for ch in text])
            if add_special_tokens:
                ids.append(self.eos_token_id)
            ids = ids[:max_length]
            rows.append(ids)
        width = max(len(row) for row in rows)
        if max_length is not None:
            width = min(width, max_length)
        out = []
        masks = []
        for row in rows:
            row = row[:width]
            pad = [self.pad_token_id] * (width - len(row))
            if self.padding_side == "left":
                padded = pad + row
                mask = [0] * len(pad) + [1] * len(row)
            else:
                padded = row + pad
                mask = [1] * len(row) + [0] * len(pad)
            out.append(padded)
            masks.append(mask)
        return {"input_ids": torch.tensor(out), "attention_mask": torch.tensor(masks)}


def make_pair():
    source = ToyLM()
    destination = ToyLM(hidden_size=8, intermediate_size=16, num_heads=4, base=source)
    return source, destination


def test_input_matrix_generates_random_and_tokenizer_bundles():
    source, _ = make_pair()
    tokenizer = ToyTokenizer()
    bundles = make_input_matrix(tokenizer, source.config, InputMatrixSpec(seq_lengths=(8,), num_random_batches_per_length=1))
    names = {bundle.name for bundle in bundles}
    assert any(name.startswith("random.seq8") for name in names)
    assert any(name.startswith("tokenizer.right.seq8") for name in names)
    assert tokenizer_fingerprint(tokenizer)["available"] is True


def test_save_reload_check_with_custom_loader():
    source, destination = make_pair()
    spec = ValidationSpec(seq_len=8, batch_size=2, num_batches=1, embedding_dim_multiplier=2, up_project_multiplier=2, num_heads_multiplier=2)
    bundles = make_input_matrix(None, source.config, InputMatrixSpec(seq_lengths=(8,), tokenizer_fixture_batches=False, num_random_batches_per_length=1))
    report = run_save_reload_check(
        source,
        destination,
        bundles[0].batches,
        spec,
        spec=SaveReloadSpec(allow_transformers_fallback=False),
        model_loader=lambda path: ToyLM.from_pretrained(path),
    )
    assert report.ok, report.failures


def test_p0_suite_without_save_reload_loader_disabled():
    source, destination = make_pair()
    tokenizer = ToyTokenizer()
    report = run_p0_validation(
        source,
        destination,
        tokenizer=tokenizer,
        spec=P0ValidationSpec(
            base=ValidationSpec(seq_len=8, batch_size=2, num_batches=1, embedding_dim_multiplier=2, up_project_multiplier=2, num_heads_multiplier=2),
            inputs=InputMatrixSpec(seq_lengths=(8,), num_random_batches_per_length=1),
            run_save_reload=False,
        ),
    )
    assert report.ok, report.failures
