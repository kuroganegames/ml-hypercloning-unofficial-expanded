'''Architecture-independent HyperCloning validation helpers.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from hypercloning.common import clone_matrix, clone_vector
from hypercloning.trace import CloneTrace
from hc_validate.model_loading import maybe_move_model, move_batch_to_model


@dataclass
class ValidationSpec:
    seq_len: int = 128
    batch_size: int = 2
    num_batches: int = 8
    seed: int = 0
    embedding_dim_multiplier: int = 2
    up_project_multiplier: int = 2
    num_heads_multiplier: Optional[int] = None
    device: str = 'cpu'
    atol: float = 1e-4
    rtol: float = 1e-4
    compare_hidden_states: bool = True


@dataclass
class TensorDiff:
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: Optional[float]


@dataclass
class ValidationReport:
    ok: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.failures.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> TensorDiff:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    d = a - b
    denom = torch.linalg.vector_norm(a).clamp_min(1e-12)
    cosine = None if a.numel() == 0 else float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
    return TensorDiff(float(d.abs().max().item()), float(d.abs().mean().item()), float((torch.linalg.vector_norm(d) / denom).item()), cosine)


def make_random_lm_batches(vocab_size: int, spec: ValidationSpec, pad_token_id: int = 0) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator(device='cpu')
    gen.manual_seed(spec.seed)
    batches = []
    for i in range(spec.num_batches):
        ids = torch.randint(1, vocab_size, (spec.batch_size, spec.seq_len), generator=gen, dtype=torch.long)
        mask = torch.ones_like(ids)
        pad_len = max(1, spec.seq_len // 4)
        if i % 3 == 1:
            ids[:, -pad_len:] = pad_token_id
            mask[:, -pad_len:] = 0
        elif i % 3 == 2:
            ids[:, :pad_len] = pad_token_id
            mask[:, :pad_len] = 0
        batches.append({'input_ids': ids, 'attention_mask': mask, 'labels': ids.clone()})
    return batches


def _filter_kwargs(model, batch: dict[str, Any]) -> dict[str, Any]:
    import inspect
    accepted = set(inspect.signature(model.forward).parameters)
    return {k: v for k, v in batch.items() if k in accepted}


def _forward_flags(model, *, hidden: bool) -> dict[str, Any]:
    import inspect
    accepted = set(inspect.signature(model.forward).parameters)
    flags = {'use_cache': False, 'return_dict': True, 'output_hidden_states': hidden, 'output_attentions': False}
    return {k: v for k, v in flags.items() if k in accepted}


def compare_repeated_hidden(src: torch.Tensor, dst: torch.Tensor, repeat: int) -> list[TensorDiff]:
    if dst.shape[-1] != src.shape[-1] * repeat:
        raise ValueError(f'hidden size mismatch: source={tuple(src.shape)} destination={tuple(dst.shape)} repeat={repeat}')
    chunks = dst.reshape(*dst.shape[:-1], repeat, src.shape[-1])
    return [tensor_diff(src, chunks[..., i, :]) for i in range(repeat)]


@torch.inference_mode()
def validate_forward(source_model, destination_model, batches: list[dict[str, torch.Tensor]], spec: ValidationSpec) -> ValidationReport:
    report = ValidationReport()
    maybe_move_model(source_model, spec.device)
    maybe_move_model(destination_model, spec.device)
    logits_diffs = []
    loss_diffs = []
    hidden_diffs = []
    for batch_i, batch in enumerate(batches):
        src_batch = move_batch_to_model(batch, source_model, fallback=spec.device)
        dst_batch = move_batch_to_model(batch, destination_model, fallback=spec.device)
        src_out = source_model(**(_filter_kwargs(source_model, src_batch) | _forward_flags(source_model, hidden=spec.compare_hidden_states)))
        dst_out = destination_model(**(_filter_kwargs(destination_model, dst_batch) | _forward_flags(destination_model, hidden=spec.compare_hidden_states)))
        if not hasattr(src_out, 'logits') or not hasattr(dst_out, 'logits'):
            report.fail('ModelOutput missing logits')
            continue
        diff = tensor_diff(src_out.logits, dst_out.logits)
        logits_diffs.append(diff)
        if diff.max_abs > spec.atol and diff.rel_l2 > spec.rtol:
            report.fail(f'batch={batch_i}: logits mismatch max_abs={diff.max_abs:.3g} rel_l2={diff.rel_l2:.3g}')
        if getattr(src_out, 'loss', None) is not None and getattr(dst_out, 'loss', None) is not None:
            loss_diffs.append(float((src_out.loss.detach().float().cpu() - dst_out.loss.detach().float().cpu()).abs().item()))
        src_hs = getattr(src_out, 'hidden_states', None)
        dst_hs = getattr(dst_out, 'hidden_states', None)
        if spec.compare_hidden_states and src_hs is not None and dst_hs is not None:
            for layer_i, (src_h, dst_h) in enumerate(zip(src_hs, dst_hs)):
                try:
                    hidden_diffs.extend(compare_repeated_hidden(src_h, dst_h, spec.embedding_dim_multiplier))
                except ValueError as exc:
                    report.fail(f'batch={batch_i} layer={layer_i}: {exc}')
    report.metrics['logits_max_abs'] = max((d.max_abs for d in logits_diffs), default=None)
    report.metrics['logits_rel_l2'] = max((d.rel_l2 for d in logits_diffs), default=None)
    report.metrics['loss_abs_diff'] = max(loss_diffs, default=None)
    report.metrics['hidden_max_abs'] = max((d.max_abs for d in hidden_diffs), default=None)
    return report


def audit_config(src_config, dst_config, spec: ValidationSpec) -> ValidationReport:
    aliases = {
        'hidden_size': ('hidden_size', 'd_model', 'n_embd', 'model_dim'),
        'intermediate_size': ('intermediate_size', 'ffn_dim', 'mlp_hidden_size', 'n_inner'),
        'num_hidden_layers': ('num_hidden_layers', 'num_layers', 'n_layer', 'n_layers'),
        'num_attention_heads': ('num_attention_heads', 'n_head', 'n_heads'),
        'vocab_size': ('vocab_size', 'n_vocab'),
    }
    checks = [('vocab_size', 1), ('num_hidden_layers', 1), ('hidden_size', spec.embedding_dim_multiplier), ('intermediate_size', spec.up_project_multiplier)]
    if spec.num_heads_multiplier is not None:
        checks.append(('num_attention_heads', spec.num_heads_multiplier))
    report = ValidationReport()
    for key, multiplier in checks:
        src_name = next((n for n in aliases[key] if hasattr(src_config, n)), None)
        dst_name = next((n for n in aliases[key] if hasattr(dst_config, n)), None)
        if src_name is None or dst_name is None:
            report.warnings.append(f'could not resolve config field {key}')
            continue
        src_value = getattr(src_config, src_name)
        dst_value = getattr(dst_config, dst_name)
        expected = src_value * multiplier
        report.metrics[f'config.{key}'] = {'source': src_value, 'destination': dst_value, 'expected': expected}
        if dst_value != expected:
            report.fail(f'{dst_name} expected {expected}, got {dst_value}')
    return report


def audit_state_with_trace(source_model, destination_model, trace: CloneTrace, *, atol: float = 1e-5, rtol: float = 1e-5) -> ValidationReport:
    report = ValidationReport(metrics={'checked_ops': 0, 'failed_ops': 0})
    src_sd = source_model.state_dict()
    dst_sd = destination_model.state_dict()
    for op in trace.ops:
        if op.src_param not in src_sd or op.dst_param not in dst_sd:
            report.fail(f'missing traced tensor: {op.src_param} -> {op.dst_param}')
            continue
        src = src_sd[op.src_param].detach().cpu()
        dst = dst_sd[op.dst_param].detach().cpu()
        if src.ndim == 2 and dst.ndim == 2 and op.kind in {'matrix', 'linear', 'embedding', 'lm_head_scaled'}:
            expected = clone_matrix(dst.shape, src, snr_db=None, normalize=op.normalize_input_repeat)
        elif src.ndim == 1 and dst.ndim == 1:
            expected = clone_vector(dst.shape, src)
        else:
            report.warnings.append(f'skipped custom op {op.kind}: {op.dst_param}')
            continue
        if op.scale is not None:
            expected = expected * op.scale
        diff = tensor_diff(expected, dst)
        report.metrics['checked_ops'] += 1
        if diff.max_abs > atol and diff.rel_l2 > rtol:
            report.metrics['failed_ops'] += 1
            report.fail(f'{op.dst_param} differs max_abs={diff.max_abs:.3g} rel_l2={diff.rel_l2:.3g}')
    return report


def run_train_smoke(model, batches: list[dict[str, torch.Tensor]], *, device: str = 'cpu', lr: float = 1e-5, steps: int = 5) -> ValidationReport:
    report = ValidationReport(metrics={'losses': [], 'grad_norms': []})
    model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        batch = {k: v.to(device) for k, v in batches[step % len(batches)].items()}
        opt.zero_grad(set_to_none=True)
        out = model(**(_filter_kwargs(model, batch) | _forward_flags(model, hidden=False)))
        loss = getattr(out, 'loss', None)
        if loss is None or not torch.isfinite(loss):
            report.fail(f'invalid loss at step {step}: {loss}')
            break
        loss.backward()
        total = torch.zeros((), device=device)
        for p in model.parameters():
            if p.grad is not None:
                total = total + p.grad.detach().float().pow(2).sum()
        grad_norm = float(total.sqrt().item())
        if not torch.isfinite(torch.tensor(grad_norm)):
            report.fail(f'invalid grad norm at step {step}')
            break
        opt.step()
        report.metrics['losses'].append(float(loss.item()))
        report.metrics['grad_norms'].append(grad_norm)
    return report


__all__ = [
    'ValidationSpec',
    'TensorDiff',
    'ValidationReport',
    'tensor_diff',
    'make_random_lm_batches',
    'compare_repeated_hidden',
    'validate_forward',
    'audit_config',
    'audit_state_with_trace',
    'run_train_smoke',
]
