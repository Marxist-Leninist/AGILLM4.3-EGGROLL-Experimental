"""Trainer integration boundary for AGILLM 4.3 EGGROLL v25.

The adapter deliberately separates three operations that v24 mixed together:

1. cheap, fused counterfactual router search;
2. a proposed aggregate router update that does not mutate the model;
3. full-model search/guard measurements supplied by the trainer before an
   update may be applied.

A proposal is never permission to update production weights.  The caller must
run the fixed-token guard path and pass the result to :func:`decide_acceptance`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from fused_router_kernel import (
    SigmaController,
    antithetic_es_update,
    counterfactual_top1_fitness,
    evaluate_antithetic_fitness,
    route_flip_fraction,
    sample_population_factors,
)

DecisionLabel = Literal[
    "invalid",
    "no_signal",
    "counterfactual_only",
    "guard_rejected",
    "accepted_measurable",
]


@dataclass(frozen=True)
class RouterEventConfig:
    """Configuration for one fused, counterfactual router event."""

    pairs: int = 64
    rank: int = 1
    sigma: float = 0.01
    learning_rate: float = 0.01
    pair_chunk: int = 16
    max_update_rms_ratio: float = 0.002
    load_balance_coefficient: float = 0.0
    line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    target_flip_fraction: float = 0.05
    sigma_gain: float = 0.25
    min_sigma: float = 1e-5
    max_sigma: float = 0.25

    def validate(self) -> None:
        if self.pairs < 1:
            raise ValueError("pairs must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.pair_chunk < 1:
            raise ValueError("pair_chunk must be >= 1")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        if self.max_update_rms_ratio <= 0:
            raise ValueError("max_update_rms_ratio must be positive")
        if self.load_balance_coefficient < 0:
            raise ValueError("load_balance_coefficient must be non-negative")
        if not self.line_search_scales:
            raise ValueError("line_search_scales must not be empty")
        if any(
            (not math.isfinite(scale)) or scale <= 0 or scale > 1
            for scale in self.line_search_scales
        ):
            raise ValueError("line-search scales must be finite and in (0, 1]")
        if len(set(self.line_search_scales)) != len(self.line_search_scales):
            raise ValueError("line_search_scales must not contain duplicates")
        if not 0 < self.target_flip_fraction < 1:
            raise ValueError("target_flip_fraction must be in (0, 1)")
        if self.sigma_gain <= 0:
            raise ValueError("sigma_gain must be positive")
        if self.min_sigma <= 0 or self.max_sigma < self.min_sigma:
            raise ValueError("invalid sigma bounds")


@dataclass(frozen=True)
class RouterEventBatch:
    """Captured inputs for one selected MoE router.

    ``expert_costs`` contains a smaller-is-better counterfactual local cost
    for every token and every expert.  Its shape must equal the router logits
    shape produced by ``activations`` and ``router_weight``.
    """

    router_id: str
    committed_step: int
    activations: Tensor
    expert_costs: Tensor
    router_weight: Tensor
    router_bias: Tensor | None = None
    fixed_tokens_sha256: str = ""
    objective_sha256: str = ""

    def validate(self) -> None:
        if not self.router_id:
            raise ValueError("router_id must not be empty")
        if self.committed_step < 0:
            raise ValueError("committed_step must be non-negative")
        if self.activations.ndim < 2:
            raise ValueError("activations must have shape [..., in_features]")
        if self.router_weight.ndim != 2:
            raise ValueError(
                "router_weight must have shape [experts, in_features]"
            )
        experts, in_features = map(int, self.router_weight.shape)
        if self.activations.shape[-1] != in_features:
            raise ValueError("activation and router input widths differ")
        expected_cost_shape = (*self.activations.shape[:-1], experts)
        if tuple(self.expert_costs.shape) != expected_cost_shape:
            raise ValueError(
                "expert_costs must have shape "
                f"{expected_cost_shape}, got {tuple(self.expert_costs.shape)}"
            )
        if self.router_bias is not None and self.router_bias.shape != (experts,):
            raise ValueError("router_bias must have shape [experts]")
        tensors: Iterable[Tensor] = (
            self.activations,
            self.expert_costs,
            self.router_weight,
        )
        for tensor in tensors:
            if tensor.device != self.router_weight.device:
                raise ValueError("event tensors must share a device")
            if tensor.dtype != self.router_weight.dtype:
                raise ValueError("event tensors must share a dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError("event tensors must be finite")
        if self.router_bias is not None:
            if self.router_bias.device != self.router_weight.device:
                raise ValueError("router_bias must share the router device")
            if self.router_bias.dtype != self.router_weight.dtype:
                raise ValueError("router_bias must share the router dtype")
            if not torch.isfinite(self.router_bias).all():
                raise ValueError("router_bias must be finite")


@dataclass(frozen=True)
class RouterUpdateProposal:
    """Auditable, non-mutating result of one fused router search event."""

    router_id: str
    committed_step: int
    update: Tensor
    sigma: float
    next_sigma: float
    pairs: int
    rank: int
    pair_flip_fraction: float
    pair_signal_mean: float
    pair_signal_std: float
    baseline_local_fitness: float
    candidate_local_fitness: float
    local_fitness_improvement: float
    selected_line_search_scale: float
    raw_update_rms: float
    applied_update_rms: float
    weight_rms: float
    update_clip_scale: float
    candidate_route_change_fraction: float
    candidate_max_expert_fraction: float
    fixed_tokens_sha256: str
    objective_sha256: str

    def audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable record without the update tensor."""

        record = asdict(self)
        record.pop("update")
        record["schema"] = "agillm43.eggroll.v25.router_proposal.v1"
        return record


@dataclass(frozen=True)
class FullModelGuardMeasurements:
    """Paired full-model loss measurements for the aggregate proposal.

    All losses are smaller-is-better.  ``baseline_repeat_noise`` is measured
    from two disabled repeats on the same fixed tokens rather than guessed.
    """

    search_loss_before: float
    search_loss_after: float
    guard_losses_before: tuple[float, ...]
    guard_losses_after: tuple[float, ...]
    baseline_repeat_noise: float
    hashes_match: bool = True

    def validate(self) -> None:
        values = (
            self.search_loss_before,
            self.search_loss_after,
            self.baseline_repeat_noise,
            *self.guard_losses_before,
            *self.guard_losses_after,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("guard measurements must be finite")
        if self.baseline_repeat_noise < 0:
            raise ValueError("baseline_repeat_noise must be non-negative")
        if not self.guard_losses_before:
            raise ValueError("at least one independent guard crop is required")
        if len(self.guard_losses_before) != len(self.guard_losses_after):
            raise ValueError("guard before/after lengths differ")


@dataclass(frozen=True)
class AcceptanceDecision:
    label: DecisionLabel
    apply_update: bool
    reason: str
    search_loss_improvement: float
    required_search_improvement: float
    worst_guard_regression: float
    allowed_guard_regression: float

    def audit_record(self) -> dict[str, object]:
        record = asdict(self)
        record["schema"] = "agillm43.eggroll.v25.acceptance.v1"
        return record


def should_run_event(
    committed_step: int,
    *,
    every_steps: int = 32,
    offset: int = 0,
) -> bool:
    """Return whether a committed optimiser step should host an ES event."""

    if committed_step < 0:
        raise ValueError("committed_step must be non-negative")
    if every_steps < 1:
        raise ValueError("every_steps must be >= 1")
    if offset < 0 or offset >= every_steps:
        raise ValueError("offset must be in [0, every_steps)")
    return committed_step >= offset and (committed_step - offset) % every_steps == 0


def select_router_round_robin(event_index: int, router_count: int) -> int:
    if event_index < 0:
        raise ValueError("event_index must be non-negative")
    if router_count < 1:
        raise ValueError("router_count must be >= 1")
    return event_index % router_count


def _max_expert_fraction(logits: Tensor) -> float:
    route = logits.argmax(dim=-1).reshape(-1)
    experts = int(logits.shape[-1])
    counts = torch.bincount(route, minlength=experts).float()
    return float((counts.max() / max(route.numel(), 1)).item())


def _route_change_fraction(before_logits: Tensor, after_logits: Tensor) -> float:
    before = before_logits.argmax(dim=-1)
    after = after_logits.argmax(dim=-1)
    return float((before != after).float().mean().item())


@torch.no_grad()
def propose_router_update(
    batch: RouterEventBatch,
    config: RouterEventConfig,
    *,
    generator: torch.Generator | None = None,
) -> RouterUpdateProposal:
    """Search a fused population and return a non-mutating aggregate proposal.

    The function never changes ``batch.router_weight``.  A cheap local
    backtracking pass chooses the best aggregate scale before the caller pays
    for full-model search and guard forwards.
    """

    batch.validate()
    config.validate()
    original_weight = batch.router_weight.detach().clone()

    factors = sample_population_factors(
        pairs=config.pairs,
        in_features=int(batch.router_weight.shape[1]),
        out_features=int(batch.router_weight.shape[0]),
        rank=config.rank,
        device=batch.router_weight.device,
        dtype=batch.router_weight.dtype,
        generator=generator,
    )

    def fitness_fn(logits: Tensor) -> Tensor:
        return counterfactual_top1_fitness(
            logits,
            batch.expert_costs,
            load_balance_coefficient=config.load_balance_coefficient,
        )

    base_logits = F.linear(
        batch.activations,
        batch.router_weight,
        batch.router_bias,
    )
    baseline_fitness = float(fitness_fn(base_logits.unsqueeze(0))[0].item())

    positive_fitness, negative_fitness = evaluate_antithetic_fitness(
        batch.activations,
        batch.router_weight,
        factors,
        sigma=config.sigma,
        bias=batch.router_bias,
        fitness_fn=fitness_fn,
        pair_chunk=min(config.pair_chunk, config.pairs),
    )
    pair_flip = route_flip_fraction(
        batch.activations,
        batch.router_weight,
        factors,
        sigma=config.sigma,
        bias=batch.router_bias,
        pair_chunk=min(config.pair_chunk, config.pairs),
    )

    raw_update, stats = antithetic_es_update(
        batch.router_weight,
        factors,
        positive_fitness,
        negative_fitness,
        sigma=config.sigma,
        learning_rate=config.learning_rate,
        max_update_rms_ratio=config.max_update_rms_ratio,
    )

    best_scale = 0.0
    best_fitness = baseline_fitness
    best_logits = base_logits
    # Largest scale wins ties, making the decision deterministic while still
    # rejecting a proposal that does not improve the local objective.
    for scale in sorted(config.line_search_scales, reverse=True):
        candidate_logits = F.linear(
            batch.activations,
            batch.router_weight + raw_update * scale,
            batch.router_bias,
        )
        candidate_fitness = float(
            fitness_fn(candidate_logits.unsqueeze(0))[0].item()
        )
        if candidate_fitness > best_fitness:
            best_scale = float(scale)
            best_fitness = candidate_fitness
            best_logits = candidate_logits

    selected_update = raw_update * best_scale
    if not torch.equal(batch.router_weight, original_weight):
        raise RuntimeError("proposal search mutated the source router weight")

    sigma_controller = SigmaController(
        target_flip_fraction=config.target_flip_fraction,
        gain=config.sigma_gain,
        min_sigma=config.min_sigma,
        max_sigma=config.max_sigma,
    )
    next_sigma = sigma_controller.update(config.sigma, pair_flip)
    applied_rms = float(selected_update.float().square().mean().sqrt().item())
    pair_signal = positive_fitness.float() - negative_fitness.float()

    return RouterUpdateProposal(
        router_id=batch.router_id,
        committed_step=batch.committed_step,
        update=selected_update,
        sigma=float(config.sigma),
        next_sigma=float(next_sigma),
        pairs=config.pairs,
        rank=config.rank,
        pair_flip_fraction=float(pair_flip),
        pair_signal_mean=float(pair_signal.mean().item()),
        pair_signal_std=float(pair_signal.std(unbiased=False).item()),
        baseline_local_fitness=baseline_fitness,
        candidate_local_fitness=best_fitness,
        local_fitness_improvement=float(best_fitness - baseline_fitness),
        selected_line_search_scale=best_scale,
        raw_update_rms=float(stats.raw_update_rms),
        applied_update_rms=applied_rms,
        weight_rms=float(stats.weight_rms),
        update_clip_scale=float(stats.clip_scale),
        candidate_route_change_fraction=_route_change_fraction(
            base_logits, best_logits
        ),
        candidate_max_expert_fraction=_max_expert_fraction(best_logits),
        fixed_tokens_sha256=batch.fixed_tokens_sha256,
        objective_sha256=batch.objective_sha256,
    )


def decide_acceptance(
    proposal: RouterUpdateProposal,
    guards: FullModelGuardMeasurements,
    *,
    search_noise_multiple: float = 2.0,
    guard_noise_multiple: float = 1.0,
    max_expert_fraction: float = 0.90,
) -> AcceptanceDecision:
    """Apply the predeclared fixed-token promotion gate to one proposal."""

    if search_noise_multiple < 0 or guard_noise_multiple < 0:
        raise ValueError("noise multiples must be non-negative")
    if not 0 < max_expert_fraction <= 1:
        raise ValueError("max_expert_fraction must be in (0, 1]")

    try:
        guards.validate()
    except ValueError as error:
        return AcceptanceDecision(
            label="invalid",
            apply_update=False,
            reason=str(error),
            search_loss_improvement=float("nan"),
            required_search_improvement=float("nan"),
            worst_guard_regression=float("nan"),
            allowed_guard_regression=float("nan"),
        )

    search_improvement = guards.search_loss_before - guards.search_loss_after
    required_search_improvement = (
        search_noise_multiple * guards.baseline_repeat_noise
    )
    guard_regressions = tuple(
        after - before
        for before, after in zip(
            guards.guard_losses_before, guards.guard_losses_after
        )
    )
    worst_guard_regression = max(guard_regressions)
    allowed_guard_regression = guard_noise_multiple * guards.baseline_repeat_noise

    finite_proposal = all(
        math.isfinite(value)
        for value in (
            proposal.local_fitness_improvement,
            proposal.applied_update_rms,
            proposal.candidate_max_expert_fraction,
        )
    ) and bool(torch.isfinite(proposal.update).all())

    if not guards.hashes_match or not finite_proposal:
        reason = (
            "fixed-token/objective hashes do not match"
            if not guards.hashes_match
            else "proposal contains non-finite values"
        )
        return AcceptanceDecision(
            label="invalid",
            apply_update=False,
            reason=reason,
            search_loss_improvement=search_improvement,
            required_search_improvement=required_search_improvement,
            worst_guard_regression=worst_guard_regression,
            allowed_guard_regression=allowed_guard_regression,
        )

    if (
        proposal.local_fitness_improvement <= 0
        or proposal.applied_update_rms <= 0
        or proposal.selected_line_search_scale <= 0
    ):
        return AcceptanceDecision(
            label="no_signal",
            apply_update=False,
            reason="aggregate update did not improve counterfactual router fitness",
            search_loss_improvement=search_improvement,
            required_search_improvement=required_search_improvement,
            worst_guard_regression=worst_guard_regression,
            allowed_guard_regression=allowed_guard_regression,
        )

    if search_improvement <= required_search_improvement:
        return AcceptanceDecision(
            label="counterfactual_only",
            apply_update=False,
            reason=(
                "router objective improved, but full-model search loss did not "
                "beat measured baseline noise"
            ),
            search_loss_improvement=search_improvement,
            required_search_improvement=required_search_improvement,
            worst_guard_regression=worst_guard_regression,
            allowed_guard_regression=allowed_guard_regression,
        )

    if proposal.candidate_max_expert_fraction > max_expert_fraction:
        return AcceptanceDecision(
            label="guard_rejected",
            apply_update=False,
            reason="candidate route load exceeds the configured skew limit",
            search_loss_improvement=search_improvement,
            required_search_improvement=required_search_improvement,
            worst_guard_regression=worst_guard_regression,
            allowed_guard_regression=allowed_guard_regression,
        )

    if worst_guard_regression > allowed_guard_regression:
        return AcceptanceDecision(
            label="guard_rejected",
            apply_update=False,
            reason="an independent guard crop regressed beyond baseline noise",
            search_loss_improvement=search_improvement,
            required_search_improvement=required_search_improvement,
            worst_guard_regression=worst_guard_regression,
            allowed_guard_regression=allowed_guard_regression,
        )

    return AcceptanceDecision(
        label="accepted_measurable",
        apply_update=True,
        reason="counterfactual, search-loss and independent guard gates passed",
        search_loss_improvement=search_improvement,
        required_search_improvement=required_search_improvement,
        worst_guard_regression=worst_guard_regression,
        allowed_guard_regression=allowed_guard_regression,
    )


@torch.no_grad()
def apply_accepted_update_(
    router_weight: Tensor,
    proposal: RouterUpdateProposal,
    decision: AcceptanceDecision,
) -> None:
    """Atomically apply a proposal only after a passing decision."""

    if not decision.apply_update or decision.label != "accepted_measurable":
        raise ValueError("refusing to apply a router update that did not pass")
    if router_weight.shape != proposal.update.shape:
        raise ValueError("router weight and proposal update shapes differ")
    if router_weight.device != proposal.update.device:
        raise ValueError("router weight and proposal update devices differ")
    if router_weight.dtype != proposal.update.dtype:
        raise ValueError("router weight and proposal update dtypes differ")
    if not torch.isfinite(proposal.update).all():
        raise ValueError("proposal update contains non-finite values")
    router_weight.add_(proposal.update)
