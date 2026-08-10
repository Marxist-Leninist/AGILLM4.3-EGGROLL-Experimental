from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from trainer_adapter import (
    FullModelGuardMeasurements,
    RouterEventBatch,
    RouterEventConfig,
    apply_accepted_update_,
    decide_acceptance,
    propose_router_update,
    select_router_round_robin,
    should_run_event,
)


def _synthetic_event() -> tuple[RouterEventBatch, RouterEventConfig]:
    activations = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.1],
            [2.0, -0.1],
            [-1.0, 0.0],
            [-2.0, 0.0],
            [-1.0, -0.1],
            [-2.0, 0.1],
        ]
    )
    # Deliberately routes every token to the wrong expert.
    weight = torch.tensor([[-0.01, 0.0], [0.0, 0.0]])
    preferred_expert = (activations[:, 0] < 0).long()
    expert_costs = torch.full((activations.shape[0], 2), 2.0)
    expert_costs[torch.arange(activations.shape[0]), preferred_expert] = 0.0
    batch = RouterEventBatch(
        router_id="block.0.router",
        committed_step=64,
        activations=activations,
        expert_costs=expert_costs,
        router_weight=weight,
        router_bias=torch.zeros(2),
        fixed_tokens_sha256="tokens",
        objective_sha256="objective",
    )
    config = RouterEventConfig(
        pairs=512,
        rank=1,
        sigma=0.1,
        learning_rate=0.05,
        pair_chunk=64,
        max_update_rms_ratio=2.0,
        line_search_scales=(1.0, 0.5, 0.25, 0.125),
    )
    return batch, config


def _good_guards() -> FullModelGuardMeasurements:
    return FullModelGuardMeasurements(
        search_loss_before=4.0,
        search_loss_after=3.7,
        guard_losses_before=(4.1, 3.9),
        guard_losses_after=(4.08, 3.91),
        baseline_repeat_noise=0.05,
    )


def test_synthetic_proposal_improves_hard_route_objective_without_mutation():
    batch, config = _synthetic_event()
    before = batch.router_weight.clone()
    generator = torch.Generator().manual_seed(123)
    proposal = propose_router_update(batch, config, generator=generator)

    torch.testing.assert_close(batch.router_weight, before)
    assert proposal.local_fitness_improvement == pytest.approx(2.0)
    assert proposal.candidate_route_change_fraction == pytest.approx(1.0)
    assert proposal.selected_line_search_scale in config.line_search_scales
    assert proposal.applied_update_rms > 0
    assert torch.isfinite(proposal.update).all()
    assert proposal.audit_record()["schema"].endswith("router_proposal.v1")


def test_acceptance_gate_requires_measurable_full_model_gain():
    batch, config = _synthetic_event()
    proposal = propose_router_update(
        batch, config, generator=torch.Generator().manual_seed(123)
    )
    decision = decide_acceptance(
        proposal,
        _good_guards(),
        max_expert_fraction=0.60,
    )
    assert decision.label == "accepted_measurable"
    assert decision.apply_update

    noisy_only = replace(_good_guards(), search_loss_after=3.95)
    decision = decide_acceptance(
        proposal,
        noisy_only,
        max_expert_fraction=0.60,
    )
    assert decision.label == "counterfactual_only"
    assert not decision.apply_update


def test_guard_regression_and_hash_mismatch_are_rejected():
    batch, config = _synthetic_event()
    proposal = propose_router_update(
        batch, config, generator=torch.Generator().manual_seed(123)
    )
    regressed = replace(_good_guards(), guard_losses_after=(4.3, 3.91))
    decision = decide_acceptance(
        proposal,
        regressed,
        max_expert_fraction=0.60,
    )
    assert decision.label == "guard_rejected"

    invalid = replace(_good_guards(), hashes_match=False)
    decision = decide_acceptance(
        proposal,
        invalid,
        max_expert_fraction=0.60,
    )
    assert decision.label == "invalid"


def test_no_signal_proposal_is_not_promoted():
    batch, config = _synthetic_event()
    proposal = propose_router_update(
        batch, config, generator=torch.Generator().manual_seed(123)
    )
    proposal = replace(
        proposal,
        update=torch.zeros_like(proposal.update),
        selected_line_search_scale=0.0,
        applied_update_rms=0.0,
        local_fitness_improvement=0.0,
    )
    decision = decide_acceptance(
        proposal,
        _good_guards(),
        max_expert_fraction=0.60,
    )
    assert decision.label == "no_signal"


def test_update_application_is_atomic_and_refuses_rejections():
    batch, config = _synthetic_event()
    proposal = propose_router_update(
        batch, config, generator=torch.Generator().manual_seed(123)
    )
    rejected = decide_acceptance(
        proposal,
        replace(_good_guards(), search_loss_after=3.95),
        max_expert_fraction=0.60,
    )
    before = batch.router_weight.clone()
    with pytest.raises(ValueError):
        apply_accepted_update_(batch.router_weight, proposal, rejected)
    torch.testing.assert_close(batch.router_weight, before)

    accepted = decide_acceptance(
        proposal,
        _good_guards(),
        max_expert_fraction=0.60,
    )
    apply_accepted_update_(batch.router_weight, proposal, accepted)
    torch.testing.assert_close(batch.router_weight, before + proposal.update)


def test_schedule_and_round_robin_are_deterministic():
    assert should_run_event(0)
    assert should_run_event(32)
    assert not should_run_event(31)
    assert should_run_event(7, every_steps=16, offset=7)
    assert select_router_round_robin(0, 14) == 0
    assert select_router_round_robin(14, 14) == 0
    assert select_router_round_robin(17, 14) == 3
