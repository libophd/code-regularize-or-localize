from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("transformers")

from jepa_lm.systems import MultiLayerSIGRegBridge, PerLayerKVPredictorBridge
from jepa_lm.systems.bridge_probe import (
    FrozenTrunkProbe,
    gate_attention_score_frobenius_error,
    run_a2_probe,
    run_a3_probe,
)
from jepa_lm.systems.kv_bridge import PostHocKVDecoderBridge
from jepa_lm.trunks.llama import HFLlamaTrunk


def _tiny_trunk(vocab_size: int = 100) -> HFLlamaTrunk:
    return HFLlamaTrunk(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=vocab_size,
        max_position_embeddings=64,
    )


def _batches(n: int, *, b: int = 2, t: int = 8, vocab: int = 100, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, vocab, (b, t), generator=g)} for _ in range(n)]


def test_a2_probe_end_to_end() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    result = run_a2_probe(
        trunk,
        n_steps=4,
        train_loader=_batches(4, seed=1),
        eval_loader=_batches(2, seed=2),
    )
    assert result.probe == "a2_per_layer_kv_predictors"
    assert result.n_steps_trained == 4
    assert result.gate_metric == "mean_per_layer_residual_norm"
    assert 0.0 <= result.gate_value < 5.0 and result.gate_threshold == 0.5
    assert isinstance(result.passed, bool)
    assert len(result.per_layer["k_residual"]) == 2
    assert len(result.per_layer["v_residual"]) == 2
    json.dumps(result.to_dict())


def test_frozen_trunk_weights_do_not_move() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    bridge = PerLayerKVPredictorBridge(n_layers=2, d_kv=16)
    before = {n: p.detach().clone() for n, p in trunk.named_parameters()}
    pred_before = bridge.k_predictors[0].weight.detach().clone()

    probe = FrozenTrunkProbe(trunk, bridge, lr=1e-2)
    probe.train(_batches(5, seed=3), n_steps=5)

    for n, p in trunk.named_parameters():
        assert torch.equal(p, before[n]), f"trunk param {n} moved — trunk not frozen"
        assert p.requires_grad is False
    assert not torch.equal(bridge.k_predictors[0].weight, pred_before), (
        "predictors must train — bridge weights should have moved"
    )


def test_a2_loss_decreases_on_repeated_batch() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    bridge = PerLayerKVPredictorBridge(n_layers=2, d_kv=16)
    probe = FrozenTrunkProbe(trunk, bridge, lr=1e-2)
    batch = _batches(1, seed=4)[0]
    first = probe.train_step(batch)
    for _ in range(20):
        last = probe.train_step(batch)
    assert last < first, f"loss should fall on a repeated batch; {first} -> {last}"


def test_frozen_probe_rejects_parameterless_bridge() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    sigreg_bridge = MultiLayerSIGRegBridge(n_layers=2, d_kv=16)
    with pytest.raises(ValueError):
        FrozenTrunkProbe(trunk, sigreg_bridge)


def test_a3_probe_end_to_end() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    result = run_a3_probe(
        trunk,
        n_steps=4,
        train_loader=_batches(4, seed=1),
        eval_loader=_batches(2, seed=2),
    )
    assert result.probe == "a3_post_hoc_kv_decoders"
    assert result.n_steps_trained == 4
    assert result.gate_metric == "mean_attention_score_frobenius_error"
    assert result.gate_threshold == 0.1
    assert 0.0 <= result.gate_value < 2.0
    assert isinstance(result.passed, bool)
    assert len(result.per_layer["attn_frobenius"]) == 2
    json.dumps(result.to_dict())


def test_a3_frozen_trunk_weights_do_not_move() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    bridge = PostHocKVDecoderBridge(n_layers=2, d_kv=16, d_proj=16)
    before = {n: p.detach().clone() for n, p in trunk.named_parameters()}
    dec_before = bridge.k_decoders[0].weight.detach().clone()

    probe = FrozenTrunkProbe(trunk, bridge, lr=1e-2, needs_z=True)
    probe.train(_batches(5, seed=3), n_steps=5)

    for n, p in trunk.named_parameters():
        assert torch.equal(p, before[n]), f"trunk param {n} moved — trunk not frozen"
        assert p.requires_grad is False
    assert not torch.equal(bridge.k_decoders[0].weight, dec_before), (
        "decoders must train — bridge weights should have moved"
    )


def test_a3_gate_improves_with_training() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    bridge = PostHocKVDecoderBridge(n_layers=2, d_kv=16, d_proj=16)
    eval_batch = _batches(1, seed=7)
    initial, per_layer = gate_attention_score_frobenius_error(trunk, bridge, eval_batch)
    assert len(per_layer["attn_frobenius"]) == 2

    probe = FrozenTrunkProbe(trunk, bridge, lr=1e-2, needs_z=True)
    probe.train(_batches(1, seed=7) * 60, n_steps=60)
    trained, _ = gate_attention_score_frobenius_error(trunk, bridge, eval_batch)
    assert trained < initial, f"A3' gate should improve with training; {initial} -> {trained}"


def test_a3_gate_rejects_nonstandard_rope() -> None:
    trunk = _tiny_trunk()
    trunk.model.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    bridge = PostHocKVDecoderBridge(n_layers=2, d_kv=16, d_proj=16)
    with pytest.raises(NotImplementedError):
        gate_attention_score_frobenius_error(trunk, bridge, _batches(1, seed=0))


def test_a2_probe_with_mlp_bridge_end_to_end() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    result = run_a2_probe(
        trunk,
        n_steps=4,
        train_loader=_batches(4, seed=1),
        eval_loader=_batches(2, seed=2),
        bridge_name="per_layer_kv_mlp_predictors",
        bridge_kwargs={"hidden_multiplier": 2},
    )
    assert result.probe == "a2_per_layer_kv_mlp_predictors"
    assert result.gate_metric == "mean_per_layer_residual_norm"
    assert 0.0 <= result.gate_value < 5.0
    assert len(result.per_layer["k_residual"]) == 2
    json.dumps(result.to_dict())


def test_a2_probe_rejects_non_predictor_bridge() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    with pytest.raises(ValueError, match="PerLayerKVPredictorBridge"):
        run_a2_probe(
            trunk,
            n_steps=1,
            train_loader=_batches(1),
            eval_loader=_batches(1),
            bridge_name="multi_layer_sigreg",
        )
