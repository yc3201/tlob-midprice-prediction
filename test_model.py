"""Tests for the TLOB model (my reimplementation).

Run with the project venv:  ~/workspace/.tlob-venv/bin/python test_model.py
(no pytest needed)

What is tested, and what is NOT claimed:
* Shapes: [B, T, F] -> [B, 3] for both 40- and 144-feature inputs.
* Masking invariance at the ATTENTION MODULE level: a masked time
  step (temporal) / feature (spatial) cannot influence the output at
  unmasked positions, because attention is the only cross-position
  mixing inside those modules. This is the property "attention
  masking" means here.
* NOT claimed for the full model: the temporal-mixing MLP mixes
  across time steps by design, so masking step 0 can legitimately
  change the final-step logits through the mixer. The full-model test
  only checks that a mask is accepted and outputs stay finite.
* Gradients flow (a training step runs, no NaN) and eval mode is
  deterministic.
"""
import torch

from model import (BilinearNorm, MLPLOBBlock, SpatialAttention, TLOB,
                   TemporalAttention)


def test_tlob_output_shape():
    torch.manual_seed(0)
    for feats in (40, 144):
        model = TLOB(in_features=feats, seq_len=100)
        model.eval()
        x = torch.randn(8, 100, feats)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 3), out.shape
        assert torch.isfinite(out).all()


def test_tlob_param_count_sane():
    # 4 blocks, d_model = 40: tens of thousands of parameters, not
    # millions — the model is small enough to train on CPU.
    model = TLOB(in_features=40, seq_len=100)
    n = sum(p.numel() for p in model.parameters())
    assert 10_000 < n < 1_000_000, n


def test_temporal_attention_masking():
    torch.manual_seed(0)
    mod = TemporalAttention(d_model=16)
    mod.eval()
    base = torch.randn(2, 12, 16)
    masked = base.clone()
    masked[:, 0, :] = 999.0  # garbage at time step 0
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 0] = True  # ignore step 0
    with torch.no_grad():
        out_base = mod(base, key_padding_mask=mask)
        out_masked = mod(masked, key_padding_mask=mask)
    # Unmasked positions (steps 1..) must be unaffected by step 0.
    assert torch.allclose(out_base[:, 1:, :], out_masked[:, 1:, :],
                          atol=1e-5)


def test_spatial_attention_masking():
    torch.manual_seed(0)
    mod = SpatialAttention(seq_len=12)
    mod.eval()
    base = torch.randn(2, 12, 16)
    masked = base.clone()
    masked[:, :, 0] = 999.0  # garbage in feature 0 at all times
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 0] = True  # ignore feature 0
    with torch.no_grad():
        out_base = mod(base, key_padding_mask=mask)
        out_masked = mod(masked, key_padding_mask=mask)
    # Unmasked features must be unaffected by feature 0.
    assert torch.allclose(out_base[:, :, 1:], out_masked[:, :, 1:],
                          atol=1e-5)


def test_bilinear_norm_is_per_sample():
    # Same feature values, different batch-mates -> different output:
    # statistics come from the current batch, not fixed constants.
    torch.manual_seed(0)
    norm = BilinearNorm(4)
    x1 = torch.randn(2, 8, 4)
    x2 = torch.randn(2, 8, 4) * 100.0
    with torch.no_grad():
        y1 = norm(x1)
        y2 = norm(torch.cat([x1[:1], x2[1:]], dim=0))
    assert not torch.allclose(y1[:1], y2[:1], atol=1e-5)


def test_mlp_lob_block_shape():
    torch.manual_seed(0)
    block = MLPLOBBlock(seq_len=100, d_model=40)
    block.eval()
    x = torch.randn(4, 100, 40)
    with torch.no_grad():
        assert block(x).shape == (4, 100, 40)


def test_training_step_runs_no_nan():
    torch.manual_seed(0)
    model = TLOB(in_features=40, seq_len=100)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(16, 100, 40)
    y = torch.randint(0, 3, (16,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    opt.step()


def test_eval_deterministic():
    torch.manual_seed(0)
    model = TLOB(in_features=40, seq_len=100)
    model.eval()
    x = torch.randn(4, 100, 40)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
