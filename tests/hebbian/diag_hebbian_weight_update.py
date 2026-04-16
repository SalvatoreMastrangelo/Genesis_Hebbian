#!/usr/bin/env python3
"""
Diagnostic: verify that Hebbian rules mechanically update hebbian.W in act().

No Genesis / no checkpoint required.  Creates a synthetic frozen actor, runs
N steps of act() with random observations, and prints how much W diverges from
W_checkpoint at each step.

Run from the repo root:
    python tests/hebbian/diag_hebbian_weight_update.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from WP2.hebbian import HebbianLastLayer
from WP2.frozen_actor import IsolatedPopulationActor


# ── synthetic actor architecture matching the real one ───────────────────────

def _build_synthetic_actor(lstm_in: int, lstm_hidden: int, hidden: int, num_actions: int):
    """Tiny ActorCriticTanh-like object with just the pieces IsolatedPopulationActor uses."""

    class _FakeMemory(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(lstm_in, lstm_hidden, batch_first=False)

    class _FakeScale(nn.Module):
        max_throttle = 1.0
        max_servo = 1.0

        def _scale(self, a: torch.Tensor) -> torch.Tensor:
            return a  # identity scale

    class _FakeModel(_FakeScale, nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_a = _FakeMemory()
            self.recurrency = True
            # Sequential matching real architecture: LSTM-out → 64 → ELU → 64 → ELU → 7
            elu = nn.ELU()
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden, hidden),
                elu,  # rsl-rl reuses the SAME ELU instance
                nn.Linear(hidden, hidden),
                elu,
                nn.Linear(hidden, num_actions),
            )

    model = _FakeModel()
    return model


# ── main diagnostic ───────────────────────────────────────────────────────────

def run(
    num_steps: int = 200,
    N: int = 4,          # total envs (K*S)
    K: int = 1,
    lstm_in: int = 32,
    lstm_hidden: int = 64,
    hidden: int = 64,
    num_actions: int = 7,
    eta: float = 1e-3,   # 10× real config for visible effect in short run
    decay: float = 1e-3,
    device: str = "cpu",
):
    S = N // K
    torch.manual_seed(0)

    print("=" * 70)
    print("Hebbian weight-update diagnostic (no Genesis, no checkpoint)")
    print("=" * 70)
    print(f"  N={N} envs  K={K}  S={S}  steps={num_steps}")
    print(f"  eta={eta}  decay={decay}")
    print(f"  arch: LSTM({lstm_in}→{lstm_hidden}) → Linear({lstm_hidden}→{hidden}) "
          f"→ ELU → Linear({hidden}→{hidden}) → ELU → Linear({hidden}→{num_actions})")
    print()

    # 1. Build synthetic actor
    model = _build_synthetic_actor(lstm_in, lstm_hidden, hidden, num_actions)
    model.eval()

    # Extract the last layer (Linear(hidden → num_actions)) ─ same logic as load_frozen_actor
    last_layer = None
    for m in reversed(list(model.actor.modules())):
        if isinstance(m, nn.Linear):
            last_layer = m
            break
    assert last_layer is not None

    # Convert weight from Parameter to buffer (mirrors load_frozen_actor)
    weight_data = last_layer.weight.data.clone()
    del last_layer.weight
    last_layer.register_buffer("weight", weight_data)

    W_ckpt = last_layer.weight.clone()  # (num_actions, hidden)

    # 2. Build non-zero Hebbian rules (simple: A=1, B=C=D=0, lam=decay)
    A = torch.ones(N, num_actions, hidden, device=device) * 0.5
    B = torch.zeros(N, num_actions, hidden, device=device)
    C = torch.zeros(N, num_actions, hidden, device=device)
    D = torch.zeros(N, num_actions, hidden, device=device)
    lam = torch.full((N, num_actions, hidden), decay, device=device)
    rules = {"A": A, "B": B, "C": C, "D": D, "lam": lam}

    # 3. Build HebbianLastLayer
    hebb = HebbianLastLayer(
        W_checkpoint=W_ckpt,
        hebbian_rules=rules,
        eta=eta,
        w_max=1e6,
        use_oja_coefficient=False,
        device=device,
        num_envs=N,
    )

    # 4. Build IsolatedPopulationActor (wraps model + hebb)
    actor = IsolatedPopulationActor(
        model=model,
        hebbian=hebb,
        K=K,
        S=S,
        stochastic=False,
        obs_normalizer=nn.Identity(),
    )
    actor.reset_episode(device=device)

    # Snapshot the plain-frozen output (zero rules, same obs sequence)
    hebb_zero = HebbianLastLayer(
        W_checkpoint=W_ckpt,
        hebbian_rules={"A": torch.zeros_like(A), "B": torch.zeros_like(B),
                       "C": torch.zeros_like(C), "D": torch.zeros_like(D),
                       "lam": torch.zeros_like(lam)},
        eta=eta, w_max=1e6, use_oja_coefficient=False,
        device=device, num_envs=N,
    )
    actor_zero = IsolatedPopulationActor(
        model=model,
        hebbian=hebb_zero,
        K=K,
        S=S,
        stochastic=False,
        obs_normalizer=nn.Identity(),
    )
    actor_zero.reset_episode(device=device)

    # 5. Run steps
    print(f"  {'step':>5}  {'W_max_drift':>14}  {'action_delta':>14}  {'W_mean_drift':>14}")
    print("  " + "-" * 55)

    obs = torch.randn(N, lstm_in, device=device)

    w_drifts = []
    action_deltas = []

    for t in range(num_steps):
        # Both actors see the SAME obs for fair comparison
        obs_clone = obs.clone()

        with torch.no_grad():
            a_hebb = actor.act(obs_clone)
            a_zero = actor_zero.act(obs_clone)

        # Drift of hebbian.W from checkpoint
        w_max_drift = (hebb.W - W_ckpt.unsqueeze(0)).abs().max().item()
        w_mean_drift = (hebb.W - W_ckpt.unsqueeze(0)).abs().mean().item()
        # Difference in actions between hebbian and plain controller
        action_delta = (a_hebb - a_zero).abs().max().item()

        w_drifts.append(w_max_drift)
        action_deltas.append(action_delta)

        if t < 10 or t % 50 == 0 or t == num_steps - 1:
            print(f"  {t:>5}  {w_max_drift:>14.6f}  {action_delta:>14.6f}  {w_mean_drift:>14.6f}")

        # Advance obs with a bit of noise so the simulation isn't static
        obs = obs + 0.01 * torch.randn_like(obs)

    print()
    final_drift = w_drifts[-1]
    final_delta = action_deltas[-1]

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if final_drift > 1e-6:
        print(f"  ✓  W drifted from checkpoint (max |W - W_ckpt| = {final_drift:.4e})")
    else:
        print(f"  ✗  W did NOT drift from checkpoint after {num_steps} steps!")

    if final_delta > 1e-6:
        print(f"  ✓  Hebbian actions differ from plain controller (max Δ = {final_delta:.4e})")
    else:
        print(f"  ✗  Actions are IDENTICAL to plain controller (Δ = {final_delta:.4e})")

    # Verify zero-rules leaves W untouched
    zero_drift = (hebb_zero.W - W_ckpt.unsqueeze(0)).abs().max().item()
    if zero_drift < 1e-8:
        print(f"  ✓  Zero-rules actor: W unchanged from checkpoint (drift = {zero_drift:.2e})")
    else:
        print(f"  ✗  Zero-rules actor: unexpected W drift = {zero_drift:.2e}")

    print()
    passed = final_drift > 1e-6 and final_delta > 1e-6 and zero_drift < 1e-8
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    return passed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
