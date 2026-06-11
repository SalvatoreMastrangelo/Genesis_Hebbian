#!/usr/bin/env python3
"""
Tests for the feed-forward actor variant (ActorCriticTanhFF) and its
integration with the WP2 Hebbian pipeline (IsolatedPopulationActor).

The FF actor is a no-LSTM twin of ActorCriticTanh: same tanh squashing,
same throttle/servo scaling, same tanh-corrected log-prob — but the actor
MLP consumes raw (normalized) observations directly.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from winged_drone_train.rl.A2C_modified import ActorCriticTanh, ActorCriticTanhFF
from WP2.frozen_actor import (
    IsolatedPopulationActor,
    is_recurrent_state_dict,
    last_actor_linear_key,
)
from WP2.hebbian import HebbianLastLayer

NUM_OBS = 10
NUM_ACTIONS = 7
HIDDEN_DIMS = [16, 8]
MAX_SERVO = 0.5
MAX_THROTTLE = 1.0


def _make_ff_model(seed: int = 0) -> ActorCriticTanhFF:
    torch.manual_seed(seed)
    return ActorCriticTanhFF(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=HIDDEN_DIMS,
        activation="elu",
        init_noise_std=0.3,
        max_servo=MAX_SERVO,
        max_throttle=MAX_THROTTLE,
    )


def _make_recurrent_model(seed: int = 0) -> ActorCriticTanh:
    torch.manual_seed(seed)
    return ActorCriticTanh(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=HIDDEN_DIMS,
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=12,
        rnn_num_layers=1,
        init_noise_std=0.3,
        max_servo=MAX_SERVO,
        max_throttle=MAX_THROTTLE,
    )


def _zero_rules(num_actions: int, hidden_dim: int) -> dict:
    shape = (num_actions, hidden_dim)
    return {
        "A": torch.zeros(shape),
        "B": torch.zeros(shape),
        "C": torch.zeros(shape),
        "D": torch.zeros(shape),
        "lam": torch.zeros(shape),
    }


def _wrap_isolated(model, K: int, S: int, stochastic: bool) -> IsolatedPopulationActor:
    last_linear = [m for m in model.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    hebbian = HebbianLastLayer(
        W_checkpoint=last_linear.weight.data,
        hebbian_rules=_zero_rules(*last_linear.weight.shape),
        eta=0.0,
        w_max=3.0,
        use_oja_coefficient=False,
        device="cpu",
        num_envs=K * S,
    )
    return IsolatedPopulationActor(model, hebbian, K=K, S=S, stochastic=stochastic)


# ---------------------------------------------------------------------------
# ActorCriticTanhFF — standalone behaviour
# ---------------------------------------------------------------------------

def test_ff_actor_has_no_memory_and_is_not_recurrent():
    model = _make_ff_model()
    assert not hasattr(model, "memory_a")
    assert not hasattr(model, "memory_c")
    assert model.recurrency is False
    assert ActorCriticTanhFF.is_recurrent is False
    assert ActorCriticTanh.is_recurrent is True


def test_ff_actor_ignores_rnn_kwargs():
    torch.manual_seed(0)
    model = ActorCriticTanhFF(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=HIDDEN_DIMS,
        activation="elu",
        init_noise_std=0.3,
        rnn_type="lstm",
        rnn_hidden_size=64,
        rnn_num_layers=1,
        critic_rnn_hidden_size=64,
    )
    assert not hasattr(model, "memory_a")


def test_ff_actor_action_shapes_and_bounds():
    model = _make_ff_model()
    obs = torch.randn(32, NUM_OBS)
    act = model.act(obs)
    assert act.shape == (32, NUM_ACTIONS)
    thr, srv = act[:, :1], act[:, 1:]
    assert (thr >= 0).all() and (thr <= MAX_THROTTLE).all()
    assert (srv >= -MAX_SERVO).all() and (srv <= MAX_SERVO).all()


def test_ff_actor_batch_size_one_keeps_batch_dim():
    model = _make_ff_model()
    obs = torch.randn(1, NUM_OBS)
    act = model.act(obs, deterministic=True)
    assert act.shape == (1, NUM_ACTIONS)


def test_ff_actor_deterministic_is_repeatable():
    model = _make_ff_model()
    obs = torch.randn(8, NUM_OBS)
    a1 = model.act(obs, deterministic=True)
    a2 = model.act(obs, deterministic=True)
    assert torch.equal(a1, a2)


def test_ff_actor_log_prob_consistency():
    model = _make_ff_model()
    obs = torch.randn(16, NUM_OBS)
    torch.manual_seed(123)
    act = model.act(obs)
    logp_from_act = model.get_actions_log_prob(act)
    logp_cached = model._last_logp.squeeze(-1)
    assert logp_from_act.shape == (16,)
    assert torch.isfinite(logp_from_act).all()
    assert torch.allclose(logp_from_act, logp_cached, atol=1e-4)


def test_ff_actor_evaluate_critic():
    model = _make_ff_model()
    obs = torch.randn(16, NUM_OBS)
    value = model.evaluate(obs)
    assert value.shape == (16, 1)


# ---------------------------------------------------------------------------
# state-dict sniffing helpers (WP2.frozen_actor)
# ---------------------------------------------------------------------------

def test_is_recurrent_state_dict():
    rec_sd = _make_recurrent_model().state_dict()
    ff_sd = _make_ff_model().state_dict()
    assert is_recurrent_state_dict(rec_sd) is True
    assert is_recurrent_state_dict(ff_sd) is False


def test_last_actor_linear_key_recurrent_arch():
    sd = _make_recurrent_model().state_dict()
    key = last_actor_linear_key(sd)
    assert key == "actor.4.weight"
    assert sd[key].shape == (NUM_ACTIONS, HIDDEN_DIMS[-1])


def test_last_actor_linear_key_ff_arch():
    # FF actor: Linear,ELU,Linear,ELU,Linear → last layer is actor.4 here too;
    # use a deeper FF net so the index actually differs from the hardcoded 4.
    torch.manual_seed(0)
    model = ActorCriticTanhFF(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[32, 16, 8],
        critic_hidden_dims=[32, 16, 8],
        activation="elu",
        init_noise_std=0.3,
    )
    sd = model.state_dict()
    key = last_actor_linear_key(sd)
    assert key == "actor.6.weight"
    assert sd[key].shape == (NUM_ACTIONS, 8)


def test_last_actor_linear_key_missing():
    assert last_actor_linear_key({"critic.0.weight": torch.zeros(1)}) is None


# ---------------------------------------------------------------------------
# IsolatedPopulationActor with a feed-forward backbone
# ---------------------------------------------------------------------------

def test_isolated_actor_ff_matches_model_act_with_zero_rules():
    """Zero rules + deterministic: the wrapper's manual forward must equal
    the model's own deterministic act(). Guards against layer-iteration bugs
    (e.g. the deduplicated-ELU bug) in the FF path."""
    model = _make_ff_model()
    K, S = 2, 3
    actor = _wrap_isolated(model, K, S, stochastic=False)
    actor.reset_episode(device="cpu")

    obs = torch.randn(K * S, NUM_OBS)
    with torch.no_grad():
        expected = model.act(obs, deterministic=True)
    got = actor.act(obs)
    assert torch.allclose(got, expected, atol=1e-6), (
        f"max abs diff {(got - expected).abs().max().item()}"
    )


def test_isolated_actor_ff_act_before_reset_raises():
    model = _make_ff_model()
    actor = _wrap_isolated(model, 1, 2, stochastic=False)
    obs = torch.randn(2, NUM_OBS)
    try:
        actor.act(obs)
        raise AssertionError("act() before reset_episode should raise")
    except RuntimeError:
        pass


def test_isolated_actor_ff_reset_individual():
    model = _make_ff_model()
    K, S = 2, 2
    actor = _wrap_isolated(model, K, S, stochastic=False)
    actor.reset_episode(device="cpu")
    # Dirty the weights of all slots, then reset only individual 0.
    actor.hebbian.W += 1.0
    actor.reset_individual(0, device="cpu")
    W_ckpt = actor.hebbian.W_checkpoint
    assert torch.allclose(actor.hebbian.W[0:S], W_ckpt.expand(S, -1, -1))
    assert not torch.allclose(actor.hebbian.W[S:], W_ckpt.expand(S, -1, -1))


def test_isolated_actor_ff_stochastic_runs():
    model = _make_ff_model()
    actor = _wrap_isolated(model, 2, 2, stochastic=True)
    actor.reset_episode(device="cpu")
    act = actor.act(torch.randn(4, NUM_OBS))
    assert act.shape == (4, NUM_ACTIONS)
    assert torch.isfinite(act).all()


# ---------------------------------------------------------------------------
# IsolatedPopulationActor with the recurrent backbone (regression guard)
# ---------------------------------------------------------------------------

def test_isolated_actor_recurrent_still_works():
    model = _make_recurrent_model()
    K, S = 2, 2
    actor = _wrap_isolated(model, K, S, stochastic=False)
    actor.reset_episode(device="cpu")
    obs = torch.randn(K * S, NUM_OBS)
    a1 = actor.act(obs)
    assert a1.shape == (K * S, NUM_ACTIONS)
    # LSTM state must have been advanced (non-zero after one step).
    assert actor._h is not None and actor._h.abs().sum() > 0
    # Same obs again should generally differ (state advanced).
    a2 = actor.act(obs)
    assert not torch.equal(a1, a2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
