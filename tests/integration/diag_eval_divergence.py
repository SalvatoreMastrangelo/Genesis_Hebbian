#!/usr/bin/env python3
"""
Diagnostic: locate where WP1 inference and IsolatedPopulationActor.act diverge.

For a single env.reset (same seed), this script computes the step-0 action
through BOTH pipelines and compares:
  (a) raw observations
  (b) normalized observations
  (c) LSTM output
  (d) MLP output before last-layer
  (e) last-layer weight + bias
  (f) pre-tanh action
  (g) scaled action

Run with the same args as test_eval_comparison.py.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.environ["GS_PARA_LEVEL"] = "3"


def _configure_cache_root() -> None:
    cache = (Path("logs") / ".cache" / "gstaichi").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[key] = str(cache)
    mpl = cache / "mpl"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl)


def _build_env(wp1_config_path, urdf_path, num_envs, device):
    from WP1.config import RunConfig
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise

    wp1_cfg = RunConfig.from_yaml(wp1_config_path)
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = wp1_cfg.to_legacy_cfgs()

    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        unique_forests_eval=True,
        growing_forest=True,
        x_upper=600,
        forest_x_limit=600,
        tree_radius=0.75,
        base_init_pos=[-50.0, 0.0, 10.0],
    ))
    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    env = WingedDroneEnv(
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        urdf_file=urdf_path, show_viewer=False, eval=True, device=device,
    )
    configure_solver_noise(env, env_cfg)
    return env, wp1_cfg


def _load_wp1_policy(checkpoint_path, wp1_config_path, env, device):
    import builtins
    from rsl_rl.runners import OnPolicyRunner
    from WP1.config import RunConfig
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh
    builtins.ActorCriticTanh = ActorCriticTanh

    wp1_cfg = RunConfig.from_yaml(wp1_config_path)
    train_cfg = copy.deepcopy(wp1_cfg.to_train_cfg())
    log_dir = str(Path(checkpoint_path).parent)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(checkpoint_path)
    return runner


def _build_hebbian_actor(checkpoint_path, wp1_config_path, num_envs, device):
    from WP2.frozen_actor import load_frozen_actor, build_isolated_population_actor

    _, _, num_actions, hidden_dim = load_frozen_actor(checkpoint_path, wp1_config_path, device=device)
    z = torch.zeros(num_actions, hidden_dim, device=device)
    zero_rules = {"A": z, "B": z, "C": z, "D": z, "lam": z}

    class _HebbCfg:
        eta = 0.0
        w_max = 1e9
        use_oja_coefficient = False
        decay = 0.0
    class _Cfg:
        hebbian = _HebbCfg()

    return build_isolated_population_actor(
        checkpoint_path=checkpoint_path,
        wp1_cfg_path=wp1_config_path,
        hebbian_rules_per_individual=[zero_rules],
        cfg=_Cfg(), K=1, S=num_envs, device=device, stochastic=False,
    )


def compare(args):
    import genesis as gs
    _configure_cache_root()
    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    print(f"[diag] Building env (num_envs={args.num_envs})")
    env, _ = _build_env(args.wp1_config, args.urdf, args.num_envs, args.device)

    print(f"[diag] Loading WP1 policy")
    runner = _load_wp1_policy(args.checkpoint, args.wp1_config, env, args.device)

    print(f"[diag] Building Hebbian actor")
    hebb_actor = _build_hebbian_actor(args.checkpoint, args.wp1_config, args.num_envs, args.device)

    # Reset both sides
    seed = args.seed
    torch.manual_seed(seed)
    runner.alg.policy.memory_a.reset()
    obs_wp1, _ = env.reset()
    obs_wp1 = obs_wp1.clone()

    print("\n" + "=" * 70)
    print("STEP-0 COMPARISON")
    print("=" * 70)

    # ============ (a) raw observations ============
    print(f"\n[a] Raw observations:")
    print(f"    shape:              {tuple(obs_wp1.shape)}")
    print(f"    min/max/mean:       {obs_wp1.min().item():.4f} / {obs_wp1.max().item():.4f} / {obs_wp1.mean().item():.4f}")

    # ============ (b) normalized observations ============
    # WP1 side: runner.obs_normalizer
    wp1_norm = runner.obs_normalizer
    hebb_norm = hebb_actor.obs_normalizer
    print(f"\n[b] Obs normalizer identity:")
    print(f"    WP1  normalizer:    {type(wp1_norm).__name__}")
    print(f"    Hebb normalizer:    {type(hebb_norm).__name__}")

    # Compare normalizer state if both EmpiricalNormalization
    if hasattr(wp1_norm, "_mean") and hasattr(hebb_norm, "_mean"):
        d_mean = (wp1_norm._mean - hebb_norm._mean).abs().max().item()
        d_std  = (wp1_norm._std  - hebb_norm._std ).abs().max().item()
        print(f"    Δ mean (max abs):   {d_mean:.3e}")
        print(f"    Δ std  (max abs):   {d_std:.3e}")
        print(f"    WP1 mean abs max:   {wp1_norm._mean.abs().max().item():.4f}")
        print(f"    WP1 std range:      [{wp1_norm._std.min().item():.4f}, {wp1_norm._std.max().item():.4f}]")
        print(f"    Hebb mean abs max:  {hebb_norm._mean.abs().max().item():.4f}")
        print(f"    Hebb std range:     [{hebb_norm._std.min().item():.4f}, {hebb_norm._std.max().item():.4f}]")
    elif isinstance(hebb_norm, nn.Identity):
        print(f"    ✗  Hebb normalizer is nn.Identity — obs not normalized!")

    obs_wp1_norm = wp1_norm(obs_wp1)
    obs_hebb_norm = hebb_norm(obs_wp1)
    d_norm = (obs_wp1_norm - obs_hebb_norm).abs().max().item()
    print(f"    Δ normalized obs:   {d_norm:.3e}")

    # ============ (c) LSTM output ============
    print(f"\n[c] LSTM output:")
    # WP1: uses memory_a which has hidden_states=None initially
    wp1_policy = runner.alg.policy
    lstm_wp1 = wp1_policy.memory_a.rnn  # nn.LSTM
    lstm_hebb = hebb_actor._rnn

    # Verify LSTM weights are identical
    for name_w, name_h in zip(lstm_wp1.state_dict().keys(), lstm_hebb.state_dict().keys()):
        t_w = lstm_wp1.state_dict()[name_w]
        t_h = lstm_hebb.state_dict()[name_h]
        d = (t_w - t_h).abs().max().item()
        if d > 1e-8:
            print(f"    ✗  LSTM param {name_w} differs by {d:.3e}")
    print(f"    LSTM weights identical.")

    # Compute LSTM output for both
    h0 = torch.zeros(lstm_wp1.num_layers, args.num_envs, lstm_wp1.hidden_size, device=args.device)
    c0 = torch.zeros_like(h0)
    with torch.no_grad():
        out_wp1, _  = lstm_wp1(obs_wp1_norm.unsqueeze(0),  (h0, c0))
        out_hebb, _ = lstm_hebb(obs_hebb_norm.unsqueeze(0), (hebb_actor._h, hebb_actor._c) if hebb_actor._h is not None else (h0, c0))
    d_lstm = (out_wp1 - out_hebb).abs().max().item()
    print(f"    Δ LSTM output:      {d_lstm:.3e}")

    # ============ (d) MLP output before last layer ============
    print(f"\n[d] MLP backbone (all layers except last Linear):")
    actor_wp1_layers = list(wp1_policy.actor.children())
    actor_hebb_layers = hebb_actor._actor_layers
    print(f"    WP1  layers:        {len(actor_wp1_layers)}")
    print(f"    Hebb layers:        {len(actor_hebb_layers)}")

    # Verify MLP weights identical (layers [:-1] contain Linear, ELU, Linear, ELU)
    for i, (lw, lh) in enumerate(zip(actor_wp1_layers, actor_hebb_layers)):
        if hasattr(lw, "weight") and hasattr(lh, "weight"):
            w_w = lw.weight.data if hasattr(lw.weight, "data") else lw.weight
            w_h = lh.weight.data if hasattr(lh.weight, "data") else lh.weight
            d = (w_w - w_h).abs().max().item()
            if d > 1e-8:
                print(f"    ✗  Layer {i} weight differs by {d:.3e}")
        if hasattr(lw, "bias") and hasattr(lh, "bias") and lw.bias is not None and lh.bias is not None:
            b_w = lw.bias.data if hasattr(lw.bias, "data") else lw.bias
            b_h = lh.bias.data if hasattr(lh.bias, "data") else lh.bias
            d = (b_w - b_h).abs().max().item()
            if d > 1e-8:
                print(f"    ✗  Layer {i} bias differs by {d:.3e}")
    print(f"    MLP weights identical.")

    # Forward through MLP[:-1]
    x_wp1 = out_wp1.squeeze(0)
    for layer in actor_wp1_layers[:-1]:
        x_wp1 = layer(x_wp1)
    x_hebb = out_hebb.squeeze(0)
    for layer in actor_hebb_layers[:-1]:
        x_hebb = layer(x_hebb)
    d_mlp = (x_wp1 - x_hebb).abs().max().item()
    print(f"    Δ post-MLP (pre-last-linear): {d_mlp:.3e}")

    # ============ (e) last-layer weight + bias comparison ============
    print(f"\n[e] Last-layer weights:")
    last_wp1 = actor_wp1_layers[-1]  # nn.Linear
    W_wp1 = last_wp1.weight.data if hasattr(last_wp1.weight, "data") else last_wp1.weight  # (out, in)
    b_wp1 = last_wp1.bias.data if last_wp1.bias is not None else None  # (out,)
    W_hebb = hebb_actor.hebbian.W  # (num_envs, out, in)
    W_ckpt = hebb_actor.hebbian.W_checkpoint  # (out, in)
    b_hebb = hebb_actor._last_layer_bias.data if hasattr(hebb_actor._last_layer_bias, "data") else hebb_actor._last_layer_bias

    print(f"    WP1 W shape:        {tuple(W_wp1.shape)}")
    print(f"    Hebb W_checkpoint:  {tuple(W_ckpt.shape)}")
    print(f"    Hebb W.shape:       {tuple(W_hebb.shape)}")
    d_ckpt = (W_wp1 - W_ckpt).abs().max().item()
    print(f"    Δ (WP1 W vs Hebb W_checkpoint):  {d_ckpt:.3e}")
    # Check if W across envs == W_checkpoint
    for e in range(min(3, W_hebb.shape[0])):
        d = (W_hebb[e] - W_ckpt).abs().max().item()
        print(f"    Δ W[env={e}] vs W_checkpoint:  {d:.3e}")
    d_per_env_max = (W_hebb - W_ckpt.unsqueeze(0)).abs().max().item()
    print(f"    Δ W across all envs:           {d_per_env_max:.3e}")

    # Bias compare
    if b_wp1 is not None and b_hebb is not None:
        d_bias = (b_wp1 - b_hebb).abs().max().item()
        print(f"    Δ bias:             {d_bias:.3e}")

    # ============ (f) pre-tanh action ============
    print(f"\n[f] Pre-tanh action (full last layer):")
    with torch.no_grad():
        # WP1 path: F.linear(x, W_wp1, b_wp1)
        y_wp1 = torch.nn.functional.linear(x_wp1, W_wp1, b_wp1)
        # Hebb path: einsum("ni,noi->no") + bias
        y_hebb = torch.einsum("ni,noi->no", x_hebb, W_hebb)
        if b_hebb is not None:
            y_hebb = y_hebb + b_hebb.unsqueeze(0)
    d_y = (y_wp1 - y_hebb).abs().max().item()
    print(f"    Δ pre-tanh action:  {d_y:.3e}")

    # ============ (g) final action through act_inference / act ============
    print(f"\n[g] Final scaled action via act_inference vs actor.act:")
    # Compare _scale parameters
    print(f"    WP1 model max_throttle={wp1_policy.max_throttle} max_servo={wp1_policy.max_servo}")
    print(f"    Hebb model max_throttle={hebb_actor.model.max_throttle} max_servo={hebb_actor.model.max_servo}")
    print(f"    WP1  len(actor.children())={len(list(wp1_policy.actor.children()))}")
    print(f"    Hebb len(_actor_layers)={len(hebb_actor._actor_layers)}")
    print(f"    WP1  actor children types:  {[type(m).__name__ for m in wp1_policy.actor.children()]}")
    print(f"    Hebb actor layers types:    {[type(m).__name__ for m in hebb_actor._actor_layers]}")

    # Reset fresh
    torch.manual_seed(seed)
    runner.alg.policy.memory_a.reset()
    obs_a, _ = env.reset()
    with torch.no_grad():
        norm_a = runner.obs_normalizer(obs_a)
        # Step through WP1 manually to capture pre-tanh
        inp_wp1 = wp1_policy.memory_a(norm_a)
        actions_mean_wp1 = wp1_policy.actor(inp_wp1.squeeze(0) if inp_wp1.dim() == 3 else inp_wp1)
        a_wp1_pretanh = actions_mean_wp1
        a_wp1_tanh = torch.tanh(actions_mean_wp1)
        a_wp1 = wp1_policy._scale(a_wp1_tanh)

    torch.manual_seed(seed)
    hebb_actor.reset_episode(device=args.device)
    obs_b, _ = env.reset()
    with torch.no_grad():
        # Step through Hebb manually (replicate actor.act, no in-place hebbian_update)
        norm_b = hebb_actor.obs_normalizer(obs_b)
        rnn_out_b, _ = hebb_actor._rnn(norm_b.unsqueeze(0), (hebb_actor._h, hebb_actor._c))
        inp_b = rnn_out_b.squeeze(0)
        x_b = inp_b
        for layer in hebb_actor._actor_layers[:-1]:
            x_b = layer(x_b)
        W_b = hebb_actor.hebbian.W
        y_b = torch.einsum("ni,noi->no", x_b, W_b)
        if hebb_actor._last_layer_bias is not None:
            y_b = y_b + hebb_actor._last_layer_bias.unsqueeze(0)
        a_hebb_pretanh = y_b
        a_hebb_tanh = torch.tanh(y_b)
        a_hebb = hebb_actor.model._scale(a_hebb_tanh)

    print(f"    Δ normalized obs (WP1 vs Hebb reset): {(norm_a - norm_b).abs().max().item():.3e}")
    print(f"    Δ pre-tanh (act_inference path):       {(a_wp1_pretanh - a_hebb_pretanh).abs().max().item():.3e}")
    print(f"    Δ post-tanh:                           {(a_wp1_tanh - a_hebb_tanh).abs().max().item():.3e}")
    print(f"    Δ post-scale:                          {(a_wp1 - a_hebb).abs().max().item():.3e}")
    print(f"    a_wp1_pretanh[0]:  {a_wp1_pretanh[0].cpu().numpy()}")
    print(f"    a_hebb_pretanh[0]: {a_hebb_pretanh[0].cpu().numpy()}")
    print(f"    a_wp1_tanh[0]:     {a_wp1_tanh[0].cpu().numpy()}")
    print(f"    a_hebb_tanh[0]:    {a_hebb_tanh[0].cpu().numpy()}")

    d_raw_obs_reset = (obs_a - obs_b).abs().max().item()
    print(f"    Δ obs after 2x reset w/ same seed: {d_raw_obs_reset:.3e}")
    d_action = (a_wp1 - a_hebb).abs().max().item()
    print(f"    Δ final action (step 0):           {d_action:.3e}")
    if d_action > 1e-5:
        print(f"    ✗  Actions differ meaningfully!")
        print(f"    Per-action-dim max Δ: {(a_wp1 - a_hebb).abs().max(0).values.cpu().numpy()}")
        print(f"    First env WP1 action:  {a_wp1[0].cpu().numpy()}")
        print(f"    First env Hebb action: {a_hebb[0].cpu().numpy()}")
    else:
        print(f"    ✓  Actions match within tolerance.")

    # ============ std check (for stochastic mode, but we're deterministic) ============
    print(f"\n[h] Policy std (unused in deterministic mode):")
    if hasattr(wp1_policy, "std"):
        print(f"    WP1 std mean: {wp1_policy.std.mean().item():.4f}")
    else:
        print(f"    WP1 has no .std attribute")

    gs.destroy()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wp1-config", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    return compare(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
