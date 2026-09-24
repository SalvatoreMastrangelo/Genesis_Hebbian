"""Zero-rule rollout of the frozen generalist on the standard drone: statistics of the
last hidden activity x (32) and of the output mean y (7).

Source of the activity figures quoted in thesis section 5.2 (squared norm of x about 5 on
average and seldom above 12; root mean square 0.4 for the features, 0.5 for the outputs).
1 024 flights, search course of the production co-design config (100 m, density 0 -> 5),
commanded speeds 10 to 20 m/s, all coefficients at zero. Output: latent_activity_stats.json.

Run from the repository root, in the local Genesis image:

    docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/workspace/bind -v "$PWD/tesis/scripts":/scratch_out \
        -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
        mygenesis:latest python /scratch_out/latent_activity_stats.py
"""
import glob, json, sys
import numpy as np, torch
from WP2.config import HebbianEvolutionConfig
from WP2.evaluate import _build_env
from WP2.frozen_actor import build_isolated_population_actor
from WP2.utils import decode_hebbian_genes, create_zero_initialized_genome
from WP1.config import RunConfig
import genesis as gs

cfg_path = glob.glob("logs/remote/outer_nsga/outer_exam_4_64_64_300_extra_r1/2026-08-26_18-19-52_*/reproducibility/config.yaml")[0]
cfg = HebbianEvolutionConfig.from_yaml(cfg_path)
dev = "cuda:0"
gs.init(backend=gs.gpu, logging_level="warning")
wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
N = 1024
env, _ = _build_env(cfg, wp1_cfg, dev, num_envs_override=N)
genome = create_zero_initialized_genome(cfg)
rules = decode_hebbian_genes(genome, cfg.hebbian, out_features=7, in_features=32)
actor = build_isolated_population_actor(cfg.checkpoint_path, cfg.checkpoint_config_path, [rules], cfg, K=1, S=N, device=dev, stochastic=True)
actor.reset_episode(device=dev)
obs, _ = env.reset()
done = torch.zeros(N, dtype=torch.bool, device=dev)
xs, ys = [], []
step = 0
with torch.no_grad():
    while not done.all() and step < 1500:
        a = actor.act(obs)
        x = actor._last_hidden_input
        y = torch.einsum("ni,noi->no", x, actor.hebbian.W) + actor._last_layer_bias
        alive = ~done
        if step % 5 == 0:
            xs.append(x[alive].cpu().numpy()); ys.append(y[alive].cpu().numpy())
        obs, _, term, _ = env.step(a)
        done |= term.bool() | env.nan_envs.bool()
        step += 1
X = np.concatenate(xs); Y = np.concatenate(ys)
dW = (actor.hebbian.W - actor.hebbian.W_checkpoint).abs().max().item()
out = dict(
    steps=step, samples=int(X.shape[0]), max_abs_dW_zero_rules=dW,
    x_mean=float(X.mean()), x_rms=float(np.sqrt((X**2).mean())), x_min=float(X.min()), x_max=float(X.max()),
    x_absmean=float(np.abs(X).mean()),
    xnorm2_mean=float((X**2).sum(1).mean()), xnorm2_p5=float(np.percentile((X**2).sum(1),5)), xnorm2_p95=float(np.percentile((X**2).sum(1),95)), xnorm2_max=float((X**2).sum(1).max()),
    x_unit_mean=X.mean(0).round(3).tolist(), x_unit_std=X.std(0).round(3).tolist(),
    x_frac_negative=float((X<0).mean()),
    y_mean=Y.mean(0).round(3).tolist(), y_std=Y.std(0).round(3).tolist(), y_absmean=float(np.abs(Y).mean()), y_rms=float(np.sqrt((Y**2).mean())), y_absmax=float(np.abs(Y).max()),
    xy_absmean=float(np.abs(Y[:, :, None]*X[:, None, :]).mean()),
    # share of variance of x that is constant over time (mean^2 / mean square)
    x_const_share=float((X.mean(0)**2).sum() / (X**2).mean(0).sum()),
)
print(json.dumps(out, indent=1))
json.dump(out, open("/scratch_out/latent_activity_stats.json", "w"), indent=1)
