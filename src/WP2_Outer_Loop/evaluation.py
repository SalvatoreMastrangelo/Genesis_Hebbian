"""
NSGA-II per-URDF evaluation pass.
==================================

After the inner CMA-ES loop finishes, we hold:

- ``best_rules``: the carried-over Hebbian-rule genome (``[0,1]^n_rules``).
- ``urdf_paths``: the outer-loop URDF population that produced it.

NSGA-II needs **per-URDF** objective values, but the inner-loop
``evaluate_population_multi_urdf`` returns metrics *averaged across URDFs*
for each rule individual. To get per-URDF scalars we evaluate URDF-by-URDF:
for each URDF we call ``evaluate_population_multi_urdf`` with
``solutions=[best_rules]`` and ``urdf_paths=[one_urdf]``, sizing
``cfg.evaluation.num_eval_envs = envs_per_urdf``. The returned metrics
(shape ``(1,)``) describe that single URDF under the best rules.

Efficiency note: this rebuilds the Genesis scene once per URDF. That's
acceptable because the NSGA-II evaluation pass runs once per outer
generation (vs. the inner CMA-ES which runs once per inner generation).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np

from WP2.config import HebbianEvolutionConfig
from WP2.evaluate import evaluate_population_multi_urdf
from WP2.frozen_actor import load_frozen_actor

from .config import OuterLoopConfig


# Mapping from outer-loop objective names → keys in the WP2 metrics dict.
_METRIC_ALIASES: Dict[str, str] = {
    "progress_m": "progresses",
    "progress": "progresses",
    "cost_of_transport": "cots",
    "cot": "cots",
    "fitness": "reward_sums",
    "velocity": "velocities",
    "crash_rate": "crash_flags",
    "velocity_deviation": "v_deviations",
}


def _lookup_metric(metrics: Dict[str, np.ndarray], name: str) -> np.ndarray:
    key = _METRIC_ALIASES.get(name, name)
    if key not in metrics:
        raise KeyError(
            f"Metric {name!r} (alias→{key!r}) not found in evaluation output. "
            f"Available: {sorted(metrics.keys())}"
        )
    return np.asarray(metrics[key], dtype=np.float64)


def evaluate_urdfs_per_individual(
    urdf_paths: Sequence[str],
    best_rules: np.ndarray,
    outer_cfg: OuterLoopConfig,
    inner_cfg_template: HebbianEvolutionConfig,
) -> np.ndarray:
    """Return a ``(P, n_obj)`` objective matrix, one row per URDF.

    Parameters
    ----------
    urdf_paths
        The P URDFs in the current outer-loop population.
    best_rules
        Best Hebbian-rule genome from the inner loop.
    outer_cfg
        Outer-loop config (objectives, num_eval_envs).
    inner_cfg_template
        Fully-populated ``HebbianEvolutionConfig`` used by the inner loop
        this outer generation. We copy it and patch ``num_eval_envs`` +
        ``catalog`` for the per-URDF evaluation.
    """
    P = len(urdf_paths)
    if P == 0:
        raise ValueError("urdf_paths is empty")

    envs_per_urdf = outer_cfg.num_eval_envs // P
    if envs_per_urdf < 1:
        raise ValueError(
            f"num_eval_envs ({outer_cfg.num_eval_envs}) < population_size "
            f"({P}); cannot give ≥1 env per URDF"
        )

    # One-shot load of the frozen actor (shared across URDF evals).
    model, last_layer, _, _ = load_frozen_actor(
        inner_cfg_template.checkpoint_path,
        inner_cfg_template.checkpoint_config_path,
        device=inner_cfg_template.device,
    )
    from WP1.config import RunConfig
    wp1_cfg = RunConfig.from_yaml(inner_cfg_template.checkpoint_config_path)
    model_and_layer = (
        model, last_layer,
        inner_cfg_template.hebbian.num_actions,
        inner_cfg_template.hebbian.hidden_dim,
    )

    n_obj = len(outer_cfg.objectives)
    objectives = np.zeros((P, n_obj), dtype=np.float64)

    for i, urdf in enumerate(urdf_paths):
        cfg_i = copy.deepcopy(inner_cfg_template)
        cfg_i.evaluation.num_eval_envs = envs_per_urdf
        # NSGA-II eval does not need baseline or forest refresh.
        cfg_i.evaluation.run_baseline = False
        cfg_i.evaluation.refresh_forests_per_generation = False
        # Force the multi-URDF code path even with N=1 for a clean API.
        cfg_i.catalog.path = ""
        cfg_i.catalog.num_urdfs = 1
        cfg_i.catalog.force_multi_urdf = True

        try:
            _fit, metrics = evaluate_population_multi_urdf(
                solutions=[np.asarray(best_rules, dtype=np.float64)],
                cfg=cfg_i,
                model_and_layer=model_and_layer,
                wp1_cfg=wp1_cfg,
                urdf_paths=[urdf],
                existing_env=None,
                verbose=False,
            )
        except Exception as exc:
            print(f"[OuterLoop] NSGA-II eval failed for URDF {i} ({urdf}): {exc}")
            # Worst-possible scores for each objective so this individual is
            # dominated in NSGA-II selection.
            for j, obj in enumerate(outer_cfg.objectives):
                objectives[i, j] = -np.inf if obj.direction == "maximize" else np.inf
            continue

        for j, obj in enumerate(outer_cfg.objectives):
            vec = _lookup_metric(metrics, obj.name)
            objectives[i, j] = float(vec[0])

    return objectives


def evaluate_carried_rules_on_new_population(
    urdf_paths: Sequence[str],
    carried_rules: np.ndarray,
    inner_cfg_template: HebbianEvolutionConfig,
    num_eval_envs: int,
) -> Dict[str, float]:
    """Sanity-check re-evaluation of carried-over rules on the new URDF pop.

    Called at the start of each outer generation ≥ 1, before the inner
    CMA-ES resumes from the carried-over rules. Returns a dict of scalars
    averaged across all URDFs (the same aggregation the inner loop uses
    for its own fitness). This number is logged as a reference point; it
    does not feed back into CMA-ES.
    """
    model, last_layer, _, _ = load_frozen_actor(
        inner_cfg_template.checkpoint_path,
        inner_cfg_template.checkpoint_config_path,
        device=inner_cfg_template.device,
    )
    from WP1.config import RunConfig
    wp1_cfg = RunConfig.from_yaml(inner_cfg_template.checkpoint_config_path)
    model_and_layer = (
        model, last_layer,
        inner_cfg_template.hebbian.num_actions,
        inner_cfg_template.hebbian.hidden_dim,
    )

    cfg = copy.deepcopy(inner_cfg_template)
    cfg.evaluation.num_eval_envs = num_eval_envs
    cfg.evaluation.run_baseline = False
    cfg.evaluation.refresh_forests_per_generation = False
    cfg.catalog.path = ""
    cfg.catalog.num_urdfs = len(urdf_paths)
    cfg.catalog.force_multi_urdf = True

    try:
        fit, metrics = evaluate_population_multi_urdf(
            solutions=[np.asarray(carried_rules, dtype=np.float64)],
            cfg=cfg,
            model_and_layer=model_and_layer,
            wp1_cfg=wp1_cfg,
            urdf_paths=list(urdf_paths),
            existing_env=None,
            verbose=False,
        )
    except Exception as exc:
        print(f"[OuterLoop] Carried-rules re-eval failed: {exc}")
        return {}

    return {
        "fitness": float(fit[0]),
        "progress_m": float(metrics["progresses"][0]),
        "cost_of_transport": float(metrics["cots"][0]),
        "velocity": float(metrics["velocities"][0]),
        "crash_rate": float(metrics["crash_flags"][0]),
    }
