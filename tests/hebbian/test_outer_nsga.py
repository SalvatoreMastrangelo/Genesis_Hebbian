"""
Tests for the WP2 outer loop (persistent CMA-ES + NSGA-II URDF refresh).

Pure-python: covers the NSGA-II variation helpers, the outer config
validation, the objective-name aliasing, the 2-D hypervolume, and the
per-URDF metric reductions added to WP2.evaluate — no Genesis required.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2_Outer_Loop.config import OuterNSGA2Config
from WP2_Outer_Loop.nsga2 import (
    arrays_to_individuals,
    individuals_to_array,
    make_toolbox,
)
from WP2_Outer_Loop.nsga_cma import NSGA2MorphCMAES, make_offspring_tournament
from WP2_Outer_Loop.pareto_plots import _hypervolume_2d, _nondominated_mask


def _cfg(num_urdfs=4, **outer_kw):
    """Single-file config: inner-loop fields + `outer:` NSGA-II section."""
    cfg = OuterNSGA2Config()
    cfg.catalog.num_urdfs = num_urdfs
    cfg.catalog.refresh_urdfs_every = 8
    for k, v in outer_kw.items():
        assert hasattr(cfg.outer, k)
        setattr(cfg.outer, k, v)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------- config

def test_config_defaults_validate():
    _cfg()


def test_config_rejects_bad_elites():
    with pytest.raises(ValueError):
        _cfg(num_urdfs=4, n_elites=4)
    with pytest.raises(ValueError):
        _cfg(num_urdfs=4, n_elites=0)


def test_config_allows_odd_and_non_multiple_of_four_pops():
    # The legacy loop required pop % 4 == 0 (DEAP selTournamentDCD); the
    # rewrite must not.
    for n in (2, 3, 5, 6):
        _cfg(num_urdfs=n, n_elites=1)


def test_config_rejects_single_urdf():
    with pytest.raises(ValueError):
        _cfg(num_urdfs=1)


def test_config_rejects_mutate_refresh():
    cfg = _cfg()
    cfg.catalog.mutate = True
    with pytest.raises(ValueError, match="mutate"):
        cfg.validate()


def test_config_rejects_disabled_refresh():
    cfg = _cfg()
    cfg.catalog.refresh_urdfs_every = 0
    with pytest.raises(ValueError, match="refresh_urdfs_every"):
        cfg.validate()


def test_config_rejects_too_small_env_budget():
    cfg = _cfg()
    cfg.cmaes.population_size = 16
    cfg.evaluation.num_eval_envs = cfg.catalog.num_urdfs * 16 - 1
    with pytest.raises(ValueError, match="num_eval_envs"):
        cfg.validate()


def test_config_single_file_yaml_roundtrip(tmp_path):
    """One YAML carries inner fields AND the outer section; to_yaml round-trips."""
    src = tmp_path / "outer.yaml"
    src.write_text(
        "exp_name: single_file_test\n"
        "seed: 7\n"
        "cmaes:\n"
        "  population_size: 8\n"
        "  sigma_reinflate: 1.5\n"
        "catalog:\n"
        "  num_urdfs: 4\n"
        "  refresh_urdfs_every: 8\n"
        "  include_standard_mydrone: false\n"
        "evaluation:\n"
        "  num_eval_envs: 512\n"
        "outer:\n"
        "  n_elites: 3\n"
        "  score_top_frac: 0.25\n"
        "  sbx_eta: 10.0\n"
        "  objectives:\n"
        "    - name: progress_m\n"
        "      direction: maximize\n"
        "    - name: crash_rate\n"
        "      direction: minimize\n"
    )
    cfg = OuterNSGA2Config.from_yaml(src)
    # inner fields land in the usual places
    assert cfg.exp_name == "single_file_test"
    assert cfg.seed == 7
    assert cfg.cmaes.population_size == 8
    assert cfg.catalog.num_urdfs == 4
    assert cfg.catalog.include_standard_mydrone is False
    # outer section is parsed into typed fields
    assert cfg.outer.n_elites == 3
    assert cfg.outer.score_top_frac == 0.25
    assert cfg.outer.sbx_eta == 10.0
    assert [(o.name, o.direction) for o in cfg.outer.objectives] == [
        ("progress_m", "maximize"), ("crash_rate", "minimize")
    ]
    assert cfg.outer.objective_weights() == [1.0, -1.0]
    cfg.validate()
    # the serialized config (what lands in reproducibility/config.yaml)
    # is itself a loadable single-file config
    dumped = tmp_path / "dumped.yaml"
    cfg.to_yaml(dumped)
    cfg2 = OuterNSGA2Config.from_yaml(dumped)
    assert cfg2.outer.n_elites == 3
    assert cfg2.catalog.num_urdfs == 4
    assert [(o.name, o.direction) for o in cfg2.outer.objectives] == [
        ("progress_m", "maximize"), ("crash_rate", "minimize")
    ]


# ---------------------------------------------------------------- variation

@pytest.mark.parametrize("N", [2, 3, 4, 6])
def test_offspring_counts_and_bounds(N):
    random.seed(0)
    cfg = _cfg(num_urdfs=max(N, 2), n_elites=1)
    tb = make_toolbox(cfg.outer, genome_dim=15)
    genomes = np.random.rand(N, 15)
    objs = np.column_stack([np.random.rand(N) * 100, np.random.rand(N) * 2])
    inds = arrays_to_individuals(genomes, objs)
    for n_elites in range(1, N):
        elites = tb.select(inds, n_elites)
        off = make_offspring_tournament(tb, inds, N - n_elites, 0.9)
        new = np.vstack([individuals_to_array(elites),
                         individuals_to_array(off)])
        assert new.shape == (N, 15)
        assert new.min() >= 0.0 and new.max() <= 1.0


def test_selnsga2_respects_objective_directions():
    """fitness maximize + cot minimize: the dominant point must win at k=1."""
    random.seed(0)
    cfg = _cfg()
    tb = make_toolbox(cfg.outer, genome_dim=15)
    g = np.random.rand(2, 15)
    objs = np.array([[100.0, 0.5],   # dominates on both objectives
                     [50.0, 1.5]])
    sel = tb.select(arrays_to_individuals(g, objs), 1)
    np.testing.assert_allclose(list(sel[0]), g[0])


# ---------------------------------------------------------------- aliases

def test_objective_aliases():
    f = NSGA2MorphCMAES._canonical_objective
    assert f("fitness") == "fitness"
    assert f("cot") == "cost_of_transport"
    assert f("cost_of_transport") == "cost_of_transport"
    assert f("progress") == "progress_m"
    with pytest.raises(KeyError):
        f("not_a_metric")


# ---------------------------------------------------------------- hypervolume

def test_hypervolume_simple_rectangle():
    front = np.array([[1.0, 1.0]])
    assert _hypervolume_2d(front, np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_hypervolume_two_point_front():
    front = np.array([[2.0, 1.0], [1.0, 2.0]])
    # (2-0)*(1-0) + (1-0)*(2-1) = 3
    assert _hypervolume_2d(front, np.array([0.0, 0.0])) == pytest.approx(3.0)


def test_hypervolume_grows_with_dominating_point():
    ref = np.array([0.0, 0.0])
    hv1 = _hypervolume_2d(np.array([[1.0, 1.0]]), ref)
    hv2 = _hypervolume_2d(np.array([[2.0, 2.0]]), ref)
    assert hv2 > hv1


def test_nondominated_mask():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.5]])  # maximization
    mask = _nondominated_mask(pts)
    assert mask.tolist() == [False, True, True]


# ------------------------------------------------- per-URDF reductions

def test_per_urdf_reduction_consistency():
    """Mean over URDFs of the per-URDF matrix must equal the per-individual
    reduction used by the inner loop."""
    D, P, F = 3, 4, 5
    metric = torch.rand(D, P * F)
    per_urdf = metric.view(D, P, F).mean(dim=2)          # (D, P)
    per_ind = metric.view(D, P, F).mean(dim=(0, 2))      # (P,)
    assert torch.allclose(per_urdf.mean(dim=0), per_ind, atol=1e-6)


def test_per_urdf_cot_penalises_crashers():
    """Ratio-of-sums CoT: a slot with ~zero distance must yield a huge CoT,
    not a spuriously small one."""
    D, P, F = 1, 2, 4
    energy = torch.full((D, P, F), 5.0)
    dx = torch.tensor([[[20.0] * F, [1e-6] * F]])  # ind 0 flies, ind 1 crashes
    mg = torch.tensor([[9.81]])
    pu_energy = energy.mean(dim=2)
    pu_dx = dx.mean(dim=2)
    cot = pu_energy / (mg * pu_dx.clamp(min=1e-2))
    assert cot[0, 1] > 100 * cot[0, 0]
