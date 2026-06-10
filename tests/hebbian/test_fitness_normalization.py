"""
Test min-max fitness normalization applied before CMA-ES tell().

The helper normalizes a generation's fitness vector to [0, 1] so the best
individual gets 1.0 and the worst 0.0. Degenerate (constant) vectors are
returned unchanged to avoid division by zero.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from WP2.evolve_cma import minmax_normalize_fitness


def test_basic_normalization():
    fitnesses = np.array([2.0, 5.0, 8.0])
    result = minmax_normalize_fitness(fitnesses)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_best_is_one_worst_is_zero():
    fitnesses = np.array([13.7, -4.2, 100.5, 0.0])
    result = minmax_normalize_fitness(fitnesses)
    assert result[np.argmax(fitnesses)] == 1.0
    assert result[np.argmin(fitnesses)] == 0.0
    assert (result >= 0.0).all() and (result <= 1.0).all()


def test_preserves_ranking():
    rng = np.random.default_rng(0)
    fitnesses = rng.normal(size=64) * 50.0
    result = minmax_normalize_fitness(fitnesses)
    np.testing.assert_array_equal(np.argsort(result), np.argsort(fitnesses))


def test_constant_vector_returned_unchanged():
    fitnesses = np.zeros(8)
    result = minmax_normalize_fitness(fitnesses)
    np.testing.assert_array_equal(result, fitnesses)

    fitnesses = np.full(8, 3.5)
    result = minmax_normalize_fitness(fitnesses)
    np.testing.assert_array_equal(result, fitnesses)


def test_negative_fitnesses():
    fitnesses = np.array([-10.0, -5.0, -20.0])
    result = minmax_normalize_fitness(fitnesses)
    np.testing.assert_allclose(result, [2.0 / 3.0, 1.0, 0.0])


def test_config_flag_defaults_off():
    from WP2.config import CMAESConfig

    assert CMAESConfig().normalize_fitness is False


if __name__ == "__main__":
    test_basic_normalization()
    test_best_is_one_worst_is_zero()
    test_preserves_ranking()
    test_constant_vector_returned_unchanged()
    test_negative_fitnesses()
    test_config_flag_defaults_off()
    print("All tests passed.")
