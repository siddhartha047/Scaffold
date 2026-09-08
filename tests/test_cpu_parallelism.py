"""Regression checks for worker masks and deterministic parallel sampling."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scaffold.algorithms import _sampling as sampling
from scaffold.utils import workers as workers_module


def test_numba_masks_are_local_to_each_pool_task():
    numba = pytest.importorskip("numba")
    barrier = threading.Barrier(2)
    ceiling = numba.config.NUMBA_NUM_THREADS

    def task(requested):
        previous = numba.get_num_threads()
        with workers_module.parallel_threads(requested) as actual:
            barrier.wait(timeout=30)
            observed = numba.get_num_threads()
            barrier.wait(timeout=30)
        return actual, observed, numba.get_num_threads(), previous

    with workers_module.parallel_threads(1):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(task, count) for count in (1, 2)]
            results = [future.result() for future in futures]
    for requested, (actual, observed, restored, previous) in zip((1, 2), results):
        assert actual == observed == min(requested, ceiling)
        assert restored == previous


def test_nested_mask_can_shrink_but_cannot_exceed_parent_and_restores_on_error():
    numba = pytest.importorskip("numba")
    previous = numba.get_num_threads()
    with workers_module.parallel_threads(2) as outer:
        with workers_module.parallel_threads(1):
            assert numba.get_num_threads() == 1
        assert numba.get_num_threads() == outer
        with pytest.raises(RuntimeError), workers_module.parallel_threads(100):
            assert numba.get_num_threads() == outer
            raise RuntimeError("restore on error")
        assert numba.get_num_threads() == outer
    assert numba.get_num_threads() == previous


@pytest.mark.parametrize("offset", [0.0, 0.125, np.nextafter(1.0, 0.0)])
def test_parallel_systematic_draw_matches_numpy_exactly(offset):
    rng = np.random.default_rng(17)
    # Includes flat segments, probability-one entries, and a partial last block.
    probabilities = rng.choice([0.0, 0.125, 0.5, 1.0], size=240_000)
    cumulative = np.cumsum(probabilities)
    count = int(cumulative[-1])
    ticks = offset + np.arange(count, dtype=np.float64)
    expected = np.minimum(
        np.searchsorted(cumulative, ticks, side="left"), cumulative.size - 1
    )
    for workers in (1, 2, 4):
        got, used = sampling.systematic_positions(cumulative, offset, count, workers)
        np.testing.assert_array_equal(got, expected)
        assert 1 <= used <= workers


def test_sampling_fallback_and_end_clipping(monkeypatch):
    cumulative = np.array([0.0, 0.0, 0.5, 1.5])
    for kernel in (sampling._scan_kernel, None):
        monkeypatch.setattr(sampling, "_scan_kernel", kernel)
        count = 70_001
        got, _ = sampling.systematic_positions(cumulative, 0.25, count, 2)
        expected = np.minimum(np.searchsorted(cumulative, 0.25 + np.arange(count)), 3)
        np.testing.assert_array_equal(got, expected)
    empty, used = sampling.systematic_positions([], 0.0, 0, 4)
    assert empty.size == 0 and used == 1
    with pytest.raises(ValueError):
        sampling.systematic_positions([], 0.0, 1, 4)
