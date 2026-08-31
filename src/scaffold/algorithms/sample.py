r"""SCAFFOLD-Sample: score every edge once, then draw a fresh graph per epoch.

The other four variants answer "which edges should I keep?". This one answers
"how important is each edge?" -- and hands you the weights.

That difference matters for GNN training. Fixing one sparse graph for all
epochs throws away the rest of the graph permanently. Re-running a greedy
sparsifier every epoch is far too slow. SCAFFOLD-Sample does the structural
work **once**, offline, and reduces the per-epoch cost to one uniform draw plus
one ``searchsorted``.

Precomputation (paper Algorithm 3)
----------------------------------
1. Build a fixed backbone ``F0`` (MaxST by default, or seeded RandST) and ``R``
   random spanning forests ``F_1..F_R``.
2. Score every non-tree edge exactly against each forest with the
   ``O(m log n + n)`` tree kernel.
3. Aggregate into a single ratio-independent weight::

       pi[e] = max(eps, freq[e] + lambda * mean_normalized_score[e])

   where ``freq[e]`` is the fraction of the random forests containing ``e``
   (structural indispensability) and the score term is how badly the forests
   serve ``e`` (repair value). One artifact serves every target ratio.
4. Order edges by the DFS preorder of their LCA in ``F0`` -- *tree locality*,
   which is what makes systematic sampling spread its picks across the graph
   rather than clumping them.

Per-epoch draw (paper Algorithm 4)
----------------------------------
* At or above the connectivity floor, a complete spanning forest is unioned in,
  so the drawn graph has exactly the components of ``G`` with probability 1 --
  not in expectation, not with high probability. Below the floor, the forest
  cannot fit and is budget-trimmed afresh on every draw as described below.
* The remaining ``k`` edges come from systematic ``pi``-ps sampling over the
  tree-locality order, which yields exactly ``k`` edges with exact marginal
  inclusion probabilities.

Below the connectivity floor ``delta_min = (n - c) / m`` no method can keep the
graph's components intact; the complete forced forest is then trimmed uniformly
at random for each draw and ``below_connectivity_floor`` is set in the metadata.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..backbone import build_backbone
from ..graph import Graph
from ..kernels import (
    build_tree_index,
    component_count,
    spanning_forest_mask,
    tree_lca,
)
from ..scoring import ScoreParams, tree_scores
from ..utils.validation import connectivity_floor, resolve_budget
from ..utils.workers import parallel_threads, resolve_workers, split_workers

DEFAULT_TREE_COUNT = 8
DEFAULT_LAMBDA = 1.0
EPS = 1e-8

# Below this many edges the per-backbone work is too small to hand to a thread
# pool; see the measured crossover quoted at the use site in ``fit``.
PARALLEL_SAMPLE_MIN_EDGES = 50_000

__all__ = ["ScaffoldSampler", "cap_and_renormalize", "run"]


# ----------------------------------------------------------------------
# inclusion probabilities
# ----------------------------------------------------------------------
def cap_and_renormalize(pi, target: int) -> np.ndarray:
    """Scale ``pi`` so ``sum(p) == target`` with every ``p <= 1``.

    The textbook ``pi``-ps fixed point: repeatedly cap the entries that would
    exceed 1 and redistribute the freed budget over the rest. Edges that end up
    capped at 1 form the *deterministic core* -- present in every draw.
    """
    pi = np.asarray(pi, dtype=np.float64)
    n = int(pi.size)
    target = int(target)
    if n == 0 or target <= 0:
        return np.zeros(n, dtype=np.float64)
    if target >= n:
        return np.ones(n, dtype=np.float64)

    p = np.zeros(n, dtype=np.float64)
    free = np.ones(n, dtype=bool)
    capped = 0
    guard = 0
    while True:
        guard += 1
        remaining = target - capped
        if remaining <= 0 or not free.any():
            break
        free_sum = float(pi[free].sum())
        if not np.isfinite(free_sum) or free_sum <= 0.0:
            p[free] = remaining / float(free.sum())
            break
        scale = remaining / free_sum
        newly = free & (pi * scale >= 1.0)
        if not newly.any() or guard > 64:
            p[free] = np.clip(pi[free] * scale, 0.0, 1.0)
            break
        p[newly] = 1.0
        free &= ~newly
        capped += int(newly.sum())
    return np.clip(p, 0.0, 1.0)


# ----------------------------------------------------------------------
# the sampler
# ----------------------------------------------------------------------
class ScaffoldSampler:
    """Precomputed SCAFFOLD edge weights, plus the per-epoch draw.

    Build one with :func:`scaffold.sample` (or construct directly and call
    :meth:`fit`), keep it for the whole training run, and call :meth:`draw`
    whenever you want a fresh view of the graph.
    """

    def __init__(
        self,
        params: Optional[ScoreParams] = None,
        tree_count: int = DEFAULT_TREE_COUNT,
        aggregate_lambda: float = DEFAULT_LAMBDA,
        backbone: str = "fixed-maxst",
        scheme: str = "systematic",
        weighted_paths: bool = False,
        seed=None,
        workers=None,
        verbose: bool = False,
    ):
        self.params = params or ScoreParams()
        self.tree_count = max(1, int(tree_count))
        self.workers = resolve_workers(workers)
        self.aggregate_lambda = float(aggregate_lambda)
        backbone = str(backbone).strip().lower().replace("_", "-")
        if backbone not in ("fixed-maxst", "fixed-randst", "rotate-randst"):
            raise ValueError(
                "backbone must be 'fixed-maxst', 'fixed-randst', or "
                "'rotate-randst', got "
                f"{backbone!r}"
            )
        self.backbone = backbone
        scheme = str(scheme).strip().lower()
        if scheme != "systematic":
            raise ValueError("scheme must be 'systematic'")
        self.scheme = scheme
        self.weighted_paths = bool(weighted_paths)
        self.seed = seed
        self.verbose = bool(verbose)

        self.graph: Optional[Graph] = None
        self.pi: Optional[np.ndarray] = None
        self.mandatory: Optional[np.ndarray] = None
        self.det_forest: Optional[np.ndarray] = None
        self.random_forests = []
        self.order: Optional[np.ndarray] = None
        self.base_components = 0
        self.delta_min = 0.0
        self.build_seconds = 0.0
        self._draw_index = 0
        self._plan = None
        self._plan_key = None

    # -- precompute ---------------------------------------------------------
    def fit(self, graph: Graph) -> ScaffoldSampler:
        """Run the one-time precompute and store the artifact on ``self``."""
        start = time.perf_counter()
        n, m = graph.num_nodes, graph.num_edges
        src, dst = graph.src, graph.dst
        weight = graph.weights_or_ones()

        self.graph = graph
        self.base_components = component_count(n, src, dst)
        self.delta_min = connectivity_floor(n, m, self.base_components)

        if m == 0:
            self.pi = np.zeros(0, dtype=np.float64)
            self.mandatory = np.zeros(0, dtype=bool)
            self.det_forest = np.zeros(0, dtype=bool)
            self.random_forests = []
            self.order = np.zeros(0, dtype=np.int64)
            self.build_seconds = time.perf_counter() - start
            return self

        # --- fixed backbone used to guarantee every draw ------------------
        if self.backbone == "fixed-randst":
            det_mask = build_backbone(graph, "randst", seed=self.seed)
        else:
            if graph.edge_weight is None:
                visit = np.arange(m, dtype=np.int64)
            else:
                visit = np.argsort(-graph.edge_weight, kind="stable").astype(
                    np.int64
                )
            det_mask = spanning_forest_mask(n, src, dst, visit)
        det_index = build_tree_index(n, src[det_mask], dst[det_mask])
        det_out = tree_scores(
            n, src, dst, det_mask, weight,
            params=self.params, tree_index=det_index,
            weighted_paths=self.weighted_paths,
            workers=self.workers,  # runs alone: give it every worker
        )

        # --- R random forests ----------------------------------------------
        # The R backbones are independent, which makes them the widest parallel
        # axis available here -- and the only one that also covers the union-find
        # forest construction, which is serial within a single backbone. When
        # there are fewer backbones than workers the leftovers go to the tree
        # scorer's own candidate-level parallelism instead of oversubscribing.
        outer, inner = split_workers(self.tree_count, self.workers)
        if m < PARALLEL_SAMPLE_MIN_EDGES:
            # Scoring a small graph takes a couple of milliseconds per
            # backbone, less than it costs to hand the work to a pool. Measured
            # on 8 backbones: 0.32x at 1.5k edges, 0.90x at 24k, 1.58x at 80k,
            # 3.59x at 320k. The threshold sits comfortably past break-even.
            outer, inner = 1, self.workers

        def score_forest(r):
            rng = np.random.default_rng(
                None if self.seed is None else int(self.seed) + 1000 * (r + 1)
            )
            mask = spanning_forest_mask(
                n, src, dst, rng.permutation(m).astype(np.int64)
            )
            out = tree_scores(
                n, src, dst, mask, weight,
                params=self.params, weighted_paths=self.weighted_paths,
                workers=inner,
            )
            return mask, out


        if outer > 1:
            # Pin the inner thread count *outside* the pool, so each worker
            # thread's tree_scores call finds the pin already in place rather
            # than racing to reset a process-global.
            with parallel_threads(inner), ThreadPoolExecutor(
                max_workers=outer
            ) as pool:
                results = list(pool.map(score_forest, range(self.tree_count)))
        else:
            # No pool, so nothing to protect -- and pinning here would override
            # each tree_scores call's own decision to drop to one thread on a
            # small candidate set, since a nested pin is a no-op by design.
            results = [score_forest(r) for r in range(self.tree_count)]

        accumulated = np.zeros(m, dtype=np.float64)
        frequency = np.zeros(m, dtype=np.float64)
        mandatory = det_out["mandatory"].copy()
        forests = []
        # Accumulated in forest order, never in completion order: these are
        # float sums, so a scheduler-dependent order would make pi depend on
        # the worker count.
        for r, (mask, out) in enumerate(results):
            values = out["score"]
            finite = np.isfinite(values)
            total = float(values[finite].sum())
            if total > 0.0:
                accumulated[finite] += values[finite] / total
            frequency += mask.astype(np.float64)
            mandatory |= out["mandatory"]
            forests.append(mask)
            if self.verbose:
                print(
                    f"[scaffold.sample] forest {r + 1}/{self.tree_count} "
                    f"edges={int(mask.sum())}"
                )

        pi = frequency / self.tree_count + self.aggregate_lambda * (
            accumulated / self.tree_count
        )
        pi = np.clip(np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0), EPS, None)

        # --- tree-locality ordering ----------------------------------------
        det_depth, det_root, det_up = det_index[0], det_index[1], det_index[2]
        det_tin = det_index[4]
        lca = tree_lca(src, dst, det_depth, det_root, det_up)
        key = np.where(lca >= 0, det_tin[np.maximum(lca, 0)], np.int64(-1))
        order = np.lexsort((np.arange(m, dtype=np.int64), key)).astype(np.int64)

        self.pi = pi
        self.mandatory = mandatory
        self.det_forest = det_mask
        self.random_forests = forests
        self.order = order
        self.total_stretch = float(det_out["total_stretch"])
        self.build_seconds = time.perf_counter() - start
        self._plan = None
        self._plan_key = None
        if self.verbose:
            print(
                f"[scaffold.sample] precompute {self.build_seconds:.2f}s "
                f"edges={m} mandatory={int(mandatory.sum())} "
                f"delta_min={self.delta_min:.4f}"
            )
        return self

    # -- derived quantities -------------------------------------------------
    def _budget(self, keep_ratio, num_edges) -> int:
        return resolve_budget(self.graph.num_edges, keep_ratio, num_edges)

    def inclusion_probabilities(self, keep_ratio=None, num_edges=None) -> np.ndarray:
        """Per-edge inclusion probability for a budget; sums to the budget.

        Above the connectivity floor, forced edges (backbone plus mandatory)
        get exactly 1.0. Below it, the full forced forest is uniformly trimmed,
        so each of its edges has probability ``target / forest_size``.
        """
        self._require_fit()
        target = self._budget(keep_ratio, num_edges)
        plan = self._get_plan(target, self._rotation())
        p = np.zeros(self.graph.num_edges, dtype=np.float64)
        if plan["trimmed"]:
            if plan["forced"].size:
                p[plan["forced"]] = target / plan["forced"].size
            return p
        p[plan["forced"]] = 1.0
        p[plan["pool_order"]] = plan["p"]
        return p

    def coverage(self, keep_ratio=None, num_edges=None, epochs=(1, 10, 100, 500)):
        """How much of the graph a training run of ``epochs`` draws would see.

        Returns the deterministic-core size and, per epoch count, the expected
        fraction of edges included at least once. This is the number to look at
        when deciding whether per-epoch resparsification is buying anything.
        """
        p = self.inclusion_probabilities(keep_ratio=keep_ratio, num_edges=num_edges)
        with np.errstate(divide="ignore"):
            first = np.where(p > 0, 1.0 / np.maximum(p, 1e-300), np.inf)
        finite = first[np.isfinite(first)]
        return {
            "always_included": int((p >= 1.0 - 1e-12).sum()),
            "median_epochs_to_first_inclusion": (
                float(np.median(finite)) if finite.size else float("inf")
            ),
            "coverage_curve": {
                int(e): float(np.mean(1.0 - np.power(1.0 - p, e))) for e in epochs
            },
        }

    # -- drawing ------------------------------------------------------------
    def _rotation(self) -> int:
        if self.backbone.startswith("fixed-"):
            return 0
        return self._draw_index % max(1, len(self.random_forests))

    def _backbone_mask(self, rotation: int) -> np.ndarray:
        if self.backbone.startswith("fixed-"):
            return self.det_forest
        return self.random_forests[rotation]

    def _get_plan(self, target_edges: int, rotation: int):
        key = (int(target_edges), self.backbone, int(rotation))
        if self._plan is not None and self._plan_key == key:
            return self._plan

        forced = self._backbone_mask(rotation).copy()
        forced |= self.mandatory
        forced_ids = np.flatnonzero(forced).astype(np.int64, copy=False)

        # Keep the complete forest in the cached plan. Below the connectivity
        # floor each draw trims it with that draw's RNG, so per-epoch
        # resparsification does not accidentally reuse one cached subset.
        trimmed = bool(forced_ids.size > target_edges)
        remaining = 0 if trimmed else int(target_edges - forced_ids.size)
        pool_order = (
            np.zeros(0, dtype=np.int64)
            if trimmed
            else self.order[~forced[self.order]]  # tree-locality order
        )
        p = cap_and_renormalize(self.pi[pool_order], remaining)

        self._plan = {
            "forced": forced_ids,
            "pool_order": pool_order,
            "p": p,
            "cum": np.cumsum(p),
            "remaining": remaining,
            "target_edges": int(target_edges),
            "trimmed": trimmed,
            "rotation": int(rotation),
        }
        self._plan_key = key
        return self._plan

    def draw(self, keep_ratio=None, num_edges=None, seed=None, advance: bool = True):
        """Draw one sparse graph. Returns a :class:`~scaffold.result.ScaffoldResult`.

        Successive calls give different graphs; pass ``seed`` to pin a specific
        one. With ``backbone="rotate-randst"`` the guaranteed forest also
        rotates, so even the always-present edges vary across epochs.
        """
        from ..result import ScaffoldResult

        self._require_fit()
        start = time.perf_counter()
        target = self._budget(keep_ratio, num_edges)
        rotation = self._rotation()
        plan = self._get_plan(target, rotation)

        draw_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(
            None
            if draw_seed is None
            else np.random.SeedSequence([int(draw_seed), int(self._draw_index)])
        )
        forced = plan["forced"]
        if plan["trimmed"]:
            drop_count = int(forced.size - target)
            forced = rng.permutation(forced)[drop_count:]
        chosen = np.unique(
            np.concatenate((forced, self._draw_pool(plan, rng)))
        )

        m = self.graph.num_edges
        if m and chosen.size != target:
            raise RuntimeError(
                "scaffold.sample budget invariant failed: selected "
                f"{chosen.size} edges, expected {target}"
            )

        mask = np.zeros(m, dtype=bool)
        mask[chosen] = True
        edge_index = self.graph.subgraph_edge_index(mask)
        edge_weight = (
            None if self.graph.edge_weight is None else self.graph.edge_weight[mask]
        )
        components = component_count(
            self.graph.num_nodes, edge_index[0], edge_index[1]
        )
        if not plan["trimmed"] and m and components != self.base_components:
            raise RuntimeError(
                "scaffold.sample connectivity invariant failed: "
                f"{components} components, expected {self.base_components}"
            )

        if advance:
            self._draw_index += 1
        return ScaffoldResult(
            graph=self.graph,
            method="sample",
            mask=mask,
            undirected_edge_index=edge_index,
            undirected_edge_weight=edge_weight,
            metadata={
                "backbone": self.backbone,
                "support_budget_mode": "full_then_random_trim",
                "rotation": int(rotation),
                "draw_index": int(self._draw_index),
                "backbone_edges": int(plan["forced"].size),
                "forced_edges": int(forced.size),
                "sampled_edges": int(plan["remaining"]),
                "target_edges": int(target),
                "selected_edges": int(chosen.size),
                "budget_trimmed": int(plan["forced"].size - forced.size),
                "components": int(components),
                "base_components": int(self.base_components),
                "delta_min": float(self.delta_min),
                "below_connectivity_floor": bool(plan["trimmed"]),
                "tree_count": int(self.tree_count),
                "aggregate_lambda": float(self.aggregate_lambda),
                "runtime": float(time.perf_counter() - start),
                "precompute_seconds": float(self.build_seconds),
            },
        )

    def _draw_pool(self, plan, rng) -> np.ndarray:
        """Systematic ``pi``-ps: exactly ``k`` edges with exact marginals."""
        remaining = plan["remaining"]
        if remaining <= 0:
            return np.zeros(0, dtype=np.int64)
        cum = plan["cum"]
        ticks = float(rng.random()) + np.arange(remaining, dtype=np.float64)
        pos = np.searchsorted(cum, ticks, side="left")
        pos = np.unique(np.minimum(pos, cum.size - 1))
        if pos.size < remaining:
            # Float-boundary degeneracy when many p_e == 1. Top up with the
            # highest-weight unselected pool entries so the budget stays exact.
            taken = np.zeros(cum.size, dtype=bool)
            taken[pos] = True
            spare = np.flatnonzero(~taken)
            extra = spare[np.argsort(-plan["p"][spare], kind="stable")]
            pos = np.concatenate((pos, extra[: remaining - pos.size]))
        return plan["pool_order"][pos]

    # -- persistence --------------------------------------------------------
    def save(self, path):
        """Save the precomputed artifact to a ``.npz`` file."""
        self._require_fit()
        forests = self.random_forests
        np.savez_compressed(
            path,
            version=np.int64(1),
            num_nodes=np.int64(self.graph.num_nodes),
            src=self.graph.src,
            dst=self.graph.dst,
            pi=self.pi,
            mandatory=self.mandatory,
            det_forest=self.det_forest,
            random_forests=(
                np.stack(forests) if forests else np.zeros((0, 0), dtype=bool)
            ),
            order=self.order,
            base_components=np.int64(self.base_components),
            delta_min=np.float64(self.delta_min),
            tree_count=np.int64(self.tree_count),
            aggregate_lambda=np.float64(self.aggregate_lambda),
            backbone=np.asarray(self.backbone),
        )
        return path

    @classmethod
    def load(cls, path, graph: Graph, **kwargs) -> ScaffoldSampler:
        """Load an artifact and verify it describes ``graph``."""
        with np.load(path, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
        if int(data["num_nodes"]) != graph.num_nodes or data["src"].size != graph.num_edges:
            raise ValueError("artifact does not match this graph (size mismatch)")
        if not (
            np.array_equal(data["src"], graph.src)
            and np.array_equal(data["dst"], graph.dst)
        ):
            raise ValueError(
                "artifact does not match this graph (different edge list). "
                "Rebuild it with scaffold.sample(graph)."
            )
        if "backbone" in data:
            kwargs.setdefault("backbone", str(data["backbone"].item()))
        sampler = cls(
            tree_count=int(data["tree_count"]),
            aggregate_lambda=float(data["aggregate_lambda"]),
            **kwargs,
        )
        sampler.graph = graph
        sampler.pi = data["pi"]
        sampler.mandatory = data["mandatory"].astype(bool)
        sampler.det_forest = data["det_forest"].astype(bool)
        forests = data["random_forests"]
        sampler.random_forests = (
            [row.astype(bool) for row in forests] if forests.size else []
        )
        sampler.order = data["order"]
        sampler.base_components = int(data["base_components"])
        sampler.delta_min = float(data["delta_min"])
        return sampler

    def _require_fit(self):
        if self.graph is None or self.pi is None:
            raise RuntimeError("call fit(graph) before using the sampler")


def run(
    graph: Graph,
    keep_ratio: Optional[float] = None,
    num_edges: Optional[int] = None,
    params: Optional[ScoreParams] = None,
    seed=None,
    tree_count: int = DEFAULT_TREE_COUNT,
    aggregate_lambda: float = DEFAULT_LAMBDA,
    backbone: str = "fixed-maxst",
    scheme: str = "systematic",
    weighted_paths: bool = False,
    workers=None,
    verbose: bool = False,
) -> ScaffoldSampler:
    """Fit and return a :class:`ScaffoldSampler` for ``graph``."""
    sampler = ScaffoldSampler(
        params=params,
        tree_count=tree_count,
        aggregate_lambda=aggregate_lambda,
        backbone=backbone,
        scheme=scheme,
        weighted_paths=weighted_paths,
        seed=seed,
        workers=workers,
        verbose=verbose,
    )
    return sampler.fit(graph)
