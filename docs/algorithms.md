# The SCAFFOLD objective, and four ways to evaluate it

## The idea

A sparsifier has to answer one question: given a budget of `M` edges, which
`M`? Most methods answer it by scoring edges in isolation — degree, similarity,
effective resistance — and keeping the top ones. That has a well-known failure
mode: the top-scoring edges can leave the graph in pieces, or funnel every path
through a handful of survivors.

SCAFFOLD answers it differently, in two stages.

**Stage 1 — guarantee the structure.** Start from a spanning forest of `G`. A
forest with `n − c` edges has, by construction, exactly the connected components
of `G`. Nothing else is guaranteed for free, and this one is cheap.

**Stage 2 — spend the rest on what hurts most.** For each edge `e = (u, v)` not
yet in the support graph `H`, ask what its absence costs:

- **dilation** — how much longer the detour is than the edge itself:

  ```
  dil[e] = dist_H(u, v) / w(e)
  ```

- **edge congestion** — how many other missing edges are routed over the same
  support edges, aggregated as a normalized `p`-norm over the detour's edges:

  ```
  eConPath[e] = ( (1/|P|) · Σ_{f ∈ P(e)} (eCon[f] + ε)^p )^(1/p)
  ```

- **node congestion** — the same over the detour's *interior* nodes, order `q`.

Each term is divided by its maximum over the candidate set — which makes the
three commensurable — and raised to a tunable exponent:

```
score(e) = ((dil      + ε) / (D_max + ε))^alpha
         · ((eConPath + ε) / (E_max + ε))^beta_edge
         · ((vConPath + ε) / (V_max + ε))^beta_node
```

Higher is more urgent. `alpha` rewards edges whose detour is long; the betas
reward edges whose detour crosses already-overloaded parts of the support, on
the grounds that adding such an edge relieves a bottleneck rather than just
shortening one path.

Edges whose endpoints lie in *different* components of `H` have no detour at
all. They get `dil = score = ∞` and are reported as **mandatory**: adding them
is the only way to preserve connectivity, so they are always taken first.

---

## Why the naive evaluation is too slow

Evaluating that objective literally means: for each of `m − |E(H)|` candidates,
run a shortest-path search in `H`, walk the resulting path to accumulate
congestion counters, and then — because adding an edge changes the paths of
every nearby candidate — do it all again after each insertion.

For a budget of `M` edges that is `O(M · m · (n + m))`. On Cora (2,708 nodes,
5,278 edges) it takes minutes. On anything larger it is hopeless.

The four variants are four answers to that problem.

---

## `scaffold.greedy` — the reference

Does exactly what the definition says. One shortest-path search per distinct
source, full rescoring after every insertion, no caching or sampling.

```python
result = scaffold.greedy(G, keep_ratio=0.2, batch_size=1)
```

Rescoring is the point, not an inefficiency: once an edge lands, nearby
candidates get shorter detours and their congestion shifts. A stale ranking is
a different algorithm.

`batch_size` commits several edges per rescoring round, trading fidelity for a
proportional speed-up — the expensive part is the rescoring, not the insertion.

**Use it for:** validating the other three, toy graphs, figures.

---

## `scaffold.heap` — lazy greedy

Almost all of Greedy's work is wasted. Adding one edge changes the detours of
candidates routed *near* it and leaves the rest of the graph untouched.

So: keep scores in a max-heap and let them go stale. Each round, pop the top
`k`, rescore only those against the current support, and commit the best. After
an insertion, mark dirty only the candidates whose recorded detour touched the
affected edges or nodes (plus a `local_radius`-hop neighbourhood) and push them
back with fresh scores. Everything else keeps its stale key.

```python
result = scaffold.heap(G, keep_ratio=0.2,
                       top_k=16,          # candidates rescored per round
                       add_per_round=1,   # edges committed per round
                       local_radius=1,    # invalidation radius
                       dirty_limit=64,    # default cap (0 = unlimited)
                       clusters=None)     # cluster-local heaps
```

### A different score form

Heap defaults to the **max**-congestion form:

```
score(e) = dil(e) · (1 + beta_edge · log1p(maxECon) + beta_node · log1p(maxVCon))
```

rather than the normalized `p`-norm product. The reason is structural: the
`p`-norm form divides by `D_max`, `E_max`, `V_max`, which shift every round —
so a stale key computed against last round's maxima is not comparable to a
fresh one. The max form has no global normalizer, so stale keys stay meaningful,
which is the entire premise of the lazy heap.

Pass `score_form="product"` for the shared `p`-norm objective if you want
Heap and Greedy to be optimizing literally the same thing.

**Use it for:** mid-sized graphs where you want to stay close to Greedy.

---

## `scaffold.fast` — one tree-prefix pass

The variant to use, and the one Algorithm 1 of the paper describes.

The observation: while the support graph is still the backbone *forest* `F`,
every term of the objective is a path aggregate on a tree — and every path
aggregate on a tree is a difference of two root-prefix sums.

### The kernel

1. **Root the forest** and build a binary-lifting ancestor table:
   `O(n log n)` once.

2. **LCA** of every candidate's endpoints, `O(log n)` each. Candidates in
   different components get `-1`, which is how mandatory edges are detected.

3. **Congestion for all `|I|` paths at once**, with no path enumeration:

   ```
   eCon[u] += 1 ;  eCon[v] += 1 ;  eCon[lca] -= 2      for every candidate
   eCon[parent[x]] += eCon[x]                          folding subtrees upward
   ```

   `eCon[x]` then holds the number of candidate detours using the tree edge
   `(x, parent[x])`. Node congestion follows from it:

   ```
   vCon[x] = eCon[x] + (x is an LCA count) − (x is an endpoint count)
   ```

   because a detour touches `x` in exactly one of two disjoint ways — via the
   edge to its parent, or as the LCA — and endpoints are not interior nodes.

4. **Root-prefix sums** of `(eCon + ε)^p` and `(vCon + ε)^q`, so every path
   aggregate becomes `S[u] + S[v] − 2·S[lca]`, in `O(1)`.

Total: `O(m log n + n)`. No shortest-path search anywhere.

The resulting scores are **identical** to what `scaffold.greedy` computes in its
first round — in round 1 the support *is* the tree, so `d_H = d_T`. That
equivalence is a test, not a claim: `tests/test_scoring.py` checks
`tree_scores` against `path_scores` across five backbones, four parameter sets,
weighted and unweighted graphs, and disconnected inputs, to `rtol=1e-9`.

### Selection

Because growth never rebuilds the tree index, **the score is static**. The
sampled round loop was only ever approximating a ranking that can be taken
exactly in one pass.

```python
scaffold.fast(G, keep_ratio=0.2, selection="topk")    # default
scaffold.fast(G, keep_ratio=0.2, selection="rounds")  # paper Algorithm 1
```

- **`"topk"`** — one global top-`M` over the static scores. Mandatory edges
  carry `+∞` and sort first, which is the priority connectivity demands.
- **`"rounds"`** — per-cluster degree-weighted sampling, top-`r` per cluster per
  round, degrees updated as edges land. Produces a more spatially spread
  selection at equal budget. It cannot beat `topk` on the objective it is
  ranking by (both read the same scores), but on a highly symmetric graph the
  spread can matter: with large blocks of exactly-tied scores, `topk` resolves
  the tie by edge id, which clusters its picks. The grid demo shows this
  clearly.

### Reusing one pass across budgets

```python
result = scaffold.fast(G, keep_ratio=0.99, return_scores=True)
scores = result.metadata["scores"]     # one array, every budget
```

**Use it for:** everything, unless you have a reason not to.

---

## `scaffold.sample` — weights, not a subgraph

The other three answer "which edges should I keep?". This one answers "how
important is each edge?" and hands you the numbers.

That difference matters for GNN training. Fixing one sparse graph for all epochs
discards the rest of the graph permanently. Re-running a greedy sparsifier every
epoch is far too slow. SCAFFOLD-Sample does the structural work once and reduces
the per-epoch cost to one uniform draw plus one `searchsorted`.

### Precomputation

1. Build a fixed backbone `F₀` (MaxST by default, or seeded `randst`) and `R`
   random spanning forests.
2. Score every non-tree edge exactly against each forest with the tree kernel —
   `O(m log n + n)` per forest.
3. Aggregate into a single **ratio-independent** weight:

   ```
   pi[e] = max(ε,  freq[e]  +  lambda · mean_normalized_score[e])
   ```

   `freq[e]` is the fraction of random forests containing `e` — structural
   indispensability. The score term is how badly the forests serve `e` — repair
   value. One artifact serves every target ratio.

4. Order edges by the DFS preorder of their LCA in `F₀`. This **tree-locality**
   ordering is what makes systematic sampling spread its picks across the graph
   instead of clumping them.

### Per-epoch draw

- At or above the connectivity floor, a complete spanning forest is unioned in,
  so every draw has exactly the components of `G` with probability 1 — not in
  expectation. Below the floor, the forest cannot fit and is uniformly
  budget-trimmed afresh on every draw.
- The remaining `k` edges come from **systematic π-ps sampling** over the
  tree-locality order: exactly `k` edges, exact marginal inclusion
  probabilities, in one `searchsorted`.

```python
scores = scaffold.sample(G, seed=0, tree_count=8, aggregate_lambda=1.0)

scores.scores                                    # pi, one per undirected edge
scores.edge_index, scores.edge_weight            # symmetric, GNN-ready
scores.inclusion_probabilities(keep_ratio=0.2)   # sums to the budget, all <= 1
scores.draw(keep_ratio=0.2)                      # one concrete sparse graph
scores.sampler.coverage(keep_ratio=0.2)          # what N epochs will cover
```

### The deterministic core

`cap_and_renormalize` scales `pi` so the probabilities sum to the budget with
every `p ≤ 1`. Edges that end up capped at exactly 1 form the *deterministic
core* — present in every single draw. Backbone edges are always in it. Checking
its size tells you how much variation per-epoch resampling can actually deliver:

```python
p = scores.inclusion_probabilities(keep_ratio=0.2)
core = (p >= 1 - 1e-12).sum()      # edges in every draw
```

With `backbone="rotate-randst"` the guaranteed forest rotates between draws,
which shrinks the core and increases the variety of the views.

**Use it for:** GNN training with per-epoch resparsification; any setting where
you want edge importances rather than one fixed subgraph.

---

## Choosing

```
Do you want one fixed graph, or a fresh one per epoch?
│
├── fresh per epoch ─────────────────────────► scaffold.sample
│
└── one fixed graph
    │
    ├── m > ~10^5 edges ─────────────────────► scaffold.fast
    ├── m < ~10^5 and you want max fidelity ─► scaffold.heap
    └── validating / a figure / a toy graph ─► scaffold.greedy
```

## Complexity

For `n` nodes, `m` edges, budget `M`, heap width `k`, forest count `R`:

| | scoring | selection | total |
|---|---|---|---|
| `greedy` | `O(m(n+m))` per round | argmax | `O(M·m·(n+m))` |
| `heap`  | `O(k(n+m))` per round | heap pop | `~O(M·k·(n+m))` |
| `fast`  | `O(m log n + n)` **once** | one top-k | `O(m log n + n)` |
| `sample` | `O(R·(m log n + n))` **once** | `O(m)` per draw | precompute + `O(m)`/epoch |

---

## Measured behaviour

Quality here is measured on the **result**, with `path_scores` against the
final sparsified subgraph — not against the backbone the score was computed
from. *mean dil* is the average detour length over the dropped edges, *max
cong* the worst traffic concentration on a surviving edge. Lower is better for
both. Single-threaded, numba enabled, `scaffold` 0.1.0.

| graph | n | m | keep | method | time (ms) | mean dil | max cong |
|---|---|---|---|---|---|---|---|
| grid 24×24 | 576 | 1104 | 0.64 | `greedy` | 1,640 | 3.83 | 3.9 |
| | | | | `heap` | 442 | 3.79 | 5.7 |
| | | | | `fast (topk)` | **0.4** | 12.06 | 26.3 |
| | | | | `fast (rounds)` | 2.6 | 10.97 | 25.9 |
| | | | | `sample` | 3.7 | **3.73** | 5.1 |
| geometric | 600 | 2932 | 0.32 | `greedy` | 9,235 | 2.40 | 14.6 |
| | | | | `heap` | 1,268 | **2.30** | 17.5 |
| | | | | `fast (topk)` | **0.8** | 2.68 | 42.0 |
| | | | | `fast (rounds)` | 8.0 | 2.47 | 29.0 |
| | | | | `sample` | 6.4 | 2.68 | **23.2** |
| Barabási–Albert | 600 | 1791 | 0.45 | `greedy` | 3,004 | 3.66 | **36.9** |
| | | | | `heap` | 591 | **3.63** | 141.7 |
| | | | | `fast (topk)` | **0.6** | 3.82 | 107.1 |
| | | | | `fast (rounds)` | 4.4 | 3.77 | 120.2 |
| | | | | `sample` | 4.8 | 3.89 | 177.9 |
| SBM (4 blocks) | 600 | 3262 | 0.30 | `greedy` | 16,354 | **5.13** | **37.1** |
| | | | | `heap` | 1,578 | 5.19 | 78.1 |
| | | | | `fast (topk)` | **0.8** | 6.92 | 697.1 |
| | | | | `fast (rounds)` | 9.2 | 5.61 | 141.8 |
| | | | | `sample` | 7.1 | 5.30 | 82.3 |

Reproduce with `python benchmarks/bench_methods.py`.

### What this says

**`fast` is 1,000–20,000× faster and usually within 5–35% of Greedy on
dilation.** On the geometric and Barabási–Albert graphs the gap is 12% and 4%.
That is the trade the method is designed to make.

**Its weakness is congestion, not dilation.** Because the score is static and
never sees the edges already added, `topk` concentrates its picks — worst on
SBM, where max congestion is 697 against Greedy's 37. `selection="rounds"` cuts
that to 142 for about 10× the time, and `sample` to 82.

**The grid is the pathological case, deliberately.** A lattice produces large
blocks of exactly-tied scores, so `topk` fills a few neighbourhoods and leaves
the rest alone: 12.06 against Greedy's 3.83. Symmetric synthetic graphs are the
worst case for one-shot scoring; real graphs are not this symmetric.

**`sample` is the quality surprise.** It matches or beats Greedy's dilation on
the grid (3.73 vs 3.83) at 400× the speed, because tree-locality ordering
spreads its picks by construction — precisely what `topk` fails to do. If you
want one fixed graph and quality matters more than the last millisecond, a
single `sample` draw is a strong option.

**`heap` needs `dirty_limit`.** With unlimited invalidation, one insertion near
a hub dirties most of the candidate set and Heap becomes slower than Greedy
(4,757 ms vs 3,004 ms on Barabási–Albert). The default `dirty_limit=64` brings
that to 591 ms with mean dilation 3.627 vs 3.630 — an 8× speed-up at no
measurable quality cost.
