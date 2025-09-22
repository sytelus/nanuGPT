from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import random
from typing import Dict, Iterable, Iterator, List, Tuple


def _enumerate_forward_dags(N: int) -> Iterable[List[int]]:
    """
    Enumerate all labeled DAGs on vertices 0..N-1 where edges are only allowed
    from lower to higher indices (i -> j only if i < j), and where each vertex
    i < N-1 has a non-empty set of successors among {i+1, ..., N-1}.

    Representation: return a list `rows` of length N, where rows[i] is an
    integer bitmask over 0..N-1 indicating the outgoing neighbors of i.
    Because edges only go forward, rows[N-1] == 0.
    """
    rows = [0] * N

    def rec(i: int) -> Iterable[List[int]]:
        if i == N - 1:
            # All earlier rows chosen; sink row is 0 by construction.
            yield list(rows)
            return
        succs = list(range(i + 1, N))
        m = len(succs)
        # Non-empty subsets of successors:
        # mask runs from 1 to (1<<m)-1
        for mask in range(1, 1 << m):
            bitset = 0
            for k in range(m):
                if (mask >> k) & 1:
                    bitset |= (1 << succs[k])
            rows[i] = bitset
            yield from rec(i + 1)
        rows[i] = 0

    yield from rec(0)


def _compute_in_neighbors(rows: List[int]) -> List[List[int]]:
    N = len(rows)
    in_nbrs: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        x = rows[i]
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            in_nbrs[j].append(i)
            x ^= lsb
    return in_nbrs


def _compute_out_neighbors(rows: List[int]) -> List[List[int]]:
    N = len(rows)
    out_nbrs: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        x = rows[i]
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            out_nbrs[i].append(j)
            x ^= lsb
    return out_nbrs


def _min_distance_to_sink(rows: List[int]) -> List[int]:
    """
    Minimal distance (number of edges) to the unique sink (vertex N-1),
    exploiting the forward-only orientation (topological order).
    dist[N-1] = 0; for i < N-1, dist[i] = 1 + min(dist[j] for j in succ(i)).
    """
    N = len(rows)
    dist = [0] * N
    dist[N - 1] = 0
    for i in range(N - 2, -1, -1):
        x = rows[i]
        best = None
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            d = dist[j]
            best = d if best is None else min(best, d)
            x ^= lsb
        # non-empty successors guaranteed; best cannot be None
        dist[i] = 1 + (best if best is not None else 0)
    return dist


def _wl_partition(rows: List[int]) -> Tuple[List[int], List[List[int]]]:
    """
    Weisfeiler–Lehman (1-dim) refinement for directed graphs, starting from
    initial colors based on (min distance to sink, indegree, outdegree).
    Returns:
      - a list of stable color ids (0..K-1) in canonical order (not used directly),
      - an ordered list of color classes (each a list of vertex indices).
    The color ids are assigned deterministically by sorting the signature keys.
    """
    N = len(rows)
    in_n = _compute_in_neighbors(rows)
    out_n = _compute_out_neighbors(rows)
    indeg = [len(in_n[i]) for i in range(N)]
    outdeg = [len(out_n[i]) for i in range(N)]
    dist = _min_distance_to_sink(rows)

    # Initial color key for each vertex; map unique sorted keys to ids deterministically.
    base_keys = [(dist[i], indeg[i], outdeg[i]) for i in range(N)]
    uniq = sorted(set(base_keys))
    color = [uniq.index(base_keys[i]) for i in range(N)]

    while True:
        # Signature = (current color, multiset of out-neighbor colors, multiset of in-neighbor colors)
        sigs = []
        for i in range(N):
            ocols = tuple(sorted(color[j] for j in out_n[i]))
            icols = tuple(sorted(color[j] for j in in_n[i]))
            sigs.append((color[i], ocols, icols))
        uniq = sorted(set(sigs))
        new_color = [uniq.index(sigs[i]) for i in range(N)]
        if new_color == color:
            break
        color = new_color

    # Build groups (color classes) in the canonical order of color ids.
    groups_dict: Dict[int, List[int]] = {}
    for v, c in enumerate(color):
        groups_dict.setdefault(c, []).append(v)
    ordered_ids = sorted(groups_dict.keys())
    ordered_groups = [groups_dict[c] for c in ordered_ids]
    return ordered_ids, ordered_groups


def _canonical_code(rows: List[int]) -> int:
    """
    Compute an isomorphism-invariant canonical code of the DAG:
      1) WL refine to get stable color classes.
      2) Fix the sink (N-1) last (unique end vertex).
      3) Enumerate permutations **within** each WL color class, keeping classes
         in the canonical order, and pick the lexicographically smallest
         row-major adjacency bitstring (N*N bits).
    The resulting integer is a canonical form; equal iff graphs are isomorphic.
    """
    N = len(rows)
    # WL refinement (classes in canonical order)
    _, groups = _wl_partition(rows)

    # Remove the sink from its WL class and place it at the very end.
    sink = N - 1
    groups_fixed: List[List[int]] = []
    sink_class_idx = None
    for gi, g in enumerate(groups):
        if sink in g:
            sink_class_idx = gi
            members = [v for v in g if v != sink]
            if members:
                groups_fixed.append(members)
        else:
            groups_fixed.append(list(g))
    if sink_class_idx is None:
        raise RuntimeError("Sink not found in WL classes (should be impossible).")

    # Build adjacency matrix for fast lookup (row-major)
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        x = rows[i]
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            A[i][j] = 1
            x ^= lsb

    best_code: int | None = None

    # Cartesian product of all permutations **inside** each class
    group_perms: List[List[Tuple[int, ...]]] = [list(permutations(g)) for g in groups_fixed]
    for choice in product(*group_perms):
        order = [v for block in choice for v in block]
        order.append(sink)  # sink fixed last

        # Row-major N x N bitstring (diagonal is always 0 for DAGs)
        code = 0
        for ii in range(N):
            oi = order[ii]
            row = A[oi]
            for jj in range(N):
                oj = order[jj]
                code = (code << 1) | row[oj]

        if best_code is None or code < best_code:
            best_code = code

    assert best_code is not None
    return best_code


@dataclass
class DAGIsomorphismSampler:
    """
    Uniform sampler over isomorphism classes of DAGs on N > 1 vertices
    with exactly one end vertex (unique sink) and the property that every
    non-sink vertex has a directed path to the sink.
    The set of possible start vertices (sources) is unrestricted—i.e., any
    non-empty subset of the N-1 non-sink vertices can occur as sources.

    Construction:
      - Enumerates all DAGs in a fixed topological labeling (edges only i->j for i<j),
        enforcing non-empty out-neighborhood for each i < N-1. This ensures:
          * the last vertex is a sink,
          * every vertex has an outward path to the sink (by induction).
      - Computes a canonical form per graph using WL refinement + permutations
        inside WL color classes (sink fixed last).
      - Deduplicates by canonical code to obtain one representative per
        isomorphism class.
      - Sampling uniformly from that representative list yields uniform sampling
        over isomorphism classes.

    Complexity notes:
      - Enumeration count is ∏_{i=1}^{N-1} (2^i - 1) = (2^1 - 1)(2^2 - 1)...(2^{N-1} - 1),
        which grows fast but is small for N ≤ 6–7.
      - Canonicalization cost is typically modest because WL strongly refines
        classes; remaining symmetries are handled by block-internal permutations.

    Example sanity check:
      - For N = 5 this discovers exactly 164 isomorphism classes (as noted).
    """
    N: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.N <= 1:
            raise ValueError("N must be > 1.")
        self._rng = random.Random(self.seed)
        # Build the set of isomorphism classes and store one labeled representative per class.
        self._reps: List[List[int]] = []
        seen: Dict[int, List[int]] = {}

        for rows in _enumerate_forward_dags(self.N):
            code = _canonical_code(rows)
            if code not in seen:
                seen[code] = rows
                self._reps.append(rows)

        # Optional: shuffle representatives so that repeated runs with the same seed
        # don't always visit classes in the same internal order (sampling is uniform anyway).
        self._rng.shuffle(self._reps)

    @property
    def num_isomorphism_classes(self) -> int:
        """Number of unlabeled isomorphism classes discovered for this N."""
        return len(self._reps)

    def _rows_to_output(self, rows: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Convert bitmask rows to the requested output format:
          A list of (node_id, [sorted successor ids]) in topological order.
        """
        N = len(rows)
        out: List[Tuple[int, List[int]]] = []
        for i in range(N):
            succs: List[int] = []
            x = rows[i]
            while x:
                lsb = x & -x
                j = (lsb.bit_length() - 1)
                succs.append(j)
                x ^= lsb
            succs.sort()
            out.append((i, succs))
        return out

    def sample_one(self) -> List[Tuple[int, List[int]]]:
        """Sample one DAG uniformly from the isomorphism classes and return it in the requested format."""
        rows = self._rng.choice(self._reps)
        return self._rows_to_output(rows)

    def __iter__(self) -> Iterator[List[Tuple[int, List[int]]]]:
        """
        Infinite iterator: each iteration yields a DAG (in the requested format)
        sampled uniformly from the isomorphism classes.
        """
        while True:
            yield self.sample_one()



def build_graph_instructions(adj:List[Tuple[int, List[int]]])-> str:
    def _format_targets(target_nodes):
        problems = [f"Problem {n+1}" for n in target_nodes]
        if not problems:
            return ""
        if len(problems) == 1:
            return problems[0]
        if len(problems) == 2:
            return f"{problems[0]} as well as {problems[1]}"
        return f"{', '.join(problems[:-1])} as well as {problems[-1]}"

    lines = []
    for node, targets in adj:
        if targets:
            targets_text = _format_targets(targets)
            lines.append(
                f"- Output of Problem {node+1} should become input for {targets_text}."
            )
        else:
            lines.append(
                "- Output of Problem {} should become the final output of the "
                "combined problem and this output value should remain exactly same "
                "as it is in original Problem {}.".format(node+1, node+1)
            )

    return "\n".join(lines)

if __name__ == "__main__":
    # Example: N=5 (there are 164 isomorphism classes with the given constraints)
    sampler = DAGIsomorphismSampler(N=4, seed=123)

    print(f"N=4 -> #classes discovered: {sampler.num_isomorphism_classes} (expected 164)")

    it = iter(sampler)
    for k in range(10):
        dag = next(it)
        print(f"\nSample {k+1}:")
        print(dag)
        print(build_graph_instructions(dag))
        print("-" * 40)
