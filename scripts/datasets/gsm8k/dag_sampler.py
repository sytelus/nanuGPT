import heapq
import random
from functools import lru_cache
from itertools import permutations
from collections import deque

# ---------- Core helpers (enumeration, checks, canonicalization) ----------

def _enumerate_forward_dags(N):
    """
    Generate all DAGs on {0,...,N-1} whose edges go only i->j for i<j.
    Every DAG has at least one topological order, so this covers at least
    one labeled representative of every unlabeled DAG.
    Yields: tuple of edges (u,v) with 0<=u<v<N.
    """
    edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    M = len(edges)
    for mask in range(1 << M):
        E, mm = [], mask
        for idx in range(M):
            if mm & 1:
                E.append(edges[idx])
            mm >>= 1
        yield tuple(E)

def _indeg_outdeg(N, edges):
    indeg = [0]*N
    outdeg = [0]*N
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
    return indeg, outdeg

def _everyone_reaches_sink(N, edges, sink):
    """
    Check that every vertex reaches 'sink'. (Redundant if 'sink' is the only sink,
    but kept for safety and clarity.)
    """
    rev = [[] for _ in range(N)]
    for u, v in edges:
        rev[v].append(u)
    seen = [False]*N
    seen[sink] = True
    dq = deque([sink])
    while dq:
        x = dq.popleft()
        for y in rev[x]:
            if not seen[y]:
                seen[y] = True
                dq.append(y)
    return all(seen)

def _is_one_sink_dag(N, edges):
    """
    Our class: exactly one sink and every vertex reaches it.
    """
    indeg, outdeg = _indeg_outdeg(N, edges)
    sinks = [i for i in range(N) if outdeg[i] == 0]
    if len(sinks) != 1:
        return False
    return _everyone_reaches_sink(N, edges, sinks[0])

def _canonical_form_with_sink_last(N, edges):
    """
    Compute a canonical labeling for 'edges' under vertex relabelings,
    restricting to permutations that map the unique sink to position N-1.
    Returns:
        key (str): lexicographically minimal adjacency bitstring (i!=j order).
        edges_can (tuple[(u,v)]): edges under the canonical permutation, with sink at N-1.
    Complexity: O((N-1)! * N^2) per graph (much faster than N! thanks to fixing the sink).
    """
    # Identify the current sink (unique, by our filter).
    _, outdeg = _indeg_outdeg(N, edges)
    sink = next(i for i in range(N) if outdeg[i] == 0)

    E = set(edges)
    best_key = None
    best_perm = None

    # Permute only the N-1 non-sink vertices; sink -> N-1.
    others = [v for v in range(N) if v != sink]
    for idx_order, p_others in enumerate(permutations(others)):
        p = [None]*N
        # Map the non-sink vertices to 0..N-2 in order of p_others; sink -> N-1.
        for new_label, old_v in enumerate(p_others):
            p[old_v] = new_label
        p[sink] = N-1

        # Build adjacency bitstring in row-major (i!=j) for the permuted graph.
        bits = []
        # Fast membership test: apply permutation and check edges
        # (u,v) maps to (p[u], p[v]).
        Pe = {(p[u], p[v]) for (u, v) in E}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                bits.append('1' if (i, j) in Pe else '0')
        s = ''.join(bits)

        if best_key is None or s < best_key:
            best_key = s
            best_perm = p

    # Apply best_perm to produce canonical edge list (tuple of (u,v))
    edges_can = tuple(sorted((best_perm[u], best_perm[v]) for (u, v) in E))
    return best_key, edges_can

# ---------- Build & cache all isomorphism-class representatives for a given N ----------

@lru_cache(maxsize=None)
def _representatives_for_N(N):
    """
    Return a tuple of canonical edge-sets (each a tuple of (u,v), 0-based with sink at N-1),
    one per isomorphism class of one-sink DAGs on N vertices.
    This cache is built once per N; subsequent calls are O(1).
    """
    if N <= 1:
        raise ValueError("N must be > 1")

    reps = []
    seen_keys = set()
    for edges in _enumerate_forward_dags(N):
        if not _is_one_sink_dag(N, edges):
            continue
        key, edges_can = _canonical_form_with_sink_last(N, edges)
        if key not in seen_keys:
            seen_keys.add(key)
            reps.append(edges_can)

    # Optional: sort by (|E|, lex) to get a stable order (not required for uniform sampling)
    reps.sort(key=lambda es: (len(es), es))
    return tuple(reps)

# ---------- Public sampler ----------

def sample_one_sink_dag_class(N, seed=None):
    """
    Uniformly sample from *isomorphism classes* of DAGs on N>1 vertices
    that have exactly one sink (global sink) and where every other vertex
    has a directed path to the sink. Sources may be any subset of {1..N-1}.

    Output format:
        A dict mapping node IDs 1..N to a sorted list of immediate successors.
        For example: {1: [2,5], 2: [5], 3: [5], 4: [5], 5: []}

    Uniformity guarantee:
        We build one canonical representative per isomorphism class and then
        choose uniformly among these representatives. Thus the distribution is
        uniform over classes (not over labeled graphs), which is exactly the
        requirement here.

    Randomness:
        Pass `seed` to make the draw reproducible for testing; otherwise uses
        the global RNG.

    Complexity:
        • First call for a given N: exponential preprocessing to enumerate
          and canonicalize all classes (feasible for small N, e.g., N≤6).
        • Subsequent calls: O(N+E) time and memory (pick a representative and
          materialize its adjacency).

    Known alternatives:
        • Burnside/Polya “orbit samplers” (a.k.a. Burnside process) can produce
          uniform unlabeled samples without enumerating all classes, but they
          require the ability to sample uniformly from Fix(π) for π∈S_N, which
          is nontrivial for DAGs with structural constraints. This simple method
          is the most maintainable exact approach for small N.
    """
    reps = _representatives_for_N(N)
    if not reps:
        raise RuntimeError("No representatives found; check implementation.")
    rng = random.Random(seed) if seed is not None else random
    edges = rng.choice(reps)  # edges are 0-based with sink at N-1 (canonical labeling)

    # Build 1-based adjacency dict
    adj = {i+1: [] for i in range(N)}
    for u, v in edges:
        adj[u+1].append(v+1)
    for i in range(1, N+1):
        adj[i].sort()
    return adj


def sample_prompt(N, seed=None):
    """Sample a one-sink DAG and describe how to chain the problems."""
    adj = sample_one_sink_dag_class(N, seed=seed)

    indegree = {node: 0 for node in adj}
    for targets in adj.values():
        for node in targets:
            indegree[node] += 1

    zero_heap = [node for node, deg in indegree.items() if deg == 0]
    heapq.heapify(zero_heap)
    topo_order = []
    while zero_heap:
        node = heapq.heappop(zero_heap)
        topo_order.append(node)
        for succ in adj[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(zero_heap, succ)

    if len(topo_order) != len(adj):
        raise ValueError("Sampled graph is not acyclic; expected a DAG.")

    def _format_targets(target_nodes):
        problems = [f"Problem {n}" for n in target_nodes]
        if not problems:
            return ""
        if len(problems) == 1:
            return problems[0]
        if len(problems) == 2:
            return f"{problems[0]} as well as {problems[1]}"
        return f"{', '.join(problems[:-1])} as well as {problems[-1]}"

    lines = []
    for node in topo_order:
        targets = adj[node]
        if targets:
            targets_text = _format_targets(targets)
            lines.append(
                f"- Output of Problem {node} should become input for {targets_text}."
            )
        else:
            lines.append(
                "- Output of Problem {} should become the final output of the "
                "combined problem and this output value should remain exactly same "
                "as it is in original Problem {}.".format(node, node)
            )

    return "\n".join(lines)


if __name__ == "__main__":
    g = sample_one_sink_dag_class(5)
    print(g)

    g1 = sample_one_sink_dag_class(5, seed=12345)
    g2 = sample_one_sink_dag_class(5, seed=12345)
    assert g1 == g2
