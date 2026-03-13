"""Potential-theory active cloaking overlay for graph environments.

Adapted from the cloaking experiments notebook (DeGiovanni & Guevara Vasquez,
2025, arXiv:2405.07961v2).  Given an RGG, computes a signal-visibility map
that models how well hints about the goal door "radiate" through the network.

In the **uncloaked** case, visibility decays naturally with graph distance
(the reference Dirichlet potential with goal=source).

In the **cloaked** case, a cloaking region Ω around the goal blocks signal
propagation via the DTN (Dirichlet-to-Neumann) operator.  Exterior nodes
see almost no information about the goal, forcing agents to explore blind
until they penetrate the cloak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CloakingOverlay:
    """Stores the computed signal-visibility map and diagnostic data."""

    signal_visibility: dict[int, float] = field(default_factory=dict)
    potential_ref: np.ndarray | None = None
    potential_cloaked: np.ndarray | None = None
    cloaking_metric: float = 0.0
    omega_nodes: list[int] = field(default_factory=list)
    partial_omega_nodes: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cloaking_overlay(
    node_positions: dict[int, tuple[float, float]],
    adjacency: dict[int, list[int]],
    goal_node_id: int,
    cloak_inner_radius: float = 0.12,
    cloak_outer_radius: float = 0.20,
    boundary_margin: float = 0.10,
) -> CloakingOverlay:
    """Build a cloaking overlay for a graph-world environment.

    Parameters
    ----------
    node_positions : {node_id: (x, y)} for every node in the graph.
    adjacency : {node_id: [neighbor_ids]} for every node.
    goal_node_id : The node that is the goal door.
    cloak_inner_radius : Radius of Ω (interior cloaking region) around goal.
    cloak_outer_radius : Radius of ∂Ω (boundary layer of cloak) around goal.
    boundary_margin : Nodes within this distance of the [0,1]² edge are
                      treated as potential-theory boundary nodes.
    """

    node_ids = sorted(node_positions.keys())
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    positions = np.zeros((n, 2))
    for nid in node_ids:
        positions[id_to_idx[nid]] = node_positions[nid]

    # --- build adjacency matrix ---
    rows, cols, data = [], [], []
    seen: set[tuple[int, int]] = set()
    for nid, nbrs in adjacency.items():
        ii = id_to_idx[nid]
        for nb in nbrs:
            jj = id_to_idx[nb]
            edge = (min(ii, jj), max(ii, jj))
            if edge not in seen:
                seen.add(edge)
                rows.extend([ii, jj])
                cols.extend([jj, ii])
                data.extend([1.0, 1.0])
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Laplacian L = D - A
    degrees = np.asarray(A.sum(axis=1)).ravel()
    L = (sp.diags(degrees) - A).tocsc()

    # --- boundary nodes: near edge of [0,1]² ---
    is_boundary = np.logical_or.reduce((
        positions[:, 0] < boundary_margin,
        positions[:, 0] > 1 - boundary_margin,
        positions[:, 1] < boundary_margin,
        positions[:, 1] > 1 - boundary_margin,
    ))
    B = set(np.nonzero(is_boundary)[0].tolist())

    goal_idx = id_to_idx[goal_node_id]
    B.discard(goal_idx)
    B_arr = np.array(sorted(B), dtype=int)
    V = np.arange(n)

    # --- reference potential field (uncloaked) ---
    # Dirichlet: potential=1 at goal, potential=0 at domain boundary
    B_with_goal = np.unique(np.append(B_arr, goal_idx))
    u_boundary = np.zeros(n)
    u_boundary[goal_idx] = 1.0

    u_ref = _solve_dirichlet(L, V, B_with_goal, u_boundary)
    u_ref_norm = _normalize(u_ref)

    # --- define cloaking region ---
    goal_pos = positions[goal_idx]
    dists = np.linalg.norm(positions - goal_pos[None, :], axis=1)

    omega = np.nonzero(dists <= cloak_inner_radius)[0]
    partial_omega = np.nonzero(
        (dists > cloak_inner_radius) & (dists <= cloak_outer_radius)
    )[0]

    # --- cloaked potential field ---
    if omega.size == 0 or partial_omega.size == 0:
        u_cloak_raw = u_ref.copy()
        cloak_metric = 0.0
    else:
        Lr = _build_cloaked_laplacian(L, V, omega, partial_omega)
        u_cloak_raw = _solve_dirichlet(Lr, V, B_with_goal, u_boundary,
                                       exclude=omega)
        u_cloak_raw[omega] = 0.0

        ext_idx = np.setdiff1d(V, np.concatenate([omega, partial_omega]))
        if ext_idx.size > 0:
            n_anom = np.linalg.norm(u_ref[ext_idx])
            n_cloak = np.linalg.norm(u_cloak_raw[ext_idx])
            cloak_metric = float(n_anom - n_cloak)
        else:
            cloak_metric = 0.0

    u_cloak_norm = _normalize(u_cloak_raw)

    # --- signal visibility map ---
    # Visibility = how much of the reference signal leaks through the cloak.
    # u_ref[node] is the "natural" signal strength; u_cloak[node] is what
    # survives cloaking.  Ratio gives per-node attenuation.
    # Override: Ω → 1.0 (inside the cloak, you see everything),
    #           ∂Ω → at least 0.5 (transition zone).
    ref_max = u_ref.max() if u_ref.max() > 1e-12 else 1.0
    signal_vis: dict[int, float] = {}
    omega_set = set(omega.tolist())
    partial_set = set(partial_omega.tolist())

    for nid in node_ids:
        idx = id_to_idx[nid]
        # Ratio of cloaked / reference potential (clipped to [0.05, 1.0])
        if u_ref[idx] > 1e-12:
            vis = float(np.clip(u_cloak_raw[idx] / u_ref[idx], 0.05, 1.0))
        else:
            vis = float(np.clip(u_cloak_raw[idx] / ref_max, 0.05, 1.0))

        if idx in omega_set:
            signal_vis[nid] = 1.0
        elif idx in partial_set:
            signal_vis[nid] = max(vis, 0.5)
        else:
            signal_vis[nid] = vis

    return CloakingOverlay(
        signal_visibility=signal_vis,
        potential_ref=u_ref_norm,
        potential_cloaked=u_cloak_norm,
        cloaking_metric=cloak_metric,
        omega_nodes=[node_ids[i] for i in omega],
        partial_omega_nodes=[node_ids[i] for i in partial_omega],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solve_dirichlet(
    L: sp.spmatrix,
    V: np.ndarray,
    B: np.ndarray,
    u_boundary: np.ndarray,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the Dirichlet problem L·u = 0 with given boundary values."""
    Vm = np.setdiff1d(V, B)
    if exclude is not None and exclude.size > 0:
        Vm = np.setdiff1d(Vm, exclude)
    if Vm.size == 0:
        return u_boundary.copy()

    L_sub = L[Vm[:, None], Vm].tocsc()
    L_bnd = L[Vm[:, None], B].toarray()
    rhs = -L_bnd @ u_boundary[B]

    u = np.zeros(len(V))
    u[B] = u_boundary[B]
    try:
        u[Vm] = spla.spsolve(L_sub, rhs)
    except Exception:
        u[Vm] = 0.0
    return u


def _build_cloaked_laplacian(
    L: sp.spmatrix,
    V: np.ndarray,
    omega: np.ndarray,
    partial_omega: np.ndarray,
) -> sp.spmatrix:
    """Build the modified Laplacian Lr that hides the cloaking region.

    The DTN (Dirichlet-to-Neumann) operator replaces direct connections
    through Ω with the Schur complement, making the interior invisible
    from the exterior.
    """
    # Schur complement: DTN = L_∂∂ - L_∂Ω · L_ΩΩ⁻¹ · L_Ω∂
    L_dd = L[partial_omega[:, None], partial_omega].toarray()
    L_dO = L[partial_omega[:, None], omega].toarray()
    L_Od = L[omega[:, None], partial_omega].toarray()
    L_OO = L[omega[:, None], omega].toarray()

    try:
        middle = np.linalg.solve(L_OO, L_Od)
        DTN = L_dd - L_dO @ middle
    except np.linalg.LinAlgError:
        DTN = L_dd

    # Replace connections through Ω with DTN at ∂Ω
    Lr = L.tolil().copy()
    for i_idx, i in enumerate(partial_omega):
        for j_idx, j in enumerate(partial_omega):
            Lr[i, j] = DTN[i_idx, j_idx]

    # Disconnect Ω from the rest of the network
    others = np.setdiff1d(V, omega)
    for o in others:
        for w in omega:
            Lr[o, w] = 0.0
            Lr[w, o] = 0.0
    for w in omega:
        Lr[w, w] = 0.0

    return Lr.tocsc()


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-12:
        return (arr - lo) / (hi - lo)
    return np.ones_like(arr)
