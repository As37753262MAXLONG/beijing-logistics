"""
solver.py
CVRP solver and route optimization models (Paper Section 3.3 / 3.4).

Multimodal modeling idea:
  In M1's cost matrix, each depot→customer arc is examined:
  If there exists a hub h such that cost(depot→h→customer) * HUB_DISCOUNT
  < cost(depot→customer), the arc is replaced by the discounted hub path,
  so Clarke-Wright naturally prefers hub-routed paths.

Models:
  M1: Dynamic multimodal (Clarke-Wright + hub discount)
  M2: Static single-modal (off_peak/normal planning + real-scenario costing)
  M3: Nearest-neighbor heuristic (greedy, no discount, no dynamic awareness)
  M4: Hybrid drone + vehicle (implemented in app.py compute_all_models)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from network_beijing import (
    DEPOT_ID, CUSTOMER_IDS, DEMANDS, HUB_IDS,
    VEHICLE_CAPACITY, ALL_NODES
)

# Hub transfer discount (mimics rail / dedicated freight corridor advantage)
HUB_TRANSFER_DISCOUNT = 0.65   # 35% discount via hub


# ── Multimodal effective cost matrix ─────────────────────────────
def build_multimodal_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    """
    For each direct arc i → j, check whether routing via any hub h
    with the discount factor yields a lower cost. If so, replace.
    """
    C = cost_matrix.copy()
    n = len(ALL_NODES)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = C[i][j]
            best_via_hub = min(
                (C[i][h] + C[h][j]) * HUB_TRANSFER_DISCOUNT
                for h in HUB_IDS
                if h != i and h != j
            )
            if best_via_hub < direct:
                C[i][j] = best_via_hub
    return C


# ── Data structures ──────────────────────────────────────────────
@dataclass
class Route:
    nodes: List[int] = field(default_factory=list)
    load: int = 0
    cost: float = 0.0

    def total_nodes(self) -> List[int]:
        return [DEPOT_ID] + self.nodes + [DEPOT_ID]


@dataclass
class CVRPSolution:
    routes: List[Route]
    total_cost: float
    scenario: Tuple[str, str]
    n_vehicles: int
    vehicle_utilization: List[float]
    model_name: str = "M1"

    def summary(self) -> str:
        lines = [
            f"[{self.model_name}] Scenario: {self.scenario[0]} | {self.scenario[1]}",
            f"  Vehicles : {self.n_vehicles}",
            f"  Total cost: {self.total_cost:.2f} CNY",
        ]
        for k, r in enumerate(self.routes):
            util = self.vehicle_utilization[k] * 100
            lines.append(
                f"  Route {k+1}: {r.total_nodes()}  "
                f"load={r.load}/{VEHICLE_CAPACITY} ({util:.0f}%)  "
                f"cost={r.cost:.2f}CNY"
            )
        return "\n".join(lines)


def _route_cost(node_sequence: List[int], cost_matrix: np.ndarray) -> float:
    full = [DEPOT_ID] + node_sequence + [DEPOT_ID]
    return sum(cost_matrix[full[i]][full[i + 1]] for i in range(len(full) - 1))


# ── Clarke-Wright Savings Algorithm ──────────────────────────────
def clarke_wright_savings(cost_matrix: np.ndarray,
                          scenario: Tuple[str, str],
                          model_name: str = "M1") -> CVRPSolution:
    """
    Clarke-Wright savings algorithm.
    cost_matrix is the effective cost matrix prepared by the caller.
    """
    C = cost_matrix

    # Init: each customer as its own route
    routes: Dict[int, Route] = {}
    for c in CUSTOMER_IDS:
        routes[c] = Route(nodes=[c], load=DEMANDS[c],
                          cost=_route_cost([c], C))

    # Compute savings s(i,j) = c(0,i) + c(j,0) - c(i,j)
    savings = []
    for i in CUSTOMER_IDS:
        for j in CUSTOMER_IDS:
            if i != j:
                s = C[DEPOT_ID][i] + C[j][DEPOT_ID] - C[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])

    node_to_route: Dict[int, int] = {c: c for c in CUSTOMER_IDS}

    for s_val, i, j in savings:
        if s_val <= 0:
            break
        ri_id = node_to_route.get(i)
        rj_id = node_to_route.get(j)
        if ri_id is None or rj_id is None or ri_id == rj_id:
            continue
        route_i = routes[ri_id]
        route_j = routes[rj_id]
        if route_i.nodes[-1] != i or route_j.nodes[0] != j:
            continue
        if route_i.load + route_j.load > VEHICLE_CAPACITY:
            continue
        merged = route_i.nodes + route_j.nodes
        routes[ri_id] = Route(nodes=merged,
                              load=route_i.load + route_j.load,
                              cost=_route_cost(merged, C))
        del routes[rj_id]
        for node in route_j.nodes:
            node_to_route[node] = ri_id

    final_routes = list(routes.values())
    return CVRPSolution(
        routes=final_routes,
        total_cost=sum(r.cost for r in final_routes),
        scenario=scenario,
        n_vehicles=len(final_routes),
        vehicle_utilization=[r.load / VEHICLE_CAPACITY for r in final_routes],
        model_name=model_name,
    )


# ── M1: Dynamic Multimodal ───────────────────────────────────────
def dynamic_multimodal(cost_matrix: np.ndarray,
                       scenario: Tuple[str, str]) -> CVRPSolution:
    """
    M1: Dynamic multimodal.
    - Uses the current scenario cost matrix (time-aware)
    - Applies hub-transfer discount (multimodal advantage)
    """
    C_mm = build_multimodal_cost_matrix(cost_matrix)
    return clarke_wright_savings(C_mm, scenario, model_name="M1-Dynamic-MM")


# ── M2: Static single-modal baseline ─────────────────────────────
def static_single_modal_baseline(cost_matrix: np.ndarray,
                                 scenario: Tuple[str, str]) -> CVRPSolution:
    """
    M2: Static single-modal baseline.
    - Route planning always uses off_peak/normal (scenario-blind)
    - No multimodal discount (road-only)
    - Re-priced at real scenario cost → reveals static planning loss
    """
    from traffic import build_cost_matrix
    from network_beijing import build_distance_matrix
    static_C = build_cost_matrix(build_distance_matrix(), "off_peak", "normal")
    sol = clarke_wright_savings(static_C, ("off_peak", "normal"),
                                model_name="M2-Static")
    for r in sol.routes:
        r.cost = _route_cost(r.nodes, cost_matrix)
    sol.total_cost = sum(r.cost for r in sol.routes)
    sol.scenario = scenario
    sol.vehicle_utilization = [r.load / VEHICLE_CAPACITY for r in sol.routes]
    return sol


# ── M3: Nearest-Neighbor heuristic baseline ──────────────────────
def heuristic_nearest_neighbor(cost_matrix: np.ndarray,
                                scenario: Tuple[str, str]) -> CVRPSolution:
    """
    M3: Nearest-neighbor heuristic.
    Greedy, no dynamic awareness, no multimodal.
    """
    C = cost_matrix
    unvisited = set(CUSTOMER_IDS)
    routes = []
    while unvisited:
        route_nodes, load, current = [], 0, DEPOT_ID
        while unvisited:
            candidates = [(C[current][j], j) for j in unvisited
                          if load + DEMANDS[j] <= VEHICLE_CAPACITY]
            if not candidates:
                break
            _, nearest = min(candidates)
            route_nodes.append(nearest)
            load += DEMANDS[nearest]
            unvisited.discard(nearest)
            current = nearest
        if route_nodes:
            routes.append(Route(nodes=route_nodes, load=load,
                                cost=_route_cost(route_nodes, C)))

    return CVRPSolution(
        routes=routes,
        total_cost=sum(r.cost for r in routes),
        scenario=scenario,
        n_vehicles=len(routes),
        vehicle_utilization=[r.load / VEHICLE_CAPACITY for r in routes],
        model_name="M3-NearestNeighbor",
    )


# ── M4: Hybrid drone + vehicle ───────────────────────────────────
def hybrid_drone_vehicle(cost_matrix: np.ndarray,
                          scenario: tuple) -> dict:
    """
    M4: Hybrid delivery.
    - Customers passing payload + range + cost dominance use drones.
    - Remaining customers use M1 dynamic multimodal CVRP.
    """
    from drone import assign_drone_or_vehicle
    import network_beijing as nb

    assignment    = assign_drone_or_vehicle(cost_matrix, scenario)
    drone_dict    = assignment['drone']
    vehicle_nodes = assignment['vehicle']
    drone_cost    = sum(info['cost'] for info in drone_dict.values())

    original = nb.CUSTOMER_IDS[:]
    nb.CUSTOMER_IDS[:] = vehicle_nodes

    if vehicle_nodes:
        C_mm = build_multimodal_cost_matrix(cost_matrix)
        vehicle_sol = clarke_wright_savings(C_mm, scenario, model_name="M4-Vehicle")
    else:
        vehicle_sol = CVRPSolution(
            routes=[], total_cost=0.0, scenario=scenario,
            n_vehicles=0, vehicle_utilization=[], model_name="M4-Vehicle")

    nb.CUSTOMER_IDS[:] = original

    return {
        'drone_result':  assignment,
        'vehicle_sol':   vehicle_sol,
        'total_cost':    drone_cost + vehicle_sol.total_cost,
        'drone_cost':    drone_cost,
        'vehicle_cost':  vehicle_sol.total_cost,
        'n_drone':       len(drone_dict),
        'n_vehicle':     len(vehicle_nodes),
        'scenario':      scenario,
    }
