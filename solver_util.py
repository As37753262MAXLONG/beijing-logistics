"""
solver_util.py — helper functions to avoid drone.py ↔ solver.py circular import
"""
import numpy as np
from network_beijing import DEPOT_ID, CUSTOMER_IDS, DEMANDS, VEHICLE_CAPACITY


def estimate_avg_vehicle_cost(cost_matrix: np.ndarray) -> float:
    """
    Quickly estimate avg per-customer vehicle delivery cost via nearest-neighbor.
    No need to import solver.py, avoids circular imports。
    """
    C = cost_matrix
    unAccessed = set(CUSTOMER_IDS)
    total_cost = 0.0
    routes_count = 0

    while unAccessed:
        nodes, load, cur = [], 0, DEPOT_ID
        while unAccessed:
            cands = [(C[cur][j], j) for j in unAccessed
                     if load + DEMANDS[j] <= VEHICLE_CAPACITY]
            if not cands:
                break
            _, nxt = min(cands)
            nodes.append(nxt)
            load += DEMANDS[nxt]
            unAccessed.discard(nxt)
            cur = nxt
        if nodes:
            full = [DEPOT_ID] + nodes + [DEPOT_ID]
            cost = sum(C[full[i]][full[i+1]] for i in range(len(full)-1))
            total_cost += cost
            routes_count += 1

    n = len(CUSTOMER_IDS)
    return total_cost / n if n > 0 else 1e9
