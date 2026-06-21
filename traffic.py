"""
traffic.py
Time-varying traffic scenarios and cost formulas.

Formulas (Paper Section 3.1.4):
  tau_ij(t,s) = d_ij / (v(t) * beta(s))         travel time
  c_ij(t,s)   = c_d * d_ij + c_t * tau_ij(t,s)  link cost
"""

import numpy as np
from network_beijing import ALL_NODES, build_distance_matrix

# Time period → base speed (km/h)
TIME_PERIODS = {
    "off_peak": 50,
    "mid_peak": 25,
    "peak":      5,
}

# Congestion state → speed multiplier
CONGESTION_STATES = {
    "normal":    1.0,
    "congested": 0.6,
    "severe":    0.2,
}

# Cost coefficients
C_D = 2.0    # Distance cost (CNY/km)
C_T = 50.0   # Time cost (CNY/h)

ALL_SCENARIOS = [
    (t, s)
    for t in TIME_PERIODS
    for s in CONGESTION_STATES
]


def travel_time(d_ij, t, s):
    """Travel time in hours."""
    eff = TIME_PERIODS[t] * CONGESTION_STATES[s]
    return d_ij / eff if eff > 0 else float('inf')


def link_cost(d_ij, t, s):
    """Link cost in CNY."""
    return C_D * d_ij + C_T * travel_time(d_ij, t, s)


def build_cost_matrix(distance_matrix, t, s):
    """Build cost matrix for given scenario (t, s)."""
    n = len(ALL_NODES)
    C = np.zeros((n, n))
    for i in ALL_NODES:
        for j in ALL_NODES:
            if i != j:
                C[i][j] = link_cost(distance_matrix[i][j], t, s)
    return C


def build_all_cost_matrices(distance_matrix):
    """Build all 9 scenario cost matrices."""
    return {(t, s): build_cost_matrix(distance_matrix, t, s)
            for t, s in ALL_SCENARIOS}
