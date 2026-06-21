"""
drone.py
Drone delivery module (Paper extension: M4 hybrid).

Drone parameters (based on JD/Meituan urban logistics drones):
  - Max payload : 10 units (small packages)
  - Max range   : 15 km (one-way)
  - Flight speed: 80 km/h (straight-line, unaffected by traffic)
  - Fixed cost  : 8 CNY/trip (includes depreciation)
  - Distance cost: 3 CNY/km (battery + maintenance)
  - Launch points: distribution center (depot) or any transfer hub

Hybrid delivery strategy:
  1. If demand <= DRONE_MAX_LOAD and straight-line distance <= DRONE_MAX_RANGE,
     the customer is drone-eligible.
  2. Among all available launch points (depot + hubs), pick the closest.
  3. If drone total cost < vehicle delivery cost, assign to drone.
  4. Remaining customers stay on vehicle CVRP routes.
"""

import math
import numpy as np
from network_beijing import (
    DEPOT_ID, HUB_IDS, CUSTOMER_IDS, ALL_NODES,
    DEMANDS, VEHICLE_CAPACITY, COORDS
)

# ── Drone parameters ─────────────────────────────────────────────
DRONE_MAX_LOAD   = 10
DRONE_MAX_RANGE  = 15.0
DRONE_SPEED      = 80.0
DRONE_FIXED_COST = 8.0
DRONE_DIST_COST  = 3.0

# Available launch points: depot + all hubs
DRONE_LAUNCH_NODES = [DEPOT_ID] + HUB_IDS


# ── Straight-line distance (Haversine, no detour factor) ─────────
def straight_line_km(n1, n2):
    """Drone flight distance = real straight-line distance (km)."""
    lat1, lon1 = COORDS[n1]
    lat2, lon2 = COORDS[n2]
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 4)


# ── Drone delivery cost ──────────────────────────────────────────
def drone_delivery_cost(launch_node, customer):
    """Single drone delivery cost (launch → customer → return)."""
    d = straight_line_km(launch_node, customer)
    return DRONE_FIXED_COST + (d * 2) * DRONE_DIST_COST


def drone_delivery_time(launch_node, customer):
    """Single drone delivery time in hours (round trip)."""
    d = straight_line_km(launch_node, customer)
    return (d * 2) / DRONE_SPEED


# ── Drone eligibility ────────────────────────────────────────────
def is_drone_eligible(customer):
    """
    Whether a customer can be served by drone:
      1. Demand <= max payload
      2. At least one launch point within range
    """
    if DEMANDS[customer] > DRONE_MAX_LOAD:
        return False
    for ln in DRONE_LAUNCH_NODES:
        if straight_line_km(ln, customer) <= DRONE_MAX_RANGE:
            return True
    return False


def best_launch_node(customer):
    """
    Return the best launch point and its cost as a tuple
    (launch_node, cost, distance_km), or None if unreachable.
    """
    best = None
    for ln in DRONE_LAUNCH_NODES:
        d = straight_line_km(ln, customer)
        if d <= DRONE_MAX_RANGE:
            cost = drone_delivery_cost(ln, customer)
            if best is None or cost < best[1]:
                best = (ln, cost, d)
    return best


# ── Hybrid assignment ────────────────────────────────────────────
def assign_drone_or_vehicle(cost_matrix, scenario):
    """
    Decide drone vs vehicle for each customer.

    Decision logic (two-step):
      Step 1: estimate avg per-customer vehicle cost using a quick CVRP run.
      Step 2: for each eligible customer, drone if drone_cost < avg_vehicle_cost,
              else vehicle.

    Returns:
      {
        'drone'  : {customer_id: {launch, cost, dist_km, time_h, demand}},
        'vehicle': [customer_id, ...]
      }
    """
    from solver_util import estimate_avg_vehicle_cost
    avg_vehicle_cost = estimate_avg_vehicle_cost(cost_matrix)

    drone_assignments = {}
    vehicle_customers = []

    for c in CUSTOMER_IDS:
        launch_info = best_launch_node(c)

        if launch_info is None:
            # Out of range or overload → must go by vehicle
            vehicle_customers.append(c)
            continue

        launch_node, d_cost, dist_km = launch_info

        if d_cost < avg_vehicle_cost:
            drone_assignments[c] = {
                'launch':   launch_node,
                'cost':     d_cost,
                'dist_km':  dist_km,
                'time_h':   drone_delivery_time(launch_node, c),
                'demand':   DEMANDS[c],
            }
        else:
            vehicle_customers.append(c)

    return {
        'drone':   drone_assignments,
        'vehicle': vehicle_customers,
    }
