"""
network_beijing.py
Beijing urban logistics network definition.

Contents:
  - Real 4th Ring Road boundary (24 vertices)
  - Distribution center + 5 real logistics hubs
  - 24 random customer nodes inside 4th Ring (Haversine distance)
"""

import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ── 4th Ring Road boundary (real lat/lon, clockwise) ─────────────
FOURTH_RING_BOUNDARY = [
    (39.998, 116.348), (39.999, 116.375), (39.998, 116.405),
    (39.995, 116.430), (39.985, 116.455), (39.970, 116.475),
    (39.955, 116.482), (39.935, 116.480), (39.915, 116.478),
    (39.900, 116.470), (39.885, 116.458), (39.875, 116.435),
    (39.870, 116.405), (39.870, 116.375), (39.875, 116.345),
    (39.885, 116.320), (39.900, 116.302), (39.915, 116.290),
    (39.935, 116.288), (39.950, 116.292), (39.965, 116.305),
    (39.980, 116.320), (39.992, 116.335), (39.998, 116.348),
]

# ── Node IDs ─────────────────────────────────────────────────────
DEPOT_ID     = 0
HUB_IDS      = list(range(1, 6))
CUSTOMER_IDS = list(range(6, 30))
ALL_NODES    = list(range(30))

NODE_TYPES = {DEPOT_ID: "depot"}
for _h in HUB_IDS:
    NODE_TYPES[_h] = "hub"
for _c in CUSTOMER_IDS:
    NODE_TYPES[_c] = "customer"

# ── Real coordinates ─────────────────────────────────────────────
DEPOT_COORD = (39.9042, 116.4074)   # Chaoyang Distribution Center

HUB_COORDS = {
    1: (39.9600, 116.3100),   # Haidian Sijiqing Logistics Park
    2: (39.9500, 116.4700),   # Chaoyang Sihui Logistics Base
    3: (39.8780, 116.4200),   # Fengtai Nanyuan Logistics Park
    4: (39.8950, 116.3150),   # Fengtai Liuliqiao Freight Hub
    5: (39.9750, 116.3900),   # Xicheng Deshengmen Node
}

HUB_NAMES = {
    1: "Haidian Sijiqing Logistics Park",
    2: "Chaoyang Sihui Logistics Base",
    3: "Fengtai Nanyuan Logistics Park",
    4: "Fengtai Liuliqiao Freight Hub",
    5: "Xicheng Deshengmen Node",
}


# ── Point-in-polygon (ray casting) ───────────────────────────────
def _in_polygon(lat, lon, poly):
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i][1], poly[i][0]
        xj, yj = poly[j][1], poly[j][0]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ── Generate random customers inside 4th Ring ────────────────────
def _generate_customers(n=24, seed=42):
    rng = random.Random(seed)
    lats = [p[0] for p in FOURTH_RING_BOUNDARY]
    lons = [p[1] for p in FOURTH_RING_BOUNDARY]
    coords, cid = {}, 6
    for _ in range(100000):
        if cid >= 6 + n:
            break
        lat = rng.uniform(min(lats), max(lats))
        lon = rng.uniform(min(lons), max(lons))
        if _in_polygon(lat, lon, FOURTH_RING_BOUNDARY):
            coords[cid] = (round(lat, 5), round(lon, 5))
            cid += 1
    return coords


CUSTOMER_COORDS = _generate_customers()

# ── Merge all coordinates ────────────────────────────────────────
COORDS = {DEPOT_ID: DEPOT_COORD}
COORDS.update(HUB_COORDS)
COORDS.update(CUSTOMER_COORDS)

# ── Haversine distance ───────────────────────────────────────────
ROAD_FACTOR = 1.35   # Detour factor (straight-line → road distance)


def haversine(n1, n2):
    """Real geographic distance between two nodes (km)."""
    lat1, lon1 = COORDS[n1]
    lat2, lon2 = COORDS[n2]
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 4)


def build_distance_matrix():
    """Return 30x30 road distance matrix (km, w/ detour factor)."""
    n = len(ALL_NODES)
    D = np.zeros((n, n))
    for i in ALL_NODES:
        for j in ALL_NODES:
            if i != j:
                D[i][j] = haversine(i, j) * ROAD_FACTOR
    return D


# ── Projected coords (km, depot as origin, for plotting) ─────────
def _to_km(lat, lon):
    rlat, rlon = DEPOT_COORD
    x = (lon - rlon) * 111.320 * math.cos(math.radians(rlat))
    y = (lat - rlat) * 110.574
    return round(x, 3), round(y, 3)


POSITIONS = {node: _to_km(*coord) for node, coord in COORDS.items()}
RING4_XY  = [_to_km(lat, lon) for lat, lon in FOURTH_RING_BOUNDARY]

# ── Demand and vehicle config ────────────────────────────────────
DEMANDS = {i: 0 for i in range(6)}
_rng = random.Random(42)
for c in CUSTOMER_IDS:
    DEMANDS[c] = _rng.randint(5, 20)

VEHICLE_CAPACITY = 100
