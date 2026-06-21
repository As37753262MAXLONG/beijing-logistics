"""
database.py — SQLite database module

Tables:
  nodes   — Node info (customer / hub / depot)
  queries — Route query history
  results — Per-query model comparison results
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "logistics.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database and create all tables."""
    conn = get_conn()
    c = conn.cursor()

    # Nodes table
    c.execute("""
    CREATE TABLE IF NOT EXISTS nodes (
        id          INTEGER PRIMARY KEY,
        node_type   TEXT NOT NULL,        -- depot / hub / customer
        name        TEXT,
        lat         REAL NOT NULL,
        lon         REAL NOT NULL,
        demand      INTEGER DEFAULT 0,
        is_active   INTEGER DEFAULT 1,
        created_at  TEXT DEFAULT (datetime('now','localtime'))
    )""")

    # Queries table
    c.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        selected_nodes  TEXT NOT NULL,    -- JSON array of node ids
        time_period     TEXT NOT NULL,    -- off_peak / mid_peak / peak
        congestion      TEXT NOT NULL,    -- normal / congested / severe
        drone_range     REAL DEFAULT 15,
        drone_load      INTEGER DEFAULT 10,
        best_model      TEXT,             -- M1 / M2 / M3 / M4
        queried_at      TEXT DEFAULT (datetime('now','localtime'))
    )""")

    # Results table
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id    INTEGER NOT NULL,
        model       TEXT NOT NULL,
        total_cost  REAL,
        drone_cost  REAL DEFAULT 0,
        vehicle_cost REAL DEFAULT 0,
        n_drone     INTEGER DEFAULT 0,
        n_vehicle   INTEGER DEFAULT 0,
        route_json  TEXT,
        FOREIGN KEY(query_id) REFERENCES queries(id)
    )""")

    conn.commit()
    conn.close()
    print("Database initialized")


def seed_nodes():
    """
    Insert built-in node data from network_beijing.py.
    IMPORTANT: only clears built-in nodes (ID < 100),
    preserves user-added custom nodes (ID >= 100).
    """
    from network_beijing import (
        DEPOT_COORD, HUB_COORDS, HUB_NAMES, CUSTOMER_COORDS, DEMANDS
    )

    conn = get_conn()
    c = conn.cursor()

    # Clear only built-in nodes; preserve custom nodes
    c.execute("DELETE FROM nodes WHERE id < 100")

    # Depot
    c.execute("INSERT INTO nodes (id, node_type, name, lat, lon, demand) VALUES (?,?,?,?,?,?)",
              (0, 'depot', 'Chaoyang DC', DEPOT_COORD[0], DEPOT_COORD[1], 0))

    # Hubs
    for hid, coord in HUB_COORDS.items():
        c.execute("INSERT INTO nodes (id, node_type, name, lat, lon, demand) VALUES (?,?,?,?,?,?)",
                  (hid, 'hub', HUB_NAMES[hid], coord[0], coord[1], 0))

    # Customers
    for cid, coord in CUSTOMER_COORDS.items():
        c.execute("INSERT INTO nodes (id, node_type, name, lat, lon, demand) VALUES (?,?,?,?,?,?)",
                  (cid, 'customer', f'Customer Node {cid}', coord[0], coord[1], DEMANDS[cid]))

    custom_count = conn.execute("SELECT COUNT(*) as n FROM nodes WHERE id >= 100").fetchone()['n']
    conn.commit()
    conn.close()

    if custom_count > 0:
        print(f"Seeded {1 + len(HUB_COORDS) + len(CUSTOMER_COORDS)} built-in nodes "
              f"(preserved {custom_count} custom)")
    else:
        print(f"Seeded {1 + len(HUB_COORDS) + len(CUSTOMER_COORDS)} nodes")


def get_all_nodes():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_customers():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE node_type='customer' AND is_active=1 ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def toggle_node(node_id: int, active: bool):
    conn = get_conn()
    conn.execute("UPDATE nodes SET is_active=? WHERE id=?", (1 if active else 0, node_id))
    conn.commit()
    conn.close()


def save_query(selected_nodes, time_period, congestion,
               drone_range, drone_load, best_model, model_results):
    """Save a query along with all of its model results."""
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    INSERT INTO queries (selected_nodes, time_period, congestion,
                         drone_range, drone_load, best_model)
    VALUES (?,?,?,?,?,?)
    """, (json.dumps(selected_nodes), time_period, congestion,
          drone_range, drone_load, best_model))

    qid = c.lastrowid

    for model, data in model_results.items():
        c.execute("""
        INSERT INTO results (query_id, model, total_cost, drone_cost,
                             vehicle_cost, n_drone, n_vehicle, route_json)
        VALUES (?,?,?,?,?,?,?,?)
        """, (qid, model,
              data.get('total_cost', 0),
              data.get('drone_cost', 0),
              data.get('vehicle_cost', 0),
              data.get('n_drone', 0),
              data.get('n_vehicle', 0),
              json.dumps(data.get('routes', []))))

    conn.commit()
    conn.close()
    return qid


def get_recent_queries(limit=20):
    conn = get_conn()
    rows = conn.execute("""
    SELECT q.*, r.total_cost as m1_cost
    FROM queries q
    LEFT JOIN results r ON r.query_id=q.id AND r.model='M1'
    ORDER BY q.id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_query_detail(query_id):
    conn = get_conn()
    q = conn.execute("SELECT * FROM queries WHERE id=?", (query_id,)).fetchone()
    rs = conn.execute("SELECT * FROM results WHERE query_id=?", (query_id,)).fetchall()
    conn.close()
    if not q:
        return None
    return {
        'query': dict(q),
        'results': [dict(r) for r in rs],
    }


def get_stats():
    """Statistics: total queries, by scenario, average M4 vs M1 savings."""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) as n FROM queries").fetchone()['n']
    by_time = conn.execute("""
        SELECT time_period, COUNT(*) as n FROM queries GROUP BY time_period
    """).fetchall()
    avg_saving = conn.execute("""
        SELECT AVG((r2.total_cost - r1.total_cost) / r2.total_cost * 100) as avg_save
        FROM results r1
        JOIN results r2 ON r1.query_id = r2.query_id
        WHERE r1.model='M4' AND r2.model='M1'
    """).fetchone()
    conn.close()
    return {
        'total_queries': total,
        'by_time': [dict(r) for r in by_time],
        'avg_saving_pct': round(avg_saving['avg_save'] or 0, 2),
    }
