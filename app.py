"""
app.py — Flask backend API
Start: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import math
import os

from database import (init_db, seed_nodes, get_all_nodes, toggle_node,
                      save_query, get_recent_queries, get_query_detail,
                      get_stats, get_conn)

app = Flask(__name__, static_folder='static')

# ── Initialize ────────────────────────────────────────────────────
init_db()
seed_nodes()

# ── Core logistics computation ──────────────────────────────────────────────
SPEEDS   = {'off_peak': 50, 'mid_peak': 25, 'peak': 5}
BETAS    = {'normal': 1.0,  'congested': 0.6, 'severe': 0.2}
CD, CT   = 2.0, 50.0
ROAD_F   = 1.35
DRONE_FIXED, DRONE_KM = 8.0, 3.0
VEHICLE_CAP = 100


def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p = math.pi / 180
    a = (math.sin((la2 - la1) * p / 2) ** 2
         + math.cos(la1 * p) * math.cos(la2 * p)
         * math.sin((lo2 - lo1) * p / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def road_dist(la1, lo1, la2, lo2):
    return haversine(la1, lo1, la2, lo2) * ROAD_F


def link_cost(la1, lo1, la2, lo2, t, s):
    d = road_dist(la1, lo1, la2, lo2)
    tau = d / (SPEEDS[t] * BETAS[s])
    return CD * d + CT * tau


def drone_cost_fn(la1, lo1, la2, lo2):
    d = haversine(la1, lo1, la2, lo2)
    return DRONE_FIXED + d * 2 * DRONE_KM, d


def nearest_neighbor_routes(customers, depot, t, s):
    unAccessed = {c['id']: c for c in customers}
    routes = []
    total_cost = 0.0

    while unAccessed:
        path, load, cur = [], 0, depot
        route_cost = 0.0
        while True:
            best_id, best_cost, best_node = None, 1e18, None
            for cid, c in unAccessed.items():
                if load + c['demand'] > VEHICLE_CAP:
                    continue
                cost = link_cost(cur['lat'], cur['lon'], c['lat'], c['lon'], t, s)
                if cost < best_cost:
                    best_cost, best_id, best_node = cost, cid, c
            if best_id is None:
                break
            route_cost += best_cost
            path.append(best_id)
            load += best_node['demand']
            unAccessed.pop(best_id)
            cur = best_node
        if path:
            route_cost += link_cost(cur['lat'], cur['lon'],
                                    depot['lat'], depot['lon'], t, s)
            routes.append({'nodes': path, 'load': load, 'cost': round(route_cost, 2)})
            total_cost += route_cost

    return routes, round(total_cost, 2)


def compute_all_models(selected_customers, all_nodes_map, t, s,
                       drone_range, drone_load):
    depot = all_nodes_map[0]
    hubs  = [all_nodes_map[i] for i in range(1, 6) if i in all_nodes_map]
    launch_nodes = [depot] + hubs

    # M1: Dynamic multimodal
    def mm_cost(la1, lo1, la2, lo2):
        direct = link_cost(la1, lo1, la2, lo2, t, s)
        best = min(
            (link_cost(la1, lo1, h['lat'], h['lon'], t, s)
             + link_cost(h['lat'], h['lon'], la2, lo2, t, s)) * 0.65
            for h in hubs
        ) if hubs else direct
        return min(direct, best)

    m1_routes, m1_total = [], 0.0
    unAccessed = {c['id']: c for c in selected_customers}
    while unAccessed:
        path, load, cur = [], 0, depot
        rc = 0.0
        while True:
            best_id, best_c, best_node = None, 1e18, None
            for cid, c in unAccessed.items():
                if load + c['demand'] > VEHICLE_CAP:
                    continue
                cost = mm_cost(cur['lat'], cur['lon'], c['lat'], c['lon'])
                if cost < best_c:
                    best_c, best_id, best_node = cost, cid, c
            if best_id is None:
                break
            rc += best_c; path.append(best_id)
            load += best_node['demand']; unAccessed.pop(best_id); cur = best_node
        if path:
            rc += mm_cost(cur['lat'], cur['lon'], depot['lat'], depot['lon'])
            m1_routes.append({'nodes': path, 'load': load, 'cost': round(rc, 2)})
            m1_total += rc
    m1_total = round(m1_total, 2)

    # M2: Static single-modal
    static_routes, _ = nearest_neighbor_routes(selected_customers, depot, 'off_peak', 'normal')
    m2_total = 0.0
    for r in static_routes:
        nodes_seq = [depot] + [all_nodes_map[nid] for nid in r['nodes'] if nid in all_nodes_map] + [depot]
        cost = sum(link_cost(nodes_seq[i]['lat'], nodes_seq[i]['lon'],
                             nodes_seq[i+1]['lat'], nodes_seq[i+1]['lon'], t, s)
                   for i in range(len(nodes_seq)-1))
        r['cost'] = round(cost, 2)
        m2_total += cost
    m2_total = round(m2_total, 2)

    # M3: Nearest-neighbor heuristic
    m3_routes, m3_total = nearest_neighbor_routes(selected_customers, depot, t, s)

    # M4: Drone + vehicle hybrid
    avg_vehicle = m1_total / max(len(selected_customers), 1)
    drone_list, vehicle_list = [], []

    for c in selected_customers:
        if c['demand'] > drone_load:
            vehicle_list.append(c)
            continue
        best_launch, best_dc, best_dist = None, 1e18, 0
        for ln in launch_nodes:
            dc, dd = drone_cost_fn(ln['lat'], ln['lon'], c['lat'], c['lon'])
            if dd <= drone_range and dc < best_dc:
                best_dc, best_launch, best_dist = dc, ln, dd
        if best_launch and best_dc < avg_vehicle:
            drone_list.append({**c, 'launch': best_launch['id'],
                                'launch_name': best_launch.get('name', ''),
                                'drone_cost': round(best_dc, 2),
                                'dist_km': round(best_dist, 2)})
        else:
            vehicle_list.append(c)

    drone_total = sum(d['drone_cost'] for d in drone_list)
    veh_routes, veh_total = nearest_neighbor_routes(vehicle_list, depot, t, s) if vehicle_list else ([], 0)
    m4_total = round(drone_total + veh_total, 2)

    drone_ids = {d['id'] for d in drone_list}
    per_node = {}
    for c in selected_customers:
        nid = c['id']
        if nid in drone_ids:
            info = next(d for d in drone_list if d['id'] == nid)
            per_node[nid] = {
                'mode': 'drone', 'cost': info['drone_cost'],
                'launch_node': info['launch'], 'launch_name': info['launch_name'],
                'dist_km': info['dist_km'], 'demand': c['demand'],
            }
        else:
            per_node[nid] = {'mode': 'vehicle', 'cost': None, 'demand': c['demand']}

    best_model = min(
        [('M1', m1_total), ('M2', m2_total), ('M3', m3_total), ('M4', m4_total)],
        key=lambda x: x[1]
    )[0]

    return {
        'M1': {'total_cost': m1_total, 'drone_cost': 0, 'vehicle_cost': m1_total,
               'n_drone': 0, 'n_vehicle': len(selected_customers), 'routes': m1_routes},
        'M2': {'total_cost': m2_total, 'drone_cost': 0, 'vehicle_cost': m2_total,
               'n_drone': 0, 'n_vehicle': len(selected_customers), 'routes': static_routes},
        'M3': {'total_cost': m3_total, 'drone_cost': 0, 'vehicle_cost': m3_total,
               'n_drone': 0, 'n_vehicle': len(selected_customers), 'routes': m3_routes},
        'M4': {'total_cost': m4_total, 'drone_cost': round(drone_total, 2),
               'vehicle_cost': round(veh_total, 2),
               'n_drone': len(drone_list), 'n_vehicle': len(vehicle_list),
               'routes': veh_routes, 'drone_details': drone_list},
        'best_model': best_model,
        'per_node': per_node,
    }


# ══════════════════════════════════════════════════════════════
# API routes - IMPORTANT: all /api/* must come before wildcard route
# ══════════════════════════════════════════════════════════════

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r


@app.route('/api/nodes', methods=['GET'])
def api_nodes():
    return jsonify(get_all_nodes())


@app.route('/api/nodes/add', methods=['POST', 'OPTIONS'])
def api_add_node():
    """User manually adds customer node"""
    if request.method == 'OPTIONS':
        return '', 204
    body = request.get_json(force=True) or {}
    lat  = float(body.get('lat', 0))
    lon  = float(body.get('lon', 0))
    name = body.get('name', 'Custom Node')
    demand = int(body.get('demand', 10))

    conn = get_conn()
    max_id = conn.execute("SELECT MAX(id) as m FROM nodes WHERE id >= 100").fetchone()['m']
    new_id = max(100, (max_id or 99) + 1)
    conn.execute(
        "INSERT INTO nodes (id, node_type, name, lat, lon, demand) VALUES (?,?,?,?,?,?)",
        (new_id, 'customer', name, lat, lon, demand)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'node_id': new_id, 'name': name})


@app.route('/api/nodes/<int:node_id>/toggle', methods=['POST', 'OPTIONS'])
def api_toggle(node_id):
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json(force=True) or {}
    toggle_node(node_id, data.get('active', True))
    return jsonify({'ok': True, 'node_id': node_id})


@app.route('/api/nodes/<int:node_id>/delete', methods=['POST', 'OPTIONS'])
def api_delete_node(node_id):
    """deleteuserCustom Node（only ID>=100）"""
    if request.method == 'OPTIONS':
        return '', 204
    if node_id < 100:
        return jsonify({'error': 'Built-in nodes cannot be deleted'}), 403
    conn = get_conn()
    conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/route', methods=['POST', 'OPTIONS'])
def api_route():
    """Core endpoint: compute 4-model optimal route"""
    if request.method == 'OPTIONS':
        return '', 204

    body = request.get_json(force=True) or {}
    selected_ids = body.get('selected_nodes', [])
    t  = body.get('time_period', 'peak')
    s  = body.get('congestion',  'severe')
    dr = float(body.get('drone_range', 15))
    dl = int(body.get('drone_load', 10))

    if not selected_ids:
        return jsonify({'error': 'Please select at least one node'}), 400

    all_nodes = {n['id']: n for n in get_all_nodes()}
    selected_customers = [all_nodes[i] for i in selected_ids if i in all_nodes]

    if not selected_customers:
        return jsonify({'error': 'Selected nodes do not exist'}), 400

    result = compute_all_models(selected_customers, all_nodes, t, s, dr, dl)

    model_results = {k: result[k] for k in ('M1','M2','M3','M4') if k in result}
    qid = save_query(selected_ids, t, s, dr, dl, result['best_model'], model_results)

    return jsonify({
        'query_id':   qid,
        'best_model': result['best_model'],
        'models':     {k: v for k, v in result.items() if k in ('M1','M2','M3','M4')},
        'per_node':   result['per_node'],
        'selected':   selected_ids,
        'scenario':   {'time_period': t, 'congestion': s,
                       'drone_range': dr, 'drone_load': dl},
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(get_recent_queries())


@app.route('/api/history/<int:qid>', methods=['GET'])
def api_history_detail(qid):
    d = get_query_detail(qid)
    if not d:
        return jsonify({'error': 'not found'}), 404
    return jsonify(d)


@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(get_stats())


# ══════════════════════════════════════════════════════════════
# Static files - MUST come after all API routes!
# ══════════════════════════════════════════════════════════════

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path and os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    return send_from_directory('static', 'index.html')


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Beijing Logistics Optimization API")
    print("  Visit: http://localhost:5000")
    print("=" * 50 + "\n")
    # List all routes for debugging
    print("Registered API endpoints:")
    for rule in app.url_map.iter_rules():
        if str(rule).startswith('/api'):
            methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
            print(f"  {methods:10s} {rule}")
    print()
    app.run(debug=True, port=5000)
