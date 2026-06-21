# Beijing Multimodal Transportation Optimization System

A Flask-based web application for dynamic multimodal logistics optimization
in Beijing's urban area, integrating vehicle routing with drone delivery.

## First-Time Setup

Install Python dependencies:

```
pip install -r requirements.txt
```

## Run the Application

```
python app.py
```

Then open in your browser: http://localhost:5000

## File Overview

| File | Purpose |
|------|---------|
| `app.py` | Flask backend — run this to start |
| `database.py` | SQLite database (auto-created) |
| `network_beijing.py` | Beijing 4th-Ring map + node coordinates |
| `traffic.py` | 9 traffic scenario definitions |
| `drone.py` | Drone delivery logic |
| `solver.py` | M1 / M2 / M3 / M4 optimization models |
| `solver_util.py` | Helper functions |
| `experiment.py` | Batch experiments (optional, generates CSV) |
| `static/index.html` | Frontend UI (auto-served) |

## Optional: Generate Paper Figures

```
python experiment.py    # Run experiments, produces results.csv
python viz_drone.py     # Generate the 10 PNG figures
```

## Windows: Chinese Font Configuration (for Chinese figure labels)

If you want figure labels in Chinese, open `viz_drone.py` and change:

```python
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
```

to:

```python
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
```

## Models Implemented

| Model | Description |
|-------|-------------|
| M1 | Dynamic multimodal (Clarke-Wright + hub discount) |
| M2 | Static single-modal baseline |
| M3 | Nearest-neighbor heuristic baseline |
| M4 | Hybrid drone + vehicle delivery |

## Key Findings

| Scenario | M4 vs M1 |
|----------|----------|
| Off-Peak / Normal | M4 is 8.9% more expensive (vehicles in batch are more efficient) |
| Peak / Congested | M4 saves 68.6% (drones bypass ground traffic) |
| Peak / Severe | M4 saves 88.7% (all 24 customers switch to drone) |

The optimal model depends on real-time traffic conditions — this is the
core value of dynamic multimodal optimization.
