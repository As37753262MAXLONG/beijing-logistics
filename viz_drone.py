"""
viz_drone.py — Drone hybrid delivery visualization (extension)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(_CJK)
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from network_beijing import (POSITIONS, NODE_TYPES, DEPOT_ID, HUB_IDS,
                              CUSTOMER_IDS, HUB_NAMES, DEMANDS, RING4_XY,
                              build_distance_matrix)
from traffic import build_cost_matrix
from solver import dynamic_multimodal, hybrid_drone_vehicle
from drone import straight_line_km

C_DEPOT = '#E74C3C'; C_HUB = '#2980B9'; C_CUSTOMER = '#27AE60'
C_RING  = '#95A5A6'; C_M1 = '#2980B9'; C_M4 = '#8E44AD'
C_DRONE_LINE = '#E74C3C'; C_VEH_LINE = '#2980B9'
ROUTE_COLORS = ['#E67E22','#16A085','#C0392B','#117A65','#8E44AD','#2471A3']


def _basemap(ax):
    xs = [p[0] for p in RING4_XY]; ys = [p[1] for p in RING4_XY]
    ax.fill(xs, ys, color='#AED6F1', alpha=0.06, zorder=0)
    ax.plot(xs, ys, color=C_RING, lw=1.5, ls='--', alpha=0.5, zorder=1)
    ax.grid(True, ls=':', alpha=0.2); ax.set_facecolor('#F4F6F7')

def _nodes(ax, drone_set=None, vehicle_set=None):
    for c in CUSTOMER_IDS:
        x, y = POSITIONS[c]
        if drone_set and c in drone_set:
            ax.scatter(x, y, s=90, c='#E74C3C', zorder=6,
                       edgecolors='white', lw=1.2, marker='^')
        elif vehicle_set and c in vehicle_set:
            ax.scatter(x, y, s=60, c=C_CUSTOMER, zorder=5,
                       edgecolors='white', lw=0.8)
        else:
            ax.scatter(x, y, s=60, c=C_CUSTOMER, zorder=5,
                       edgecolors='white', lw=0.8, alpha=0.4)
    for h in HUB_IDS:
        x, y = POSITIONS[h]
        ax.scatter(x, y, s=150, c=C_HUB, marker='D', zorder=6,
                   edgecolors='white', lw=1.2)
        ax.text(x+0.2, y+0.15, f'H{h}', fontsize=6,
                color=C_HUB, fontweight='bold')
    x, y = POSITIONS[DEPOT_ID]
    ax.scatter(x, y, s=260, c=C_DEPOT, marker='*', zorder=7,
               edgecolors='white', lw=1.5)
    ax.text(x+0.2, y+0.25, 'DC', fontsize=8,
            color=C_DEPOT, fontweight='bold')

def _extent(ax):
    ax.set_xlim(-15, 16); ax.set_ylim(-8, 10)
    ax.set_xlabel('East-West (km)', fontsize=8)
    ax.set_ylabel('North-South (km)', fontsize=8)
    ax.set_aspect('equal')


# ══════════════════════════════════════════════════════════════
# Figure 7: M4 hybrid route map (two scenarios)
# ══════════════════════════════════════════════════════════════
def plot_hybrid_routes():
    D = build_distance_matrix()
    scenarios = [("off_peak","normal"), ("peak","severe")]
    titles    = ["Off-Peak/Normal  Off-Peak / Normal",
                 "Peak/SevereCongested  Peak / Severe"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (t, s), title in zip(axes, scenarios, titles):
        C  = build_cost_matrix(D, t, s)
        m4 = hybrid_drone_vehicle(C, (t, s))

        drone_nodes   = set(m4['drone_result']['drone'].keys())
        vehicle_nodes = set(m4['drone_result']['vehicle'])

        _basemap(ax)

        # 1) Drone routes (dashed arcs)
        for c, info in m4['drone_result']['drone'].items():
            lx, ly = POSITIONS[info['launch']]
            cx, cy = POSITIONS[c]
            ax.annotate("", xy=(cx, cy), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-|>", color=C_DRONE_LINE,
                                lw=1.2, alpha=0.65, linestyle='dashed',
                                mutation_scale=10))

        # 2) Vehicle routes (solid arrows)
        for k, route in enumerate(m4['vehicle_sol'].routes):
            color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
            full  = route.total_nodes()
            for m in range(len(full)-1):
                xi, yi = POSITIONS[full[m]]
                xj, yj = POSITIONS[full[m+1]]
                ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.0, alpha=0.8, mutation_scale=12))

        _nodes(ax, drone_set=drone_nodes, vehicle_set=vehicle_nodes)
        _extent(ax)

        drone_pct = len(drone_nodes)/len(CUSTOMER_IDS)*100
        ax.set_title(
            f'{title}\n'
            f'Drones: {len(drone_nodes)}customer(s) ({drone_pct:.0f}%)  '
            f'Vehicles: {len(vehicle_nodes)} customers  '
            f'M4Total cost: {m4["total_cost"]:.0f}¥',
            fontsize=9, pad=8)

    # Unified legend
    legend_elems = [
        Line2D([0],[0], color=C_DRONE_LINE, lw=1.5, ls='--',
               marker='>', markersize=6, label='Drone Route'),
        Line2D([0],[0], color=ROUTE_COLORS[0], lw=2,
               marker='>', markersize=6, label='Vehicle Route'),
        plt.scatter([],[], s=90, c='#E74C3C', marker='^',
                    label='Drone Customer'),
        plt.scatter([],[], s=60, c=C_CUSTOMER,
                    label='Vehicle Customer'),
        plt.scatter([],[], s=150, c=C_HUB, marker='D',
                    label='Transfer Hub'),
        plt.scatter([],[], s=260, c=C_DEPOT, marker='*',
                    label='Distribution Center'),
    ]
    axes[1].legend(handles=legend_elems, loc='lower right',
                   fontsize=7, framealpha=0.9)

    fig.suptitle('M4 Hybrid Routes: Drone + Vehicle Coordination\n'
                 'Hybrid Drone-Vehicle Routing (M4 Model)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('/home/claude/fig7_hybrid_routes.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Hybrid drone-vehicle route map")


# ══════════════════════════════════════════════════════════════
# Figure 8: Four-model cost comparison
# ══════════════════════════════════════════════════════════════
def plot_four_model_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Total Cost Comparison: Four Models (M1/M2/M3/M4)\n'
                 'Total Cost Comparison: All Four Models',
                 fontsize=12, y=1.02)

    tps = ['off_peak','mid_peak','peak']
    tp_labels = ['Off-Peak\nOff-Peak','Mid-Peak\nMid-Peak','Peak\nPeak']
    ss  = ['normal','congested','severe']
    s_labels = ['Normal','Congested','Severe']
    x = np.arange(3); w = 0.20

    colors = {'M1': C_M1, 'M2': '#E74C3C', 'M3': '#F39C12', 'M4': C_M4}

    for i, (tp, tl) in enumerate(zip(tps, tp_labels)):
        ax = axes[i]
        sub = df[df['Time Period']==tp]
        vals = {m: sub[f'{m}_Cost'].values for m in ['M1','M2','M3','M4']}

        for j, (m, c) in enumerate(colors.items()):
            offset = (j - 1.5) * w
            bars = ax.bar(x + offset, vals[m], w, label=m,
                          color=c, alpha=0.85, edgecolor='white')

        # M4 line emphasis
        ax.plot(x, vals['M4'], 'o--', color=C_M4,
                lw=2, ms=6, zorder=5, alpha=0.9)

        ax.set_title(tl, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(s_labels, fontsize=9)
        ax.set_ylabel('Total Cost (CNY)' if i==0 else '', fontsize=9)
        ax.grid(axis='y', ls='--', alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        if i==2: ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig('/home/claude/fig8_four_models.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Four-model cost comparison")


# ══════════════════════════════════════════════════════════════
# Figure 9: Drone allocation vs congestion
# ══════════════════════════════════════════════════════════════
def plot_drone_adoption(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Drone Adoption Analysis\nDrone Adoption Analysis',
                 fontsize=12, y=1.02)

    tps = ['off_peak','mid_peak','peak']
    ss  = ['normal','congested','severe']
    colors_t = {'off_peak':'#3498DB','mid_peak':'#E67E22','peak':'#E74C3C'}
    labels_t  = {'off_peak':'Off-Peak','mid_peak':'Mid-Peak','peak':'Peak'}

    # Left: drone customer count
    x = np.arange(3)
    for tp in tps:
        sub = df[df['Time Period']==tp]
        ax1.plot(x, sub['M4_nDrone'].values, 'o-',
                 color=colors_t[tp], lw=2, ms=8, label=labels_t[tp])
    ax1.axhline(len(CUSTOMER_IDS), color='gray', ls=':', lw=1,
                label=f'Total customers ({len(CUSTOMER_IDS)})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Normal','Congested','Severe'], fontsize=9)
    ax1.set_ylabel('Drone-Served Customers', fontsize=10)
    ax1.set_title('Drone Service Count vs Congestion', fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(ls='--', alpha=0.3)
    ax1.set_facecolor('#F8F9FA')

    # Right: M4 vs M1 savings rate
    for tp in tps:
        sub = df[df['Time Period']==tp]
        ax2.plot(x, sub['M4_vs_M1'].values, 's-',
                 color=colors_t[tp], lw=2, ms=8, label=labels_t[tp])
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.fill_between(x, [0]*3, [0]*3, alpha=0)  # placeholder
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Normal','Congested','Severe'], fontsize=9)
    ax2.set_ylabel('M4 Savings vs M1 (%)', fontsize=10)
    ax2.set_title('Hybrid vs Pure-Vehicle Savings', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(ls='--', alpha=0.3)
    ax2.set_facecolor('#F8F9FA')

    # Annotate key values
    peak_severe_saving = df[(df['Time Period']=='peak')&
                            (df['Congestion']=='severe')]['M4_vs_M1'].values[0]
    ax2.annotate(f'+{peak_severe_saving:.1f}%',
                 xy=(2, peak_severe_saving),
                 xytext=(1.5, peak_severe_saving-15),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C'),
                 fontsize=9, color='#E74C3C', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/claude/fig9_drone_adoption.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("✓ Figure 9: Drone adoption analysis")


# ══════════════════════════════════════════════════════════════
# Figure 10: Cost breakdown stacked（M4 drone vs vehicle cost share）
# ══════════════════════════════════════════════════════════════
def plot_cost_breakdown(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#F8F9FA')

    labels = [f"{row['Time Period']}\n{row['Congestion']}"
              for _, row in df.iterrows()]
    x = np.arange(len(labels))
    w = 0.35

    # M1 single color
    ax.bar(x - w/2, df['M1_Cost'], w, label='M1 Vehicle Total',
           color=C_M1, alpha=0.8, edgecolor='white')

    # M4 stack: drone + vehicle
    ax.bar(x + w/2, df['M4_DroneCost'], w, label='M4 Drone Cost',
           color='#E74C3C', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, df['M4_VehCost'], w,
           bottom=df['M4_DroneCost'], label='M4 Vehicle Cost',
           color='#8E44AD', alpha=0.85, edgecolor='white')

    # Annotate M4 total cost
    for i, (drone, veh) in enumerate(zip(df['M4_DroneCost'], df['M4_VehCost'])):
        total = drone + veh
        ax.text(i + w/2, total + total*0.02, f'{total:.0f}',
                ha='center', fontsize=6.5, color='#4A235A', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Total Cost (CNY)', fontsize=10)
    ax.set_title('M1 vs M4 Cost Breakdown（M4 = Drone Cost + Vehicle Cost）\n'
                 'Cost Breakdown: M1 Vehicle-Only vs M4 Hybrid',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', ls='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig('/home/claude/fig10_cost_breakdown.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("✓ Figure 10: Cost breakdown stacked bar")


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    from experiment import run_all_scenarios
    df = run_all_scenarios()

    plot_hybrid_routes()
    plot_four_model_comparison(df)
    plot_drone_adoption(df)
    plot_cost_breakdown(df)

    print("\n✓ All drone visualization figures generated.")
