"""
experiment.py — Full scenario experiment (M1 / M2 / M3 / M4)
"""
import numpy as np
import pandas as pd
from network_beijing import build_distance_matrix
from traffic import build_all_cost_matrices, ALL_SCENARIOS
from solver import (dynamic_multimodal, static_single_modal_baseline,
                    heuristic_nearest_neighbor, hybrid_drone_vehicle)


def run_all_scenarios() -> pd.DataFrame:
    D  = build_distance_matrix()
    CM = build_all_cost_matrices(D)
    records = []

    for (t, s) in ALL_SCENARIOS:
        C  = CM[(t, s)]
        m1 = dynamic_multimodal(C, (t, s))
        m2 = static_single_modal_baseline(C, (t, s))
        m3 = heuristic_nearest_neighbor(C, (t, s))
        m4 = hybrid_drone_vehicle(C, (t, s))

        records.append({
            "Time Period":  t,
            "Congestion":   s,
            "M1_Cost":      round(m1.total_cost, 2),
            "M2_Cost":      round(m2.total_cost, 2),
            "M3_Cost":      round(m3.total_cost, 2),
            "M4_Cost":      round(m4['total_cost'], 2),
            "M4_DroneCost": round(m4['drone_cost'], 2),
            "M4_VehCost":   round(m4['vehicle_cost'], 2),
            "M4_nDrone":    m4['n_drone'],
            "M4_nVehicle":  m4['n_vehicle'],
            "M1_Vehicles":  m1.n_vehicles,
            "M1_AvgUtil":   round(np.mean(m1.vehicle_utilization)*100, 1),
            "M1_vs_M2":     round((m2.total_cost-m1.total_cost)/m2.total_cost*100, 2),
            "M1_vs_M3":     round((m3.total_cost-m1.total_cost)/m3.total_cost*100, 2),
            "M4_vs_M1":     round((m1.total_cost-m4['total_cost'])/m1.total_cost*100, 2),
            "M4_vs_M2":     round((m2.total_cost-m4['total_cost'])/m2.total_cost*100, 2),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = run_all_scenarios()
    cols = ["Time Period","Congestion","M1_Cost","M2_Cost","M3_Cost","M4_Cost",
            "M4_nDrone","M4_nVehicle","M4_vs_M1","M1_vs_M2"]
    print(df[cols].to_string(index=False))
    print(f"\nM4 avg savings vs M1: {df['M4_vs_M1'].mean():.1f}%")
    print(f"M1 avg savings vs M2: {df['M1_vs_M2'].mean():.1f}%")
    df.to_csv("/home/claude/results.csv", index=False)
    print("✓ results.csv saved")
