# Simple region scheduler for training jobs (executable).
regions = {                                   # region: (power_W, water_L_per_s, carbon_g_per_kWh)
    "coastal": [2e9, 500.0, 50.0],
    "inland":  [1.2e9, 100.0, 300.0],
}
PUE = {"coastal": 1.15, "inland": 1.3}
jobs = [                                       # (job_id, power_W, water_L_per_s, flops_required)
    ("trainA", 5e6, 1.0, 1e20),
    ("trainB", 2e7, 5.0, 5e20),
    ("trainC", 1e6, 0.2, 2e19),
]
# admission: greedy by lowest regional carbon intensity per job energy
admitted = []
for job_id, p_req, w_req, _ in sorted(jobs, key=lambda j: min(regions[r][2] for r in regions)):
    for r, (p_avail, w_avail, _c) in regions.items():
        p_margin = p_avail / PUE[r]
        if p_req <= p_margin and w_req <= w_avail:
            admitted.append((job_id, r))
            regions[r][0] -= p_req * PUE[r]            # consume power budget
            regions[r][1] -= w_req                    # consume water budget
            break
# print admission decisions
for j in admitted:
    print("Admitted", j[0], "to", j[1])