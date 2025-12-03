import numpy as np

# Parameters (tunable for deployment)
e_electrical_per_bit = 1e-12   # J/bit (electrical SerDes)
e_optical_per_bit = 2e-13     # J/bit (co-packaged optics)
latency_optical_per_bit = 1e-9 # s/bit
latency_local = 5e-6          # s for NMC compute setup
w_energy, w_latency = 0.7, 0.3 # cost weights

def task_cost(data_bytes, ops, use_optical=True):
    bits = data_bytes*8
    if use_optical:
        move_e = bits * e_optical_per_bit
        move_t = bits * latency_optical_per_bit
        compute_e = ops * 1e-12  # approximate compute energy per op
        compute_t = ops * 1e-9   # approximate compute time per op
        total_e = move_e + compute_e
        total_t = move_t + compute_t
    else: # near-memory compute (NMC)
        move_e = 0.0
        move_t = latency_local
        compute_e = ops * 2e-12  # local compute may be slightly less efficient
        compute_t = ops * 2e-9
        total_e = move_e + compute_e
        total_t = move_t + compute_t
    return w_energy*total_e + w_latency*total_t, total_e, total_t

# Pipeline: perception -> fusion -> cognition (example tasks)
tasks = [
    {'name':'perception', 'data':1024, 'ops':1e6},
    {'name':'fusion', 'data':8192, 'ops':5e6},
    {'name':'cognition','data':4096, 'ops':2e7},
]

for t in tasks:
    cost_o, e_o, t_o = task_cost(t['data'], t['ops'], use_optical=True)
    cost_n, e_n, t_n = task_cost(t['data'], t['ops'], use_optical=False)
    choice = 'optical' if cost_o < cost_n else 'near-memory'
    print(f"{t['name']}: choose {choice}; optical E={e_o:.3e} J, NMC E={e_n:.3e} J")