import math

def energy_per_flop(t, e0=1e-9, alpha=0.2, e_min=1e-11):
    # t: years since baseline
    return e0 * math.exp(-alpha*t) + e_min

def job_budget(F, t, p_elec=0.05, pue=1.2, amort_per_kwh=0.02):
    e_flop = energy_per_flop(t)
    E_j = F * e_flop * pue            # joules
    kwh = E_j / 3.6e6                 # convert to kWh
    cost_energy = kwh * p_elec
    cost_amort = kwh * amort_per_kwh
    return {'kwh': kwh, 'cost_total': cost_energy + cost_amort}

# Example: 1e24 FLOPs, 2 years after baseline
if __name__ == '__main__':
    F = 1e24
    out = job_budget(F, t=2.0)
    print(f"Energy (kWh): {out['kwh']:.3e}, Total cost ($): {out['cost_total']:.2f}")