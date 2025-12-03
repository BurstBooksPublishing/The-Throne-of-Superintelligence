import pandas as pd
import numpy as np

df = pd.read_csv('decision_log.csv')            # load trace
w = df['criticality'].astype(float).values      # task weights
h = df['human_intervention'].astype(int).values

# Autonomy metric (Eq. 1)
A = (w * (1 - h)).sum() / w.sum()

# Discovery efficacy components (Eq. 2)
N = df['novelty'].clip(0,1).mean()              # normalized novelty
R = df['replication'].mean()                   # reproducibility rate
V = df['validation'].clip(0,1).mean()          # validation quality
C = df['cost'].sum()
T_v = df.loc[df['time_to_validation']>0,'time_to_validation'].median()
beta = 0.01                                     # time-to-cost weight
E = (N * R * V) / (C + beta * (T_v if not np.isnan(T_v) else 0))

# Information efficiency (Eq. 3)
eta = df['info_gain'].sum() / (df['cost'].sum() + 1e-9)

# Oversight metrics
B = df['human_time'].sum() / max(len(df),1)
L = df.loc[df['human_time']>0,'human_time'].median()

print(f"Autonomy A={A:.3f}, Efficacy E={E:.3e}, InfoEff eta={eta:.3e}")