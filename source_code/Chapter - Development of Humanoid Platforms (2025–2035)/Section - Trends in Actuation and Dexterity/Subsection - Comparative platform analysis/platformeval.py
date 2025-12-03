import numpy as np

# Platform database (values are illustrative)
platforms = {
  'QDD': {'torque':120,'bw':100,'backdrive':0.3,'eff':0.90,'costk':8,'mtbf':10000},
  'SEA': {'torque':60,'bw':50,'backdrive':0.8,'eff':0.85,'costk':6,'mtbf':15000},
  'TDV': {'torque':80,'bw':30,'backdrive':0.9,'eff':0.75,'costk':10,'mtbf':12000}
}

# Normalization bounds
bounds = {'torque':(10,150),'bw':(5,150),'eff':(0.50,0.95),'costk':(4,12),'mtbf':(5000,20000)}
w = np.array([0.25,0.20,0.15,0.15,0.10,0.15])  # weights

def normalize(x,key):
    lo,hi = bounds[key]
    return np.clip((x-lo)/(hi-lo),0,1)

def cost_score(c):
    # invert cost: lower cost -> higher score
    lo,hi = bounds['costk']
    return 1.0 - np.clip((c-lo)/(hi-lo),0,1)

def score_and_feasible(p,req):
    m = np.array([
      normalize(p['torque'],'torque'),
      normalize(p['bw'],'bw'),
      p['backdrive'],
      normalize(p['eff'],'eff'),
      cost_score(p['costk']),
      normalize(p['mtbf'],'mtbf')
    ])
    S = float(w.dot(m))
    feasible = (p['torque'] >= req['tau_req']) and (p['bw'] >= req['f_req'])
    return S, feasible

# task requirements
task = {'tau_req':50,'f_req':40}
for name,p in platforms.items():
    s,ok = score_and_feasible(p,task)
    print(f"{name}: score={s:.3f}, feasible={ok}")