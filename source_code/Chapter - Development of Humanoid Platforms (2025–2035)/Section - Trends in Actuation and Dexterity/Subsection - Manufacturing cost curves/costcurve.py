import numpy as np
import matplotlib.pyplot as plt

# Parameters for two actuator types (F, v, b)
actuators = {
    'tendon': {'F':50e6, 'v':8e3, 'b':0.20},
    'smart_servo': {'F':5e6, 'v':4e3, 'b':0.10}
}

N = np.logspace(0,6,500)  # cumulative units
plt.figure(figsize=(6,4))
for name,p in actuators.items():
    c = p['F']/N + p['v']*N**(-p['b'])  # unit cost model
    plt.loglog(N, c, label=name)
    # print production needed to reach $6k unit cost
    target = 6000
    idx = np.where(c <= target)[0]
    if idx.size:
        print(name, 'reach', target, 'at N=', int(N[idx[0]]))
    else:
        print(name, 'does not reach', target, 'within simulated N range')
plt.xlabel('Cumulative units N')
plt.ylabel('Unit cost (USD)')
plt.legend()
plt.grid(True,which='both',ls=':')
plt.show()