import numpy as np
import matplotlib.pyplot as plt

W = 1e15            # FLOPs per step (toy)
f = 1e13            # FLOPs per device
S = 1e10            # bytes to sync per step
B_net = 1e10        # bytes/s per effective network pipe
L = 1e-6            # s latency term
def T(N):
    # ring all-reduce approx: g(N)=1, h(N)=0
    return W/(N*f) + S/B_net + L*0
Ns = np.arange(1,129)
times = [T(n) for n in Ns]
plt.plot(Ns, times); plt.xlabel('Devices'); plt.ylabel('Per-step time (s)')
plt.grid(True); plt.show()
# comment: shows flattening when S/B_net dominates compute term