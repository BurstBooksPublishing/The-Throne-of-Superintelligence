import torch, torch.nn as nn, torch.optim as optim
import random
# simple 1D dynamics simulator
def step(s,a): return s + 0.1*a + 0.01*random.gauss(0,1)
# model: predict next state from state and action
class Dyn(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(2,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x): return self.net(x)
model, opt = Dyn(), optim.Adam(Dyn().parameters(), lr=1e-3)  # trainable model
# dataset buffer
buffer = []
for epoch in range(200):  # collect and train online
    s = random.uniform(-1,1)
    a = random.uniform(-1,1)  # exploratory action
    s1 = step(s,a)
    buffer.append((s,a,s1))
    if len(buffer)>64:
        batch = random.sample(buffer,64)
        x = torch.tensor([[b[0],b[1]] for b in batch], dtype=torch.float32)
        y = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)
        pred = model(x)
        loss = ((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # novelty score: prediction error for intrinsic reward
    with torch.no_grad():
        inp = torch.tensor([[s,a]], dtype=torch.float32)
        err = ((model(inp) - torch.tensor([[s1]]) )**2).item()
    # log or use err to bias next action sampling (higher err -> explore)
    # (policy update omitted for brevity)
print("final prediction error", err)