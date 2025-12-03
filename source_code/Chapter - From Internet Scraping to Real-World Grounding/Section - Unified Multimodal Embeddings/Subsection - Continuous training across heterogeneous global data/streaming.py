import torch
import torch.nn as nn
import torch.optim as optim
import random

# Minimal encoder and adapter definitions (executable)
class CoreEncoder(nn.Module):
    def __init__(self, dim=256): super().__init__(); self.net = nn.Linear(512, dim)
    def forward(self, x): return nn.functional.normalize(self.net(x), dim=-1)

class Adapter(nn.Module):
    def __init__(self, dim=256): super().__init__(); self.a = nn.Linear(dim, dim)
    def forward(self, h): return nn.functional.relu(self.a(h))

# Reservoir sampler for bounded replay
class Reservoir:
    def __init__(self, capacity): self.capacity=capacity; self.buffer=[]; self.count=0
    def add(self, item):
        self.count += 1
        if len(self.buffer) < self.capacity: self.buffer.append(item)
        else:
            i = random.randrange(self.count)
            if i < self.capacity: self.buffer[i] = item
    def sample(self, k):
        return random.sample(self.buffer, min(k, len(self.buffer)))

# Contrastive loss (temp=0.1)
def contrastiveLoss(zA, zB, temp=0.1):
    logits = zA @ zB.t()
    logits = logits / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
    return loss

# Instantiate models, optimizer, and replay
core = CoreEncoder()
targetCore = CoreEncoder()  # for EMA
adapter = Adapter()
opt = optim.Adam(list(adapter.parameters()) + list(core.net.parameters()), lr=1e-4)
replay = Reservoir(capacity=10000)
tau = 0.995  # EMA factor

# Streaming training loop (skeleton)
for step, (modalA, modalB, provenance) in enumerate(streamingDataGenerator()):
    # preprocess -> feature vectors (application-specific)
    xA = preprocess(modalA); xB = preprocess(modalB)
    hA = core(xA); hB = core(xB)
    # apply adapters for domain adaptation
    ha = adapter(hA); hb = adapter(hB)
    # primary loss + replay loss
    lossMain = contrastiveLoss(ha, hb)
    replayBatch = replay.sample(64)
    lossReplay = torch.tensor(0.0)
    if replayBatch:
        rA = torch.stack([r[0] for r in replayBatch]); rB = torch.stack([r[1] for r in replayBatch])
        lossReplay = contrastiveLoss(core(rA), core(rB))
    # regularization as L2 deviation from EMA target
    reg = 0.0
    for p, q in zip(core.parameters(), targetCore.parameters()):
        reg = reg + ((p - q).pow(2).sum())
    loss = lossMain + 1e-3 * reg + 0.5 * lossReplay
    opt.zero_grad(); loss.backward(); opt.step()
    # update EMA target
    with torch.no_grad():
        for p, q in zip(core.parameters(), targetCore.parameters()):
            q.data.mul_(tau).add_(p.data, alpha=1-tau)
    # add representative pair to replay (store processed features)
    replay.add((xA.detach(), xB.detach()))
    # periodic validation, gating, and checkpointing (omitted for brevity)