import torch
import torch.nn as nn
import torch.optim as optim

# Perception encoder (simulated)
encoder = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32))

# Small adapter for TTT (only this is updated online)
adapter = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
adapter_opt = optim.SGD(adapter.parameters(), lr=0.01)

# Simple vector store (embeddings and associated traces)
db_embeddings = torch.randn(100,32)  # stored context embeddings
db_traces = torch.randn(100,10)      # short action/state traces

# Two experts
expert_a = nn.Linear(32,16)  # generalist
expert_b = nn.Linear(32,16)  # specialist

# Gating network for routing
gating = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,2), nn.Softmax(dim=-1))

def retrieve(q_emb, k=5):
    # nearest neighbors by dot product (simulated)
    scores = db_embeddings @ q_emb
    idx = scores.topk(k).indices
    return db_traces[idx], db_embeddings[idx]

def ttt_adapt(q_emb, traces, steps=2):
    # self-supervised prediction: predict next trace vector
    for _ in range(steps):
        pred = adapter(q_emb)               # predict using adapter
        loss = ((pred - traces.mean(dim=0))**2).mean()  # simple target
        adapter_opt.zero_grad()
        loss.backward()
        adapter_opt.step()

def inference_step(raw_input):
    q = encoder(raw_input)                 # perception
    traces, ctx_embs = retrieve(q)         # retrieval
    ttt_adapt(q, traces)                   # test-time training (adapter updated)
    q_post = adapter(q)                    # adapted representation
    gate = gating(q_post)                  # routing probabilities
    out_a = expert_a(q_post)
    out_b = expert_b(q_post)
    out = gate[0]*out_a + gate[1]*out_b    # fused expert output
    return out

# Simulate one query
raw = torch.randn(128)
result = inference_step(raw)
print(result.shape)  # (16,)