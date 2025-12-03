import torch, math
torch.manual_seed(0)
# Simple utility nets: old and new (simulated candidate update)
class Net(torch.nn.Module):
    def __init__(self): super().__init__(); self.lin=torch.nn.Linear(8,1)
    def forward(self,x): return self.lin(x).squeeze(-1)

old = Net(); new = Net()
# Simulate small update: copy then perturb new parameters
new.load_state_dict(old.state_dict())
for p in new.parameters(): p.data += 0.02*torch.randn_like(p)  # candidate change

# Critical state set S_crit (canonicalized features)
S = torch.randn(200,8)  # replace with domain-specific scenarios
U_old = old(S).detach()
U_new = new(S).detach()

# Coherence metrics
max_abs_change = torch.max(torch.abs(U_new - U_old)).item()
# Pairwise sign-preservation fraction (sample pairs)
idx = torch.randperm(len(S))[:100]  # sample indices
pairs = [(i,j) for i,j in zip(idx[::2], idx[1::2])]
preserve = 0
for i,j in pairs:
    preserve += (torch.sign(U_old[i]-U_old[j]) == torch.sign(U_new[i]-U_new[j])).item()
preserve_rate = preserve/len(pairs)

print(f"max_abs_change={max_abs_change:.4f}, preserve_rate={preserve_rate:.3f}")
# Decision gate
EPS_MAG, EPS_RANK = 0.5, 0.95
if max_abs_change <= EPS_MAG and preserve_rate >= EPS_RANK:
    print("PASS: issue certificate and allow update")
else:
    print("FAIL: reject update or escalate to human review")