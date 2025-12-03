import torch, torch.nn as nn, torch.optim as optim

# Perception encoders (image and text)
class Perception(nn.Module):
    def __init__(self, img_dim=128, txt_dim=64, emb=256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, emb)   # image embedding
        self.txt_proj = nn.Linear(txt_dim, emb)   # text embedding

    def forward(self, img, txt):
        return self.img_proj(img), self.txt_proj(txt)

# Fusion + reasoning (single transformer block)
class FusionReasoning(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb, num_heads=8)
        self.ff = nn.Sequential(nn.Linear(emb, emb*4), nn.ReLU(), nn.Linear(emb*4, emb))

    def forward(self, img_emb, txt_emb):
        # concatenate as sequence: [img, txt]
        seq = torch.stack([img_emb, txt_emb], dim=0)
        attn_out, _ = self.attn(seq, seq, seq)
        return self.ff(attn_out.mean(dim=0))  # pooled context

# Action head with safety gating (simple threshold)
class ActionHead(nn.Module):
    def __init__(self, emb=256, action_dim=8):
        super().__init__()
        self.mlp = nn.Linear(emb, action_dim)
    def forward(self, ctx):
        action = self.mlp(ctx)
        # safety gate: clip magnitude (hardware risk control)
        return torch.clamp(action, -1.0, 1.0)

# Assemble pipeline
model = nn.Sequential(
    Perception(),           # returns tuple; handle by simple wrapper below
)

# Wrap full step for clarity
perception = Perception()
fusion = FusionReasoning()
action = ActionHead()
params = list(perception.parameters()) + list(fusion.parameters()) + list(action.parameters())
opt = optim.Adam(params, lr=1e-4)
loss_fn = nn.MSELoss()

# Dummy dataset (engineer-provided provenance tags)
for epoch in range(10):
    img = torch.randn(32, 128)    # batch of image features
    txt = torch.randn(32, 64)     # batch of text features
    target_action = torch.randn(32, 8)  # supervision from logged behavior

    img_emb, txt_emb = perception(img, txt)
    ctx = fusion(img_emb, txt_emb)
    pred = action(ctx)
    loss = loss_fn(pred, target_action)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # diagnostic control
    opt.step()

    if epoch % 2 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")  # simple monitor