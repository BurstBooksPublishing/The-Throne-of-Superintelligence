import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Simulator stub (replace with real API) ---
def simulate_event(n_samples, bias=0.2):
    # returns (rgb, depth, imu, label, sim_density)
    X_rgb = np.random.randn(n_samples, 3, 64, 64).astype(np.float32) # mock images
    X_depth = np.random.randn(n_samples, 1, 64, 64).astype(np.float32)
    X_imu = np.random.randn(n_samples, 6).astype(np.float32)
    # label: 1 for near-miss, 0 otherwise; bias oversamples near-miss
    labels = (np.random.rand(n_samples) < bias).astype(np.int64)
    # sim_density: probability under simulator sampling policy
    sim_density = np.clip(0.5 + 0.5*(labels - 0.5), 1e-6, 1.0)
    return X_rgb, X_depth, X_imu, labels, sim_density

# --- Estimate target density (placeholder KDE or domain estimate) ---
def estimate_target_density(labels):
    # domain expert assigns low prior to rare events
    return np.where(labels==1, 0.01, 0.99)

# --- Small fusion model ---
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_conv = nn.Sequential(nn.Conv2d(4,8,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(8+6,32), nn.ReLU(), nn.Linear(32,2))
    def forward(self, rgb, depth, imu):
        x = torch.cat([rgb, depth], dim=1)
        v = self.img_conv(x).view(x.size(0), -1)
        z = torch.cat([v, imu], dim=1)
        return self.fc(z)

# --- Pipeline ---
X_rgb, X_depth, X_imu, labels, sim_density = simulate_event(1024, bias=0.3)
target_density = estimate_target_density(labels)
weights = target_density / sim_density                         # importance weights
weights = np.clip(weights, 1e-3, 10.0)

# convert to tensors
rgb_t = torch.from_numpy(X_rgb)
depth_t = torch.from_numpy(X_depth)
imu_t = torch.from_numpy(X_imu)
y_t = torch.from_numpy(labels)
w_t = torch.from_numpy(weights.astype(np.float32))

model = FusionNet()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(reduction='none')

# one epoch of weighted training
model.train()
for i in range(0, len(y_t), 64):
    rgb_b = rgb_t[i:i+64]; depth_b = depth_t[i:i+64]; imu_b = imu_t[i:i+64]
    y_b = y_t[i:i+64]; w_b = w_t[i:i+64]
    logits = model(rgb_b, depth_b, imu_b)
    loss = loss_fn(logits, y_b)
    weighted_loss = (loss * w_b).mean()                          # apply importance weights
    opt.zero_grad(); weighted_loss.backward(); opt.step()