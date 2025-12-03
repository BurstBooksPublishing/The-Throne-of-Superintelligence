import torch
import torch.nn as nn
# coarse predictor: small MLP for long-horizon structure
class CoarseNet(nn.Module):
    def __init__(self, inp, out): super().__init__(); self.net=nn.Sequential(nn.Linear(inp,64),nn.ReLU(),nn.Linear(64,out))
    def forward(self,x): return self.net(x)
# fine refiner: PINN-style network takes spatial coord and time
class FinePINN(nn.Module):
    def __init__(self, inp, out): super().__init__(); self.net=nn.Sequential(nn.Linear(inp,128),nn.Tanh(),nn.Linear(128,out))
    def forward(self,x): return self.net(x)
# PDE residual for 1D heat equation u_t - k u_xx = 0
def pde_residual(u, x, t, k=0.1):
    # compute derivatives via autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - k * u_xx
# instantiate
coarse = CoarseNet(inp=10, out=20)
fine = FinePINN(inp=2, out=1)  # (x,t) -> temperature
opt = torch.optim.Adam(list(coarse.parameters())+list(fine.parameters()), lr=1e-3)
mse = nn.MSELoss()
for batch in dataloader:  # dataloader provides coarse_inputs, y_coarse, x_coord, t_coord, y_fine
    coarse_inputs, y_coarse, x, t, y_fine = batch
    opt.zero_grad()
    # coarse loss
    y_hat_coarse = coarse(coarse_inputs)
    L_coarse = mse(y_hat_coarse, y_coarse)
    # fine predictions and physics residual
    x.requires_grad_(True); t.requires_grad_(True)
    inp = torch.cat([x,t], dim=-1)
    u_pred = fine(inp)
    L_fine = mse(u_pred, y_fine)  # data term
    R = pde_residual(u_pred, x, t)  # physics residual
    L_phys = torch.mean(R**2)
    # consistency term (simple upsample match)
    y_hat_upsampled = upsample(y_hat_coarse)  # implement upsample mapping
    L_cons = mse(y_hat_upsampled, u_pred.detach())  # one-way consistency
    loss = 1.0*L_coarse + 10.0*L_fine + 100.0*L_phys + 1.0*L_cons
    loss.backward(); opt.step()  # joint update