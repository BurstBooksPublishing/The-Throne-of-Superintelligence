import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def circ_conv(a, b):                    # HRR circular convolution (numpy)
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return np.real(np.fft.ifft(fa * fb))

def circ_corr(c, b):                    # HRR circular correlation (numpy)
    fc = np.fft.fft(c)
    fb = np.fft.fft(b)
    return np.real(np.fft.ifft(fc * np.conj(fb)))

# Small differentiable implication: maps condition vector to consequence vector
class ImplicationNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)   # learnable implication operator
    def forward(self, cond):
        return F.normalize(self.lin(cond), dim=-1)

# Pipeline demo
dim = 512
# sensor embedding -> symbolic projection (here random for demo)
sensor_emb = torch.randn(dim)            # perceptual embedding
role = torch.from_numpy(np.random.randn(dim)).float()
filler = sensor_emb
# bind using FFT via torch (convert to numpy for HRR conv demo)
a = role.numpy(); b = filler.numpy()
c = circ_conv(a, b)                      # bound vector stored in memory
memory = torch.from_numpy(c).float()
# retrieve candidate filler via correlation
retrieved = torch.from_numpy(circ_corr(memory.numpy(), role.numpy())).float()
# apply differentiable implication
impl = ImplicationNet(dim)
consequence = impl(retrieved)            # predicted consequence in VSA space
# similarity check for downstream action selection
score = F.cosine_similarity(consequence.unsqueeze(0), torch.randn(1,dim), dim=-1)
print('action_score', score.item())