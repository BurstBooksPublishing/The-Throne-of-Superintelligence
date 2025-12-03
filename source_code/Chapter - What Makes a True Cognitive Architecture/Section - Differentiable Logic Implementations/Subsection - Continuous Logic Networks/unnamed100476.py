class ContinuousLogicGate(torch.nn.Module):
    def __init__(self, input_dim, gate_type='AND'):
        super().__init__()
        self.gate_type = gate_type
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, inputs):
        if self.gate_type == 'AND':
            # Min-like operation
            return torch.prod(inputs, dim=-1, keepdim=True)
        elif self.gate_type == 'OR':
            # Max-like operation  
            return 1 - torch.prod(1 - inputs, dim=-1, keepdim=True)
        elif self.gate_type == 'NOT':
            return 1 - inputs
        else:
            # Learned gate
            return self.net(inputs)