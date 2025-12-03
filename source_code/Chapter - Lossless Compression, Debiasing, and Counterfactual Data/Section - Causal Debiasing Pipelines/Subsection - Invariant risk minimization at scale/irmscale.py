import torch
# encoder: phi, classifier: f, optimizer declared externally
def irm_step(batch_by_env, encoder, classifier, optimizer, lambda_penalty):
    # batch_by_env: dict(env_id -> (x,y) tensors) on local shard
    total_loss = 0.0
    penalty = 0.0
    for env, (x, y) in batch_by_env.items():
        z = encoder(x)                         # shared representation
        logits = classifier(z)                 # base classifier output
        loss_e = torch.nn.functional.cross_entropy(logits, y)
        total_loss += loss_e
        # scalar scale trick for IRMv1
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        scaled_logits = logits * scale
        loss_scaled = torch.nn.functional.cross_entropy(scaled_logits, y)
        grad_w = torch.autograd.grad(loss_scaled, scale, create_graph=True)[0]
        penalty += grad_w.pow(2)
    loss = total_loss + lambda_penalty * penalty
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), penalty.item()