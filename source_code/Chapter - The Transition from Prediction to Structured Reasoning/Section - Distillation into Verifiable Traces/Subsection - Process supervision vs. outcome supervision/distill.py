import torch, torch.nn as nn

# teacher: returns (action, trace) given observation
# student: predicts action and trace components
def distill_step(obs, teacher, student, verifier, optimizer, alpha=0.3):
    action_teacher, trace_teacher = teacher(obs)           # teacher rollout
    action_pred, trace_pred = student(obs)                # student prediction

    # outcome loss (e.g., cross-entropy on action)
    loss_outcome = nn.CrossEntropyLoss()(action_pred, action_teacher)

    # process loss: per-step binary verification plus L2 on symbolic embeddings
    per_step_ok = torch.tensor([verifier(step) for step in trace_pred]).float()
    loss_process = 1.0 - per_step_ok.mean()                # penalize unverifiable steps

    # combined loss and optimization
    loss = (1 - alpha) * loss_outcome + alpha * loss_process
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    return {'L': loss.item(), 'S': per_step_ok.mean().item()}