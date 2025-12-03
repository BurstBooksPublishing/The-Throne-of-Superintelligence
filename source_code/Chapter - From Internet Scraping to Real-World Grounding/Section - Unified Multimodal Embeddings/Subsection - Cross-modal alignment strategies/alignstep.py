# Simplified single-step update demonstrating pipeline components.
# enc_v, enc_t, enc_s: encoders; proj_*: projection heads; dyn_pred: latent forward model
# optimizer is already created.
def train_step(batch):
    v, t, s, a, s_next = batch  # tensors: images, text tokens, state, action, next_state
    z_v = proj_v(enc_v(v))      # image -> common latent
    z_t = proj_t(enc_t(t))      # text -> common latent
    z_s = proj_s(enc_s(s))      # proprio-> common latent
    z_s_next = proj_s(enc_s(s_next))

    # Contrastive (InfoNCE) between vision and text
    logits = z_v @ z_t.T / temperature
    labels = torch.arange(len(v), device=logits.device)
    loss_nce = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

    # Latent forward prediction: predict z_s_next from z_s and action
    pred = dyn_pred(torch.cat([z_s, a], dim=-1))          # predict next latent
    loss_pred = F.mse_loss(pred, z_s_next)                # grounding error

    # Cycle consistency (simple symmetric reconstruction)
    recon_t = dec_t(inv_proj_t(z_v))                      # optional decoder/inverse
    loss_cycle = F.cross_entropy(recon_t_logits(recon_t), t_labels)

    loss = loss_nce + lambda_dyn * loss_pred + lambda_cycle * loss_cycle
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()