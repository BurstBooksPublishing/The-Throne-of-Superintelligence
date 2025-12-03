import torch, random
# model: base_transformer + adapter bank + routing_net (simple interfaces)
# env: step(obs) -> (next_obs, reward, done)
buffer = []  # replay buffer
for stage in ["pretrain","grounding","online_adapt"]:
    for episode in range(stage_length[stage]):
        obs = env.reset()
        done = False
        while not done:
            # perception: encode sensors
            enc = base_transformer.encode(obs)  # fixed backbone encoding
            # routing: select adapters (sparse gating)
            gate = routing_net(enc)             # returns adapter index
            action = adapter_bank[gate].policy(enc)  # task-specific policy
            next_obs, reward, done = env.step(action)
            buffer.append((obs,action,reward,next_obs,gate))
            obs = next_obs
        # periodic update: prioritized sampling and constrained adapter updates
        batch = sample_prioritized(buffer, batch_size=64)  # importance sampled
        loss = compute_loss(batch, base_transformer, adapter_bank)
        # freeze backbone during grounding; allow adapters to train
        for p in adapter_parameters(adapter_bank): p.grad = None
        loss.backward()
        optimizer.step()  # optimizer targets adapter params only