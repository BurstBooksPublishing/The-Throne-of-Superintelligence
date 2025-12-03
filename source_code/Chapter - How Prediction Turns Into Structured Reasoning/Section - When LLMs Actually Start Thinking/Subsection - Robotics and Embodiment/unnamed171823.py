def verify_reasoning_chain(chain, num_verifiers=32):
    step_confidences = []
    for step_idx, step in enumerate(chain):
        context = chain[:step_idx]
        verifier_scores = [verifier.verify(context, step) 
                          for _ in range(num_verifiers)]
        step_confidence = np.mean(verifier_scores)
        step_confidences.append(step_confidence)
    return min(step_confidences) >= 0.98 and np.prod(step_confidences) > 0.90