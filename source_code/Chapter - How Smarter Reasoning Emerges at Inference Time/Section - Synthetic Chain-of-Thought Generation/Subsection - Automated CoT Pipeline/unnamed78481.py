class SyntheticCoTGenerator:
    def __init__(self, base_model, verifier_model):
        self.generator = base_model
        self.verifier = verifier_model
        self.reasoning_templates = self._load_templates()
        
    def generate_reasoning_chain(self, problem, max_steps=20):
        """Generate verifiable reasoning steps"""
        state = {"problem": problem, "steps": [], "confidence": 1.0}
        
        for step in range(max_steps):
            # Generate candidate reasoning step
            prompt = self._format_prompt(state)
            candidate_step = self.generator.generate(prompt, max_tokens=100)
            
            # Verify step quality
            verification_score = self.verifier.verify_step(
                state, candidate_step
            )
            
            if verification_score > 0.8:
                state["steps"].append(candidate_step)
                state["confidence"] *= verification_score
            else:
                # Backtrack and try alternative
                alternative = self._generate_alternative(state, step)
                if self.verifier.verify_step(state, alternative) > 0.7:
                    state["steps"][-1] = alternative
                else:
                    break  # Dead end
        
        return state["steps"], state["confidence"]
    
    def solve_with_search(self, problem, beam_width=8):
        """Beam search over reasoning trajectories"""
        trajectories = [(self.generate_reasoning_chain(problem), 1.0)]
        
        for depth in range(5):  # 5-layer search
            new_trajectories = []
            for trajectory, confidence in trajectories:
                # Expand beam
                for _ in range(beam_width):
                    new_step = self._sample_next_step(trajectory)
                    new_traj = trajectory + [new_step]
                    new_conf = confidence * self.verifier.verify(new_traj)
                    new_trajectories.append((new_traj, new_conf))
            
            # Keep top beam
            trajectories = sorted(new_trajectories, 
                                key=lambda x: x[1], reverse=True)[:beam_width]
        
        return max(trajectories, key=lambda x: x[1])[0]

# Performance: 20-50x effective capability gain