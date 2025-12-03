class CausalCounterfactualGenerator:
    def generate_rare_events(self, base_scenario, interventions):
        counterfactuals = []
        for intervention in interventions:
            intervened = self.causal_model.do(base_scenario, intervention)
            trajectory = self.physics_engine.simulate(intervened)
            observations = self.render_trajectory(trajectory)
            counterfactuals.append({
                'intervention': intervention,
                'observations': observations,
                'ground_truth': self.extract_labels(trajectory)
            })
        return counterfactuals  # 100x diverse causal data