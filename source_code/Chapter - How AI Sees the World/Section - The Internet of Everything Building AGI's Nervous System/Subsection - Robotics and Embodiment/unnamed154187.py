class FederatedSensorFusion:
    def __init__(self, global_model):
        self.global_model = global_model
        self.device_models = {}
        
    def local_update(self, device_id, local_data):
        if device_id not in self.device_models:
            self.device_models[device_id] = copy.deepcopy(self.global_model)
        
        # Train on local data
        local_model = self.device_models[device_id]
        optimizer = torch.optim.Adam(local_model.parameters())
        
        for batch in local_data:
            prediction = local_model(batch['inputs'])
            loss = self.correction_loss(prediction, batch['targets'])
            loss.backward()
            optimizer.step()
        
        return local_model.state_dict()
    
    def global_aggregation(self, updates):
        # Average model updates
        global_state = {}
        for param_name in self.global_model.state_dict():
            global_state[param_name] = torch.mean(
                torch.stack([update[param_name] for update in updates]),
                dim=0
            )
        self.global_model.load_state_dict(global_state)