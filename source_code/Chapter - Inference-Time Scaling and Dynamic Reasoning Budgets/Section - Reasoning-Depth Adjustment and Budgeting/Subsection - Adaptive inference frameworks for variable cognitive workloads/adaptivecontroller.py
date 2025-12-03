import time, math, random

# Simulated primitives (replace with real model APIs)
def perceive(sensor): return {"features": sensor}            # perception
def retrieve(query, k): return ["doc"+str(i) for i in range(k)]  # retrievals
def reason(state, depth, docs):
    # returns (answer, confidence, trace_steps)
    base_conf = 0.6 + 0.08*depth + 0.05*min(len(docs),2)
    noise = random.normalvariate(0, 0.02)
    return ("answer", min(0.999, base_conf+noise), ["step"]*depth)
def verify(answer, trace): return random.random() < 0.9  # simple verifier

class AdaptiveController:
    def __init__(self, max_latency=2.0, lambda_cost=1.0):
        self.max_latency = max_latency
        self.lambda_cost = lambda_cost

    def marginal_gain(self, prev_conf, new_conf, value=1.0):
        return (new_conf - prev_conf) * value

    def cost_of_step(self): return 0.1  # seconds per reasoning step
    def cost_of_retrieval(self, k): return 0.05*k

    def run(self, sensor):
        start = time.time()
        state = perceive(sensor)
        depth = 1
        docs = retrieve(state, k=1)
        ans, conf, trace = reason(state, depth, docs)
        # iterative refinement loop
        while True:
            elapsed = time.time() - start
            if elapsed + self.cost_of_step() > self.max_latency: break
            # simulate candidate next step
            depth += 1
            candidate_ans, candidate_conf, candidate_trace = reason(state, depth, docs)
            mg = self.marginal_gain(conf, candidate_conf)
            cost = self.lambda_cost * self.cost_of_step()
            if mg <= cost: break  # stop if marginal gain not worth cost
            ans, conf, trace = candidate_ans, candidate_conf, candidate_trace
            # optionally trigger extra retrievals
            if conf < 0.8:
                docs += retrieve(state, k=1)                  # expand context
                time.sleep(self.cost_of_retrieval(1))
            if not verify(ans, trace): break  # stop if verification fails
        return {"answer": ans, "confidence": conf, "depth": depth, "elapsed": time.time()-start}

# Run controller
if __name__ == "__main__":
    ctrl = AdaptiveController(max_latency=1.0, lambda_cost=0.8)
    out = ctrl.run("sensor_blob")  # execute pipeline, real systems pass tensors
    print(out)  # observe depth, confidence, elapsed