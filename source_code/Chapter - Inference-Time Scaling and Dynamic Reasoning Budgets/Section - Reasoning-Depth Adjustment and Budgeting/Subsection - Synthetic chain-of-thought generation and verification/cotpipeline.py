import random, math
# Mock LLM API wrappers (replace with real SDK calls).
def generate_traces(llm, context, n): return [llm.sample(context) for _ in range(n)]
def verify_trace(verifier, trace, context): return verifier.score(trace, context)  # 0..1
def syntactic_check(trace): return 1.0 if "consistency" in trace else 0.5  # simple check
def execute_check(simulator, trace): return simulator.run(trace)  # returns 0..1 empirical pass

def select_best_trace(context, llm, verifier, simulator, budget_lambda=0.01):
    traces = generate_traces(llm, context, n=8)                    # stochastic diversity
    scored = []
    for t in traces:
        s_syn = syntactic_check(t)                                 # cheap
        s_ver = verify_trace(verifier, t, context)                 # learned verifier
        s_exe = execute_check(simulator, t)                        # optional probe
        cost = len(t) * 0.001                                       # proxy compute cost
        utility = 0.4*s_syn + 0.5*s_ver + 0.1*s_exe - budget_lambda*cost
        scored.append((utility, t))
    return max(scored, key=lambda x: x[0])[1]                       # choose highest utility

# Example usage with mock objects
# action = plan_executor.execute(select_best_trace(sensor_context, llm_api, verifier_api, sim))