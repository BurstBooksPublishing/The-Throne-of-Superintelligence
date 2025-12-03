# Minimal example: generate chain-of-thought, synthesize a python expression, execute and verify.
import json, subprocess, sys

def simple_generator(prompt):
    # Lightweight deterministic "generator" for demo; replace with a real LLM call.
    # Handles arithmetic word problems of the form "If A has X and B Y..."
    if "If" in prompt and "have" in prompt:
        # parse numbers and produce steps
        nums = [int(s) for s in prompt.split() if s.isdigit()]
        expr = "+".join(map(str, nums))
        cot = f"Step1: add numbers -> {expr}\nStep2: evaluate -> {eval(expr)}"
        prog = f"result = {expr}"
        return cot, prog
    return "No chain", "result = None"

def execute_program(prog):
    # Execute synthesized program in a controlled namespace.
    ns = {}
    exec(prog, {"__builtins__": {}}, ns)  # restricted exec for safety in examples
    return ns.get("result")

def verify_with_alternative(prompt, main_ans):
    # Simple verification: recompute via a different decomposition.
    # In practice use independent model or symbolic solver.
    cot2, prog2 = simple_generator(prompt)  # independent trace in real system
    alt_ans = execute_program(prog2)
    return main_ans == alt_ans, alt_ans

# Demo problem
problem = "If Alice has 3 apples and Bob has 4 apples, how many apples do they have together?"
cot, program = simple_generator(problem)      # cognition: generate CoT + program
answer = execute_program(program)             # action: execute
ok, alt = verify_with_alternative(problem, answer)  # governance: cross-verify
print(json.dumps({"cot": cot, "program": program, "answer": answer, "verified": ok}))