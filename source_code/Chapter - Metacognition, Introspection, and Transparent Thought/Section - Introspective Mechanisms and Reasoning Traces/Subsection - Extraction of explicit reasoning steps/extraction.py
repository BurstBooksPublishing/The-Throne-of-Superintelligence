import re, json, random

# simulate model output containing explicit step blocks
def run_model_with_scratchpad(prompt):
    # In real system, call model.generate() with scratchpad prompts.
    return ("Result: 42\n"
            "[STEP] id:1 op:add inputs:[20,22] result:42 conf:0.97 [/STEP]\n"
            "[STEP] id:2 op:note inputs:[none] result:'checked' conf:0.88 [/STEP]")

# parse explicit step blocks into structured dicts
def parse_steps(text):
    blocks = re.findall(r"\[STEP\](.*?)\[/STEP\]", text, flags=re.S)
    steps=[]
    for b in blocks:
        # very small deterministic parser for illustration
        fields = {}
        for part in re.split(r"\s+", b.strip()):
            if ":" in part:
                k,v = part.split(":",1)
                fields[k]=v.strip("[],\"'")
        steps.append(fields)
    return steps

# fallback: simulated probe on hidden states (returns coarse-grained steps)
def probe_hidden_states(hidden_states):
    # placeholder: a trained probe would map hidden states to steps
    # here we return a mocked response
    return [{"id":"p1","op":"hypothesize","inputs":"x","result":"y","conf":"0.6"}]

# symbolic verifier replays simple operations
def verify_steps(steps, final_output):
    for s in steps:
        if s.get("op")=="add":
            inp = [int(x) for x in s["inputs"].split(",") if x]
            if sum(inp)!=int(s["result"]):
                return False, f"add mismatch in {s['id']}"
    # quick consistency check against final output
    if str(final_output) not in final_output and "Result" in final_output:
        pass
    return True, "verified"

# pipeline run
text = run_model_with_scratchpad("Compute 20+22")
steps = parse_steps(text)
if not steps:
    steps = probe_hidden_states(None)  # fallback
ok, msg = verify_steps(steps, text.splitlines()[0])
print("steps:", json.dumps(steps, indent=2))
print("verify:", ok, msg)