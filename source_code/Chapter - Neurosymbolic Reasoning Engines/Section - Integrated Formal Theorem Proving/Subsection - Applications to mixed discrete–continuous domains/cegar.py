import z3          # SMT for discrete plan synthesis
import cvxpy as cp # convex solver for continuous feasibility

# discrete vars: mode_i True if mode A at step i
N = 5
modes = [z3.Bool(f"mode_{i}") for i in range(N)]

s = z3.Solver()
# initial symbolic constraints (example): must start in mode A
s.add(modes[0] == z3.BoolVal(True))

blocked = []  # store blocked assignments

def solve_discrete():
    for b in blocked: s.add(z3.Not(b))
    if s.check() != z3.sat: return None
    model = s.model()
    return [bool(model.eval(m)) for m in modes]

def check_continuous(plan):
    # simple convex relaxation: positions x_i and control u_i
    x = cp.Variable(N) 
    u = cp.Variable(N-1)
    constraints = []
    # dynamics: x_{i+1} = x_i + u_i
    for i in range(N-1):
        constraints += [x[i+1] == x[i] + u[i]]
    # mode-dependent bounds: if mode True tighter bounds (as example)
    for i,m in enumerate(plan):
        if m: constraints += [x[i] >= 0, x[i] <= 5]
        else:  constraints += [x[i] >= -5, x[i] <= 10]
    # obstacle avoidance convexified as interval exclusion (approx)
    constraints += [cp.abs(x[2] - 3.0) >= 0.5] 
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraints)
    try:
        prob.solve()
    except Exception:
        return False, None
    return prob.status == 'optimal', x.value

# main loop
for it in range(20):
    plan = solve_discrete()
    if plan is None:
        print("No discrete plan remains"); break
    feasible, traj = check_continuous(plan)
    if feasible:
        print("Feasible hybrid solution found", plan); break
    # block this discrete plan (concrete blocking clause)
    clause = z3.And(*[modes[i] if plan[i] else z3.Not(modes[i]) for i in range(N)])
    blocked.append(clause)
else:
    print("Exceeded iterations without feasible hybrid plan")