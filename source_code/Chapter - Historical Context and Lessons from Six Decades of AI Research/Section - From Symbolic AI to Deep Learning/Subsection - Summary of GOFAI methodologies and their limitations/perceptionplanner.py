import heapq
# simple classifier returns predicates from sensor features (placeholder)
def classify(sensor): return {'at(robot,A)', 'clear(box)'}  # mock labels

def successors(state):
    # yield (action, new_state, cost)
    if 'at(robot,A)' in state and 'clear(box)' in state:
        yield ('pick', frozenset((s for s in state if s!='clear(box)')) | {'holding(box)'}, 1)
    # add more domain rules...
    
def heuristic(state, goal):
    return 0 if goal.issubset(state) else 1  # admissible simple heuristic

def astar(start, goal):
    pq = [(0, start)]; came = {start: None}; cost = {start:0}
    while pq:
        _, cur = heapq.heappop(pq)
        if goal.issubset(cur): break
        for a, nx, c in successors(cur):
            nc = cost[cur]+c
            if nx not in cost or nc < cost[nx]:
                cost[nx]=nc; came[nx]=(cur,a); heapq.heappush(pq,(nc+heuristic(nx,goal),nx))
    # reconstruct plan
    plan=[]; s=next((s for s in came if goal.issubset(s)), None)
    while s and came[s]:
        s_prev,a = came[s]
        plan.append(a); s=s_prev
    return list(reversed(plan))

# pipeline usage
sensors = ...  # real sensor input
state = classify(sensors)
goal = {'holding(box)'}
plan = astar(frozenset(state), goal)
print(plan)  # execute with robot controllers