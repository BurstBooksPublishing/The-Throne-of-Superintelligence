import random, math
# Agent and task models
class Agent:
    def __init__(self,i,energy,capabilities,pos):
        self.i=i; self.energy=energy; self.capabilities=capabilities; self.pos=pos
    def utility(self,task):
        # fused perceptual score (capability match)
        match = sum(min(self.capabilities.get(k,0),task['req'].get(k,0)) for k in task['req'])
        dist = math.hypot(self.pos[0]-task['pos'][0], self.pos[1]-task['pos'][1])
        # LLM score simulated by randomness conditioned on task difficulty
        llm_score = 1.0/(1+task['difficulty']) + random.uniform(-0.1,0.1)
        return 1.0*match - 0.5*(dist/(self.energy+1e-3)) + 0.8*llm_score
# initialize
agents=[Agent(i,energy=100.0,capabilities={'weld':i%2,'inspect':1},pos=(i,i%3)) for i in range(6)]
tasks=[{'id':j,'req':{'weld':1} if j%2 else {'inspect':1},'pos':(j%3,j),'difficulty':j%4} for j in range(6)]
# decentralized rounds: each agent bids on top task locally, broadcasts top-1 (simulated)
assignments={}  # task_id -> agent_id
for round in range(5):
    bids=[]
    for a in agents:
        # local perception fusion + LLM reasoning -> score list
        scores=[(t['id'], a.utility(t)) for t in tasks if t['id'] not in assignments]
        if not scores: continue
        best_id, best_score = max(scores, key=lambda x:x[1])
        bids.append((best_id, a.i, best_score))
    # simulate local resolution: highest bid wins each task
    for task_id in set(b[0] for b in bids):
        winners=[b for b in bids if b[0]==task_id]
        winner=max(winners, key=lambda x:x[2])
        assignments[task_id]=winner[1]
    # simple execution update
    for t,a_id in list(assignments.items()):
        # decrement energy and remove completed tasks probabilistically
        agent=agents[a_id]; agent.energy-=10
        if random.random()<0.8: tasks=[tk for tk in tasks if tk['id']!=t] 
print("final assignments",assignments)