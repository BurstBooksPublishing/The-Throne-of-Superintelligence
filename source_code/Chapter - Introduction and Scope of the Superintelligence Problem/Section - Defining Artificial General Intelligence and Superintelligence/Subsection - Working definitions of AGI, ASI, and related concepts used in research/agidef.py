import numpy as np
import random
# sensor fusion (simulated) -> state estimate
def fuse_sensors(cam, lidar): return np.concatenate([cam, lidar])
# lightweight planner: queries an LLM-like oracle (simulated here)
def planner(state, goal):
    # return discrete action sequence; replace with real LLM call
    if state.mean() < goal: return ["move_forward","grip"]
    return ["turn","observe"]
# executor: applies actions and returns reward (task perf)
def execute(actions, env): 
    # simple environment step; real system would call robot APIs
    reward = env.step(actions)
    return reward
# evaluation loop across tasks in distribution T
def evaluate(agent, tasks, env_factory):
    scores=[]
    for task in tasks:
        env=env_factory(task)            # construct task environment
        state=fuse_sensors(env.cam, env.lidar)
        actions=planner(state, task.goal) # cognition step
        perf=execute(actions, env)       # action step
        scores.append(perf)
    return np.mean(scores)