import pybullet as p
import pybullet_data
import json, random, time
# connect headless
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def run_episode(urdf="r2d2.urdf", steps=240, param_seed=None):
    random.seed(param_seed)
    p.resetSimulation()
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF(urdf, basePosition=[0,0,0.1])
    # counterfactual parameterization
    gravity = random.uniform(-12.0, -6.0)  # vary gravity
    friction = random.uniform(0.2, 1.2)    # vary contact friction
    p.setGravity(0,0,gravity)
    p.changeDynamics(plane, -1, lateralFriction=friction)
    p.changeDynamics(robot, -1, lateralFriction=friction)
    traces = []
    for t in range(steps):
        # simple open-loop or randomized action for diversity
        force = [random.uniform(-10,10) for _ in range(3)]
        p.applyExternalForce(robot, -1, force, [0,0,0], p.LINK_FRAME)
        p.stepSimulation()
        pos, orn = p.getBasePositionAndOrientation(robot)
        # simple sensor model: noisy range to origin
        range_meas = (pos[0]**2+pos[1]**2+pos[2]**2)**0.5 + random.gauss(0,0.01)
        traces.append({"t":t,"pos":pos,"orn":orn,"range":range_meas})
    meta = {"gravity":gravity,"friction":friction,"seed":param_seed,"urdf":urdf}
    return {"meta":meta,"traces":traces}

# example: generate and save N episodes emphasizing low-friction rare events
dataset = []
for i in range(50):
    seed = int(time.time()*1000) % 2**31
    ep = run_episode(param_seed=seed)
    # importance weight: overweight low friction episodes
    weight = 2.0 if ep["meta"]["friction"]<0.4 else 1.0
    ep["meta"]["weight"]=weight
    dataset.append(ep)
with open("counterfactual_dataset.json","w") as f:
    json.dump(dataset,f)
p.disconnect()