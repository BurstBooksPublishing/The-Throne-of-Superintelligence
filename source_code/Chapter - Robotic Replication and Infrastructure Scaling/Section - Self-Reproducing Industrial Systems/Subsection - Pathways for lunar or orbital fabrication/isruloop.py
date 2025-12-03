import numpy as np
# Sensors: lidar, hyperspec, thermal -> fused map (mock data)
lidar = np.load("lidar.npy")            # point cloud
hyperspec = np.load("hyperspec.npy")    # spectral cubes
thermal = np.load("thermal.npy")        # temperature map

def fuse_sensors(lidar, hyperspec, thermal):
    # coarse fusion: elevation + composition + temp per grid cell
    grid = {}  # {cell_id: {'elev':..., 'spec':..., 'temp':...}}
    # ... implement voxel aggregation and feature extraction ...
    return grid

def llm_plan_task(grid_cell):
    # call to fine-tuned LLM planner (pseudo-call)
    # outputs: sequence of actions with preconditions and verification checks
    plan = [
      {"action":"excavate","duration":3600},
      {"action":"transfer","duration":600},
      {"action":"process_regolith","method":"electrolysis","duration":7200},
    ]
    return plan

def execute_plan(plan):
    for step in plan:
        # check sensors, power, and safety monitors before each step
        # send low-level motor commands to manipulators (omitted)
        pass

grid = fuse_sensors(lidar, hyperspec, thermal)
cell = select_best_cell(grid)            # heuristic selection (highest yield)
plan = llm_plan_task(cell)
execute_plan(plan)                       # real-time monitored execution