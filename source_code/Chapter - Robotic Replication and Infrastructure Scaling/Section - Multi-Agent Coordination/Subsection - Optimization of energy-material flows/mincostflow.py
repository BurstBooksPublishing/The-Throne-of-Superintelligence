import networkx as nx

# Build directed graph: capacities and unit energy/material costs (weight)
G = nx.DiGraph()
# add edges with capacity and cost (weight)
G.add_edge('source', 'depotA', capacity=100, weight=2)  # cost per unit
G.add_edge('depotA', 'fab1', capacity=50, weight=1)
G.add_edge('depotA', 'fab2', capacity=60, weight=2)
G.add_edge('fab1', 'storage', capacity=40, weight=1)
G.add_edge('fab2', 'storage', capacity=40, weight=1)

# supply/demand at nodes (positive = supply, negative = demand)
demand = {'source': 100, 'storage': -90, 'fab1': 0, 'fab2': 0, 'depotA': 0}

# assign node demands to graph
for n, dem in demand.items():
    G.nodes[n]['demand'] = dem

# compute min-cost flow (NetworkX implements integer-cost min-cost flow)
flow = nx.min_cost_flow(G)
# calculate cost and print flows
total_cost = nx.cost_of_flow(G, flow)
print(f"Total cost: {total_cost}")
for u in flow:
    for v, f in flow[u].items():
        if f > 0:
            print(f"{u} -> {v}: {f}")  # inline result logging