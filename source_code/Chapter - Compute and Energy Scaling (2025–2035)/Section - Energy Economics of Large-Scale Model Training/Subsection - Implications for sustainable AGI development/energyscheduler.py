import numpy as np
# simple forecasts (hourly for 24h)
price = np.array([0.12,0.11,0.10,0.09,0.08,0.09,0.13,0.20,0.25,0.22,0.18,0.15,
                  0.14,0.13,0.12,0.11,0.10,0.09,0.10,0.11,0.12,0.13,0.14,0.13])
ci = price*100  # proxy: carbon intensity correlated with price (gCO2eq/kWh)
# parameters
train_energy_per_hour = 2.0  # MWh when full precision
edge_infer_energy = 0.0002   # MWh per inference (edge)
cloud_infer_energy = 0.0001  # MWh per inference (cloud, more efficient)
latency_edge = 50  # ms
latency_cloud = 120  # ms
latency_req = 100
# schedule: pick training hours with price below threshold
price_threshold = 0.12
train_hours = np.where(price < price_threshold)[0]
# adjust precision to meet energy budget; higher precision consumes more energy
def precision_scale(hour):
    # lower precision during high CI to reduce emissions (example policy)
    return 0.75 if ci[hour] > 12 else 1.0
# inference routing decision by hour
def route_inference(hour):
    # prefer cloud if latency still acceptable and ci lower
    if latency_cloud <= latency_req and ci[hour] < 15:
        return 'cloud'
    return 'edge'
# simulate decisions
schedule = []
for h in range(24):
    decide_train = h in train_hours
    prec = precision_scale(h)
    route = route_inference(h)
    schedule.append((h, decide_train, prec, route))
# print compact plan
for h,dt,prec,route in schedule:
    print(f"hour {h}: train={dt}, precision={prec:.2f}, route={route}")