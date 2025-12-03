import time, math, random

# hardware interface stub (replace with real GPIO/serial)
class HardwareInterface:
    def engage_hardware_cutoff(self): 
        print("HARDWARE CUTOFF ENGAGED")  # real code would toggle relay

hw = HardwareInterface()

def read_imu(): return (0.0 + random.gauss(0,0.01))  # simulated sensors
def read_vo():  return (0.01 + random.gauss(0,0.01))
def read_enc(): return (0.0 + random.gauss(0,0.02))

def median(values): return sorted(values)[len(values)//2]

WATCHDOG_TIMEOUT = 0.2  # seconds
RESIDUAL_THRESHOLD = 0.05

last_heartbeat = time.time()
while True:
    t0 = time.time()
    imu = read_imu(); vo = read_vo(); enc = read_enc()
    estimates = [imu, vo, enc]
    fused = median(estimates)               # simple voting/fusion
    residuals = [abs(x - fused) for x in estimates]
    # diagnostic checks
    if max(residuals) > RESIDUAL_THRESHOLD:
        hw.engage_hardware_cutoff()         # hardware-enforced fail-stop
        break
    # watchdog heartbeat check (heartbeat must be refreshed by planner)
    if time.time() - last_heartbeat > WATCHDOG_TIMEOUT:
        hw.engage_hardware_cutoff()
        break
    # simulate heartbeat update from cognition occasionally
    if random.random() < 0.95: last_heartbeat = time.time()
    time.sleep(0.01)  # loop cadence