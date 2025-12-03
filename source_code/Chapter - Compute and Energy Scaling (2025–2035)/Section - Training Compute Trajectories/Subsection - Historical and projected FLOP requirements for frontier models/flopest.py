#!/usr/bin/env python3
import math

# Inputs: model size and raw data collection assumptions
N_params = 175e9              # model parameters (example 175B)
sensor_hours = 1000.0         # total sensor-hours collected
tokens_per_sec_per_channel = 10  # token equivalent per second per sensor channel
channels = 4                  # multimodal channels (vision, lidar, text, audio)

# Algorithmic constant (forward+backward work per param per token)
c = 6.0

# Convert sensor-hours to total tokens
seconds = sensor_hours * 3600.0
N_tokens = seconds * tokens_per_sec_per_channel * channels

# Compute FLOPs
FLOPs = c * N_params * N_tokens

# Energy projection: hardware efficiency (FLOPs per joule)
flops_per_joule = 1e12  # example: 1 TFLOP/J (varies by accelerator and precision)
energy_J = FLOPs / flops_per_joule
energy_kWh = energy_J / 3.6e6

print(f"Model params: {N_params:.3e}")
print(f"Tokens: {N_tokens:.3e}")
print(f"Total FLOPs: {FLOPs:.3e}")
print(f"Energy: {energy_kWh:.3f} kWh")

# Simple multi-year projection using eq. (2)
def project_F(F0, years, gamma_raw, gamma_algo):
    return F0 * ((1+gamma_raw)/(1+gamma_algo))**years

F0 = FLOPs
for yrs in (1, 3, 5, 10):
    F_proj = project_F(F0, yrs, gamma_raw=1.0, gamma_algo=0.2)  # conservative scenario
    print(f"{yrs} years -> FLOPs: {F_proj:.3e}")