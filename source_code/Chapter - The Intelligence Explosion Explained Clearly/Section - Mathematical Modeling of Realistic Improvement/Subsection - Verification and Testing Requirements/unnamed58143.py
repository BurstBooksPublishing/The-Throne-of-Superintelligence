import numpy as np
import matplotlib.pyplot as plt

def realistic_improvement_trajectory(years=10, quarters_per_year=4):
    """Model self-improvement with physical constraints"""
    t = np.arange(years * quarters_per_year)
    
    # Initialize
    fitness = np.zeros(len(t))
    fitness[0] = 1.0  # GPT-5 baseline
    
    # Resource trajectories (external growth)
    compute_growth = 1.15  # 50% annual compute growth
    data_growth = 1.08     # 38% annual data growth
    
    compute_available = 1.0
    data_available = 1.0
    
    for i in range(1, len(t)):
        # Algorithmic improvement (compute-limited)
        algo_gain = 0.05 * fitness[i-1] * compute_available
        
        # Data-driven improvement
        data_gain = 0.03 * fitness[i-1] * data_available
        
        # Rare architectural breakthroughs
        arch_gain = 0.10 if i % 8 == 0 else 0.0  # Every 2 years
        
        # Total improvement per quarter
        total_gain = algo_gain + data_gain + arch_gain
        
        fitness[i] = fitness[i-1] * (1 + total_gain)
        
        # External resource growth
        compute_available *= compute_growth ** (1/4)
        data_available *= data_growth ** (1/4)
    
    return t/4, fitness

# Generate 10-year trajectory
time_years, capability = realistic_improvement_trajectory(10)

# Plot results
plt.figure(figsize=(10, 6))
plt.semilogy(time_years, capability, 'b-', linewidth=2, label='Realistic Growth')
plt.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='10x Human')
plt.axhline(y=100, color='g', linestyle='--', alpha=0.7, label='100x Human')
plt.xlabel('Years from 2025')
plt.ylabel('Capability Multiplier')
plt.title('Realistic Self-Improvement Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"10-year improvement factor: {capability[-1]:.1f}x")