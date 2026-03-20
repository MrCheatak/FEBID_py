import numpy as np

# Manually run a simple test
# Parameters
S = 1
Phi = 1700
n0 = 2.7
tau = 0.0001
sigma = 0.022
J0 = 9.2e5
beam_sigma = 8

# Grid
R = 100
N = 800
r = np.linspace(0, R, N+1)

# Initial condition (start_full=True)
k_min = S * Phi / n0 + 1.0 / tau
theta_init = S * Phi / k_min
print(f"Initial theta = {theta_init:.4f}")

# Beam profile
J = J0 * np.exp(-0.5 * (r / beam_sigma) ** 2)

# Analytical solution at t=0.002
t = 0.002
k = S * Phi / n0 + (1.0 / tau) + sigma * J
theta_inf = np.divide(S * Phi, k, out=np.zeros_like(k), where=k>0)
theta_analytical = theta_inf + (theta_init - theta_inf) * np.exp(-k * t)

print(f"\nD=0 analytical solution at t={t}:")
print(f"  theta(r=0) = {theta_analytical[0]:.4f}")
print(f"  theta_inf(r=0) = {theta_inf[0]:.4f}")
print(f"  k(r=0) = {k[0]:.2f}")
print(f"  min theta = {theta_analytical.min():.4f}")
print(f"  max theta = {theta_analytical.max():.4f}")

