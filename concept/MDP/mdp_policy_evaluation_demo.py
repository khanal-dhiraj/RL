"""
mdp_policy_evaluation_demo.py
"""

import numpy as np

# ---------- 1. States & actions ----------
states = ["Home", "Work", "Gym", "Bar"]
actions = ["Primary", "Alternate"]

S, A = len(states), len(actions)

# ---------- 2. Build transition tensor P[s,a,s′] ----------
P = np.zeros((S, A, S))

# -- From Home
P[0, 0] = [0.05, 0.80, 0.10, 0.05]   # Primary = Commute
P[0, 1] = [0.60, 0.00, 0.00, 0.40]   # Alternate = Chill
# -- From Work
P[1, 0] = [0.10, 0.70, 0.15, 0.05]   # Primary = Work Hard
P[1, 1] = [0.60, 0.00, 0.20, 0.20]   # Alternate = Go Home
# -- From Gym
P[2, 0] = [0.05, 0.30, 0.50, 0.15]   # Primary = Fitness
P[2, 1] = [0.60, 0.10, 0.00, 0.30]   # Alternate = Go Home
# -- From Bar
P[3, 0] = [0.40, 0.10, 0.00, 0.50]   # Primary = Drink
P[3, 1] = [0.80, 0.10, 0.00, 0.10]   # Alternate = Head Home

# ---------- 3. Rewards (state-only for simplicity) ----------
R = np.array([1, 5, 3, 0])
gamma = 0.9

# ---------- 4. Define a fixed policy π ----------
# 70 % take Primary action, 30 % Alternate, everywhere
pi = np.full((S, A), [0.7, 0.3])

# ---------- 5. Pre-compute “MRP under π” ----------
# Weighted transition matrix under policy π
P_pi = np.einsum('sa,saj->sj', pi, P)   # shape (S, S′)

# If rewards depended on actions you’d need R_pi[s] = Σ_a π(s,a) R(s,a)
R_pi = R                                 # state-only reward here

# ---------- 6. Analytic policy evaluation ----------
I = np.eye(S)
V_exact = np.linalg.solve(I - gamma * P_pi, R_pi)

print("Exact state values under π:")
for s, v in zip(states, V_exact):
    print(f"  Vπ({s}) = {v:0.3f}")

# ---------- 7. Iterative policy evaluation ----------
def policy_eval(P_pi, R_pi, gamma, tol=1e-6, max_iter=1_000):
    V = np.zeros(len(R_pi))
    for i in range(max_iter):
        V_new = R_pi + gamma * P_pi.dot(V)
        if np.max(np.abs(V_new - V)) < tol:
            print(f"Converged after {i+1} iterations")
            return V_new
        V = V_new
    raise RuntimeError("Didn't converge")

V_iter = policy_eval(P_pi, R_pi, gamma)
assert np.allclose(V_exact, V_iter, atol=1e-5)

# ---------- 8. Simulate a run following π ----------
rng = np.random.default_rng(1)
s = 0          # start at Home
G = 0
print("\nSample trajectory following π:")
for t in range(10):
    r = R[s]
    G += (gamma ** t) * r
    print(f" t={t:2d} | state={states[s]:<4} | reward={r:+} | return={G:5.2f}")
    # choose action by π
    a = rng.choice(A, p=pi[s])
    s = rng.choice(S, p=P[s, a])
