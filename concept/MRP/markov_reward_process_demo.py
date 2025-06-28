"""
markov_reward_process_demo.py
"""

# -------- 1. Imports ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# -------- 2. Define a tiny, concrete MRP ------------------------------------
# States
states = ["Home", "Work", "Gym", "Bar"]

# Transition matrix P (rows = current state, cols = next state)
P = np.array([
#To: Home  Work  Gym   Bar
    [0.10, 0.60, 0.20, 0.10],   # From Home
    [0.20, 0.60, 0.10, 0.10],   # From Work
    [0.30, 0.50, 0.10, 0.10],   # From Gym
    [0.40, 0.40, 0.05, 0.15],   # From Bar
])

# Immediate (expected) rewards for being in each state
R = np.array([+1, +5, +3, +0])      # earn salary at Work, endorphins at Gym …

# Discount factor
gamma = 0.9

print("States:", states)
print("Immediate rewards R:", R)
print("Transition matrix P:\n", P)

# -------- 3. Analytic value function: solve (I - γP)V = R --------------------
I = np.eye(len(states))
V_exact = np.linalg.solve(I - gamma * P, R)

print("\nExact state values:")
for s, v in zip(states, V_exact):
    print(f"  V({s}) = {v:0.3f}")

# -------- 4. Numerical value iteration (iterative policy evaluation) ---------
def value_iteration(P, R, gamma, tol=1e-6, max_iter=1_000):
    V = np.zeros(len(R))
    for i in range(max_iter):
        V_new = R + gamma * P.dot(V)
        if np.max(np.abs(V_new - V)) < tol:
            print(f"\nConverged after {i+1} iterations")
            return V_new
        V = V_new
    raise RuntimeError("Value iteration did not converge")

V_est = value_iteration(P, R, gamma)

# Verify the two methods match
assert np.allclose(V_exact, V_est, atol=1e-5)

# -------- 5. Simulate a random trajectory ------------------------------------
def simulate(P, R, start_state=0, steps=15, gamma=0.9, seed=42):
    rng = np.random.default_rng(seed)
    s = start_state
    history = []
    G = 0.0           # discounted return
    for t in range(steps):
        r = R[s]
        G += (gamma ** t) * r
        history.append((t, states[s], r, G))
        s = rng.choice(len(states), p=P[s])
    return history

trajectory = simulate(P, R, start_state=0, steps=10)

print("\nSample trajectory:")
for t, s, r, G in trajectory:
    print(f" t={t:2d} | state={s:<4} | reward={r:+.0f} | running return={G:5.2f}")

# -------- 6. (Optional) Plot value estimates vs iteration --------------------
def plot_convergence(P, R, gamma, max_iter=30):
    V = np.zeros(len(R))
    history = [V.copy()]
    for _ in range(max_iter):
        V = R + gamma * P.dot(V)
        history.append(V.copy())
    history = np.stack(history)

    for i, s in enumerate(states):
        plt.plot(history[:, i], label=s, marker="o")
    plt.title("Value iteration convergence")
    plt.xlabel("Iteration")
    plt.ylabel("V(s)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_convergence(P, R, gamma)
