# Markov Reward Process (MRP) — Quick Notes

## 1. MRP in One Sentence

A Markov Reward Process is a **state machine** that assigns a _score_ (reward) to every state and tells you the _probability_ of hopping to the next state.
The goal: compute **how much long-term goodness** you’ll collect if you start in any state and follow the machine forever (with future rewards discounted).

---

## 2. Formal Ingredients

| Symbol        | What it is                                                             | In the demo                      |
| ------------- | ---------------------------------------------------------------------- | -------------------------------- |
| $\mathcal{S}$ | Finite set of **states**                                               | `["Home", "Work", "Gym", "Bar"]` |
| $P$           | **Transition matrix**; row $s$ = probs of next state                   | `P` 4 × 4 array                  |
| $R$           | **Reward vector**; one number per state                                | `R = [1, 5, 3, 0]`               |
| $\gamma$      | **Discount factor**, $0\le\gamma\le1$                                  | `gamma = 0.9`                    |
| $V(s)$        | **State value** = total expected discounted reward if you start in $s$ | computed by code                 |

---

## 3. Intuition for $V(s)$

1. **Grab today’s reward** $R(s)$.
2. **Roll the dice**: step to a new state according to row $s$ of $P$.
3. **Repeat forever**, but shrink the value of future rewards by $\gamma^{t}$.
4. **Average** over all random trajectories. That _average_ is $V(s)$.

Mathematically (Bellman expectation equation):

$$
V(s) \;=\; R(s)\;+\;\gamma \sum_{s'} P_{s,s'}\,V(s')
$$

---

## 4. How Value Iteration Finds $V$

1. **Initialize** $V^{(0)} = 0$.
2. **Loop** until change $<\text{tol}$:

   $$
   V^{(k+1)} \;\leftarrow\; R \;+\; \gamma\,P\,V^{(k)}
   $$

3. **Converged** $V^{(\infty)}$ satisfies the Bellman equation above.

The Python demo prints:
Converged after 144 iterations

meaning the maximum change between sweeps fell below `tol = 1e-6`.

---

## 5. Interpreting the Demo Output

| Console line          | What it tells you                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Exact state values:` | Final $V(s)$ from linear-algebra solve—ground truth.                                                            |
| `Converged after …`   | How many sweeps the iterative method needed to match that truth.                                                |
| `Sample trajectory:`  | One random 10-step walk beginning at **Home**. `running return` shows $\sum_{t=0}^k \gamma^{\,t} R_{t}$ so far. |

**Example**
`V(Work) = 35.665` → If you teleport to **Work** right now and then let life unfold probabilistically, the average discounted sum of salaries, endorphins, etc. you’ll rack up is ≈ 35.7 reward-units.

---

## 6. First 10 Value-Iteration Sweeps (from the table)

- **Iteration 0**: $V^{(0)} = [0,0,0,0]$.
- **Iteration 1**: one-step look-ahead, so $V^{(1)} = R$.
- Each subsequent iteration mixes in today’s reward plus discounted tomorrow’s guess.
- By Iteration 10 the numbers resemble the final values; the remaining ~130 sweeps just polish decimals.

---

## 7. Things to Experiment With

- **Change $\gamma$**
  - $\gamma = 0$ → “live for today,” so $V(s)=R(s)$.
  - $\gamma = 1$ → “care equally about all future,” watch some state values blow up or diverge if the Markov chain isn’t _proper_.
- **Edit rewards or transitions** to model new scenarios (student life, website clicks…).
- **Monte-Carlo estimate** $V(s)$: run thousands of trajectories, average their returns, compare to analytic $V$.

---
