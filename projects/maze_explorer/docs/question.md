# Modular Project: **Curriculum-Based Maze Explorer**

Design an agent that learns to solve procedurally generated mazes while scaling from toy problems to large, partially-observable ones. Each "module" below is a self-contained set of **questions / challenges** that build logically on the previous work. Tackle them in order or skip around—the dependencies are noted.

## **Module 0 – Warm-up & Scaffolding**

1. **Environment spec:** _What state, action, and reward structure will you use for a grid-maze world?_
2. **Procedural generation:** _How will you algorithmically create mazes of tunable size and difficulty (e.g., DFS vs. Prim)?_
3. **Evaluation harness:** _What metrics (return, steps, success rate) and logging tools will you add from day one?_

## **Module 1 – Planning Baseline** (Dependency: 0)

1. _With full knowledge of the transition graph, implement Value Iteration._
2. _How does convergence speed change with γ and maze size?_
3. _What's the theoretical bound on the number of Bellman updates you need for ε-optimality in this domain?_

## **Module 2 – Tabular Model-Free Control** (Dependency: 1)

1. _Implement SARSA and Q-Learning; which explores more efficiently on small mazes?_
2. _Investigate ε-decay schedules: Which one hits a 90% success-rate fastest?_
3. _Why might SARSA outperform Q-Learning in a "windy" maze with stochastic slips?_

## **Module 3 – Function Approximation** (Dependency: 2)

1. _Replace the table with tile coding or radial basis features—what breaks?_
2. _Demonstrate divergence (or stability) when you raise α; can you explain it via the semi-gradient update rule?_
3. _Design a tiny neural network approximator: Which layer width offers the best bias-variance trade-off on 15×15 mazes?_

## **Module 4 – Deep Q-Networks** (Dependency: 3)

1. _Build a classic DQN with replay & target network._
2. _Quantify over-estimation bias and implement Double DQN—how much does it shrink the Q-value gap?_
3. _Try Dueling DQN: Does it help on sparse-reward mazes beyond 25×25? Why or why not?_

## **Module 5 – Exploration & Intrinsic Motivation** (Dependency: 4)

1. _Add a simple count-based bonus; how does sample complexity scale with state-space size?_
2. _Swap in Random Network Distillation (RND). On what maze sizes does RND begin to pay off?_
3. _Devise an ablation to separate "better exploration" from "extra reward shaping."_

## **Module 6 – Curriculum Learning** (Dependency: 5)

1. _Design a level-generator that gradually increases maze diameter and branching factor._
2. _Compare three curricula (size-based, bottleneck-based, random). Which one yields the steepest learning curve?_
3. _Prove (empirically or theoretically) whether the curriculum induces faster Bellman error shrinkage._

## **Module 7 – Transfer & Generalization** (Dependency: 6)

1. _Freeze the agent after curriculum training and test on mazes with novel wall textures or layouts: how does zero-shot performance drop?_
2. _Fine-tune for 10k steps—does catastrophic forgetting appear?_
3. _Which representation layers hold environment-agnostic features? How can you measure that?_

## **Module 8 – Model-Based RL & MPC** (Dependency: 4, optional 6)

1. _Learn a probabilistic dynamics model; what loss do you minimize?_
2. _Use Model-Predictive Control with the learned model: how many planning rollouts are cost-effective versus latency?_
3. _Compare PETS-style ensembles to a single deterministic model on out-of-distribution mazes._

## **Module 9 – Partial Observability & Memory** (Dependency: 5)

1. _Mask 70% of the maze with "fog of war." Which RNN architecture (GRU, LSTM, Transformer) best captures necessary memory?_
2. _Does adding a learned belief state (Bayesian filter) beat pure recurrent policies?_
3. _What failure modes emerge when the observation noise level rises?_

## **Module 10 – Hierarchical RL Capstone** (Depends on anything ≥ 5)

1. _Define subgoal options (e.g., "reach key," "unlock door"). How do you discover them automatically?_
2. _Implement Option-Critic or hierarchical PPO: do temporally extended actions improve sample efficiency on 40×40 mazes?_
3. _Evaluate transfer: can options learned on one maze topology accelerate learning on a totally new generator family?_
