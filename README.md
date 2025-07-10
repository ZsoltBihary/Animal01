# Animal Project

## General Goal
To understand basic **Reinforcement Learning (RL)** concepts through a concrete, simple example. Implement a simple environment and a decision-making agent that interacts with the environment, performs actions, and receives rewards.

---

## Minimum Program

1. Study RL literature, understand **Q-learning (QL)** and **Deep Q-learning (DQL)** algorithms.
2. Implement QL and DQL for our simple world.
3. Observe the behavior of our animal, and have fun.

---

## Proposed Simple World

### Agent ("Animal")

- **Actions**: `STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT` (5 possible actions)
- **Decision model ("Brain")**: Based on the Q(s, a) action-value function.
- **Decision algorithm**:
  - Given a state `s`, use the Q-function to obtain the 5 q(s, a) values.
  - Generate policy probabilities via **softmax** (with very low temperature).
  - Select an action according to the policy (balances resolution between near-equal Q-values).
  - During rollout, combine this with ε-uniform random exploration.

### Environment ("Terrain")

```
 ------------------------------
| X           o              X |
| o    (A)                     |
|    X              X        X |
| X                            |
| X                    X     o |
|    o  o        o             |
|       o                 X    |
 ------------------------------
```

- A rectangular grid (H × W) with 3 types of cells:
  - `EMPTY`
  - `FOOD`
  - `POISON`
- The animal's position is tracked.
- **Observation**: A square window of size (K × K), centered on the animal.
- **Action resolution**:
  - Animal moves according to the action (with **periodic boundary conditions**).
  - Rewards are given as defined below.
  - Consumed FOOD or POISON is regenerated randomly on an EMPTY cell far from the animal.

### Reward Rules

- Every action except `STAY`: **−1**
- Landing on `FOOD`: **+100**
- Landing on `POISON`: **−100**
- Reward is **normalized** with `(1 - γ)` (see literature).

---

## Animal / Brain Types

| Type     | Behavior Description |
|----------|----------------------|
| **Sponge** | `action = STAY` |
| **Amoeba** | `action = random_action` |
| **Insect** | `action = act(observation)` — stateless, reactive |
| **Reptile** | `brain_state = observe(brain_state, observation)`<br>`action = act(brain_state)` — has memory |
| **Mammal** | Same as Reptile, but with hard-wired architecture for memory/preprocessing |
| **Human** | Internally simulates the world.<br>If rules are known → **AlphaZero**.<br>If rules are learned → **MuZero** |

💡 **Focus for now**: Implement **Amoeba** (benchmark) and **Insect**. Reptile support may be considered in the interface.

---

## Algorithms to Implement

### 1. Q-Learning (QL) with Table

- Q(s, a) is stored in a lookup table.
- Feasible for small observation windows (K = 3), up to the Insect level.
- If converged, gives optimal policy.

### 2. Q-Learning with Heuristics

- Q(s, a) is modeled as a **parametrized function**.
- Optimize parameters through training.
- Effectively equivalent to DQL training pipeline.

### 3. Deep Q-Learning (DQL)

- Q(s, a) is modeled as a **neural network**:
  - `Q: s → q_a`
- Enables scalable, generalizable learning.

### Practical Deep Q-Learning Training Strategy

- **Rollout policy**: Based on the `Q_online` model
- **Experience buffer**: Stores `(state, action, TD-target)` tuples during rollout
- **Training**:
  - Performed asynchronously or periodically, decoupled from rollout
  - Only a few epochs per batch (e.g. 1–4), depending on throughput
  - Uses buffered data with standard TD loss:  
  L = (Q_online(s, a; θ) − y)², where  
  y = r + γ * maxₐ′ Q_target(s′, a′)
- **Q_target update**: Occasional hard or soft copy of `Q` to `Q_target`
- **Parallel rollout**: Hundreds of environment instances generate rollouts concurrently for high throughput

---

## Literature Review

### Core RL Texts

- **Reinforcement Learning: An Introduction** (Sutton & Barto) – Chapters 1–6
- **A Brief Survey of Deep Reinforcement Learning** (Arulkumaran et al., 2017)

### DQL Resources

- [Minh 2017 DQL Slides (UIUC)](https://courses.grainger.illinois.edu/cs546/sp2018/Slides/Apr05_Minh.pdf)

### World Simulation Models

- [AlphaZero Paper (DeepMind)](https://arxiv.org/abs/1712.01815)
- [MuZero Paper (DeepMind)](https://arxiv.org/abs/1911.08265)
- [World Models (2018)](https://arxiv.org/abs/1803.10122)
- [World Models (2023)](https://arxiv.org/abs/2301.04104)

### MuZero in Complex Environments

- [MuZero for Stochastic / Partially Observable Envs (ICLR 2022)](https://iclr.cc/virtual/2022/spotlight/6833?utm_source=chatgpt.com)
- [OpenReview: MuZero (2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1)

### Model Optimization

- [Vector Quantized Models (Ozair et al., 2021)](https://proceedings.mlr.press/v139/ozair21a/ozair21a.pdf)
- [EfficientZero: How It Works (LessWrong)](https://www.lesswrong.com/posts/mRwJce3npmzbKfxws/efficientzero-how-it-works?utm_source=chatgpt.com#5__Conclusions)

### GitHub Projects

- [MuZero General (Gomoku example)](https://github.com/werner-duvaud/muzero-general/blob/master/games/gomoku.py)
- [MuZero in gym-minigrid](https://github.com/mit-acl/gym-minigrid)

---

## License

_This project is intended for educational and research purposes._

---
