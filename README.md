# Animal Project

## General Goal
To understand basic **Reinforcement Learning (RL)** concepts through a concrete, simple example. 
Implement a simple environment and a decision-making agent that interacts with the environment, 
performs actions, and receives rewards.

---

## Minimum Program

1. Study RL literature, understand **Q-learning (QL)** and **Deep Q-learning (DQL)** algorithms.
2. Implement QL and DQL for our simple world.
3. Observe the behavior of our animal, and have fun.

---

## Proposed Simple World

### Agent ("Animal")

- **Actions**: `STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT` (5 possible actions)
- **Decision model**: Based on the Q(s, a) action-value function.
- **Decision algorithm**:
  - Given a state `s`, use the Q-function to obtain the 5 q_a(s) values.
  - Generate policy probabilities via **softmax** (with low temperature).
  - Select an action according to the policy (temperature balances decision between near-equal q_a values).
  - During rollout, combine this with ε-uniform random exploration.

---

### Animal Taxonomy

```
Animal
├── Protozoan          ← includes agents that can act, but cannot learn
│   ├── * Sponge
│   └── * Amoeba
└── Metazoan           ← includes all Q-learning-capable agents
    ├── Arthropod      ← includes stateless Q-learning-capable agents
    │   └── * Insect
    └── Vertebrate     ← includes memory/state-based Q-learning-capable agents
        ├── * Reptile
        └── * Mammal
```

This taxonomy hierarchy is implemented as abstract and derived classes. They represent not only biological,
but also algorithmic differences. Here is a detailed breakdown for the leaf classes:

| Class       | Behavior Description                                                                                                                                              |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Sponge**  | `action = STAY` — trivial                                                                                                                                         |
| **Amoeba**  | `action = random_action` — random benchmark                                                                                                                       |
| **Insect**  | `state = encode(observation)`<br/>`q_a = Q_model(state)`<br/>`action = select(q_a)` — reactive; no internal brain state, no memory                                |
| **Reptile** | `brain_state = imprint(brain_state, observation)`<br>`q_a, brain_state = Q_model(brain_state)`<br/>`action = select(q_a)` — thinking; has brain state, and memory |
| **Mammal**  | Same as Reptile, but with hard-wired architecture                                                                                                                 |
| **Human**   | Internally simulates the world.<br>If rules are known → **AlphaZero**.<br>If rules are learned → **MuZero**                                                       |

💡 **Focus for now**: Implement **Amoeba** (benchmark) and **Insect**. Reptile support may be considered in the interface.


### Environment ("Grid")

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

- A rectangular grid (H × W) with 4 types of cells:
  - `EMPTY`
  - `FOOD`
  - `POISON`
  - `ANIMAL`
- The animal's position is also tracked.
- **Observation**: A square window of size (K × K), centered on the animal.
- **Action resolution**:
  - Animal moves according to the action (with **periodic boundary conditions**).
  - Rewards are given as defined below.
  - Consumed FOOD or POISON is regenerated randomly far from the animal, to keep their density roughly constant.

### Reward Rules

- Every action except `STAY`: **−1**
- Landing on `FOOD`: **+100**
- Landing on `POISON`: **−100**
- Reward is **normalized** with `(1 - γ)`, so integrated reward/step is on the same scale as the per step rewards.

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
- **Replay buffer**: Stores `(state, action, q_target)` tuples during rollout
- **Training**:
  - Can be moved to GPU
  - Performed asynchronously or periodically, decoupled from rollout
  - Uses buffered data with standard TD loss:  
  L = (Q_online(s, a; θ) − y)², where  
  y = r + γ * max_a' (Q_target(s′, a′))
- **Q_target update**: Occasional hard or soft copy of `Q_online` to `Q_target`
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
