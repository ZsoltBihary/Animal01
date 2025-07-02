# 🧠 Animal01

**Animal01** is a biologically inspired neural simulation framework designed for agent-based learning and behavior modeling.  
It features a sparse neural architecture with explicitly defined perceptor, hidden, and motor neurons.

Work in progress ...

---

## 📁 Project Structure

```plaintext
.
├── world/
│   ├── helper.py         # Constants for cell types and actions
│   ├── print_world.py    # Pretty printing of the world state
│   └── grid_world.py     # Main environment logic
├── animal/
│   ├── custom_layers.py  # SparseLinear layer with fixed connectivity
│   └── brain.py          # BrainModel and Brain state manager
└── README.md             # You're here!
```

---

## 🧱 World Module

### `helper.py`

Defines constants:
```python
# Cell types
EMPTY, WALL, FOOD, ANIMAL = 0, 1, 2, 3
# Agent actions
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
```

---

### `print_world.py`

Renders a single board using unicode characters for visual inspection.

```bash
(A) - animal
 o  - food
 X  - wall
```

Example:
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

---

### `grid_world.py`

Core environment class:
- Manages multiple 2D grid worlds in parallel.
- Allows agents to move, interact, and receive rewards.
- Automatically respawns food over time.

Key functions:
- `reset()` — initializes new worlds.
- `step_animal(actions)` — applies agent actions and returns rewards.
- `step_environment()` — adds food if needed.

---

## 🧠 Brain Module

### `custom_layers.py`

Defines `SparseLinear`, a sparse connectivity layer:
- Each neuron connects to a random subset of `K` input neurons.
- Efficient and biologically inspired.

```python
SparseLinear(in_features=1000, out_features=500, K=20)
```

---

### `brain.py`

Two main classes:

#### `BrainModel`

A sparse neural network composed of:
- **Perceptor neurons** (receive observations)
- **Hidden + Motor neurons** (evolve over time)
- Q-value output via a linear readout

```python
model = BrainModel(n_neurons=1000, n_perceptors=50, n_motors=10, n_actions=5, K=20)
```

#### `Brain`

Manages the agent’s internal state:
- Holds a full state vector (neuron activations)
- Supports `observe()` and `think_and_update()` operations

---

## 🔁 Simulation Loop

Example (see `grid_world.py` main block):

```python
for step in range(10):
    actions = torch.randint(0, 5, (B,), device=device)
    rewards = env.step_animal(actions)
    env.step_environment()
    env.print_board(b=0)
```

---

## ✅ Features

- ✅ Batched environments (parallel simulations)
- ✅ Toroidal (wrap-around) grid behavior
- ✅ Sparse Q-learning-ready neural controller
- ✅ Modular code for extensibility
- ✅ Human-readable debug printing

---

## 🚧 Planned Extensions

- [ ] Observation model for visual field / local patch
- [ ] Learning rule for evolving brain weights
- [ ] Brain-Environment interaction logs
- [ ] RL training loop

---

## 📦 Requirements

- Python 3.8+
- PyTorch

Install dependencies:
```bash
pip install torch
```

---

## 🧪 Run Tests

Each module includes sanity checks under `__main__`:
```bash
python world/print_world.py
python world/grid_world.py
python animal/custom_layers.py
python animal/brain.py
```

---

## 🧠 Inspiration

This project is inspired by:
- Sparse evolutionary algorithms
- Biological brains (perceptors, motors, internal state)
- Reinforcement learning agents

---

## 📜 License

MIT License

---

## 🤝 Contributions

Welcome! The project is intentionally minimal — let us discuss.
