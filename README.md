# 🧠 Animal01

**Animal01** is a biologically inspired neural simulation framework designed for agent-based learning and behavior modeling.  
It features a sparse neural architecture with explicitly defined perceptor, hidden, and motor neurons.

Work in progress ...

---

## 📚 Literature Review: Reinforcement Learning Algorithms

This section provides an accessible overview of foundational and advanced Reinforcement Learning (RL) approaches, suitable for beginners looking for clear conceptual understanding without heavy mathematical prerequisites.

---

### 🧠 Core Concepts

| Category                      | Description                                                                                                    |                                                                  
| ----------------------------- |----------------------------------------------------------------------------------------------------------------| 
| **Value-Based**               | Learns the value of being in a certain state (e.g., `V(s)`), often used in dynamic programming and TD methods. |                                                                
| **Action-Value (Q-learning)** | Learns the expected reward of taking action `a` in state `s`, denoted as `Q(s, a)`.                            |                                                                  
| **Policy-Based**              | Learns a policy function directly that maps states to actions without using a value function.                  |
| **Temporal Difference (TD)**  | Updates estimates based partly on other learned estimates.                                                     |                                                                  
| **Actor–Critic**              | Combines value and policy learning using separate structures: an actor (policy) and a critic (value function). |                                                                  
| **Model-Based RL**            | Learns a model of the environment to plan and improve learning efficiency (e.g., MuZero).                      |                                                                  

---

### 📘 Recommended Resources

| Topic                | Resource                                                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Foundations**      | [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html) – Chapters 1–6 |
| **Deep RL Overview** | [A Brief Survey of Deep Reinforcement Learning (Arulkumaran et al., 2017)](https://arxiv.org/abs/1708.05866)                 |
| **TD Learning**      | [Temporal Difference Learning – Wikipedia](https://en.wikipedia.org/wiki/Temporal_difference_learning)                       |
| **Policy Gradient**  | [Policy Gradient Methods – Wikipedia](https://en.wikipedia.org/wiki/Policy_gradient_method)                                  |
| **Actor–Critic**     | [Actor–Critic Algorithms – Wikipedia](https://en.wikipedia.org/wiki/Actor-critic_algorithm)                                  |
| **Model-Based RL**   | [Model-Based Reinforcement Learning: A Survey (Moerland et al., 2020)](https://arxiv.org/abs/2006.16712)                     |

---

### 🤖 Advanced Architectures

| Architecture             | Description                                                                                  | Resource                                                                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DQN (Deep Q-Network)** | Neural extension of Q-learning for high-dimensional inputs (e.g. Atari).                     | [Arulkumaran et al., 2017](https://arxiv.org/abs/1708.05866)                                                                                                                           |
| **AlphaZero**            | Combines deep neural networks with Monte Carlo Tree Search (MCTS); used in Chess, Go, Shogi. | [AlphaZero – Wikipedia](https://en.wikipedia.org/wiki/AlphaZero)                                                                                                                       |
| **MuZero**               | Learns both the model and policy/value function; does not require a known environment model. | [MuZero – Wikipedia](https://en.wikipedia.org/wiki/MuZero), [MuZero 101 – Medium](https://medium.com/data-science/muzero-101-a-brief-introduction-to-deepminds-latest-ai-a2f1b3aa5275) |
| **EfficientZero**        | A sample-efficient version of MuZero optimized for real-world environments.                  | [EfficientZero Explained – LessWrong](https://www.lesswrong.com/posts/mRwJce3npmzbKfxws/efficientzero-how-it-works)                                                                    |

---

### 🧭 Suggested Reading Path

1. **Start** with Sutton & Barto (Ch. 1–6): MDPs, TD, Q-learning
2. **Explore** deep RL via [Arulkumaran et al.](https://arxiv.org/abs/1708.05866)
3. **Understand** policy gradients and actor–critic on Wikipedia
4. **Dive into** model-based planning via [Moerland et al.](https://arxiv.org/abs/2006.16712)
5. **Finish** with AlphaZero, MuZero, and EfficientZero

---

## 📁 Project Structure

---

## ✅ Features

- ✅ Batched environments (parallel simulations)
- ✅ Toroidal (wrap-around) grid behavior
- ✅ Sparse Q-learning-ready neural controller
- ✅ Modular code for extensibility
- ✅ Human-readable debug printing


Example grid world:
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

Test scripts are in experiments/.
Current working test scripts:

```bash
python experiments/run_simulation01.py
python experiments/run_simulation02.py
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
