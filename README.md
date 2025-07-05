# 🧠 Animal01

**Animal01** is a biologically inspired neural simulation framework designed for agent-based learning and behavior modeling.  
It features a sparse neural architecture with explicitly defined perceptor, hidden, and motor neurons.

Work in progress ...

---

## 📁 Project Structure


---

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

Test scripts are in experiments/
Current working test scripts:

```bash
python experiments/run_simulation01.py
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
