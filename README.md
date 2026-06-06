# The Princess Quest

> A fun reinforcement learning project disguised as a game.

A grid-based desktop chase game where the monster chasing you isn't
scripted — it's a **trained Deep Reinforcement Learning agent** that
learns your movement patterns in real time and gets better at catching
you the longer you play.

Built from scratch as a personal project to explore applied Deep RL
in a fun, interactive context. Every part of the pipeline — environment
design, agent training, model export, and game development — is
implemented manually without high-level RL frameworks.

---

## The Game

You control the **Hero** on a 10×10 grid. Your goal is simple: reach
the **Princess** before the **Monster** catches you.

Nothing stays still. The Princess jumps across the board every few
seconds to keep you moving. The Monster gets faster as time passes.
Walls are randomised every round so no two games feel the same.

### Win & Lose

| Condition                      | Result   |
| ------------------------------ | -------- |
| Hero reaches the Princess cell | You win  |
| Monster reaches the Hero cell  | You lose |

### Difficulty

| Time Elapsed | Monster Speed | Label   |
| ------------ | ------------- | ------- |
| 0 — 20s      | 4 moves/sec   | Medium  |
| 7 — 45s      | 6 moves/sec   | Hard    |
| 20 — 70s     | 7 moves/sec   | Extreme |
| 45s+         | 8 moves/sec   | Insane  |

---

## Meet the Characters

### Hero

_That's you. Move fast._

<img width="435" height="592" alt="hero" src="https://github.com/user-attachments/assets/cc4a733e-c840-46db-8ce3-0e13d0e968e4" />


---

### Monster (AI)

_Trained with Double DQN. Learns while you play._

<img width="442" height="617" alt="monster" src="https://github.com/user-attachments/assets/f0f9a1c9-625d-4ab1-99ba-c54ef5127d42" />


---

### Princess

_Your goal — but she won't let you catch her that easy!_

<img width="442" height="626" alt="princess" src="https://github.com/user-attachments/assets/0c9257bc-4803-42cc-8d81-7843197082a8" />


---

## 🧠 Deep Reinforcement Learning — How It Works

### Why Reinforcement Learning?

The monster's intelligence is not hand-coded. There are no if-statements
telling it to move toward the hero. Instead, it learned how to hunt by
playing thousands of simulated episodes and figuring out — through
trial, error, and reward — that catching the hero is good and hitting
walls is bad.

### Algorithm — Double DQN

The monster is trained using **Double Deep Q-Network (Double DQN)**,
an improvement over vanilla DQN that fixes a known instability called
Q-value overestimation.

In vanilla DQN, the same network both _selects_ and _evaluates_ the
best next action — this causes the agent to be overconfident in
suboptimal moves. Double DQN fixes this by splitting the two
responsibilities:

- The **online network** selects which action looks best
- The **target network** evaluates how good that action actually is

This small change produces significantly more stable training.

---

## 🚀 How to Play

### Windows — Download & Play (No Python needed)

1. Go to the [Releases](../../releases) page
2. Download `ThePrincessQuest.exe`
3. Double click — done

> First launch may take a few seconds while the `.exe` extracts.
> No installation, no Python, no dependencies.

---

### Linux & Mac — Run from Source

**Requirements:** Python 3.10+

**Step 1 — Clone the repo**

```bash
git clone https://github.com/ArhanTibro/The-Princess-Quest.git
cd the-princess-quest
```

**Step 2 — Create and activate virtual environment**

```bash
# Linux / Mac
python -m venv env
source env/bin/activate

# Windows (if running from source instead of .exe)
python -m venv env
env\Scripts\activate
```

**Step 3 — Install PyTorch (CPU only, saves ~1.5GB)**

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Step 4 — Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Run the game**

```bash
python game/main.py
```

The trained model (`model/monster.npz`) is already committed to the
repo — no retraining needed. The game runs immediately.

## 🎯 Controls

| Key         | Action     |
| ----------- | ---------- |
| ↑ Arrow / W | Move Up    |
| ↓ Arrow / S | Move Down  |
| ← Arrow / A | Move Left  |
| → Arrow / D | Move Right |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

_Built with Python, PyTorch, and Pygame._
_A fun project exploring Deep Reinforcement Learning through gameplay._
