import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Network ──────────────────────────────────────────────────────────────────
class DQNetwork(nn.Module):
    """
    Two-layer MLP: 4 inputs → 128 → 128 → 4 Q-values.
    Input  : [monster_r, monster_c, hero_r, hero_c]  (all normalised 0–1)
    Output : Q-value for each of 4 actions (Up, Down, Left, Right)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Fixed-size circular buffer storing experience tuples.
    (state, action, reward, next_state, done)
    """
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Double DQN Agent ─────────────────────────────────────────────────────────
class DoubleDQNAgent:
    """
    Double DQN agent controlling the monster.

    Double DQN fix:
      - Online network  selects the best next action
      - Target network  evaluates that action's Q-value
    This decoupling eliminates Q-value overestimation from vanilla DQN.
    """

    def __init__(
        self,
        lr             = 0.0005,
        gamma          = 0.95,
        epsilon_start  = 1.0,
        epsilon_end    = 0.05,
        epsilon_decay  = 0.995,
        batch_size     = 64,
        buffer_capacity= 50_000,
        target_sync    = 500,
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_sync   = target_sync
        self.steps_done    = 0            # total training steps taken

        # Networks
        self.online = DQNetwork()
        self.target = DQNetwork()
        self._sync_target()               # start with identical weights

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_capacity)

    # ── Action selection ─────────────────────────────────────────────────────
    def select_action(self, state):
        """
        Epsilon-greedy policy.
        - With probability epsilon  → random action  (explore)
        - Otherwise                 → greedy action  (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals  = self.online(state_t)
            return int(q_vals.argmax(dim=1).item())

    # ── Store experience ─────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ────────────────────────────────────────────────────────
    def train_step(self):
        """
        One gradient update using a random batch from the replay buffer.
        Returns loss value (float) for logging. Returns None if buffer
        doesn't have enough samples yet.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # ── Current Q-values ─────────────────────────────────────────────────
        # Q(s, a) for the actions actually taken
        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Double DQN target ────────────────────────────────────────────────
        with torch.no_grad():
            # Step 1: online network picks the best next action
            best_next_actions = self.online(next_states).argmax(dim=1)

            # Step 2: target network evaluates that action's value
            next_q = self.target(next_states).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)

            # Bellman target: if done, no future reward
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # ── Gradient update ──────────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients on rare bad batches
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Bookkeeping ──────────────────────────────────────────────────────
        self.steps_done += 1

        # Decay epsilon after every training step
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

        # Sync target network periodically
        if self.steps_done % self.target_sync == 0:
            self._sync_target()

        return loss.item()

    # ── Target network sync ──────────────────────────────────────────────────
    def _sync_target(self):
        """Copy online network weights into target network."""
        self.target.load_state_dict(self.online.state_dict())

    # ── Save & load ──────────────────────────────────────────────────────────
    def save(self, path):
        torch.save({
            "online_state_dict" : self.online.state_dict(),
            "target_state_dict" : self.target.state_dict(),
            "optimizer_state"   : self.optimizer.state_dict(),
            "epsilon"           : self.epsilon,
            "steps_done"        : self.steps_done,
        }, path)
        print(f"[Agent] Saved → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon    = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        print(f"[Agent] Loaded ← {path}")

    # ── Convenience: current mean Q ──────────────────────────────────────────
    def mean_q(self, states):
        """
        Returns mean max Q-value over a batch of states.
        Used for TensorBoard logging to track Q-value growth.
        """
        with torch.no_grad():
            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            return self.online(states_t).max(dim=1).values.mean().item()