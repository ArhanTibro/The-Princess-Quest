import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from game.config import (
    ONLINE_BUFFER_SIZE,
    ONLINE_BATCH_SIZE,
    ONLINE_LR,
    ONLINE_GAMMA,
    ONLINE_TRAIN_EVERY,
    ONLINE_TARGET_SYNC,
    ONLINE_EPSILON,
)


# ── Minimal online network ────────────────────────────────────────────────────
# Mirrors DQNetwork exactly — same architecture, same weights loaded from .npz
class _OnlineNetwork(nn.Module):
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

    def load_from_numpy(self, layers):
        """
        Initialise network weights from a NumPy layer dict
        (W0,b0,W1,b1,W2,b2) — the same format as monster.npz.
        """
        linear_idx = 0
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    module.weight.copy_(
                        torch.tensor(layers[f"W{linear_idx}"])
                    )
                    module.bias.copy_(
                        torch.tensor(layers[f"b{linear_idx}"])
                    )
                    linear_idx += 1

    def to_numpy(self):
        """
        Export current weights back to a NumPy layer dict.
        Called after each gradient update to push weights
        back into AIInference for live inference.
        """
        layers = {}
        linear_idx = 0
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    layers[f"W{linear_idx}"] = module.weight.numpy().copy()
                    layers[f"b{linear_idx}"] = module.bias.numpy().copy()
                    linear_idx += 1
        return layers


# ── Online trainer ────────────────────────────────────────────────────────────
class OnlineTrainer:
    """
    Runs a Double DQN fine-tuning loop in a background thread
    while the player is playing.

    Flow:
      main.py pushes experience tuples via push_experience()
      after every monster step.

      The background thread pulls from the queue, fills the
      replay buffer, and runs gradient updates every
      ONLINE_TRAIN_EVERY steps.

      After each update, updated weights are pushed back into
      AIInference via set_layers() so inference immediately
      reflects the improved policy — no restart needed.

    The trainer starts with the same weights as the offline
    trained model, so the monster is competent from round 1
    and only gets better as the session continues.
    """

    def __init__(self, ai_inference):
        """
        ai_inference : AIInference instance from game/ai_inference.py
                       Used to seed initial weights and push updates back.
        """
        self._inference    = ai_inference
        self._exp_queue    = queue.Queue()        # game loop → trainer thread
        self._buffer       = deque(maxlen=ONLINE_BUFFER_SIZE)
        self._lock         = threading.Lock()     # guards weight writes
        self._stop_event   = threading.Event()
        self._steps        = 0                    # gradient updates done
        self._thread       = None

        # ── Build online and target networks seeded from trained .npz ────────
        self._online = _OnlineNetwork()
        self._target = _OnlineNetwork()
        self._online.load_from_numpy(ai_inference.get_layers())
        self._target.load_from_numpy(ai_inference.get_layers())

        self._optimizer = optim.Adam(self._online.parameters(), lr=ONLINE_LR)
        self._loss_fn   = nn.MSELoss()

    # ── Public API (called from main.py / game loop thread) ──────────────────

    def start(self):
        """Start the background training thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._training_loop,
            daemon=True,              # dies automatically when game exits
            name="OnlineTrainer"
        )
        self._thread.start()
        print("[OnlineTrainer] Background training thread started.")

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print(f"[OnlineTrainer] Stopped. Total gradient updates: {self._steps}")

    def push_experience(self, state, action, reward, next_state, done):
        """
        Called by main.py after every monster step.
        Non-blocking — puts the experience on a queue for the
        background thread to consume.

        state, next_state : float32 np.array of shape (4,)
        action            : int (0–3)
        reward            : float
        done              : bool
        """
        self._exp_queue.put((state, action, reward, next_state, done))

    @property
    def updates_done(self):
        """How many gradient updates have been completed this session."""
        return self._steps

    # ── Background training loop ──────────────────────────────────────────────

    def _training_loop(self):
        """
        Runs in the background thread.
        Drains the experience queue, fills replay buffer,
        and triggers gradient updates.
        """
        train_counter = 0    # counts experiences since last gradient update

        while not self._stop_event.is_set():

            # Drain all available experiences from queue
            drained = False
            while not self._exp_queue.empty():
                try:
                    exp = self._exp_queue.get_nowait()
                    self._buffer.append(exp)
                    train_counter += 1
                    drained = True
                except queue.Empty:
                    break

            # Run gradient update every ONLINE_TRAIN_EVERY experiences
            if (drained
                    and train_counter >= ONLINE_TRAIN_EVERY
                    and len(self._buffer) >= ONLINE_BATCH_SIZE):
                self._gradient_update()
                train_counter = 0

            # Small sleep to avoid spinning the CPU when queue is empty
            if not drained:
                self._stop_event.wait(timeout=0.005)   # 5ms sleep

    # ── Gradient update ───────────────────────────────────────────────────────

    def _gradient_update(self):
        """
        One Double DQN gradient update on a random batch
        sampled from the replay buffer.
        Pushes updated weights back to AIInference immediately.
        """
        # Sample batch
        batch  = list(self._buffer)
        if len(batch) < ONLINE_BATCH_SIZE:
            return
        indices = np.random.choice(len(batch), ONLINE_BATCH_SIZE, replace=False)
        samples = [batch[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        states_t      = torch.tensor(np.array(states),      dtype=torch.float32)
        actions_t     = torch.tensor(actions,                dtype=torch.long)
        rewards_t     = torch.tensor(rewards,                dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t       = torch.tensor(dones,                  dtype=torch.float32)

        # Current Q values
        current_q = self._online(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_next   = self._online(next_states_t).argmax(dim=1)
            next_q      = self._target(next_states_t).gather(
                1, best_next.unsqueeze(1)
            ).squeeze(1)
            target_q    = rewards_t + ONLINE_GAMMA * next_q * (1.0 - dones_t)

        # Gradient step
        loss = self._loss_fn(current_q, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=1.0)
        self._optimizer.step()

        self._steps += 1

        # Sync target network periodically
        if self._steps % ONLINE_TARGET_SYNC == 0:
            self._target.load_state_dict(self._online.state_dict())
            print(f"[OnlineTrainer] Target network synced at update {self._steps}.")

        # ── Push updated weights back to inference ────────────────────────────
        new_layers = self._online.to_numpy()
        with self._lock:
            self._inference.set_layers(new_layers)