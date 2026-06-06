import numpy as np
from game.config import N, MODEL_PATH


# ── NumPy forward pass ────────────────────────────────────────────────────────
def _relu(x):
    return np.maximum(0.0, x)


def _forward(layers, state):
    """
    Pure NumPy forward pass through the trained MLP.
    Identical math to DQNetwork.forward() in dqn_agent.py.

    layers : dict with keys W0,b0,W1,b1,W2,b2 (loaded from monster.npz)
    state  : 1D float32 array of shape (4,)
    returns: 1D float32 array of shape (4,) — Q-value per action
    """
    out = state.astype(np.float32)
    out = _relu(np.dot(layers["W0"], out) + layers["b0"])   # hidden 1
    out = _relu(np.dot(layers["W1"], out) + layers["b1"])   # hidden 2
    out =       np.dot(layers["W2"], out) + layers["b2"]    # output (no activation)
    return out


# ── Inference engine ──────────────────────────────────────────────────────────
class AIInference:
    """
    Loads monster.npz once at startup and provides get_action()
    for the game loop to call every monster tick.

    Also exposes get_layers() and set_layers() so online_trainer.py
    can update the weights in-place as the monster keeps learning
    during gameplay — no file I/O, no restart needed.
    """

    def __init__(self):
        self._layers = {}          # weight dict — shared with online trainer
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self):
        """
        Load weight arrays from monster.npz.
        Keys: W0, b0, W1, b1, W2, b2
        All cast to float32 for consistent inference.
        """
        try:
            data = np.load(MODEL_PATH)
            self._layers = {k: data[k].astype(np.float32) for k in data.files}
            print(f"[AIInference] Loaded model from {MODEL_PATH}")
            self._log_shapes()
        except FileNotFoundError:
            print(f"[AIInference] ERROR: model not found at {MODEL_PATH}")
            print("  Run: python -m model.export_weights")
            raise

    def _log_shapes(self):
        for k, v in self._layers.items():
            print(f"  {k}: {v.shape}  dtype={v.dtype}")

    # ── State builder ─────────────────────────────────────────────────────────
    @staticmethod
    def build_state(monster_pos, hero_pos):
        """
        Build normalised state vector from game positions.
        monster_pos, hero_pos: (row, col) tuples
        Returns float32 array of shape (4,) with all values in [0, 1].
        """
        return np.array([
            monster_pos[0] / (N - 1),
            monster_pos[1] / (N - 1),
            hero_pos[0]    / (N - 1),
            hero_pos[1]    / (N - 1),
        ], dtype=np.float32)

    # ── Action selection ──────────────────────────────────────────────────────
    def get_action(self, monster_pos, hero_pos, valid_actions=None):
    
    # Run inference and return the best VALID action.
    # If the greedy action is blocked, falls back to the
    # next best Q-value action that isn't blocked.
    
    # valid_actions: list of action ints (0-3) that aren't walls.
    #               If None, all 4 actions are considered.
    
        state  = self.build_state(monster_pos, hero_pos)
        q_vals = _forward(self._layers, state)

        if valid_actions is not None and len(valid_actions) > 0:
            # Pick highest Q-value among only valid (non-wall) actions
            best_action = max(valid_actions, key=lambda a: q_vals[a])
            return best_action

        return int(np.argmax(q_vals))

    def get_q_values(self, monster_pos, hero_pos):
        """
        Returns raw Q-values for all 4 actions.
        Used by online_trainer.py for the Double DQN update
        and optionally by main.py for a debug HUD overlay.
        """
        state = self.build_state(monster_pos, hero_pos)
        return _forward(self._layers, state)

    # ── Weight access for online trainer ──────────────────────────────────────
    def get_layers(self):
        """
        Returns reference to the internal weight dict.
        online_trainer.py holds this reference and writes
        updated weights directly into it after each gradient step.
        """
        return self._layers

    def set_layers(self, new_layers):
        """
        Replace all weight arrays in-place.
        Called by online_trainer.py after a gradient update.
        Uses in-place numpy copy to avoid breaking any reference
        that might still be mid-computation.
        """
        for k in self._layers:
            np.copyto(self._layers[k], new_layers[k])