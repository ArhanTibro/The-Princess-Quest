import os
import sys
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ── Path setup ───────────────────────────────────────────────────────────────
# Allow running from project root or from training/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.grid_env   import GridEnv
from training.dqn_agent  import DoubleDQNAgent

# ── Config ───────────────────────────────────────────────────────────────────
TOTAL_STEPS      = 200_000   # total environment steps to train for
TRAIN_EVERY      = 4         # run one gradient update every N steps
LOG_EVERY        = 100       # log to TensorBoard every N episodes
SAVE_EVERY       = 10_000    # save checkpoint every N steps
WARMUP_STEPS     = 1_000     # fill buffer before training starts

MODEL_DIR        = os.path.join(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))), "model")
LOG_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

CHECKPOINT_PATH  = os.path.join(MODEL_DIR, "monster_checkpoint.pth")
FINAL_PATH       = os.path.join(MODEL_DIR, "monster.pth")
BEST_PATH        = os.path.join(MODEL_DIR, "monster_best.pth")

SEED             = 42


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Logging helpers ──────────────────────────────────────────────────────────
def log_episode(writer, ep_stats, episode):
    """Write per-episode metrics to TensorBoard."""
    writer.add_scalar("Episode/Reward",       ep_stats["reward"],   episode)
    writer.add_scalar("Episode/Length",       ep_stats["length"],   episode)
    writer.add_scalar("Episode/Caught",       ep_stats["caught"],   episode)
    writer.add_scalar("Agent/Epsilon",        ep_stats["epsilon"],  episode)

def log_step(writer, step_stats, step):
    """Write per-step metrics to TensorBoard."""
    if step_stats["loss"] is not None:
        writer.add_scalar("Train/Loss",       step_stats["loss"],   step)
    if step_stats["mean_q"] is not None:
        writer.add_scalar("Train/MeanQ",      step_stats["mean_q"], step)


# ── Main training loop ───────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    env    = GridEnv()
    agent  = DoubleDQNAgent()
    writer = SummaryWriter(log_dir=LOG_DIR)

    print("=" * 60)
    print("  THE PRINCESS QUEST — Double DQN Training")
    print(f"  Total steps : {TOTAL_STEPS:,}")
    print(f"  Warmup steps: {WARMUP_STEPS:,}")
    print(f"  Model dir   : {MODEL_DIR}")
    print(f"  TensorBoard : tensorboard --logdir {LOG_DIR}")
    print("=" * 60)

    # ── Tracking variables ───────────────────────────────────────────────────
    total_steps     = 0
    episode         = 0
    best_avg_reward = -float("inf")

    # Rolling window for smoothed metrics (last 100 episodes)
    recent_rewards  = []
    recent_lengths  = []
    recent_caught   = []

    # Q-value logging buffer
    q_log_states    = []

    # ── Progress bar over total steps ────────────────────────────────────────
    pbar = tqdm(total=TOTAL_STEPS, unit="step", dynamic_ncols=True)

    while total_steps < TOTAL_STEPS:

        # ── Episode reset ────────────────────────────────────────────────────
        state       = env.reset()
        ep_reward   = 0.0
        ep_length   = 0
        ep_caught   = False

        while True:
            # ── Select & apply action ────────────────────────────────────────
            action                           = agent.select_action(state)
            next_state, reward, done, info   = env.step(action)

            agent.store(state, action, reward, next_state, done)

            # Collect states for Q-value logging
            if len(q_log_states) < 512:
                q_log_states.append(state)

            state        = next_state
            ep_reward   += reward
            ep_length   += 1
            total_steps += 1
            pbar.update(1)

            # ── Train ────────────────────────────────────────────────────────
            loss    = None
            mean_q  = None

            if total_steps > WARMUP_STEPS and total_steps % TRAIN_EVERY == 0:
                loss = agent.train_step()

                # Log mean Q every 500 training steps
                if agent.steps_done % 500 == 0 and len(q_log_states) >= 64:
                    mean_q = agent.mean_q(q_log_states[-64:])

                log_step(writer, {"loss": loss, "mean_q": mean_q}, total_steps)

            # ── Checkpoint save ──────────────────────────────────────────────
            if total_steps % SAVE_EVERY == 0:
                agent.save(CHECKPOINT_PATH)
                pbar.write(f"[Step {total_steps:,}] Checkpoint saved.")

            if done:
                ep_caught = info.get("result") == "caught"
                break

        # ── Episode bookkeeping ──────────────────────────────────────────────
        episode += 1
        recent_rewards.append(ep_reward)
        recent_lengths.append(ep_length)
        recent_caught.append(int(ep_caught))

        # Keep rolling window at 100
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lengths.pop(0)
            recent_caught.pop(0)

        # ── TensorBoard logging ──────────────────────────────────────────────
        if episode % LOG_EVERY == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_caught = np.mean(recent_caught)   # catch rate 0–1

            log_episode(writer, {
                "reward":  avg_reward,
                "length":  avg_length,
                "caught":  avg_caught,
                "epsilon": agent.epsilon,
            }, episode)

            pbar.write(
                f"[Ep {episode:>5} | Step {total_steps:>7,}] "
                f"AvgReward: {avg_reward:>7.2f}  "
                f"AvgLen: {avg_length:>5.1f}  "
                f"CatchRate: {avg_caught*100:>5.1f}%  "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            # ── Save best model ──────────────────────────────────────────────
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(BEST_PATH)
                pbar.write(f"  ★ New best avg reward: {best_avg_reward:.2f} — saved.")

    # ── Training complete ────────────────────────────────────────────────────
    pbar.close()
    agent.save(FINAL_PATH)
    writer.close()

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Final model → {FINAL_PATH}")
    print(f"  Best model  → {BEST_PATH}")
    print(f"  Episodes    : {episode:,}")
    print(f"  Total steps : {total_steps:,}")
    print(f"  Best avg reward (last 100 eps): {best_avg_reward:.2f}")
    print("=" * 60)
    print("\nNext step: python model/export_weights.py")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()