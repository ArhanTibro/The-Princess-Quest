import os
import sys
import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dqn_agent import DQNetwork

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
BEST_PATH    = os.path.join(MODEL_DIR, "monster_best.pth")
FINAL_PATH   = os.path.join(MODEL_DIR, "monster.pth")
EXPORT_PATH  = os.path.join(MODEL_DIR, "monster.npz")


# ── Load checkpoint ──────────────────────────────────────────────────────────
def load_network(pth_path):
    """Load online network weights from a saved checkpoint."""
    checkpoint = torch.load(pth_path, map_location="cpu")

    net = DQNetwork()
    net.load_state_dict(checkpoint["online_state_dict"])
    net.eval()
    return net


# ── Extract weights ──────────────────────────────────────────────────────────
def extract_weights(net):
    """
    Pull every Linear layer's weight matrix and bias vector
    out of the network as plain NumPy arrays.

    Network structure:
      net.net[0]  → Linear(4  → 128)   W0, b0
      net.net[2]  → Linear(128→ 128)   W1, b1
      net.net[4]  → Linear(128→ 4  )   W2, b2
    Indices 1, 3 are ReLU — no parameters.
    """
    layers = {}
    linear_idx = 0
    for module in net.net:
        if isinstance(module, torch.nn.Linear):
            layers[f"W{linear_idx}"] = module.weight.detach().numpy()  # shape (out, in)
            layers[f"b{linear_idx}"] = module.bias.detach().numpy()    # shape (out,)
            linear_idx += 1
    return layers


# ── Parity test ──────────────────────────────────────────────────────────────
def forward_numpy(layers, x):
    """
    Pure NumPy forward pass — identical to DQNetwork.forward().
    x: 1D array of shape (4,)
    Returns: 1D array of shape (4,) — Q-value per action
    """
    out = x.astype(np.float32)
    out = np.dot(layers["W0"], out) + layers["b0"]   # Linear 0
    out = np.maximum(0.0, out)                        # ReLU
    out = np.dot(layers["W1"], out) + layers["b1"]   # Linear 1
    out = np.maximum(0.0, out)                        # ReLU
    out = np.dot(layers["W2"], out) + layers["b2"]   # Linear 2  (no activation)
    return out


def parity_test(net, layers, n_tests=2000, tol=1e-5):
    """
    Run n_tests random states through both PyTorch and NumPy forward pass.
    Assert all outputs match within tolerance.
    Prints pass/fail summary.
    """
    print(f"\nRunning parity test ({n_tests} random states, tol={tol}) ...")
    max_diff   = 0.0
    failures   = 0

    net.eval()
    with torch.no_grad():
        for i in range(n_tests):
            state_np = np.random.rand(4).astype(np.float32)

            # PyTorch output
            state_t   = torch.tensor(state_np).unsqueeze(0)
            torch_out = net(state_t).squeeze(0).numpy()

            # NumPy output
            numpy_out = forward_numpy(layers, state_np)

            diff = np.max(np.abs(torch_out - numpy_out))
            max_diff = max(max_diff, diff)

            if diff > tol:
                failures += 1
                if failures <= 3:    # print first 3 failures for debugging
                    print(f"  [FAIL #{failures}] state={state_np}")
                    print(f"    PyTorch : {torch_out}")
                    print(f"    NumPy   : {numpy_out}")
                    print(f"    Max diff: {diff:.2e}")

    if failures == 0:
        print(f"  PASSED — all {n_tests} states match.")
        print(f"  Max absolute difference: {max_diff:.2e}")
    else:
        print(f"  FAILED — {failures}/{n_tests} states exceeded tolerance.")
        print(f"  Max absolute difference: {max_diff:.2e}")
        sys.exit(1)


# ── Action sanity check ──────────────────────────────────────────────────────
def action_sanity_check(layers):
    """
    Quick sanity check: monster directly above hero should prefer DOWN (1).
    Monster directly below hero should prefer UP (0).
    Not a strict requirement but good to verify the agent learned something.
    """
    print("\nAction sanity check ...")

    # Monster at top, hero at bottom → monster should go DOWN
    state1  = np.array([0.1, 0.5, 0.9, 0.5], dtype=np.float32)
    q1      = forward_numpy(layers, state1)
    action1 = int(np.argmax(q1))
    label1  = ["UP","DOWN","LEFT","RIGHT"][action1]
    hint1   = "✓" if action1 == 1 else "?"     # 1 = DOWN
    print(f"  Monster top,   hero bottom → action: {label1} {hint1}")

    # Monster at bottom, hero at top → monster should go UP
    state2  = np.array([0.9, 0.5, 0.1, 0.5], dtype=np.float32)
    q2      = forward_numpy(layers, state2)
    action2 = int(np.argmax(q2))
    label2  = ["UP","DOWN","LEFT","RIGHT"][action2]
    hint2   = "✓" if action2 == 0 else "?"     # 0 = UP
    print(f"  Monster bottom, hero top  → action: {label2} {hint2}")

    # Monster left, hero right → monster should go RIGHT
    state3  = np.array([0.5, 0.1, 0.5, 0.9], dtype=np.float32)
    q3      = forward_numpy(layers, state3)
    action3 = int(np.argmax(q3))
    label3  = ["UP","DOWN","LEFT","RIGHT"][action3]
    hint3   = "✓" if action3 == 3 else "?"     # 3 = RIGHT
    print(f"  Monster left,  hero right → action: {label3} {hint3}")

    print("  (? means unexpected — not a failure, walls affect learned policy)")


# ── Save ─────────────────────────────────────────────────────────────────────
def save_npz(layers, path):
    np.savez_compressed(path, **layers)
    size_kb = os.path.getsize(path) / 1024
    print(f"\nExported → {path}  ({size_kb:.1f} KB)")
    print(f"  Keys: {list(layers.keys())}")
    print(f"  Shapes:")
    for k, v in layers.items():
        print(f"    {k}: {v.shape}  dtype={v.dtype}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Prefer best model, fall back to final
    if os.path.exists(BEST_PATH):
        src = BEST_PATH
        print(f"Loading best model: {BEST_PATH}")
    elif os.path.exists(FINAL_PATH):
        src = FINAL_PATH
        print(f"Best model not found — loading final: {FINAL_PATH}")
    else:
        print("ERROR: No trained model found.")
        print(f"  Expected: {BEST_PATH}")
        print(f"  Run training first: python -m training.train")
        sys.exit(1)

    # Load → extract → test → save
    net    = load_network(src)
    layers = extract_weights(net)

    parity_test(net, layers)
    action_sanity_check(layers)
    save_npz(layers, EXPORT_PATH)

    print("\nDone. Next step: python game/main.py")