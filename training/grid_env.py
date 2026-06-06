import numpy as np
import random
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────
N            = 10          # grid side length
NUM_WALLS    = N           # wall count = N
MAX_STEPS    = 400         # episode ends if monster hasn't caught hero by then

# Reward values
R_CATCH      =  10.0
R_STEP       =  -0.01
R_WALL_HIT   =  -0.10
R_CLOSER     =   0.05

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
DELTAS  = {UP: (-1,0), DOWN: (1,0), LEFT: (0,-1), RIGHT: (0,1)}


# ── Helpers ──────────────────────────────────────────────────────────────────
def manhattan(a, b): #manhattan distance between two (row, col) cells
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def flood_fill(grid, start):
    """Returns set of all reachable (non-wall) cells from start."""
    visited = set()
    queue   = deque([start])
    while queue:
        cell = queue.popleft()
        if cell in visited:
            continue
        visited.add(cell)
        r, c = cell
        for dr, dc in DELTAS.values():
            nr, nc = r+dr, c+dc
            if 0 <= nr < N and 0 <= nc < N and (nr,nc) not in visited:
                if grid[nr][nc] != 'W':
                    queue.append((nr, nc))
    return visited


def place_walls(exclude):
    """
    Randomly place NUM_WALLS walls on the grid, never on excluded cells.
    Retries until flood-fill confirms full connectivity.
    exclude: set of (row, col) tuples that must stay wall-free (spawn cells).
    Returns: (grid, wall_set)
    """
    all_cells = [(r, c) for r in range(N) for c in range(N)]

    for _ in range(20):                          # retry up to 20 times
        candidates = [cell for cell in all_cells if cell not in exclude]
        walls      = set(random.sample(candidates, NUM_WALLS))

        # Build grid
        grid = [['.' for _ in range(N)] for _ in range(N)]
        for (r, c) in walls:
            grid[r][c] = 'W'

        # Flood-fill from any non-wall, non-excluded cell
        start = next(cell for cell in all_cells if cell not in walls)
        reachable = flood_fill(grid, start)

        # All non-wall cells must be reachable
        non_wall = set(cell for cell in all_cells if cell not in walls)
        if non_wall == reachable:
            return grid, walls

    # Fallback: no walls (should never happen in practice)
    grid = [['.' for _ in range(N)] for _ in range(N)]
    return grid, set()


# ── GridEnv ──────────────────────────────────────────────────────────────────
class GridEnv:
    """
    Headless 10x10 grid environment for Double DQN training.
    No Pygame. No display. Pure logic.

    Walls randomise every episode — same distribution as gameplay —
    so the agent generalises to any wall layout it sees during actual play.
    """

    def __init__(self):
        self.n         = N
        self.grid      = None
        self.walls     = set()
        self.monster   = None
        self.hero      = None
        self.princess  = None
        self.steps     = 0
        self.done      = False

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self):
        """
        Start a new episode:
          - Place monster and hero at random non-overlapping cells
          - Place princess far from both
          - Randomise walls (excluding spawn cells)
        Returns normalised state vector.
        """
        all_cells = [(r, c) for r in range(N) for c in range(N)]

        # Spawn monster and hero at least 3 steps apart
        while True:
            self.monster  = random.choice(all_cells)
            self.hero     = random.choice(all_cells)
            if self.monster != self.hero and manhattan(self.monster, self.hero) >= 3:
                break

        # Spawn princess as far as possible from both
        self.princess = max(
            all_cells,
            key=lambda c: manhattan(c, self.monster) + manhattan(c, self.hero)
        )
        # Make sure princess isn't on monster or hero
        if self.princess in (self.monster, self.hero):
            candidates = [c for c in all_cells if c not in (self.monster, self.hero)]
            self.princess = random.choice(candidates)

        # Place walls — never on spawn cells
        exclude = {self.monster, self.hero, self.princess}
        self.grid, self.walls = place_walls(exclude)

        self.steps = 0
        self.done  = False

        return self._get_state()

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Monster takes one action.
        Returns (next_state, reward, done, info)
        """
        assert not self.done, "Call reset() before stepping after episode end."

        prev_dist = manhattan(self.monster, self.hero)

        # Attempt monster move
        dr, dc    = DELTAS[action]
        nr, nc    = self.monster[0] + dr, self.monster[1] + dc
        wall_hit  = False

        if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in self.walls:
            self.monster = (nr, nc)
        else:
            wall_hit = True          # stayed in place

        # Hero random walk (training opponent)
        self._hero_random_step()

        # Princess random walk
        self._princess_random_step()

        self.steps += 1

        # ── Reward ───────────────────────────────────────────────────────────
        reward = R_STEP

        if wall_hit:
            reward += R_WALL_HIT

        curr_dist = manhattan(self.monster, self.hero)
        if curr_dist < prev_dist:
            reward += R_CLOSER

        # Terminal: caught hero
        if self.monster == self.hero:
            reward    += R_CATCH
            self.done  = True
            return self._get_state(), reward, True, {"result": "caught"}

        # Terminal: max steps
        if self.steps >= MAX_STEPS:
            self.done = True
            return self._get_state(), reward, True, {"result": "timeout"}

        return self._get_state(), reward, False, {}

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _get_state(self):
        """
        Normalised state vector: [monster_r, monster_c, hero_r, hero_c]
        All values in [0, 1].
        """
        return np.array([
            self.monster[0]  / (N - 1),
            self.monster[1]  / (N - 1),
            self.hero[0]     / (N - 1),
            self.hero[1]     / (N - 1),
        ], dtype=np.float32)

    def _hero_random_step(self):
        """Hero takes a random valid step (used during training)."""
        random.shuffle(ACTIONS)
        for a in ACTIONS:
            dr, dc = DELTAS[a]
            nr, nc = self.hero[0]+dr, self.hero[1]+dc
            if 0 <= nr < N and 0 <= nc < N and (nr,nc) not in self.walls:
                self.hero = (nr, nc)
                return
        # All directions blocked — stay in place (very rare with 10 walls)

    def _princess_random_step(self):
        """Princess takes a random valid step, never onto monster cell."""
        shuffled = ACTIONS[:]
        random.shuffle(shuffled)
        for a in shuffled:
            dr, dc = DELTAS[a]
            nr, nc = self.princess[0]+dr, self.princess[1]+dc
            if (0 <= nr < N and 0 <= nc < N
                    and (nr,nc) not in self.walls
                    and (nr,nc) != self.monster):
                self.princess = (nr, nc)
                return

    # ── Render (debug only) ──────────────────────────────────────────────────
    def render(self):
        """ASCII render for debugging. Not used during training."""
        symbols = {self.monster: 'M', self.hero: 'H', self.princess: 'P'}
        print(f"\nStep {self.steps}")
        for r in range(N):
            row = ""
            for c in range(N):
                cell = (r, c)
                if cell in self.walls:
                    row += "█ "
                else:
                    row += symbols.get(cell, ". ") if cell in symbols else ". "
            print(row)
        print(f"Monster:{self.monster}  Hero:{self.hero}  Princess:{self.princess}")