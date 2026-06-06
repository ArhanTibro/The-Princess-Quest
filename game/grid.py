import random
from collections import deque
from game.config import (
    N, NUM_WALLS, CELL_SIZE, MARGIN, HUD_HEIGHT,
    cell_to_pixel, entity_rect, ENTITY_RADIUS, WALL_RADIUS,
    C_CELL_EMPTY, C_GRID_LINE, C_WALL, C_WALL_BORDER,
)

# ── Directions ────────────────────────────────────────────────────────────────
DELTAS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}


# ── Flood fill ────────────────────────────────────────────────────────────────
def _flood_fill(walls, start):
    """
    BFS from start cell.
    Returns set of all reachable non-wall cells.
    """
    visited = set()
    queue   = deque([start])
    while queue:
        cell = queue.popleft()
        if cell in visited:
            continue
        visited.add(cell)
        r, c = cell
        for dr, dc in DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in visited:
                if (nr, nc) not in walls:
                    queue.append((nr, nc))
    return visited


# ── Wall placement ────────────────────────────────────────────────────────────
def _place_walls(exclude):
    """
    Randomly place NUM_WALLS walls, never on excluded spawn cells.
    Retries until flood-fill confirms full grid connectivity.
    Returns a frozenset of wall (row, col) tuples.
    """
    all_cells = [(r, c) for r in range(N) for c in range(N)]

    for _ in range(30):
        candidates = [cell for cell in all_cells if cell not in exclude]
        walls      = set(random.sample(candidates, NUM_WALLS))

        # Flood-fill from any non-wall non-excluded cell
        start     = next(c for c in all_cells if c not in walls)
        reachable = _flood_fill(walls, start)
        non_wall  = set(c for c in all_cells if c not in walls)

        if reachable == non_wall:
            return frozenset(walls)

    # Extremely rare fallback — return no walls
    return frozenset()


# ── Manhattan distance ────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Grid class ────────────────────────────────────────────────────────────────
class Grid:
    """
    Manages the game grid for one round:
      - Random wall placement with flood-fill validation
      - Spawn position selection for all three entities
      - Move validation (boundary + wall check)
      - Collision detection
      - Pygame rendering of cells, walls, and grid lines
    """

    def __init__(self):
        self.walls   = frozenset()
        self.surface = None        # set by main.py after pygame.init()

    # ── Round reset ───────────────────────────────────────────────────────────
    def reset(self):
        """
        Generate a fresh layout for a new round.
        Returns spawn positions: (monster, hero, princess) as (row, col) tuples.
        """
        all_cells = [(r, c) for r in range(N) for c in range(N)]

        # ── Spawn monster and hero ────────────────────────────────────────────
        # At least 4 cells apart so monster doesn't catch hero immediately
        while True:
            monster = random.choice(all_cells)
            hero    = random.choice(all_cells)
            if monster != hero and manhattan(monster, hero) >= 4:
                break

        # ── Spawn princess ────────────────────────────────────────────────────
        # Maximally far from both monster and hero
        # Filter out cells too close to either entity
        princess_candidates = [
            c for c in all_cells
            if c != monster
            and c != hero
            and manhattan(c, monster) >= 3
            and manhattan(c, hero)    >= 3
        ]
        if princess_candidates:
            # Among valid candidates pick the one farthest from both
            princess = max(
                princess_candidates,
                key=lambda c: manhattan(c, monster) + manhattan(c, hero)
            )
        else:
            # Fallback — just pick any cell that isn't monster or hero
            princess = random.choice(
                [c for c in all_cells if c not in (monster, hero)]
            )

        # ── Place walls ───────────────────────────────────────────────────────
        # Exclude all three spawn cells so nobody starts on a wall
        exclude    = {monster, hero, princess}
        self.walls = _place_walls(exclude)

        return monster, hero, princess

    # ── Move validation ───────────────────────────────────────────────────────
    def is_valid(self, row, col):
        """
        Returns True if (row, col) is inside the grid and not a wall.
        Used by hero input, monster AI, and princess movement.
        """
        return (
            0 <= row < N
            and 0 <= col < N
            and (row, col) not in self.walls
        )

    def apply_action(self, pos, action):
        """
        Attempt to move from pos using action (0=UP,1=DOWN,2=LEFT,3=RIGHT).
        Returns (new_pos, hit_wall):
          - new_pos   : position after move (unchanged if wall hit)
          - hit_wall  : True if the move was blocked
        """
        dr, dc   = DELTAS[action]
        nr, nc   = pos[0] + dr, pos[1] + dc
        if self.is_valid(nr, nc):
            return (nr, nc), False
        return pos, True                 # blocked — stay in place

    def valid_neighbours(self, pos):
        """
        Returns list of valid adjacent cells from pos.
        Used by princess random walk to find moveable directions.
        """
        neighbours = []
        for dr, dc in DELTAS.values():
            nr, nc = pos[0] + dr, pos[1] + dc
            if self.is_valid(nr, nc):
                neighbours.append((nr, nc))
        return neighbours

    # ── Rendering ─────────────────────────────────────────────────────────────
    def draw(self, surface):
        """
        Draw all grid cells and walls onto the given pygame surface.
        Entities are drawn separately by main.py on top of this.
        """
        import pygame

        for r in range(N):
            for c in range(N):
                x, y = cell_to_pixel(r, c)
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if (r, c) in self.walls:
                    # Wall — filled rounded rect with border
                    pygame.draw.rect(surface, C_WALL,        rect, border_radius=WALL_RADIUS)
                    pygame.draw.rect(surface, C_WALL_BORDER, rect, width=1, border_radius=WALL_RADIUS)
                else:
                    # Empty cell
                    pygame.draw.rect(surface, C_CELL_EMPTY, rect)
                    # Subtle grid line border
                    pygame.draw.rect(surface, C_GRID_LINE,  rect, width=1)

    def draw_grid_border(self, surface):
        """
        Draw a single border rectangle around the entire grid.
        Called after draw() so it sits on top of cell borders.
        """
        import pygame

        border_rect = pygame.Rect(
            MARGIN,
            MARGIN + HUD_HEIGHT,
            N * CELL_SIZE,
            N * CELL_SIZE,
        )
        pygame.draw.rect(surface, C_GRID_LINE, border_rect, width=2)