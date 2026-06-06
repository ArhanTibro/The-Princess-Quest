import random
import pygame
from game import grid
from game.config import (
    C_HERO, C_HERO_BORDER,
    C_MONSTER, C_MONSTER_BORDER,
    C_PRINCESS, C_PRINCESS_BORDER,
    ENTITY_RADIUS, entity_rect,
)

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ── Actions ───────────────────────────────────────────────────────────────────
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

KEY_ACTION_MAP = {
    pygame.K_UP    : UP,
    pygame.K_DOWN  : DOWN,
    pygame.K_LEFT  : LEFT,
    pygame.K_RIGHT : RIGHT,
    pygame.K_w     : UP,
    pygame.K_s     : DOWN,
    pygame.K_a     : LEFT,
    pygame.K_d     : RIGHT,
}


# ── Base entity ───────────────────────────────────────────────────────────────
class Entity:
    """
    Base class for all three entities.
    Stores position and provides the shared draw method.
    """
    def __init__(self, pos, fill_colour, border_colour):
        self.pos           = pos          # (row, col)
        self.fill_colour   = fill_colour
        self.border_colour = border_colour

    def draw(self, surface):
        rect = entity_rect(self.pos[0], self.pos[1])
        pygame.draw.rect(surface, self.fill_colour,   rect, border_radius=ENTITY_RADIUS)
        pygame.draw.rect(surface, self.border_colour, rect, width=2, border_radius=ENTITY_RADIUS)


# ── Hero ──────────────────────────────────────────────────────────────────────
class Hero(Entity):
    """
    Player-controlled entity.

    Movement is input-driven — main.py passes a pygame key event
    to handle_key(), which returns an action integer if the key
    is a movement key, or None if it isn't.
    The actual position update happens via grid.apply_action()
    in main.py so wall validation stays in one place.
    """

    def __init__(self, pos):
        super().__init__(pos, C_HERO, C_HERO_BORDER)

    def handle_key(self, key):
        """
        Maps a pygame key constant to an action integer.
        Returns None if the key is not a movement key.
        Supports both arrow keys and WASD.
        """
        return KEY_ACTION_MAP.get(key, None)

    def draw(self, surface):
        """
        Hero draws as a rounded rectangle with an inner highlight
        to make it visually distinct from other entities.
        """
        rect = entity_rect(self.pos[0], self.pos[1])

        # Base fill + border
        pygame.draw.rect(surface, self.fill_colour,   rect, border_radius=ENTITY_RADIUS)
        pygame.draw.rect(surface, self.border_colour, rect, width=2, border_radius=ENTITY_RADIUS)

        # Small inner highlight — top-left corner glow
        highlight = pygame.Rect(
            rect.x + 4, rect.y + 4,
            rect.width // 3, rect.height // 3
        )
        highlight_colour = (
            min(self.fill_colour[0] + 80, 255),
            min(self.fill_colour[1] + 80, 255),
            min(self.fill_colour[2] + 80, 255),
        )
        pygame.draw.rect(surface, highlight_colour, highlight, border_radius=4)


# ── Monster ───────────────────────────────────────────────────────────────────
class Monster(Entity):
    """
    AI-controlled entity driven by the Double DQN agent.

    Monster does not decide its own moves — main.py asks
    ai_inference.get_action() for an action, then calls
    grid.apply_action() and updates monster.pos.

    Monster tracks whether its last move hit a wall so
    online_trainer.py can apply the wall-hit penalty correctly.
    """

    def __init__(self, pos):
        super().__init__(pos, C_MONSTER, C_MONSTER_BORDER)
        self.last_hit_wall = False     # set by main.py after each move

    def draw(self, surface):
        """
        Monster draws as a rounded rectangle with a pulsing-style
        darker inner shadow to look menacing.
        """
        rect = entity_rect(self.pos[0], self.pos[1])

        # Base fill + border
        pygame.draw.rect(surface, self.fill_colour,   rect, border_radius=ENTITY_RADIUS)
        pygame.draw.rect(surface, self.border_colour, rect, width=2, border_radius=ENTITY_RADIUS)

        # Dark inner shadow — inset rectangle
        shadow = pygame.Rect(
            rect.x + 6, rect.y + 6,
            rect.width  - 12,
            rect.height - 12,
        )
        shadow_colour = (
            max(self.fill_colour[0] - 60, 0),
            max(self.fill_colour[1] - 60, 0),
            max(self.fill_colour[2] - 60, 0),
        )
        pygame.draw.rect(surface, shadow_colour, shadow, border_radius=4)


# ── Princess ──────────────────────────────────────────────────────────────────
class Princess(Entity):
    """
    Autonomously moving entity — the player's goal.

    Princess moves to a random valid adjacent cell every
    PRINCESS_MOVE_INTERVAL seconds. Movement is triggered
    by main.py based on elapsed time; Princess.move() is
    called with the grid and monster position so it can
    avoid walls and the monster cell.

    If all adjacent cells are blocked (wall or monster),
    princess stays in place for that tick.
    """

    def __init__(self, pos):
        super().__init__(pos, C_PRINCESS, C_PRINCESS_BORDER)

    def move(self, grid, monster_pos, hero_pos=None):
        """
        Small move — random valid adjacent cell.
        Never onto monster or hero cell.
        """
        neighbours = grid.valid_neighbours(self.pos)
        safe = [
            cell for cell in neighbours
            if cell != monster_pos
            and cell != hero_pos
        ]
        if safe:
            self.pos = random.choice(safe)

    def jump(self, grid, monster_pos, hero_pos):
        """
        Big jump — teleports to a distant empty cell at least
        PRINCESS_JUMP_MIN_DIST away from current position.
        Prefers cells far from both hero and monster.
        Never jumps onto hero or monster cell.
        Stays put if no valid jump target exists.
        """
        from game.config import N, PRINCESS_JUMP_MIN_DIST

        all_cells = [(r, c) for r in range(N) for c in range(N)]

        candidates = [
            cell for cell in all_cells
            if cell not in grid.walls
            and cell != monster_pos
            and cell != hero_pos
            and cell != self.pos
            and manhattan_dist(cell, self.pos) >= PRINCESS_JUMP_MIN_DIST
        ]

        if not candidates:
            return   # nowhere far enough — stay put

    # Among candidates pick the one farthest from hero
    # so the player has to travel across the board
        self.pos = max(candidates, key=lambda c: manhattan_dist(c, hero_pos))

    def draw(self, surface):
        """
        Princess draws as a rounded rectangle with a diamond
        shape indicator inside to distinguish from other entities.
        """
        rect = entity_rect(self.pos[0], self.pos[1])

        # Base fill + border
        pygame.draw.rect(surface, self.fill_colour,   rect, border_radius=ENTITY_RADIUS)
        pygame.draw.rect(surface, self.border_colour, rect, width=2, border_radius=ENTITY_RADIUS)

        # Diamond indicator — four points around center
        cx = rect.centerx
        cy = rect.centery
        half = rect.width // 4
        diamond = [
            (cx,        cy - half),   # top
            (cx + half, cy        ),   # right
            (cx,        cy + half),   # bottom
            (cx - half, cy        ),   # left
        ]
        inner_colour = (
            min(self.fill_colour[0] + 40, 255),
            min(self.fill_colour[1] - 60,   0) if self.fill_colour[1] < 60 else self.fill_colour[1] - 60,
            max(self.fill_colour[2] - 80,   0),
        )
        pygame.draw.polygon(surface, inner_colour, diamond)