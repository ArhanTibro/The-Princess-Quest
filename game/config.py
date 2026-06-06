import os
import pygame

# ── Grid ─────────────────────────────────────────────────────────────────────
N              = 10          # grid side length (10×10 = 100 cells)
NUM_WALLS      = N           # wall count per round = N


pygame.init()

# ── Auto-fit to screen ────────────────────────────────────────────────────────
_info          = pygame.display.Info()
_SCREEN_H      = _info.current_h          # actual monitor height in pixels

MARGIN         = 36
HUD_HEIGHT     = 70
BOTTOM_MARGIN  = 36

# Calculate max cell size that fits vertically with some breathing room
_AVAILABLE_H   = int(_SCREEN_H * 0.88) - HUD_HEIGHT - MARGIN - BOTTOM_MARGIN
CELL_SIZE      = min(70, _AVAILABLE_H // N)   # cap at 70, shrink if needed

GRID_PIXEL     = N * CELL_SIZE
WINDOW_WIDTH   = GRID_PIXEL + MARGIN * 2
WINDOW_HEIGHT  = GRID_PIXEL + MARGIN + HUD_HEIGHT + BOTTOM_MARGIN

FPS            = 60

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(ROOT_DIR, "model", "monster.npz")
ASSETS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ── Difficulty — monster ticks per second ────────────────────────────────────
# Each tier defines: (min_elapsed_seconds, ticks_per_second, label)
# NEW
DIFFICULTY_TIERS = [
    (0,  4, "Medium"),     # starts at 4 ticks/sec immediately
    (20, 6, "Hard"),       # ramps up to 6 ticks/sec after 20 seconds
    (45, 7, "Extreme"),    # brutal 
    (70, 8, "Insane"),     # nearly unplayable — good luck
]

# ── Princess move interval ───────────────────────────────────────────────────
PRINCESS_MOVE_INTERVAL = 0.1   # moves every 0.3 seconds 




# ── Colours  (R, G, B) ───────────────────────────────────────────────────────
# Background & grid
C_BACKGROUND       = (18,  18,  28)     # deep navy — main window bg
C_GRID_LINE        = (40,  40,  60)     # subtle grid lines
C_CELL_EMPTY       = (24,  24,  38)     # empty cell fill

# Entities
C_HERO             = (72,  160, 255)    # bright blue
C_HERO_BORDER      = (140, 200, 255)    # lighter blue border
C_MONSTER          = (220, 50,  50)     # vivid red
C_MONSTER_BORDER   = (255, 120, 120)    # light red border
C_PRINCESS         = (255, 200, 50)     # gold
C_PRINCESS_BORDER  = (255, 230, 140)    # light gold border
C_WALL             = (55,  55,  75)     # muted dark slate
C_WALL_BORDER      = (70,  70,  95)     # slightly lighter slate border

# HUD
C_HUD_BG           = (12,  12,  22)     # darker than window bg
C_HUD_TEXT         = (200, 200, 220)    # soft white
C_HUD_ACCENT       = (140, 100, 255)    # purple accent for labels

# Difficulty label colours — one per tier
C_DIFFICULTY = {
    "Medium"  : (255, 200, 50),
    "Hard"    : (255, 130, 40),
    "Extreme" : (220, 50,  50),
    "Insane"  : (180, 0,   220),   # purple — ominous
}

# Screens
C_OVERLAY          = (0,   0,   0)      # used with alpha for dim overlay
C_WIN_TEXT         = (80,  200, 120)    # green
C_LOSE_TEXT        = (220, 50,  50)     # red
C_BUTTON_BG        = (60,  60,  90)     # play/replay button background
C_BUTTON_HOVER     = (90,  90,  130)    # button hover state
C_BUTTON_TEXT      = (230, 230, 255)    # button label

# ── Font sizes ───────────────────────────────────────────────────────────────
FONT_HUD_SMALL     = 18
FONT_HUD_LARGE     = 22
FONT_TITLE         = 52
FONT_SUBTITLE      = 22
FONT_BUTTON        = 20
FONT_RESULT        = 46

# ── Entity render sizes ───────────────────────────────────────────────────────
# Drawn as rounded rectangles inside each cell
ENTITY_PADDING     = 6     # pixels inset from cell edge
ENTITY_RADIUS      = 8     # corner radius for rounded rect
WALL_RADIUS        = 4     # slightly less rounded for walls

# ── Online training config ───────────────────────────────────────────────────
# These mirror the training hyperparameters exactly
ONLINE_BUFFER_SIZE = 50_000
ONLINE_BATCH_SIZE  = 64
ONLINE_LR          = 0.0005
ONLINE_GAMMA       = 0.95
ONLINE_TRAIN_EVERY = 4         # gradient update every N monster steps
ONLINE_TARGET_SYNC = 500       # target network sync every N training steps
ONLINE_EPSILON     = 0.05      # fixed low epsilon during gameplay (mostly exploit)


# ── Helper: get current difficulty tier ─────────────────────────────────────
def get_difficulty(elapsed_seconds):
    """
    Returns (ticks_per_second, label) for the given elapsed time.
    Walks tiers in reverse so the highest applicable tier wins.
    """
    ticks, label = DIFFICULTY_TIERS[0][1], DIFFICULTY_TIERS[0][2]
    for min_sec, t, lbl in DIFFICULTY_TIERS:
        if elapsed_seconds >= min_sec:
            ticks, label = t, lbl
    return ticks, label


# ── Helper: cell top-left pixel position ────────────────────────────────────
def cell_to_pixel(row, col):
    """
    Returns (x, y) pixel coordinate of the top-left corner of a grid cell.
    Accounts for margin and HUD height offset.
    """
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + HUD_HEIGHT + row * CELL_SIZE
    return x, y


# ── Helper: entity rect inside a cell ───────────────────────────────────────
def entity_rect(row, col):
    """
    Returns a pygame.Rect for drawing an entity inside its cell,
    inset by ENTITY_PADDING on all sides.
    """
    x, y = cell_to_pixel(row, col)
    return pygame.Rect(
        x + ENTITY_PADDING,
        y + ENTITY_PADDING,
        CELL_SIZE - ENTITY_PADDING * 2,
        CELL_SIZE - ENTITY_PADDING * 2,
    )