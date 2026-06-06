import sys
import os
import time
import pygame

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.config import (
    N, CELL_SIZE, MARGIN, HUD_HEIGHT,
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS,
    FONT_HUD_SMALL, FONT_HUD_LARGE, FONT_TITLE,
    FONT_SUBTITLE, FONT_BUTTON, FONT_RESULT,
    C_BACKGROUND, C_HUD_BG, C_HUD_TEXT, C_HUD_ACCENT,
    C_DIFFICULTY, C_WIN_TEXT, C_LOSE_TEXT,
    C_BUTTON_BG, C_BUTTON_HOVER, C_BUTTON_TEXT, C_OVERLAY,
    PRINCESS_MOVE_INTERVAL,
    get_difficulty, entity_rect,
)
from game.grid            import Grid, manhattan
from game.entities        import Hero, Monster, Princess, KEY_ACTION_MAP
from game.ai_inference    import AIInference
from game.online_trainer  import OnlineTrainer


# ── Game states ───────────────────────────────────────────────────────────────
STATE_START    = "start"
STATE_PLAYING  = "playing"
STATE_WIN      = "win"
STATE_LOSE     = "lose"


# ══ Utility — draw text centred ═══════════════════════════════════════════════
def draw_text_centred(surface, text, font, colour, cx, cy):
    surf = font.render(text, True, colour)
    rect = surf.get_rect(center=(cx, cy))
    surface.blit(surf, rect)


def draw_text_left(surface, text, font, colour, x, y):
    surf = font.render(text, True, colour)
    surface.blit(surf, (x, y))


# ══ Button ════════════════════════════════════════════════════════════════════
class Button:
    """Simple clickable rectangle button."""

    def __init__(self, text, cx, cy, w=180, h=44):
        self.text = text
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = (cx, cy)

    def draw(self, surface, font):
        mouse = pygame.mouse.get_pos()
        colour = C_BUTTON_HOVER if self.rect.collidepoint(mouse) else C_BUTTON_BG
        pygame.draw.rect(surface, colour,       self.rect, border_radius=8)
        pygame.draw.rect(surface, C_HUD_ACCENT, self.rect, width=2, border_radius=8)
        draw_text_centred(surface, self.text, font, C_BUTTON_TEXT,
                          self.rect.centerx, self.rect.centery)

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


# ══ HUD ═══════════════════════════════════════════════════════════════════════
def draw_hud(surface, fonts, elapsed, difficulty_label, updates_done):
    """
    Draw the info bar above the grid.
    Shows: elapsed time | difficulty | online training updates.
    """
    hud_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HUD_HEIGHT)
    pygame.draw.rect(surface, C_HUD_BG, hud_rect)

    # Separator line below HUD
    pygame.draw.line(
        surface, C_HUD_ACCENT,
        (MARGIN, HUD_HEIGHT - 1),
        (WINDOW_WIDTH - MARGIN, HUD_HEIGHT - 1),
        1
    )

    minutes  = int(elapsed) // 60
    seconds  = int(elapsed) % 60
    time_str = f"{minutes:02d}:{seconds:02d}"

    # Left — time
    draw_text_left(surface, "TIME", fonts["small"], C_HUD_ACCENT,
                   MARGIN, HUD_HEIGHT // 2 - 18)
    draw_text_left(surface, time_str, fonts["large"], C_HUD_TEXT,
                   MARGIN, HUD_HEIGHT // 2)

    # Centre — difficulty
    diff_colour = C_DIFFICULTY.get(difficulty_label, C_HUD_TEXT)
    draw_text_centred(surface, "DIFFICULTY", fonts["small"], C_HUD_ACCENT,
                      WINDOW_WIDTH // 2, HUD_HEIGHT // 2 - 18)
    draw_text_centred(surface, difficulty_label, fonts["large"], diff_colour,
                      WINDOW_WIDTH // 2, HUD_HEIGHT // 2)

    # Right — AI updates
    draw_text_left(surface, "AI UPDATES", fonts["small"], C_HUD_ACCENT,
                   WINDOW_WIDTH - MARGIN - 90, HUD_HEIGHT // 2 - 18)
    draw_text_left(surface, str(updates_done), fonts["large"], C_HUD_TEXT,
                   WINDOW_WIDTH - MARGIN - 90, HUD_HEIGHT // 2)


# ══ Screens ═══════════════════════════════════════════════════════════════════
def draw_start_screen(surface, fonts, play_button, sprites):
    surface.fill(C_BACKGROUND)

    cx = WINDOW_WIDTH  // 2
    cy = WINDOW_HEIGHT // 2

    # ── Title ─────────────────────────────────────────────────────────────
    draw_text_centred(surface, "THE PRINCESS QUEST",
                      fonts["title"], C_HUD_ACCENT, cx, cy - 200)

    draw_text_centred(surface, "Reach the princess before the monster catches you.",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 148)
    draw_text_centred(surface, "Arrow keys or WASD to move.",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 122)

    # ── Entity showcase — image or coloured rect + label ──────────────────
    entities_info = [
        ("hero",     "Hero",          (72,  160, 255)),
        ("monster",  "Monster (AI)",  (220, 50,  50 )),
        ("princess", "Princess",      (255, 200, 50 )),
    ]

    sprite_size   = 64                          # display size on start screen
    total_width   = len(entities_info) * 120    # spacing between each entity
    start_x       = cx - total_width // 2 + 20
    entity_y      = cy - 60

    for i, (name, label, colour) in enumerate(entities_info):
        ex = start_x + i * 120

        sprite = sprites.get(name) if sprites else None

        if sprite:
            # Scale sprite to showcase size (may differ from in-game size)
            showcase_sprite = pygame.transform.scale(sprite, (sprite_size, sprite_size))
            surface.blit(showcase_sprite, (ex - sprite_size // 2, entity_y))
        else:
            # Fallback — draw coloured square
            rect = pygame.Rect(ex - sprite_size // 2, entity_y, sprite_size, sprite_size)
            pygame.draw.rect(surface, colour, rect, border_radius=10)

        # Label below each entity
        draw_text_centred(surface, label, fonts["small"], C_HUD_TEXT,
                          ex, entity_y + sprite_size + 14)

    # ── Wall note ─────────────────────────────────────────────────────────
    draw_text_centred(surface, "Grey blocks are walls — navigate around them.",
                      fonts["small"], C_HUD_TEXT, cx, entity_y + sprite_size + 46)

    # ── Play button ───────────────────────────────────────────────────────
    play_button.rect.center = (cx, entity_y + sprite_size + 100)
    play_button.draw(surface, fonts["button"])

    pygame.display.flip()

def draw_end_screen(surface, fonts, win, elapsed, replay_button, quit_button):
    # Dim overlay
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((*C_OVERLAY, 180))
    surface.blit(overlay, (0, 0))

    cx = WINDOW_WIDTH  // 2
    cy = WINDOW_HEIGHT // 2

    if win:
        draw_text_centred(surface, "YOU SAVED THE PRINCESS!",
                          fonts["result"], C_WIN_TEXT, cx, cy - 80)
    else:
        draw_text_centred(surface, "CAUGHT BY THE MONSTER!",
                          fonts["result"], C_LOSE_TEXT, cx, cy - 80)

    minutes  = int(elapsed) // 60
    seconds  = int(elapsed) % 60
    draw_text_centred(surface, f"Time: {minutes:02d}:{seconds:02d}",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 30)

    replay_button.draw(surface, fonts["button"])
    quit_button.draw(surface,   fonts["button"])
    pygame.display.flip()


# ══ Entity rendering ══════════════════════════════════════════════════════════
def draw_entities(surface, hero, monster, princess, sprites):
    # Draw each entity — sprite if available, coloured rect as fallback
    for entity, name in [(hero, "hero"), (princess, "princess"), (monster, "monster")]:
        sprite = sprites.get(name)
        if sprite:
            x, y = entity_rect(entity.pos[0], entity.pos[1]).topleft
            surface.blit(sprite, (x, y))
        else:
            entity.draw(surface)


# ══ Main game class ═══════════════════════════════════════════════════════════
class Game:

    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("The Princess Quest")
        self.clock   = pygame.time.Clock()
        self._init_fonts()

        self.sprites = self._load_sprites()

        # Core systems — created once, reused across rounds
        self.grid      = Grid()
        self.inference = AIInference()
        self.trainer   = OnlineTrainer(self.inference)

        # Game entities — set per round in _new_round()
        self.hero      = None
        self.monster   = None
        self.princess  = None

        # State
        self.state     = STATE_START
        self.elapsed   = 0.0
        self.princess_timer = 0.0

        # Monster tick accumulator
        self.monster_tick_acc = 0.0

        # Buttons
        cx = WINDOW_WIDTH // 2
        self.play_button   = Button("PLAY",       cx, WINDOW_HEIGHT // 2 + 110)
        self.replay_button = Button("PLAY AGAIN", cx, WINDOW_HEIGHT // 2 + 20)
        self.quit_button   = Button("QUIT",       cx, WINDOW_HEIGHT // 2 + 74)

    def _init_fonts(self):
        self.fonts = {
            "small"   : pygame.font.SysFont("Arial", FONT_HUD_SMALL),
            "large"   : pygame.font.SysFont("Arial", FONT_HUD_LARGE,  bold=True),
            "title"   : pygame.font.SysFont("Arial", FONT_TITLE,      bold=True),
            "subtitle": pygame.font.SysFont("Arial", FONT_SUBTITLE),
            "button"  : pygame.font.SysFont("Arial", FONT_BUTTON,     bold=True),
            "result"  : pygame.font.SysFont("Arial", FONT_RESULT,     bold=True),
        }

    def _load_sprites(self):
        # Load and scale entity sprites to fit inside a cell.
        # Falls back to None if image file is missing —
        # entities will draw as coloured rectangles instead.

        sprite_size = CELL_SIZE - 8    # slight inset so it doesn't touch grid lines
        sprites     = {}
        for name in ("hero", "monster", "princess"):
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets", f"{name}.png"
            )
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                sprites[name] = pygame.transform.scale(img, (sprite_size, sprite_size))
                print(f"[Sprites] Loaded {name}.png")
            else:
                sprites[name] = None
                print(f"[Sprites] {name}.png not found — using coloured rect")
        return sprites

    # ── Round setup ───────────────────────────────────────────────────────────
    def _new_round(self):
        monster_pos, hero_pos, princess_pos = self.grid.reset()
        self.hero     = Hero(hero_pos)
        self.monster  = Monster(monster_pos)
        self.princess = Princess(princess_pos)

        self.elapsed          = 0.0
        self.princess_timer   = 0.0
        self.monster_tick_acc = 0.0

    # ── Monster tick ──────────────────────────────────────────────────────────
    def _monster_tick(self):
        """
        Ask the AI for an action, apply it, record experience,
        push to online trainer.
        """
        prev_dist = manhattan(self.monster.pos, self.hero.pos)
        state     = AIInference.build_state(self.monster.pos, self.hero.pos)

        # Get action from trained model (greedy, epsilon=0.05 in online trainer)
        action = self.inference.get_action(self.monster.pos, self.hero.pos)

        # Apply move
        new_pos, hit_wall           = self.grid.apply_action(self.monster.pos, action)
        self.monster.pos            = new_pos
        self.monster.last_hit_wall  = hit_wall

        # ── Build reward for online training ──────────────────────────────────
        curr_dist = manhattan(self.monster.pos, self.hero.pos)
        caught    = self.monster.pos == self.hero.pos

        reward  = -0.01                              # step penalty
        if hit_wall:
            reward -= 0.10                           # wall penalty
        if curr_dist < prev_dist:
            reward += 0.05                           # shaped: moved closer
        if caught:
            reward += 10.0                           # terminal catch reward

        next_state = AIInference.build_state(self.monster.pos, self.hero.pos)
        self.trainer.push_experience(state, action, reward, next_state, caught)

        return caught

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.trainer.start()

        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0      # delta time in seconds

            # ── Event handling ────────────────────────────────────────────────
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                # ── Start screen ──────────────────────────────────────────────
                if self.state == STATE_START:
                    if self.play_button.is_clicked(event):
                        self._new_round()
                        self.state = STATE_PLAYING

                # ── Playing ───────────────────────────────────────────────────
                elif self.state == STATE_PLAYING:
                    if event.type == pygame.KEYDOWN:
                        action = self.hero.handle_key(event.key)
                        if action is not None:
                            new_pos, _ = self.grid.apply_action(
                                self.hero.pos, action
                            )
                            self.hero.pos = new_pos

                            # Win check — hero reaches princess
                            if self.hero.pos == self.princess.pos:
                                self.state = STATE_WIN

                # ── End screens ───────────────────────────────────────────────
                elif self.state in (STATE_WIN, STATE_LOSE):
                    if self.replay_button.is_clicked(event):
                        self._new_round()
                        self.state = STATE_PLAYING
                    if self.quit_button.is_clicked(event):
                        running = False

            # ── Game logic (playing only) ─────────────────────────────────────
            if self.state == STATE_PLAYING:
                self.elapsed        += dt
                self.princess_timer += dt

                # ── Princess move ─────────────────────────────────────────────
                if self.princess_timer >= PRINCESS_MOVE_INTERVAL:
                    self.princess.move(self.grid, self.monster.pos)
                    self.princess_timer = 0.0

                    # Win check after princess moves onto hero's cell (rare)
                    if self.princess.pos == self.hero.pos:
                        self.state = STATE_WIN

                # ── Monster tick scheduler ────────────────────────────────────
                ticks_per_sec, diff_label = get_difficulty(self.elapsed)
                tick_interval             = 1.0 / ticks_per_sec
                self.monster_tick_acc    += dt

                if self.monster_tick_acc >= tick_interval:
                    self.monster_tick_acc -= tick_interval
                    caught = self._monster_tick()
                    if caught:
                        self.state = STATE_LOSE

            # ── Rendering ─────────────────────────────────────────────────────
            if self.state == STATE_START:
                
                draw_start_screen(
                    self.screen, self.fonts, self.play_button, self.sprites
                )

            elif self.state == STATE_PLAYING:
                self.screen.fill(C_BACKGROUND)
                ticks_per_sec, diff_label = get_difficulty(self.elapsed)

                draw_hud(
                    self.screen, self.fonts,
                    self.elapsed, diff_label,
                    self.trainer.updates_done
                )
                self.grid.draw(self.screen)
                self.grid.draw_grid_border(self.screen)
                draw_entities(self.screen, self.hero, self.monster, self.princess, self.sprites)
                pygame.display.flip()

            elif self.state in (STATE_WIN, STATE_LOSE):
                # Keep last game frame visible under the overlay
                ticks_per_sec, diff_label = get_difficulty(self.elapsed)
                draw_hud(
                    self.screen, self.fonts,
                    self.elapsed, diff_label,
                    self.trainer.updates_done
                )
                draw_end_screen(
                    self.screen, self.fonts,
                    win     = (self.state == STATE_WIN),
                    elapsed = self.elapsed,
                    replay_button = self.replay_button,
                    quit_button   = self.quit_button,
                )

        # ── Cleanup ───────────────────────────────────────────────────────────
        self.trainer.stop()
        pygame.quit()
        sys.exit()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Game().run()