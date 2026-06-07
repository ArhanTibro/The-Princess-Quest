import sys
import os
import pygame

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.config import (
    ASSETS_DIR, N, CELL_SIZE, MARGIN, HUD_HEIGHT,
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS,
    FONT_HUD_SMALL, FONT_HUD_LARGE, FONT_TITLE,
    FONT_SUBTITLE, FONT_BUTTON, FONT_RESULT,
    C_BACKGROUND, C_HUD_BG, C_HUD_TEXT, C_HUD_ACCENT,
    C_DIFFICULTY, C_WIN_TEXT, C_LOSE_TEXT,
    C_BUTTON_BG, C_BUTTON_HOVER, C_BUTTON_TEXT, C_OVERLAY,
    PRINCESS_MOVE_INTERVAL, PRINCESS_JUMP_INTERVAL,
    get_difficulty, entity_rect,
)
from game.grid           import DELTAS, Grid, manhattan
from game.entities       import Hero, Monster, Princess, KEY_ACTION_MAP
from game.ai_inference   import AIInference
from game.online_trainer import OnlineTrainer


# ── Game states ───────────────────────────────────────────────────────────────
STATE_START   = "start"
STATE_PLAYING = "playing"
STATE_WIN     = "win"
STATE_LOSE    = "lose"


# ══ Utilities ═════════════════════════════════════════════════════════════════
def draw_text_centred(surface, text, font, colour, cx, cy):
    surf = font.render(text, True, colour)
    rect = surf.get_rect(center=(cx, cy))
    surface.blit(surf, rect)


def draw_text_left(surface, text, font, colour, x, y):
    surf = font.render(text, True, colour)
    surface.blit(surf, (x, y))


# ══ Button ════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, text, cx, cy, w=180, h=44):
        self.text = text
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = (cx, cy)

    def draw(self, surface, font):
        mouse  = pygame.mouse.get_pos()
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
    hud_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HUD_HEIGHT)
    pygame.draw.rect(surface, C_HUD_BG, hud_rect)

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


# ══ Start screen ══════════════════════════════════════════════════════════════
def draw_start_screen(surface, fonts, play_button, sprites):
    surface.fill(C_BACKGROUND)

    cx = WINDOW_WIDTH  // 2
    cy = WINDOW_HEIGHT // 2

    draw_text_centred(surface, "THE PRINCESS QUEST",
                      fonts["title"], C_HUD_ACCENT, cx, cy - 200)
    draw_text_centred(surface, "Reach the princess before the monster catches you.",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 148)
    draw_text_centred(surface, "Arrow keys or WASD to move.",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 122)

    # ── Entity showcase ───────────────────────────────────────────────────────
    entities_info = [
        ("hero",     "Hero",         (72,  160, 255)),
        ("monster",  "Monster (AI)", (220, 50,  50 )),
        ("princess", "Princess",     (255, 200, 50 )),
    ]

    sprite_size = 64
    total_width = len(entities_info) * 120
    start_x     = cx - total_width // 2 + 20
    entity_y    = cy - 60

    for i, (name, label, colour) in enumerate(entities_info):
        ex     = start_x + i * 120
        sprite = sprites.get(name) if sprites else None

        if sprite:
            showcase = pygame.transform.scale(sprite, (sprite_size, sprite_size))
            surface.blit(showcase, (ex - sprite_size // 2, entity_y))
        else:
            rect = pygame.Rect(ex - sprite_size // 2, entity_y,
                               sprite_size, sprite_size)
            pygame.draw.rect(surface, colour, rect, border_radius=10)

        draw_text_centred(surface, label, fonts["small"], C_HUD_TEXT,
                          ex, entity_y + sprite_size + 14)

    draw_text_centred(surface, "Grey blocks are walls — navigate around them.",
                      fonts["small"], C_HUD_TEXT,
                      cx, entity_y + sprite_size + 46)

    play_button.rect.center = (cx, entity_y + sprite_size + 100)
    play_button.draw(surface, fonts["button"])
    pygame.display.flip()


# ══ End screen ════════════════════════════════════════════════════════════════
def draw_end_screen(surface, fonts, win, elapsed, replay_button, quit_button):
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

    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60
    draw_text_centred(surface, f"Time: {minutes:02d}:{seconds:02d}",
                      fonts["subtitle"], C_HUD_TEXT, cx, cy - 30)

    replay_button.draw(surface, fonts["button"])
    quit_button.draw(surface,   fonts["button"])
    pygame.display.flip()


# ══ Entity rendering ══════════════════════════════════════════════════════════
def draw_entities(surface, hero, monster, princess, sprites):
    for entity, name in [
        (hero,    "hero"),
        (princess,"princess"),
        (monster, "monster"),
    ]:
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
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("The Princess Quest")
        self.clock  = pygame.time.Clock()
        self._init_fonts()
        self.sprites = self._load_sprites()

        # ── Core systems ──────────────────────────────────────────────────────
        self.grid      = Grid()
        self.inference = AIInference()
        self.trainer   = OnlineTrainer(self.inference)

        # ── Entities — set per round ──────────────────────────────────────────
        self.hero     = None
        self.monster  = None
        self.princess = None

        # ── Timers & accumulators ─────────────────────────────────────────────
        self.elapsed             = 0.0
        self.princess_timer      = 0.0
        self.princess_jump_timer = 0.0
        self.monster_tick_acc    = 0.0

        # ── Game state ────────────────────────────────────────────────────────
        self.state = STATE_START

        # ── Buttons ───────────────────────────────────────────────────────────
        cx = WINDOW_WIDTH // 2
        self.play_button   = Button("PLAY",       cx, WINDOW_HEIGHT // 2 + 110)
        self.replay_button = Button("PLAY AGAIN", cx, WINDOW_HEIGHT // 2 + 20)
        self.quit_button   = Button("QUIT",       cx, WINDOW_HEIGHT // 2 + 74)

    # ── Font init ─────────────────────────────────────────────────────────────
    def _init_fonts(self):
        self.fonts = {
            "small"   : pygame.font.SysFont("Arial", FONT_HUD_SMALL),
            "large"   : pygame.font.SysFont("Arial", FONT_HUD_LARGE,  bold=True),
            "title"   : pygame.font.SysFont("Arial", FONT_TITLE,      bold=True),
            "subtitle": pygame.font.SysFont("Arial", FONT_SUBTITLE),
            "button"  : pygame.font.SysFont("Arial", FONT_BUTTON,     bold=True),
            "result"  : pygame.font.SysFont("Arial", FONT_RESULT,     bold=True),
        }

    # ── Sprite loading ────────────────────────────────────────────────────────
    def _load_sprites(self):
        sprite_size = CELL_SIZE - 8
        sprites     = {}
        for name in ("hero", "monster", "princess"):
            path = os.path.join(ASSETS_DIR, f"{name}.png")
            if os.path.exists(path):
                img           = pygame.image.load(path).convert_alpha()
                sprites[name] = pygame.transform.scale(img, (sprite_size, sprite_size))
                print(f"[Sprites] Loaded {name}.png")
            else:
                sprites[name] = None
                print(f"[Sprites] {name}.png not found at {path}")
        return sprites

    # ── Round reset ───────────────────────────────────────────────────────────
    def _new_round(self):
        monster_pos, hero_pos, princess_pos = self.grid.reset()
        self.hero     = Hero(hero_pos)
        self.monster  = Monster(monster_pos)
        self.princess = Princess(princess_pos)

        self.elapsed             = 0.0
        self.princess_timer      = 0.0
        self.princess_jump_timer = 0.0
        self.monster_tick_acc    = 0.0

    # ── Monster tick ──────────────────────────────────────────────────────────
    def _monster_tick(self):
        prev_dist = manhattan(self.monster.pos, self.hero.pos)
        state     = AIInference.build_state(self.monster.pos, self.hero.pos)

        # Only pass valid (non-wall) actions — monster never wastes a tick
        valid_actions = [
            a for a, (dr, dc) in DELTAS.items()
            if self.grid.is_valid(
                self.monster.pos[0] + dr,
                self.monster.pos[1] + dc
            )
        ]
        action = self.inference.get_action(
            self.monster.pos, self.hero.pos, valid_actions
        )

        new_pos, hit_wall          = self.grid.apply_action(self.monster.pos, action)
        self.monster.pos           = new_pos
        self.monster.last_hit_wall = hit_wall

        # ── Reward for online training ────────────────────────────────────────
        curr_dist  = manhattan(self.monster.pos, self.hero.pos)
        caught     = self.monster.pos == self.hero.pos

        reward = -0.01
        if hit_wall:
            reward -= 0.10
        if curr_dist < prev_dist:
            reward += 0.05
        if caught:
            reward += 10.0

        next_state = AIInference.build_state(self.monster.pos, self.hero.pos)
        self.trainer.push_experience(state, action, reward, next_state, caught)

        return caught

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.trainer.start()

        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                if self.state == STATE_START:
                    if self.play_button.is_clicked(event):
                        self._new_round()
                        self.state = STATE_PLAYING

                elif self.state == STATE_PLAYING:
                    if event.type == pygame.KEYDOWN:
                        action = self.hero.handle_key(event.key)
                        if action is not None:
                            new_pos, _ = self.grid.apply_action(
                                self.hero.pos, action
                            )
                            self.hero.pos = new_pos
                            if self.hero.pos == self.princess.pos:
                                self.state = STATE_WIN

                elif self.state in (STATE_WIN, STATE_LOSE):
                    if self.replay_button.is_clicked(event):
                        self._new_round()
                        self.state = STATE_PLAYING
                    if self.quit_button.is_clicked(event):
                        running = False

            # ── Game logic ────────────────────────────────────────────────────
            if self.state == STATE_PLAYING:

                self.elapsed             += dt
                self.princess_timer      += dt
                self.princess_jump_timer += dt

                # ── Princess small move ───────────────────────────────────────
                if self.princess_timer >= PRINCESS_MOVE_INTERVAL:
                    self.princess.move(
                        self.grid, self.monster.pos, self.hero.pos
                    )
                    self.princess_timer = 0.0

                # ── Princess big jump ─────────────────────────────────────────
                if self.princess_jump_timer >= PRINCESS_JUMP_INTERVAL:
                    self.princess.jump(
                        self.grid, self.monster.pos, self.hero.pos
                    )
                    self.princess_jump_timer = 0.0

                # ── Monster tick scheduler ────────────────────────────────────
                ticks_per_sec, diff_label = get_difficulty(self.elapsed)
                tick_interval             = 1.0 / ticks_per_sec
                self.monster_tick_acc    += dt

                if self.monster_tick_acc >= tick_interval:
                    self.monster_tick_acc -= tick_interval
                    if self._monster_tick():
                        self.state = STATE_LOSE

            # ── Rendering ─────────────────────────────────────────────────────
            if self.state == STATE_START:
                draw_start_screen(
                    self.screen, self.fonts,
                    self.play_button, self.sprites
                )

            elif self.state == STATE_PLAYING:
                self.screen.fill(C_BACKGROUND)
                _, diff_label = get_difficulty(self.elapsed)
                draw_hud(
                    self.screen, self.fonts,
                    self.elapsed, diff_label,
                    self.trainer.updates_done
                )
                self.grid.draw(self.screen)
                self.grid.draw_grid_border(self.screen)
                draw_entities(
                    self.screen, self.hero,
                    self.monster, self.princess, self.sprites
                )
                pygame.display.flip()

            elif self.state in (STATE_WIN, STATE_LOSE):
                _, diff_label = get_difficulty(self.elapsed)
                draw_hud(
                    self.screen, self.fonts,
                    self.elapsed, diff_label,
                    self.trainer.updates_done
                )
                draw_end_screen(
                    self.screen, self.fonts,
                    win           = (self.state == STATE_WIN),
                    elapsed       = self.elapsed,
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