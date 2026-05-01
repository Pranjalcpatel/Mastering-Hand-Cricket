import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
from agents_fight import AdaptiveFiniteAgent, OptimalAgent, RandomAgent

try:
    import pygame
    import pygame.gfxdraw
except ImportError as exc:
    raise SystemExit(
        "pygame is required for visualization. Install with: pip install pygame"
    ) from exc


@dataclass
class BallEvent:
    innings: int
    ball_index: int
    striker: str
    bowler: str
    bat_symbol: int
    bowl_symbol: int
    is_out: bool
    first_score: int
    second_needed: int
    balls_left: int
    message: str


def _infer_param(agent1, agent2, name: str, explicit):
    if explicit is not None:
        return explicit
    a1 = hasattr(agent1, name)
    a2 = hasattr(agent2, name)
    if a1 and a2:
        v1 = getattr(agent1, name)
        v2 = getattr(agent2, name)
        if v1 != v2:
            raise ValueError(f"Mismatch for {name}: agent1={v1}, agent2={v2}")
        return v1
    if a1:
        return getattr(agent1, name)
    if a2:
        return getattr(agent2, name)
    raise ValueError(f"Could not infer {name}; pass it explicitly.")


def generate_match_events(
    agent1, agent2, agent1_name, agent2_name, agent1_bats_first: bool, T: int, max_score: int
) -> Tuple[List[BallEvent], int, int]:
    if hasattr(agent1, "reset_match"):
        agent1.reset_match()
    if hasattr(agent2, "reset_match"):
        agent2.reset_match()

    events: List[BallEvent] = []
    t = T
    s = 0
    ball1 = 0

    # Use short names for scoreboard
    p1_short = agent1_name[:3].upper()
    p2_short = agent2_name[:3].upper()

    while t > 0:
        ball1 += 1
        state = s
        if agent1_bats_first:
            striker_name, bowler_name = p1_short, p2_short
            bat = int(agent1.act("bat_first", t, s))
            bowl = int(agent2.act("bowl_first", t, s))
            if hasattr(agent1, "observe"):
                agent1.observe("bat_first", t, state, bat, bowl)
            if hasattr(agent2, "observe"):
                agent2.observe("bowl_first", t, state, bowl, bat)
        else:
            striker_name, bowler_name = p2_short, p1_short
            bat = int(agent2.act("bat_first", t, s))
            bowl = int(agent1.act("bowl_first", t, s))
            if hasattr(agent2, "observe"):
                agent2.observe("bat_first", t, state, bat, bowl)
            if hasattr(agent1, "observe"):
                agent1.observe("bowl_first", t, state, bowl, bat)

        is_out = bat == bowl
        if not is_out:
            s = min(s + bat, max_score)
            t -= 1
            msg = f"+{bat}"
        else:
            msg = "W"

        events.append(
            BallEvent(1, ball1, striker_name, bowler_name, bat, bowl, is_out, s, 0, t, msg)
        )
        if is_out:
            break

    target = min(s + 1, max_score)
    t = T
    k = target
    ball2 = 0

    while t > 0 and k > 0:
        ball2 += 1
        state = k
        if agent1_bats_first:
            striker_name, bowler_name = p2_short, p1_short
            bat = int(agent2.act("bat_second", t, k))
            bowl = int(agent1.act("bowl_second", t, k))
            if hasattr(agent2, "observe"):
                agent2.observe("bat_second", t, state, bat, bowl)
            if hasattr(agent1, "observe"):
                agent1.observe("bowl_second", t, state, bowl, bat)
        else:
            striker_name, bowler_name = p1_short, p2_short
            bat = int(agent1.act("bat_second", t, k))
            bowl = int(agent2.act("bowl_second", t, k))
            if hasattr(agent1, "observe"):
                agent1.observe("bat_second", t, state, bat, bowl)
            if hasattr(agent2, "observe"):
                agent2.observe("bowl_second", t, state, bowl, bat)

        is_out = bat == bowl
        if not is_out:
            k -= bat
            t -= 1
            msg = f"+{bat}"
        else:
            msg = "W"

        events.append(
            BallEvent(2, ball2, striker_name, bowler_name, bat, bowl, is_out, s, max(k, 0), t, msg)
        )
        if is_out:
            break

    if k <= 0:
        winner_idx = 2 if agent1_bats_first else 1
    else:
        winner_idx = 1 if agent1_bats_first else 2

    return events, winner_idx, target


def _draw_label(surface, font, text, x, y, color, center=False):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(rendered, rect)


def _draw_glow(surface, color, x, y, radius):
    """Draws a soft glowing orb."""
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for r in range(radius, 0, -2):
        alpha = int(50 * (1 - (r / radius)))
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), r)
    surface.blit(surf, (x - radius, y - radius), special_flags=pygame.BLEND_PREMULTIPLIED)


def _draw_hand_realistic(surface, cx, cy, symbol, facing, reveal):
    """Draws a much more natural, shaded vector-style hand."""
    reveal = max(0.0, min(1.0, reveal))
    sign = -1 if facing == "left" else 1

    # Colors
    skin_base = (245, 205, 160) # Slightly brighter skin tone
    skin_shadow = (215, 170, 125)
    skin_dark = (180, 135, 95)
    outline = (50, 40, 35)

    # Shake effect before revealing
    if reveal < 0.2:
        shake = math.sin(pygame.time.get_ticks() * 0.05) * 4
        cx += shake
        cy += shake

    # Arm
    pygame.draw.polygon(surface, skin_shadow, [
        (cx - sign * 80, cy + 30), (cx - sign * 140, cy + 40),
        (cx - sign * 140, cy - 20), (cx - sign * 80, cy - 10)
    ])
    pygame.draw.line(surface, outline, (cx - sign * 80, cy + 30), (cx - sign * 140, cy + 40), 3)
    pygame.draw.line(surface, outline, (cx - sign * 80, cy - 10), (cx - sign * 140, cy - 20), 3)

    # Palm back
    palm_rect = pygame.Rect(0, 0, 90, 85)
    palm_rect.center = (cx, cy + 5)
    pygame.draw.ellipse(surface, skin_base, palm_rect)
    
    # Palm depth/shadow
    pygame.draw.arc(surface, skin_dark, palm_rect.inflate(4, 4), 
                    math.radians(180) if sign==1 else 0, 
                    math.radians(360) if sign==1 else math.radians(180), 4)

    up_count = max(0, min(symbol, 5))

    # Fingers (Thumb, Index, Middle, Ring, Pinky)
    finger_data = [
        {"id": "pinky",  "x": -35, "y": -20, "len": 45, "thick": 16},
        {"id": "ring",   "x": -15, "y": -28, "len": 55, "thick": 18},
        {"id": "middle", "x": 8,   "y": -32, "len": 60, "thick": 18},
        {"id": "index",  "x": 30,  "y": -25, "len": 52, "thick": 18},
    ]

    for i, fd in enumerate(finger_data):
        is_up = i < up_count
        # Uncurling animation math
        curl_factor = 1.0 - reveal if is_up else 1.0
        
        fx = cx + fd["x"] * sign
        fy = cy + fd["y"]
        
        # Base joint
        pygame.draw.circle(surface, skin_shadow, (fx, fy), fd["thick"]//2)

        # Calculate fingertip position based on curl
        tip_y = fy - (fd["len"] * (1 - curl_factor*0.8))
        tip_x = fx + (sign * curl_factor * 15) # Fingers curl inward

        # Draw finger segment
        f_rect = pygame.Rect(0, 0, fd["thick"], abs(fy - tip_y) + fd["thick"])
        f_rect.midbottom = (fx, fy + fd["thick"]//2)
        
        pygame.draw.rect(surface, skin_shadow if curl_factor > 0.5 else skin_base, f_rect, border_radius=fd["thick"]//2)
        pygame.draw.rect(surface, outline, f_rect, width=2, border_radius=fd["thick"]//2)

        # Knuckle lines
        if curl_factor < 0.2:
            pygame.draw.line(surface, skin_dark, (fx - 5, tip_y + 15), (fx + 5, tip_y + 15), 2)

    # Thumb
    thumb_up = up_count == 5
    t_curl = 1.0 - reveal if thumb_up else 1.0
    tx, ty = cx + sign * 40, cy + 10
    
    # Thumb angled out
    t_tip_x = tx + sign * (30 * (1 - t_curl * 0.5))
    t_tip_y = ty - (25 * (1 - t_curl))
    
    pygame.draw.line(surface, skin_base, (tx, ty+10), (t_tip_x, t_tip_y), 20)
    pygame.draw.line(surface, outline, (tx, ty+10), (t_tip_x, t_tip_y), 24)
    pygame.draw.line(surface, skin_base, (tx, ty+10), (t_tip_x, t_tip_y), 20) # Redraw inside over outline
    pygame.draw.circle(surface, skin_base, (t_tip_x, t_tip_y), 10)
    pygame.draw.circle(surface, outline, (t_tip_x, t_tip_y), 10, 2)


def run_pygame_simulation(
    agent1, agent2, agent1_name, agent2_name, agent1_bats_first=True, T=None, max_score=None, step_delay_ms=900
):
    T = _infer_param(agent1, agent2, "T", T)
    max_score = _infer_param(agent1, agent2, "max_score", max_score)

    pygame.init()
    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Hand Cricket Pro")
    clock = pygame.time.Clock()

    # Sleek Dark Palette (Updated for brighter text)
    c_bg = (15, 23, 42)         # Deep slate
    c_panel = (30, 41, 59)      # Elevated slate
    c_white = (255, 255, 255)   # Pure white for max contrast
    c_text = (240, 245, 250)    # Very bright off-white for main text
    c_sub = (180, 190, 210)     # Brighter gray for secondary text
    c_accent = (34, 197, 94)    # Vivid Green (Runs)
    c_danger = (239, 68, 68)    # Neon Red (Out)
    c_pitch = (71, 85, 105)     # Brighter pitch lines

    font_huge = pygame.font.SysFont("impact, trebuchet ms", 72, bold=True)
    font_large = pygame.font.SysFont("trebuchet ms", 48, bold=True)
    font_score = pygame.font.SysFont("trebuchet ms", 36, bold=True)
    font_body = pygame.font.SysFont("trebuchet ms", 20)
    font_small = pygame.font.SysFont("trebuchet ms", 16, bold=True)

    def setup_new_match():
        evts, winner_idx, target = generate_match_events(agent1, agent2, agent1_name, agent2_name, agent1_bats_first, T, max_score)
        return evts, winner_idx, target, 0, True, 0

    events, winner_idx, target, idx, autoplay, last_tick = setup_new_match()
    running = True

    while running:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif event.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(events))
                    last_tick = now
                elif event.key == pygame.K_r:
                    events, winner_idx, target, idx, autoplay, last_tick = setup_new_match()

        if autoplay and idx < len(events) and now - last_tick >= step_delay_ms:
            idx += 1
            last_tick = now

        screen.fill(c_bg)

        shown = events[idx - 1] if idx > 0 else None
        
        # --- SCOREBOARD (Top Bar) ---
        pygame.draw.rect(screen, c_panel, (0, 0, 1000, 85))
        pygame.draw.line(screen, c_pitch, (0, 85), (1000, 85), 3)

        if shown:
            # Calculate current score based on innings
            cur_score = shown.first_score if shown.innings == 1 else (target - shown.second_needed)
            # Adjust score if the ball wasn't a wicket
            if not shown.is_out and shown.innings == 2:
                 cur_score = (target - shown.second_needed) + shown.bat_symbol # Re-add current run if not out to show live score

            b_left = shown.balls_left
            
            _draw_label(screen, font_score, f"{shown.striker} (BAT)", 30, 25, c_text)
            # Center Score display
            score_txt = f"{cur_score}"
            if shown.innings == 2:
                 score_txt += f" / {target}"
            _draw_label(screen, font_large, score_txt, 500, 35, c_white, center=True)

            _draw_label(screen, font_score, f"{shown.bowler} (BOWL)", 970, 25, c_sub, center=True)
            
            status_txt = ""
            if shown.innings == 2:
                needed = max(shown.second_needed, 0)
                status_txt = f"NEED {needed} RUNS OFF {b_left} BALLS"
            else:
                status_txt = f"INNINGS 1 | BALLS LEFT: {b_left}"
            
            _draw_label(screen, font_body, status_txt, 500, 70, c_accent if shown.innings==2 else c_sub, center=True)

        else:
            _draw_label(screen, font_score, "MATCH READY - PRESS SPACE", 500, 42, c_white, center=True)

        # --- MAIN ARENA ---
        reveal = 1.0 if idx == 0 else min(1.0, (now - last_tick) / 250.0)
        sym_l = shown.bat_symbol if shown else 0
        sym_r = shown.bowl_symbol if shown else 0

        # Draw glowing numbers in background
        if shown and reveal > 0.5:
            color = c_danger if shown.is_out else c_accent
            # Glow effect behind the central text
            _draw_glow(screen, color, 500, 340, 160)
            # Main outcome text
            _draw_label(screen, font_huge, shown.message, 500, 340, c_white, center=True)

        _draw_hand_realistic(screen, 300, 370, sym_l, "right", reveal)
        _draw_hand_realistic(screen, 700, 370, sym_r, "left", reveal)

        # --- TIMELINE (Bottom Bar) ---
        pygame.draw.rect(screen, c_panel, (20, 530, 960, 70), border_radius=15)
        
        recent = events[max(0, idx - 15):idx]
        for i, ev in enumerate(recent):
            cx = 60 + i * 60
            cy = 565
            col = c_danger if ev.is_out else c_panel
            # Ensure text on the timeline is bright
            txt_col = c_white if ev.is_out else c_accent 
            
            pygame.draw.circle(screen, col, (cx, cy), 22)
            pygame.draw.circle(screen, c_pitch, (cx, cy), 22, 2)
            lbl = "W" if ev.is_out else str(ev.bat_symbol)
            _draw_label(screen, font_score, lbl, cx, cy, txt_col, center=True)
            
            # Innings marker separator
            if i > 0 and recent[i-1].innings != ev.innings:
                pygame.draw.line(screen, c_sub, (cx-30, cy-25), (cx-30, cy+25), 3)

        # --- GAME OVER OVERLAY ---
        if idx >= len(events):
            overlay = pygame.Surface((1000, 620), pygame.SRCALPHA)
            overlay.fill((10, 15, 30, 200)) # Darker, more opaque overlay
            screen.blit(overlay, (0, 0))
            
            winning_name = agent1_name if winner_idx == 1 else agent2_name
            
            # Victory message using pure white
            _draw_label(screen, font_huge, f"{winning_name} WINS!", 500, 300, c_white, center=True)
            _draw_label(screen, font_score, "Press 'R' to Rematch", 500, 380, c_sub, center=True)

        # Controls hint
        _draw_label(screen, font_small, "SPACE: Play/Pause | RIGHT: Step | R: Restart | ESC: Quit", 500, 610, c_sub, center=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _build_agent(kind: str, T: int, M: int, max_score: int):
    if kind == "optimal":
        return OptimalAgent(T, M, max_score)
    if kind == "adaptive":
        return AdaptiveFiniteAgent(T, M, max_score)
    if kind == "random":
        return RandomAgent(M)
    raise ValueError(f"Unknown agent type: {kind}")


def main():
    parser = argparse.ArgumentParser(description="Pygame hand-cricket simulation.")
    parser.add_argument("--T", type=int, default=20, help="Balls per innings.")
    parser.add_argument("--M", type=int, default=6, help="Number symbols (1..M).")
    parser.add_argument("--max-score", type=int, default=120, help="Score cap.")
    parser.add_argument("--agent1", choices=["adaptive", "optimal", "random"], default="optimal", help="Agent 1 type.")
    parser.add_argument("--agent2", choices=["adaptive", "optimal", "random"], default="random", help="Agent 2 type.")
    parser.add_argument("--agent1-bats-first", action="store_true", help="Set this flag if Agent 1 bats first.")
    parser.add_argument("--delay-ms", type=int, default=1000, help="Delay per ball in autoplay mode.")
    args = parser.parse_args()

    agent1 = _build_agent(args.agent1, args.T, args.M, args.max_score)
    agent2 = _build_agent(args.agent2, args.T, args.M, args.max_score)

    # Pass uppercase agent names to the simulation
    run_pygame_simulation(
        agent1=agent1,
        agent2=agent2,
        agent1_name=args.agent1.upper(),
        agent2_name=args.agent2.upper(),
        agent1_bats_first=args.agent1_bats_first,
        T=args.T,
        max_score=args.max_score,
        step_delay_ms=max(args.delay_ms, 50),
    )


if __name__ == "__main__":
    sys.exit(main())
