import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from agents_fight import OptimalAgent, RandomAgent

try:
    import pygame
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
    agent1, agent2, agent1_bats_first: bool, T: int, max_score: int
) -> Tuple[List[BallEvent], int, int]:
    events: List[BallEvent] = []

    t = T
    s = 0
    ball1 = 0

    while t > 0:
        ball1 += 1
        if agent1_bats_first:
            striker_name, bowler_name = "Agent 1", "Agent 2"
            bat = int(agent1.act("bat_first", t, s))
            bowl = int(agent2.act("bowl_first", t, s))
        else:
            striker_name, bowler_name = "Agent 2", "Agent 1"
            bat = int(agent2.act("bat_first", t, s))
            bowl = int(agent1.act("bowl_first", t, s))

        is_out = bat == bowl
        if not is_out:
            s = min(s + bat, max_score)
            t -= 1
            msg = f"Runs scored: {bat}"
        else:
            msg = "Wicket! Innings ends."

        events.append(
            BallEvent(
                innings=1,
                ball_index=ball1,
                striker=striker_name,
                bowler=bowler_name,
                bat_symbol=bat,
                bowl_symbol=bowl,
                is_out=is_out,
                first_score=s,
                second_needed=0,
                balls_left=t,
                message=msg,
            )
        )

        if is_out:
            break

    target = min(s + 1, max_score)

    t = T
    k = target
    ball2 = 0

    while t > 0 and k > 0:
        ball2 += 1
        if agent1_bats_first:
            striker_name, bowler_name = "Agent 2", "Agent 1"
            bat = int(agent2.act("bat_second", t, k))
            bowl = int(agent1.act("bowl_second", t, k))
        else:
            striker_name, bowler_name = "Agent 1", "Agent 2"
            bat = int(agent1.act("bat_second", t, k))
            bowl = int(agent2.act("bowl_second", t, k))

        is_out = bat == bowl
        if not is_out:
            k -= bat
            t -= 1
            msg = f"Chase scores {bat}, needs {max(k, 0)} more."
        else:
            msg = "Wicket! Chase ends."

        events.append(
            BallEvent(
                innings=2,
                ball_index=ball2,
                striker=striker_name,
                bowler=bowler_name,
                bat_symbol=bat,
                bowl_symbol=bowl,
                is_out=is_out,
                first_score=s,
                second_needed=max(k, 0),
                balls_left=t,
                message=msg,
            )
        )

        if is_out:
            break

    if k <= 0:
        winner = 2 if agent1_bats_first else 1
    else:
        winner = 1 if agent1_bats_first else 2

    return events, winner, target


def _draw_label(surface, font, text, x, y, color):
    surface.blit(font.render(text, True, color), (x, y))


def _draw_hand(
    surface,
    center_x: int,
    center_y: int,
    symbol: int,
    facing: str,
    reveal: float,
    skin=(230, 189, 140),
    outline=(33, 33, 38),
):
    reveal = max(0.0, min(1.0, reveal))
    facing_sign = -1 if facing == "left" else 1

    palm_w, palm_h = 120, 90
    arm_w, arm_h = 100, 42
    finger_w = 16
    finger_base_h = 46
    finger_tip_r = 8

    # Arm (behind palm)
    arm_rect = pygame.Rect(0, 0, arm_w, arm_h)
    arm_rect.center = (center_x - facing_sign * 95, center_y + 20)
    pygame.draw.rect(surface, skin, arm_rect, border_radius=14)
    pygame.draw.rect(surface, outline, arm_rect, width=2, border_radius=14)

    # Palm
    palm_rect = pygame.Rect(0, 0, palm_w, palm_h)
    palm_rect.center = (center_x, center_y)
    pygame.draw.ellipse(surface, skin, palm_rect)
    pygame.draw.ellipse(surface, outline, palm_rect, width=3)

    # Thumb
    thumb_len = int(34 * (0.35 + 0.65 * reveal))
    thumb_rect = pygame.Rect(0, 0, 16, thumb_len)
    thumb_rect.center = (
        center_x + facing_sign * 54,
        center_y + 10 - thumb_len // 3,
    )
    pygame.draw.rect(surface, skin, thumb_rect, border_radius=8)
    pygame.draw.rect(surface, outline, thumb_rect, width=2, border_radius=8)

    # Finger columns from index->little.
    finger_offsets = [-34, -18, 0, 18, 34]
    up_count = max(0, min(symbol, 5))
    for i, dx in enumerate(finger_offsets):
        is_up = i < up_count
        finger_h = int((finger_base_h if is_up else 20) * (0.3 + 0.7 * reveal))
        finger_rect = pygame.Rect(0, 0, finger_w, finger_h)
        finger_rect.center = (center_x + dx, center_y - 34 - finger_h // 2)
        pygame.draw.rect(surface, skin, finger_rect, border_radius=8)
        pygame.draw.rect(surface, outline, finger_rect, width=2, border_radius=8)
        pygame.draw.circle(surface, skin, finger_rect.midtop, finger_tip_r)
        pygame.draw.circle(surface, outline, finger_rect.midtop, finger_tip_r, width=2)

def _draw_pitch(surface, rect, line_color):
    pygame.draw.rect(surface, (191, 154, 109), rect, border_radius=16)
    pygame.draw.rect(surface, line_color, rect, width=3, border_radius=16)
    mid_x = rect.x + rect.width // 2
    pygame.draw.line(surface, line_color, (mid_x, rect.y + 8), (mid_x, rect.y + rect.height - 8), 3)
    pygame.draw.line(
        surface,
        line_color,
        (rect.x + 22, rect.y + rect.height // 2),
        (rect.x + rect.width - 22, rect.y + rect.height // 2),
        2,
    )


def run_pygame_simulation(
    agent1, agent2, agent1_bats_first=True, T=None, max_score=None, step_delay_ms=900
):
    T = _infer_param(agent1, agent2, "T", T)
    max_score = _infer_param(agent1, agent2, "max_score", max_score)

    pygame.init()
    screen = pygame.display.set_mode((1000, 620))
    pygame.display.set_caption("Hand Cricket Nash Simulation")
    clock = pygame.time.Clock()

    bg = (18, 24, 33)
    panel = (28, 37, 50)
    panel_2 = (23, 31, 42)
    text = (235, 239, 244)
    accent = (72, 187, 120)
    warning = (245, 101, 101)
    dim = (161, 177, 196)
    pitch_line = (54, 66, 84)

    title_font = pygame.font.SysFont("consolas", 34, bold=True)
    h_font = pygame.font.SysFont("consolas", 24, bold=True)
    body_font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 17)
    badge_font = pygame.font.SysFont("consolas", 26, bold=True)

    def setup_new_match():
        evts, winner, target = generate_match_events(
            agent1, agent2, agent1_bats_first, T, max_score
        )
        return evts, winner, target, 0, True, 0

    events, winner, target, idx, autoplay, last_tick = setup_new_match()
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
                    events, winner, target, idx, autoplay, last_tick = setup_new_match()

        if autoplay and idx < len(events) and now - last_tick >= step_delay_ms:
            idx += 1
            last_tick = now

        screen.fill(bg)
        pygame.draw.rect(screen, panel, (28, 22, 944, 576), border_radius=14)
        pygame.draw.rect(screen, panel_2, (38, 118, 924, 392), border_radius=12)
        _draw_pitch(screen, pygame.Rect(270, 196, 460, 232), pitch_line)

        _draw_label(screen, title_font, "Hand Cricket: Two-Innings Simulation", 46, 36, text)
        _draw_label(
            screen,
            small_font,
            "Controls: Space=Play/Pause  Right=Step  R=Restart  Esc=Quit",
            48,
            78,
            dim,
        )

        shown = events[idx - 1] if idx > 0 else None
        innings = shown.innings if shown is not None else 1
        first_score = shown.first_score if shown is not None else 0
        second_needed = shown.second_needed if shown is not None and innings == 2 else target
        balls_left = shown.balls_left if shown is not None else T

        _draw_label(screen, h_font, f"Innings: {innings}", 52, 130, text)
        _draw_label(screen, body_font, f"First-innings score: {first_score}", 52, 168, text)
        _draw_label(screen, body_font, f"Target: {target}", 52, 202, text)
        _draw_label(screen, body_font, f"Runs needed (2nd): {max(second_needed, 0)}", 52, 236, text)
        _draw_label(screen, body_font, f"Balls left: {balls_left}", 52, 270, text)

        reveal = 1.0 if idx == 0 else min(1.0, (now - last_tick) / 220.0)
        left_symbol = shown.bat_symbol if shown is not None else 0
        right_symbol = shown.bowl_symbol if shown is not None else 0
        left_role = "Batter"
        right_role = "Bowler"
        left_name = shown.striker if shown is not None else "Left Hand"
        right_name = shown.bowler if shown is not None else "Right Hand"

        _draw_hand(screen, 380, 312, left_symbol, facing="right", reveal=reveal)
        _draw_hand(screen, 620, 312, right_symbol, facing="left", reveal=reveal)
        pygame.draw.circle(screen, (42, 58, 78), (468, 246), 30)
        pygame.draw.circle(screen, accent, (468, 246), 30, width=3)
        pygame.draw.circle(screen, (42, 58, 78), (532, 246), 30)
        pygame.draw.circle(screen, warning, (532, 246), 30, width=3)
        _draw_label(screen, badge_font, str(left_symbol), 458, 232, text)
        _draw_label(screen, badge_font, str(right_symbol), 522, 232, text)
        _draw_label(screen, body_font, f"{left_name} ({left_role})", 290, 430, text)
        _draw_label(screen, body_font, f"{right_name} ({right_role})", 555, 430, text)

        if shown is None:
            _draw_label(screen, h_font, "Press Space to start playback.", 52, 328, accent)
            _draw_label(screen, h_font, "Hands are ready...", 398, 162, dim)
        else:
            _draw_label(
                screen,
                h_font,
                f"Ball {shown.ball_index} | {shown.striker} batting vs {shown.bowler}",
                52,
                328,
                text,
            )
            _draw_label(
                screen,
                body_font,
                shown.message,
                52,
                374,
                warning if shown.is_out else accent,
            )
            duel_text = (
                "OUT!" if shown.is_out else f"{shown.bat_symbol} vs {shown.bowl_symbol}"
            )
            duel_color = warning if shown.is_out else accent
            _draw_label(screen, h_font, duel_text, 462, 162, duel_color)

        if idx >= len(events):
            _draw_label(screen, h_font, f"Match Over | Winner: Agent {winner}", 52, 486, accent)

        log_x, log_y = 540, 130
        _draw_label(screen, h_font, "Recent Balls", log_x, log_y, text)
        recent = events[max(0, idx - 8):idx]
        for i, ev in enumerate(recent):
            status = "OUT" if ev.is_out else ev.message
            line = (
                f"I{ev.innings} B{ev.ball_index}: "
                f"{ev.striker}({ev.bat_symbol}) vs {ev.bowler}({ev.bowl_symbol}) -> {status}"
            )
            _draw_label(screen, small_font, line, log_x, log_y + 34 + i * 26, dim)

        progress = 0.0 if len(events) == 0 else idx / len(events)
        bar_w = 900
        pygame.draw.rect(screen, (51, 65, 85), (52, 548, bar_w, 18), border_radius=8)
        pygame.draw.rect(
            screen,
            accent,
            (52, 548, int(bar_w * progress), 18),
            border_radius=8,
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _build_agent(kind: str, T: int, M: int, max_score: int):
    if kind == "optimal":
        return OptimalAgent(T, M, max_score)
    if kind == "random":
        return RandomAgent(M)
    raise ValueError(f"Unknown agent type: {kind}")


def main():
    parser = argparse.ArgumentParser(description="Pygame hand-cricket simulation.")
    parser.add_argument("--T", type=int, default=20, help="Balls per innings.")
    parser.add_argument("--M", type=int, default=6, help="Number symbols (1..M).")
    parser.add_argument("--max-score", type=int, default=120, help="Score cap.")
    parser.add_argument(
        "--agent1",
        choices=["optimal", "random"],
        default="optimal",
        help="Agent 1 type.",
    )
    parser.add_argument(
        "--agent2",
        choices=["optimal", "random"],
        default="random",
        help="Agent 2 type.",
    )
    parser.add_argument(
        "--agent1-bats-first",
        action="store_true",
        help="Set this flag if Agent 1 bats first.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=900,
        help="Delay per ball in autoplay mode.",
    )
    args = parser.parse_args()

    agent1 = _build_agent(args.agent1, args.T, args.M, args.max_score)
    agent2 = _build_agent(args.agent2, args.T, args.M, args.max_score)

    run_pygame_simulation(
        agent1=agent1,
        agent2=agent2,
        agent1_bats_first=args.agent1_bats_first,
        T=args.T,
        max_score=args.max_score,
        step_delay_ms=max(args.delay_ms, 50),
    )


if __name__ == "__main__":
    sys.exit(main())
