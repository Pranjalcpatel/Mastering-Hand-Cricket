import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_fight import BestResponseFiniteAgent, OptimalAgent, RandomAgent, RequiredRunRateAgent, simulate_full_game
from nash_full_game import PHASE_NAMES
from pressure_bucket_adaptive.pressure_agents import PressureAdaptiveFiniteAgent


ROLE_ORDER = ("bat_first", "bowl_first", "bat_second", "bowl_second")


def uniform_phase_policy(M):
    uniform = np.full(M, 1.0 / M)
    return {
        role: {phase: uniform.copy() for phase in PHASE_NAMES}
        for role in ROLE_ORDER
    }


def build_opponent(kind, T, M, max_score, tie_value):
    if kind == "random":
        return RandomAgent(M)
    if kind == "required_run_rate":
        return RequiredRunRateAgent(T, M, max_score)
    if kind == "optimal":
        return OptimalAgent(T, M, max_score, tie_value=tie_value, verbose=False)
    if kind == "best_response_uniform":
        return BestResponseFiniteAgent(
            T,
            M,
            max_score,
            opponent_phase_policies=uniform_phase_policy(M),
            tie_value=tie_value,
            verbose=False,
        )
    raise ValueError(f"Unknown opponent kind: {kind}")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def flatten_trace_rows(trace, M):
    rows = []
    for entry in trace:
        row = {
            "decision_index": entry["decision_index"],
            "match_ball_index": entry["match_ball_index"],
            "innings": entry["innings"],
            "innings_ball_index": entry["innings_ball_index"],
            "role": entry["role"],
            "t": entry["t"],
            "state": entry["state"],
            "bucket": entry["bucket"],
            "blend_weight": entry["blend_weight"],
            "chosen_action": entry["chosen_action"],
            "opponent_action": entry["opponent_action"],
            "pre_observation_count": entry["pre_observation_count"],
            "post_observation_count": entry["post_observation_count"],
        }
        for symbol in range(1, M + 1):
            idx = symbol - 1
            row[f"nash_p_{symbol}"] = float(entry["nash_probs"][idx])
            row[f"br_p_{symbol}"] = float(entry["best_response_probs"][idx])
            row[f"mixed_p_{symbol}"] = float(entry["mixed_probs"][idx])
        row["l1_from_nash"] = float(np.sum(np.abs(entry["mixed_probs"] - entry["nash_probs"])))
        rows.append(row)
    return rows


def save_trace_csv(rows, csv_path):
    if not rows:
        return
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _role_color(role):
    return {
        "bat_first": "#d97706",
        "bowl_first": "#2563eb",
        "bat_second": "#b45309",
        "bowl_second": "#1d4ed8",
    }[role]


def add_role_background(ax, trace_rows):
    if not trace_rows:
        return
    start_x = trace_rows[0]["match_ball_index"] - 0.5
    current_role = trace_rows[0]["role"]
    for prev_row, next_row in zip(trace_rows, trace_rows[1:]):
        if next_row["role"] != current_role:
            end_x = prev_row["match_ball_index"] + 0.5
            ax.axvspan(start_x, end_x, color=_role_color(current_role), alpha=0.08)
            start_x = next_row["match_ball_index"] - 0.5
            current_role = next_row["role"]
    end_x = trace_rows[-1]["match_ball_index"] + 0.5
    ax.axvspan(start_x, end_x, color=_role_color(current_role), alpha=0.08)


def plot_full_match_timeline(trace_rows, M, output_dir):
    if not trace_rows:
        return

    xs = [row["match_ball_index"] for row in trace_rows]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    for ax in axes:
        add_role_background(ax, trace_rows)

    for symbol in range(1, M + 1):
        mixed_key = f"mixed_p_{symbol}"
        nash_key = f"nash_p_{symbol}"
        ys_mixed = [row[mixed_key] for row in trace_rows]
        ys_nash = [row[nash_key] for row in trace_rows]
        (line,) = axes[0].plot(xs, ys_mixed, marker="o", label=f"{symbol} adaptive")
        axes[0].plot(xs, ys_nash, linestyle="--", color=line.get_color(), alpha=0.75, label=f"{symbol} Nash")

    axes[0].set_title("Pressure-aware adaptive distribution vs Nash over full match")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)

    l1_vals = [row["l1_from_nash"] for row in trace_rows]
    blend_vals = [row["blend_weight"] for row in trace_rows]
    axes[1].plot(xs, l1_vals, marker="o", label="L1 distance from Nash")
    axes[1].plot(xs, blend_vals, marker="s", label="Blend weight")
    axes[1].set_ylabel("Distance / weight")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    chosen = [row["chosen_action"] for row in trace_rows]
    opponent = [row["opponent_action"] if row["opponent_action"] is not None else np.nan for row in trace_rows]
    axes[2].plot(xs, chosen, marker="o", label="Chosen action")
    axes[2].plot(xs, opponent, marker="x", label="Opponent action")
    axes[2].set_ylabel("Action")
    axes[2].set_yticks(range(1, M + 1))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    role_to_y = {role: idx for idx, role in enumerate(ROLE_ORDER)}
    ys = [role_to_y[row["role"]] for row in trace_rows]
    state_text = [
        f"I{row['innings']} B{row['innings_ball_index']} t={row['t']} s={row['state']} {row['bucket']}"
        for row in trace_rows
    ]
    axes[3].scatter(xs, ys, c=[_role_color(row["role"]) for row in trace_rows], s=70)
    for x, y, label in zip(xs, ys, state_text):
        axes[3].annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[3].set_yticks(range(len(ROLE_ORDER)))
    axes[3].set_yticklabels(ROLE_ORDER)
    axes[3].set_xlabel("Match ball index from adaptive agent perspective")
    axes[3].set_ylabel("Role")
    axes[3].grid(True, axis="x", alpha=0.3)

    fig.savefig(output_dir / "full_match_distribution_timeline.png", dpi=160)
    plt.close(fig)


def run_single_match_visualization(agent, opponent, T, M, max_score, output_dir, agent1_bats_first=True):
    ensure_dir(output_dir)
    winner = simulate_full_game(
        agent,
        opponent,
        agent1_bats_first=agent1_bats_first,
        T=T,
        max_score=max_score,
    )
    trace_rows = flatten_trace_rows(agent.get_policy_trace(), M)
    trace_csv = output_dir / "adaptive_policy_trace.csv"
    save_trace_csv(trace_rows, trace_csv)
    plot_full_match_timeline(trace_rows, M, output_dir)
    return winner, trace_csv


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize one match for the pressure-aware adaptive finite-horizon agent."
    )
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--max-score", type=int, default=None)
    parser.add_argument("--tie-value", type=float, default=0.5)
    parser.add_argument(
        "--opponent",
        choices=["random", "required_run_rate", "optimal", "best_response_uniform"],
        default="required_run_rate",
    )
    parser.add_argument("--agent1-bats-first", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pressure_bucket_adaptive") / "outputs" / "single_match",
    )
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    max_score = args.max_score if args.max_score is not None else args.T * args.M
    agent = PressureAdaptiveFiniteAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)
    opponent = build_opponent(args.opponent, args.T, args.M, max_score, args.tie_value)
    winner, trace_csv = run_single_match_visualization(
        agent,
        opponent,
        args.T,
        args.M,
        max_score,
        args.output_dir,
        agent1_bats_first=args.agent1_bats_first,
    )
    print(f"Winner code: {winner}")
    print(f"Saved trace to {trace_csv}")
    print(f"Saved plot to {args.output_dir / 'full_match_distribution_timeline.png'}")


if __name__ == "__main__":
    main()
