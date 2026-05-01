import argparse
import csv
from pathlib import Path

import numpy as np

from agents_fight import AdaptiveFiniteAgent, RequiredRunRateAgent, simulate_full_game
from visualize_single_match_policy import run_single_match_visualization


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the adaptive finite-horizon agent against the required-run-rate heuristic."
    )
    parser.add_argument("--T", type=int, default=12, help="Balls per innings.")
    parser.add_argument("--M", type=int, default=6, help="Number of symbols.")
    parser.add_argument("--trials", type=int, default=200, help="Number of evaluation matches.")
    parser.add_argument("--tie-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs") / "adaptive_vs_rrr",
        help="Directory to store the summary and winning sample plot.",
    )
    return parser


def save_summary_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    rng = np.random.default_rng(args.seed)
    max_score = args.T * args.M

    wins = 0
    losses = 0
    ties = 0
    rows = []
    sample_seed = None
    sample_bats_first = None

    for match_idx in range(1, args.trials + 1):
        match_seed = int(rng.integers(0, 2**31 - 1))
        adaptive_bats_first = bool(rng.random() < 0.5)
        np.random.seed(match_seed)

        adaptive = AdaptiveFiniteAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)
        rrr = RequiredRunRateAgent(args.T, args.M, max_score)
        winner = simulate_full_game(
            adaptive,
            rrr,
            agent1_bats_first=adaptive_bats_first,
            T=args.T,
            max_score=max_score,
        )

        if winner == 1:
            wins += 1
            if sample_seed is None:
                sample_seed = match_seed
                sample_bats_first = adaptive_bats_first
        elif winner == 2:
            losses += 1
        else:
            ties += 1

        rows.append(
            {
                "match_index": match_idx,
                "seed": match_seed,
                "adaptive_bats_first": adaptive_bats_first,
                "winner_code": winner,
            }
        )

    summary_csv = args.output_dir / "adaptive_vs_rrr_match_log.csv"
    save_summary_csv(rows, summary_csv)

    score_rate = (wins + args.tie_value * ties) / args.trials
    print(
        f"Adaptive vs RequiredRunRate over {args.trials} matches: "
        f"wins={wins}, losses={losses}, ties={ties}, score_rate={score_rate:.4f}"
    )
    print(f"Saved match log to {summary_csv}")

    if sample_seed is None:
        print("No adaptive win found in this batch, so no sample winning plot was generated.")
        return

    np.random.seed(sample_seed)
    sample_agent = AdaptiveFiniteAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)
    sample_opponent = RequiredRunRateAgent(args.T, args.M, max_score)
    sample_dir = args.output_dir / "winning_sample_match"
    winner, trace_csv = run_single_match_visualization(
        sample_agent,
        sample_opponent,
        args.T,
        args.M,
        max_score,
        sample_dir,
        agent1_bats_first=sample_bats_first,
    )

    print(
        f"Saved winning sample match with seed={sample_seed}, "
        f"adaptive_bats_first={sample_bats_first}, winner_code={winner}"
    )
    print(f"Saved winning sample trace to {trace_csv}")
    print(f"Saved winning sample plots to {sample_dir}")


if __name__ == "__main__":
    main()
