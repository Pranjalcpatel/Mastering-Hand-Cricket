import argparse
import csv
from pathlib import Path
import sys

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_fight import RequiredRunRateAgent, simulate_full_game, OptimalAgent
from pressure_bucket_adaptive.pressure_agents import PressureAdaptiveFiniteAgent
from pressure_bucket_adaptive.visualize_single_match_pressure_policy import run_single_match_visualization


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_summary_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the pressure-aware adaptive agent against the required-run-rate heuristic."
    )
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--tie-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pressure_bucket_adaptive") / "outputs" / "vs_rrr",
    )
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    rng = np.random.default_rng(args.seed)
    max_score = args.T * args.M
    wins = 0
    losses = 0
    ties = 0
    nash_wins = 0
    nash_losses = 0
    nash_ties = 0
    rows = []
    sample_seed = None
    sample_bats_first = None

    optimal_agent = OptimalAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)

    progress = tqdm(
        range(1, args.trials + 1),
        desc="PressureAdaptive vs RRR",
        unit="match",
    )

    for match_idx in progress:
        match_seed = int(rng.integers(0, 2**31 - 1))
        adaptive_bats_first = bool(rng.random() < 0.5)
        np.random.seed(match_seed)

        adaptive = PressureAdaptiveFiniteAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)
        opponent = RequiredRunRateAgent(args.T, args.M, max_score)
        winner = simulate_full_game(
            adaptive,
            opponent,
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

        np.random.seed(match_seed)
        nash_opponent = RequiredRunRateAgent(args.T, args.M, max_score)
        nash_winner = simulate_full_game(
            optimal_agent,
            nash_opponent,
            agent1_bats_first=adaptive_bats_first,
            T=args.T,
            max_score=max_score,
        )

        if nash_winner == 1:
            nash_wins += 1
        elif nash_winner == 2:
            nash_losses += 1
        else:
            nash_ties += 1

        rows.append(
            {
                "match_index": match_idx,
                "seed": match_seed,
                "adaptive_bats_first": adaptive_bats_first,
                "winner_code": winner,
                "nash_winner_code": nash_winner,
            }
        )

        played = wins + losses + ties
        progress.set_postfix(
            ad_score=f"{(wins + args.tie_value * ties) / played:.3f}",
            na_score=f"{(nash_wins + args.tie_value * nash_ties) / played:.3f}",
        )

    summary_csv = args.output_dir / "pressure_adaptive_vs_rrr_match_log.csv"
    save_summary_csv(rows, summary_csv)
    score_rate = (wins + args.tie_value * ties) / args.trials
    nash_score_rate = (nash_wins + args.tie_value * nash_ties) / args.trials

    print(
        f"PressureAdaptive vs RequiredRunRate over {args.trials} matches: "
        f"wins={wins}, losses={losses}, ties={ties}, score_rate={score_rate:.4f}"
    )
    print(
        f"OptimalAgent (Nash) vs RequiredRunRate over {args.trials} matches: "
        f"wins={nash_wins}, losses={nash_losses}, ties={nash_ties}, score_rate={nash_score_rate:.4f}"
    )
    print(f"Improvement over Nash: {score_rate - nash_score_rate:.4f}")
    print(f"Saved match log to {summary_csv}")

    if sample_seed is None:
        print("No pressure-adaptive win found in this batch, so no sample winning plot was generated.")
        return

    np.random.seed(sample_seed)
    sample_agent = PressureAdaptiveFiniteAgent(args.T, args.M, max_score, tie_value=args.tie_value, verbose=False)
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
    print(f"Saved winning sample plot to {sample_dir / 'full_match_distribution_timeline.png'}")


if __name__ == "__main__":
    main()
