import argparse
import csv
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute win/loss/tie rates from a pressure-adaptive match log CSV."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the match log CSV.",
    )
    parser.add_argument(
        "--tie-value",
        type=float,
        default=0.5,
        help="Tie payoff used for score-rate calculation.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    with args.csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("CSV has no match rows.")

    wins = sum(1 for row in rows if int(row["winner_code"]) == 1)
    losses = sum(1 for row in rows if int(row["winner_code"]) == 2)
    ties = sum(1 for row in rows if int(row["winner_code"]) == 0)
    total = len(rows)
    score_rate = (wins + args.tie_value * ties) / total

    print(f"Matches: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
    print(f"Win rate: {wins / total:.4f}")
    print(f"Loss rate: {losses / total:.4f}")
    print(f"Tie rate: {ties / total:.4f}")
    print(f"Score rate: {score_rate:.4f}")


if __name__ == "__main__":
    main()
