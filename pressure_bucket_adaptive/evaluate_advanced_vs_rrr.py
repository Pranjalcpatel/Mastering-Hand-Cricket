import argparse
from pathlib import Path
import sys
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_fight import RequiredRunRateAgent, simulate_full_game
from pressure_bucket_adaptive.advanced_pressure_agents import AdvancedPressureAdaptiveAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    T = 12
    M = 6
    max_score = T * M
    tie_value = 0.5
    
    wins = 0
    losses = 0
    ties = 0

    progress = tqdm(range(args.trials), desc="Advanced vs RRR")
    
    for _ in progress:
        adaptive = AdvancedPressureAdaptiveAgent(T, M, max_score, tie_value=tie_value, verbose=False)
        opponent = RequiredRunRateAgent(T, M, max_score)
        
        winner = simulate_full_game(
            adaptive,
            opponent,
            agent1_bats_first=True,
            T=T,
            max_score=max_score,
        )
        
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            ties += 1

    played = wins + losses + ties
    print(f"\nAdvancedAgent vs RRR over {played} matches: wins={wins}, losses={losses}, ties={ties}")

if __name__ == "__main__":
    main()
