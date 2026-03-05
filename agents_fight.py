from nash_full_game import solve_full_game
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os


class RandomAgent:
    def __init__(self, M):
        self.M = M

    def act(self, role, t, state):
        return np.random.randint(1, self.M + 1)


class OptimalAgent:
    def __init__(self, T, M, max_score, tie_value=0.5):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value

        print("Precomputing equilibrium...")
        self.V, self.W, self.V_strat, self.W_strat = solve_full_game(
            T, M, max_score, tie_value=tie_value
        )
        print("Done.")

    def act(self, role, t, state):
        state = int(max(0, min(state, self.max_score)))

        if role == "bat_first":
            p_opt, _ = self.W_strat[(t, state)]
            return np.random.choice(np.arange(1, self.M + 1), p=p_opt)

        elif role == "bowl_first":
            _, q_opt = self.W_strat[(t, state)]
            return np.random.choice(np.arange(1, self.M + 1), p=q_opt)

        elif role == "bat_second":
            p_opt, _ = self.V_strat[(t, state)]
            return np.random.choice(np.arange(1, self.M + 1), p=p_opt)

        elif role == "bowl_second":
            _, q_opt = self.V_strat[(t, state)]
            return np.random.choice(np.arange(1, self.M + 1), p=q_opt)

        else:
            raise ValueError("Invalid role")

    def game_value(self):
        return self.W[self.T][0]
    
class InfiniteOptimalAgent:
    def __init__(self, M, max_score, tie_value=0.5):
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value

        print("Solving infinite-balls equilibrium...")
        self.V, self.W, self.V_strat, self.W_strat = solve_infinite_game(
            M, max_score, tie_value=tie_value
        )
        print("Done.")

    def act(self, role, state):

        state = int(max(0, min(state, self.max_score)))

        if role == "bat_first":
            p_opt, _ = self.W_strat[state]
            return np.random.choice(np.arange(1, self.M + 1), p=p_opt)

        elif role == "bowl_first":
            _, q_opt = self.W_strat[state]
            return np.random.choice(np.arange(1, self.M + 1), p=q_opt)

        elif role == "bat_second":
            p_opt, _ = self.V_strat[state]
            return np.random.choice(np.arange(1, self.M + 1), p=p_opt)

        elif role == "bowl_second":
            _, q_opt = self.V_strat[state]
            return np.random.choice(np.arange(1, self.M + 1), p=q_opt)

        else:
            raise ValueError("Invalid role")

    def game_value(self):
        return self.W[0]


def _resolve_game_param(agent1, agent2, attr_name, explicit_value):
    if explicit_value is not None:
        return explicit_value

    has_1 = hasattr(agent1, attr_name)
    has_2 = hasattr(agent2, attr_name)

    if has_1 and has_2:
        v1 = getattr(agent1, attr_name)
        v2 = getattr(agent2, attr_name)
        if v1 != v2:
            raise ValueError(
                f"Agent mismatch on {attr_name}: agent1={v1}, agent2={v2}"
            )
        return v1
    if has_1:
        return getattr(agent1, attr_name)
    if has_2:
        return getattr(agent2, attr_name)

    raise ValueError(
        f"Cannot infer {attr_name}. Pass it explicitly to simulate_full_game."
    )


# ============================================================
# 4. Full Game Simulator
# ============================================================

def simulate_full_game(agent1, agent2, agent1_bats_first=True, T=None, max_score=None):
    T = _resolve_game_param(agent1, agent2, "T", T)
    max_score = _resolve_game_param(agent1, agent2, "max_score", max_score)

    # FIRST INNINGS
    t = T
    s = 0

    while t > 0:

        if agent1_bats_first:
            bat = agent1.act("bat_first", t, s)
            bowl = agent2.act("bowl_first", t, s)
        else:
            bat = agent2.act("bat_first", t, s)
            bowl = agent1.act("bowl_first", t, s)

        if bat == bowl:
            break

        s += bat
        s = min(s, max_score)
        t -= 1

    target = min(s + 1, max_score)

    # SECOND INNINGS
    t = T
    k = target

    while t > 0 and k > 0:

        if agent1_bats_first:
            bat = agent2.act("bat_second", t, k)
            bowl = agent1.act("bowl_second", t, k)
        else:
            bat = agent1.act("bat_second", t, k)
            bowl = agent2.act("bowl_second", t, k)

        if bat == bowl:
            break

        k -= bat
        t -= 1

    if k <= 0:
        winner = 2 if agent1_bats_first else 1
    elif k == 1:
        # Chasing side finished level with first-innings score.
        winner = 0
    else:
        winner = 1 if agent1_bats_first else 2

    return winner

def simulate_infinite_game(agent1, agent2, agent1_bats_first=True, max_score=None):

    max_score = _resolve_game_param(agent1, agent2, "max_score", max_score)

    # FIRST INNINGS
    s = 0

    while True:

        if agent1_bats_first:
            bat = agent1.act("bat_first", s)
            bowl = agent2.act("bowl_first", s)
        else:
            bat = agent2.act("bat_first", s)
            bowl = agent1.act("bowl_first", s)

        if bat == bowl:
            break

        s += bat
        s = min(s, max_score)

    target = min(s + 1, max_score)

    # SECOND INNINGS
    k = target

    while k > 0:

        if agent1_bats_first:
            bat = agent2.act("bat_second", k)
            bowl = agent1.act("bowl_second", k)
        else:
            bat = agent1.act("bat_second", k)
            bowl = agent2.act("bowl_second", k)

        if bat == bowl:
            break

        k -= bat

    if k <= 0:
        winner = 2 if agent1_bats_first else 1
    elif k == 1:
        winner = 0
    else:
        winner = 1 if agent1_bats_first else 2

    return winner

def compute_win_rate(T, M, trials=300, tie_value=0.5):
    max_score = T * M
    agent_opt = OptimalAgent(T, M, max_score, tie_value=tie_value)
    agent_rand = RandomAgent(M)

    score = 0.0
    for _ in range(trials):
        winner = simulate_full_game(
            agent_opt,
            agent_rand,
            agent1_bats_first=np.random.rand() < 0.5,
        )
        if winner == 1:
            score += 1.0
        elif winner == 0:
            score += tie_value

    return score / trials



def main():

    trials = 500

    def ensure_csv(path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def append_csv(path, row):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def load_csv_rows(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    # -------------------------------------------------
    # 1️⃣ Win rate vs Number of Balls (T)
    # -------------------------------------------------

    M_fixed = 6
    T_values = list(range(2, 50))
    csv_t_path = "win_rate_vs_T_results.csv"
    ensure_csv(csv_t_path, ["T", "M_fixed", "win_rate"])

    win_rate_by_T = {}
    for row in load_csv_rows(csv_t_path):
        try:
            row_t = int(row["T"])
            row_m = int(row["M_fixed"])
            row_wr = float(row["win_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        if row_m == M_fixed:
            win_rate_by_T[row_t] = row_wr

    for T in T_values:
        if T in win_rate_by_T:
            print(f"Using cached win rate for T={T}, M={M_fixed}")
            continue
        print(f"Computing win rate for T={T}, M={M_fixed}")
        win_rate_by_T[T] = compute_win_rate(T, M_fixed, trials)
        append_csv(csv_t_path, [T, M_fixed, win_rate_by_T[T]])

    plot_T = [T for T in T_values if T in win_rate_by_T]
    plot_wr_T = [win_rate_by_T[T] for T in plot_T]
    plt.figure()
    plt.plot(plot_T, plot_wr_T)
    plt.xlabel("Number of Balls (T)")
    plt.ylabel("Win Rate vs Random")
    plt.title("Win Rate vs Number of Balls")
    plt.show()
    plt.savefig("win_rate_vs_T.png")

    # -------------------------------------------------
    # 2️⃣ Win rate vs Number of Symbols (M)
    # -------------------------------------------------

    T_fixed_values = [5, 7, 9]
    M_values = list(range(2, 11))
    csv_m_path = "win_rate_vs_M_results.csv"
    ensure_csv(csv_m_path, ["T_fixed", "M", "win_rate"])

    win_rates_M_by_T = {T_fixed: {} for T_fixed in T_fixed_values}
    valid_t_values = set(T_fixed_values)

    for row in load_csv_rows(csv_m_path):
        try:
            row_t = int(row["T_fixed"])
            row_m = int(row["M"])
            row_wr = float(row["win_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        if row_t in valid_t_values:
            win_rates_M_by_T[row_t][row_m] = row_wr

    for T_fixed in T_fixed_values:
        for M in M_values:
            if M in win_rates_M_by_T[T_fixed]:
                print(f"Using cached win rate for T={T_fixed}, M={M}")
                continue
            print(f"Computing win rate for T={T_fixed}, M={M}")
            win_rates_M_by_T[T_fixed][M] = compute_win_rate(T_fixed, M, trials)
            append_csv(csv_m_path, [T_fixed, M, win_rates_M_by_T[T_fixed][M]])

    plt.figure()
    for T_fixed in T_fixed_values:
        plot_M = [M for M in M_values if M in win_rates_M_by_T[T_fixed]]
        plot_wr_M = [win_rates_M_by_T[T_fixed][M] for M in plot_M]
        plt.plot(plot_M, plot_wr_M, label=f"T={T_fixed}")
    plt.xlabel("Number of Symbols (M)")
    plt.ylabel("Win Rate vs Random")
    plt.title("Win Rate vs Number of Symbols (T fixed at 5, 7, 9)")
    plt.legend()
    plt.show()
    plt.savefig("win_rate_vs_M.png")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
