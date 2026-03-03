from nash_full_game import solve_full_game
import numpy as np
import matplotlib.pyplot as plt 

class RandomAgent:
    def __init__(self, M):
        self.M = M

    def act(self, role, t, state):
        return np.random.randint(1, self.M + 1)


class OptimalAgent:
    def __init__(self, T, M, max_score):
        self.T = T
        self.M = M
        self.max_score = max_score

        print("Precomputing equilibrium...")
        self.V, self.W, self.V_strat, self.W_strat = solve_full_game(
            T, M, max_score
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
    else:
        winner = 1 if agent1_bats_first else 2

    return winner


def compute_win_rate(T, M, trials=300):
    max_score = T * M
    agent_opt = OptimalAgent(T, M, max_score)
    agent_rand = RandomAgent(M)

    wins = 0
    for _ in range(trials):
        winner = simulate_full_game(
            agent_opt,
            agent_rand,
            agent1_bats_first=np.random.rand() < 0.5,
        )
        if winner == 1:
            wins += 1

    return wins / trials


def main():

    trials = 300

    # -------------------------------------------------
    # 1️⃣ Win rate vs Number of Balls (T)
    # -------------------------------------------------

    M_fixed = 6
    T_values = list(range(2, 11))
    win_rates_T = []

    for T in T_values:
        print(f"Computing win rate for T={T}, M={M_fixed}")
        win_rates_T.append(compute_win_rate(T, M_fixed, trials))

    plt.figure()
    plt.plot(T_values, win_rates_T)
    plt.xlabel("Number of Balls (T)")
    plt.ylabel("Win Rate vs Random")
    plt.title("Win Rate vs Number of Balls")
    plt.show()
    plt.savefigq("win_rate_vs_T.png")

    # -------------------------------------------------
    # 2️⃣ Win rate vs Number of Symbols (M)
    # -------------------------------------------------

    T_fixed = 5
    M_values = list(range(2, 11))
    win_rates_M = []

    for M in M_values:
        print(f"Computing win rate for T={T_fixed}, M={M}")
        win_rates_M.append(compute_win_rate(T_fixed, M, trials))

    plt.figure()
    plt.plot(M_values, win_rates_M)
    plt.xlabel("Number of Symbols (M)")
    plt.ylabel("Win Rate vs Random")
    plt.title("Win Rate vs Number of Symbols")
    plt.show()
    plt.savefig("win_rate_vs_M.png")

    # -------------------------------------------------
    # 3️⃣ Theoretical value vs Number of Balls
    # -------------------------------------------------

    theoretical_values_T = []

    for T in T_values:
        max_score = T * M_fixed
        agent_opt = OptimalAgent(T, M_fixed, max_score)
        theoretical_values_T.append(agent_opt.game_value())

    plt.figure()
    plt.plot(T_values, theoretical_values_T)
    plt.xlabel("Number of Balls (T)")
    plt.ylabel("Theoretical Game Value")
    plt.title("Theoretical Value vs Number of Balls")
    plt.show()
    plt.savefig("theoretical_value_vs_T.png")

    # -------------------------------------------------
    # 4️⃣ Theoretical value vs Number of Symbols
    # -------------------------------------------------

    theoretical_values_M = []

    for M in M_values:
        max_score = T_fixed * M
        agent_opt = OptimalAgent(T_fixed, M, max_score)
        theoretical_values_M.append(agent_opt.game_value())

    plt.figure()
    plt.plot(M_values, theoretical_values_M)
    plt.xlabel("Number of Symbols (M)")
    plt.ylabel("Theoretical Game Value")
    plt.title("Theoretical Value vs Number of Symbols")
    plt.show()
    plt.savefig("theoretical_value_vs_M.png")

    print("Analysis complete.")


if __name__ == "__main__":
    main()