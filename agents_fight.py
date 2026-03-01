from nash_full_game import solve_full_game
import numpy as np


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


# ============================================================
# 5. Monte Carlo Test
# ============================================================

if __name__ == "__main__":

    T = 6
    M = 6
    max_score = 36
    trials = 5000

    agent1 = OptimalAgent(T, M, max_score)
    agent2 = RandomAgent(M)

    wins_agent1 = 0

    for _ in range(trials):
        winner = simulate_full_game(
            agent1, agent2, agent1_bats_first=True
        )
        if winner == 1:
            wins_agent1 += 1

    print("Empirical win rate agent1:", wins_agent1 / trials)
    print("Theoretical value:", agent1.game_value())
