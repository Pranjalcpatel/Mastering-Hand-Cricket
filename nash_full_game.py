import numpy as np
from nash_chase import solve_zero_sum_game

def solve_full_game(T, M, max_score, tie_value=0.5):
    """Solve the full two-innings finite-horizon hand-cricket game.

    State conventions:
    - Second innings V[t][k]: payoff for the chasing batter with t balls left and
      k runs needed, with k capped in [0, max_score].
      Win = 1, tie = tie_value, loss = 0.
    - First innings W[t][s]: payoff for the first-innings batter with t balls left
      and current score s, with s capped in [0, max_score].
    """

    # -------------------------
    # Step 1: Solve second innings
    # -------------------------
    V = np.zeros((T + 1, max_score + 1))

    # Terminal condition with explicit tie handling:
    # k == 0 -> chase win, k == 1 -> tie, k >= 2 -> chase loss
    V[0, 0] = 1.0
    if max_score >= 1:
        V[0, 1] = tie_value

    V_strat = {}

    for t in range(1, T + 1):
        for k in range(max_score + 1):

            if k <= 0:
                V[t, k] = 1.0
                continue

            A = np.zeros((M, M))

            for i in range(M):
                for j in range(M):
                    if i == j:
                        # Wicket ends innings immediately.
                        # If exactly one run was needed, final scores are tied.
                        A[i, j] = tie_value if k == 1 else 0.0
                    else:
                        new_k = k - (i + 1)
                        if new_k <= 0:
                            A[i, j] = 1
                        else:
                            A[i, j] = V[t - 1, min(new_k, max_score)]

            p_opt, q_opt, val = solve_zero_sum_game(A)
            V[t, k] = val
            V_strat[(t, k)] = (p_opt, q_opt)

    # -------------------------
    # Step 2: Solve first innings
    # -------------------------
    W = np.zeros((T + 1, max_score + 1))
    W_strat = {}

    # Terminal value: no balls left, immediate transition to chase.
    for s in range(max_score + 1):
        chase_k = min(s + 1, max_score)
        W[0, s] = 1 - V[T, chase_k]

    for t in range(1, T + 1):
        for s in range(max_score + 1):

            A = np.zeros((M, M))

            for i in range(M):
                for j in range(M):

                    if i == j:
                        chase_k = min(s + 1, max_score)
                        A[i, j] = 1 - V[T, chase_k]
                    else:
                        new_s = s + (i + 1)
                        new_s = min(new_s, max_score)
                        A[i, j] = W[t - 1, new_s]

            p_opt, q_opt, val = solve_zero_sum_game(A)
            W[t, s] = val
            W_strat[(t, s)] = (p_opt, q_opt)

    return V, W, V_strat, W_strat
