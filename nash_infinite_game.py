import numpy as np
from nash_chase import solve_zero_sum_game


def solve_infinite_game(M, max_score, tie_value=0.5, tol=1e-8, max_iter=1):
    """
    Solve the infinite-balls two-innings hand-cricket game.

    State conventions (stationary):
    - Second innings V[k]: payoff for chasing batter needing k runs.
      Win = 1, tie = tie_value, loss = 0.
    - First innings W[s]: payoff for first-innings batter with score s.

    Strategies depend only on score state (no ball count).
    """

    # -------------------------------------------------
    # Step 1: Solve infinite-horizon second innings
    # -------------------------------------------------

    V = np.zeros(max_score + 1)
    V[0] = 1.0
    if max_score >= 1:
        V[1] = tie_value

    V_strat = {}

    for iteration in range(max_iter):
        print(f"Iteration {iteration}, V: {V}")
        V_new = V.copy()

        for k in range(2, max_score + 1):

            A = np.zeros((M, M))

            for i in range(M):
                for j in range(M):

                    if i == j:
                        # Wicket
                        A[i, j] = tie_value if k == 1 else 0.0

                    else:
                        new_k = k - (i + 1)

                        if new_k <= 0:
                            A[i, j] = 1.0
                        else:
                            new_k = min(new_k, max_score)
                            A[i, j] = V[new_k]

            p_opt, q_opt, val = solve_zero_sum_game(A)

            V_new[k] = val
            V_strat[k] = (p_opt, q_opt)

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            break

    # -------------------------------------------------
    # Step 2: Solve infinite-horizon first innings
    # -------------------------------------------------

    W = np.zeros(max_score + 1)
    W_strat = {}

    for iteration in range(max_iter):
        W_new = W.copy()

        for s in range(max_score + 1):

            A = np.zeros((M, M))

            for i in range(M):
                for j in range(M):

                    if i == j:
                        chase_k = min(s + 1, max_score)
                        A[i, j] = 1 - V[chase_k]

                    else:
                        new_s = s + (i + 1)
                        new_s = min(new_s, max_score)
                        A[i, j] = W[new_s]

            p_opt, q_opt, val = solve_zero_sum_game(A)

            W_new[s] = val
            W_strat[s] = (p_opt, q_opt)

        diff = np.max(np.abs(W_new - W))
        W = W_new

        if diff < tol:
            break

    return V, W, V_strat, W_strat