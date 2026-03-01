import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_game(A):
    """
    Solve zero-sum matrix game with payoff matrix A (row player maximizes).
    Returns:
        p_opt  : optimal mixed strategy for row player
        q_opt  : optimal mixed strategy for column player
        value  : game value
    """
    m, n = A.shape

    # ---- Row player LP ----
    # maximize v
    # s.t. A^T p >= v
    #      sum p = 1
    #      p >= 0

    c = np.zeros(m + 1)
    c[-1] = -1  # maximize v -> minimize -v

    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])

    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Row-player LP failed: {res.message}")

    p_opt = res.x[:m]
    value = res.x[-1]
    p_opt = np.clip(p_opt, 0.0, None)
    p_opt /= p_opt.sum()

    # ---- Column player strategy (dual) ----
    c_dual = np.zeros(n + 1)
    c_dual[-1] = 1

    A_ub_dual = np.hstack([A, -np.ones((m, 1))])
    b_ub_dual = np.zeros(m)

    A_eq_dual = np.zeros((1, n + 1))
    A_eq_dual[0, :n] = 1
    b_eq_dual = np.array([1])

    bounds_dual = [(0, None)] * n + [(None, None)]

    res_dual = linprog(
        c_dual,
        A_ub=A_ub_dual,
        b_ub=b_ub_dual,
        A_eq=A_eq_dual,
        b_eq=b_eq_dual,
        bounds=bounds_dual,
        method="highs",
    )
    if not res_dual.success:
        raise RuntimeError(f"Column-player LP failed: {res_dual.message}")

    q_opt = res_dual.x[:n]
    q_opt = np.clip(q_opt, 0.0, None)
    q_opt /= q_opt.sum()

    return p_opt, q_opt, value


# def solve_finite_hand_cricket(T, M, max_score):
#     """
#     Solve finite-horizon hand cricket.

#     Parameters:
#         T         : number of balls
#         M         : number of symbols (1..M)
#         max_score : max possible score difference

#     Returns:
#         V         : value table V[t][k]
#         strategies: dict with (t,k) -> (p_opt, q_opt)
#     """

#     # Value function V[t][k]
#     V = np.zeros((T + 1, max_score + 1))

#     # Terminal condition
#     for k in range(max_score + 1):
#         if k <= 0:
#             V[0][k] = 1
#         else:
#             V[0][k] = 0

#     strategies = {}

#     for t in range(1, T + 1):
#         for k in range(max_score + 1):

#             # If already won
#             if k <= 0:
#                 V[t][k] = 1
#                 continue

#             A = np.zeros((M, M))

#             for i in range(M):        # batter choice
#                 for j in range(M):    # bowler choice
#                     if i == j:
#                         A[i, j] = 0  # out
#                     else:
#                         new_k = k - (i + 1)
#                         if new_k <= 0:
#                             A[i, j] = 1
#                         else:
#                             A[i, j] = V[t - 1][min(new_k, max_score)]

#             p_opt, q_opt, val = solve_zero_sum_game(A)

#             V[t][k] = val
#             strategies[(t, k)] = (p_opt, q_opt)

#     return V, strategies


# # =============================
# # Example usage
# # =============================

# # T = 5          # balls remaining
# # M = 6          # symbols
# # max_score = 30 # max score difference tracked

# # V, strategies = solve_finite_hand_cricket(T, M, max_score)

# # t = 1
# # k = 5

# # p_opt, q_opt = strategies[(t, k)]

# # print(f"Game value at (t={t}, k={k}): {V[t][k]:.4f}")
# # print("Optimal batter strategy:", p_opt)
# # print("Optimal bowler strategy:", q_opt)
