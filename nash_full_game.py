import numpy as np

from nash_chase import solve_zero_sum_game


PHASE_NAMES = ("early", "mid", "late")
ROLE_NAMES = ("bat_first", "bowl_first", "bat_second", "bowl_second")


def phase_for_ball_count(t, T):
    if T <= 0:
        return "late"
    ratio = t / T
    if ratio > (2.0 / 3.0):
        return "early"
    if ratio > (1.0 / 3.0):
        return "mid"
    return "late"


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


def _normalize_probs(probs, M):
    arr = np.asarray(probs, dtype=float)
    if arr.shape != (M,):
        raise ValueError(f"Expected shape ({M},), got {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        return np.full(M, 1.0 / M)
    return arr / total


def _argmax_distribution(values, tol=1e-12):
    values = np.asarray(values, dtype=float)
    best = np.max(values)
    winners = np.flatnonzero(values >= best - tol)
    probs = np.zeros_like(values, dtype=float)
    probs[winners] = 1.0 / len(winners)
    return probs, float(best)


def _build_phase_policy_lookup(M, opponent_phase_policies):
    uniform = np.full(M, 1.0 / M)
    lookup = {}

    for role in ROLE_NAMES:
        phase_map = {}
        supplied = opponent_phase_policies.get(role, {}) if opponent_phase_policies else {}
        for phase in PHASE_NAMES:
            phase_map[phase] = _normalize_probs(supplied.get(phase, uniform), M)
        lookup[role] = phase_map

    return lookup


def solve_best_response_full_game(T, M, max_score, opponent_phase_policies, tie_value=0.5):
    """Solve a finite-horizon best response against phase-conditioned opponent policies.

    The opponent policy is assumed stationary within each `(role, phase)` bucket.
    The returned values are from the controlled agent's perspective, and each
    policy is an action distribution for the controlled agent in the given role.
    """

    phase_policies = _build_phase_policy_lookup(M, opponent_phase_policies)

    values = {
        "bat_second": np.zeros((T + 1, max_score + 1)),
        "bowl_second": np.zeros((T + 1, max_score + 1)),
        "bat_first": np.zeros((T + 1, max_score + 1)),
        "bowl_first": np.zeros((T + 1, max_score + 1)),
    }
    strategies = {
        "bat_second": {},
        "bowl_second": {},
        "bat_first": {},
        "bowl_first": {},
    }

    values["bat_second"][0, 0] = 1.0
    values["bowl_second"][0, 0] = 0.0
    if max_score >= 1:
        values["bat_second"][0, 1] = tie_value
        values["bowl_second"][0, 1] = tie_value
    if max_score >= 2:
        values["bowl_second"][0, 2:] = 1.0

    # Second innings when the controlled agent is the chasing batter.
    for t in range(1, T + 1):
        phase = phase_for_ball_count(t, T)
        opp_bowl_probs = phase_policies["bowl_second"][phase]

        for k in range(max_score + 1):
            if k <= 0:
                values["bat_second"][t, k] = 1.0
                continue

            wicket_payoff = tie_value if k == 1 else 0.0
            action_values = np.zeros(M)

            for action_idx in range(M):
                new_k = k - (action_idx + 1)
                continuation = 1.0 if new_k <= 0 else values["bat_second"][t - 1, new_k]
                hit_prob = opp_bowl_probs[action_idx]
                action_values[action_idx] = hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation

            probs, val = _argmax_distribution(action_values)
            values["bat_second"][t, k] = val
            strategies["bat_second"][(t, k)] = probs

    # Second innings when the controlled agent is defending with the ball.
    for t in range(1, T + 1):
        phase = phase_for_ball_count(t, T)
        opp_bat_probs = phase_policies["bat_second"][phase]

        for k in range(max_score + 1):
            if k <= 0:
                values["bowl_second"][t, k] = 0.0
                continue

            wicket_payoff = tie_value if k == 1 else 1.0
            continuation = np.zeros(M)

            for bat_idx in range(M):
                new_k = k - (bat_idx + 1)
                continuation[bat_idx] = 0.0 if new_k <= 0 else values["bowl_second"][t - 1, new_k]

            base_value = float(np.dot(opp_bat_probs, continuation))
            action_values = base_value + opp_bat_probs * (wicket_payoff - continuation)

            probs, val = _argmax_distribution(action_values)
            values["bowl_second"][t, k] = val
            strategies["bowl_second"][(t, k)] = probs

    for s in range(max_score + 1):
        chase_k = min(s + 1, max_score)
        values["bat_first"][0, s] = values["bowl_second"][T, chase_k]
        values["bowl_first"][0, s] = values["bat_second"][T, chase_k]

    # First innings when the controlled agent is batting first.
    for t in range(1, T + 1):
        phase = phase_for_ball_count(t, T)
        opp_bowl_probs = phase_policies["bowl_first"][phase]

        for s in range(max_score + 1):
            chase_k = min(s + 1, max_score)
            wicket_payoff = values["bowl_second"][T, chase_k]
            action_values = np.zeros(M)

            for action_idx in range(M):
                new_s = min(s + action_idx + 1, max_score)
                continuation = values["bat_first"][t - 1, new_s]
                hit_prob = opp_bowl_probs[action_idx]
                action_values[action_idx] = hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation

            probs, val = _argmax_distribution(action_values)
            values["bat_first"][t, s] = val
            strategies["bat_first"][(t, s)] = probs

    # First innings when the controlled agent is bowling first.
    for t in range(1, T + 1):
        phase = phase_for_ball_count(t, T)
        opp_bat_probs = phase_policies["bat_first"][phase]

        for s in range(max_score + 1):
            chase_k = min(s + 1, max_score)
            wicket_payoff = values["bat_second"][T, chase_k]
            continuation = np.zeros(M)

            for bat_idx in range(M):
                new_s = min(s + bat_idx + 1, max_score)
                continuation[bat_idx] = values["bowl_first"][t - 1, new_s]

            base_value = float(np.dot(opp_bat_probs, continuation))
            action_values = base_value + opp_bat_probs * (wicket_payoff - continuation)

            probs, val = _argmax_distribution(action_values)
            values["bowl_first"][t, s] = val
            strategies["bowl_first"][(t, s)] = probs

    return values, strategies
