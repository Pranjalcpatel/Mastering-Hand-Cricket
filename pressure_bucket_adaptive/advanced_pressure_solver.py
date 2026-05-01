import numpy as np

FIRST_INNINGS_BUCKETS = ("early", "mid", "late")
PRESSURE_BUCKETS = ("low", "medium", "high")
ROLE_NAMES = ("bat_first", "bowl_first", "bat_second", "bowl_second")


def get_soft_bucket_for_role(role, t, state, T, M):
    """
    Returns a dictionary of {bucket_name: weight} for the given state.
    This implements Soft Bucketing to mitigate state aliasing.
    """
    if role in ("bat_first", "bowl_first"):
        if T <= 0:
            return {"late": 1.0}
        x = t / T
        # Centers: late=1/6, mid=1/2, early=5/6
        if x <= 1/6:
            return {"late": 1.0}
        elif x < 1/2:
            mid_w = (x - 1/6) / (1/2 - 1/6)
            return {"late": 1.0 - mid_w, "mid": mid_w}
        elif x < 5/6:
            early_w = (x - 1/2) / (5/6 - 1/2)
            return {"mid": 1.0 - early_w, "early": early_w}
        else:
            return {"early": 1.0}

    if role in ("bat_second", "bowl_second"):
        if t <= 0:
            return {"high": 1.0}
        # Centers: low=0.175, medium=0.5, high=0.825
        R = (state / t) / max(M, 1)
        if R <= 0.175:
            return {"low": 1.0}
        elif R < 0.5:
            med_w = (R - 0.175) / (0.5 - 0.175)
            return {"low": 1.0 - med_w, "medium": med_w}
        elif R < 0.825:
            high_w = (R - 0.5) / (0.825 - 0.5)
            return {"medium": 1.0 - high_w, "high": high_w}
        else:
            return {"high": 1.0}

    raise ValueError(f"Unknown role: {role}")


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


def _build_bucket_policy_lookup(M, opponent_bucket_policies):
    uniform = np.full(M, 1.0 / M)
    lookup = {}

    for role in ROLE_NAMES:
        bucket_map = {}
        supplied = opponent_bucket_policies.get(role, {}) if opponent_bucket_policies else {}
        buckets = FIRST_INNINGS_BUCKETS if role in ("bat_first", "bowl_first") else PRESSURE_BUCKETS
        for bucket in buckets:
            bucket_map[bucket] = _normalize_probs(supplied.get(bucket, uniform), M)
        lookup[role] = bucket_map

    return lookup


def _get_blended_policy(lookup, role, t, state, T, M):
    weights = get_soft_bucket_for_role(role, t, state, T, M)
    blended = np.zeros(M)
    for bucket, w in weights.items():
        blended += w * lookup[role][bucket]
    return blended


def solve_pressure_aware_best_response_subgame(
    role_curr,
    t_curr,
    state_curr,
    T_total,
    M,
    max_score,
    opponent_bucket_policies,
    tie_value=0.5,
):
    """
    Subgame solver that strictly computes the reachable DP grid.
    If role_curr is second innings, it only computes up to (t_curr, state_curr).
    If role_curr is first innings, it computes first innings up to t_curr and the
    required target range in the second innings.
    """
    lookup = _build_bucket_policy_lookup(M, opponent_bucket_policies)

    values = {}
    strategies = {}

    is_second_innings = role_curr in ("bat_second", "bowl_second")

    if is_second_innings:
        # We only need to solve up to t_curr and k_curr
        k_curr = min(state_curr, max_score)
        values["bat_second"] = np.zeros((t_curr + 1, k_curr + 1))
        values["bowl_second"] = np.zeros((t_curr + 1, k_curr + 1))
        strategies["bat_second"] = {}
        strategies["bowl_second"] = {}

        # Boundary conditions
        values["bat_second"][0, 0] = 1.0
        values["bowl_second"][0, 0] = 0.0
        if k_curr >= 1:
            values["bat_second"][0, 1] = tie_value
            values["bowl_second"][0, 1] = tie_value
        if k_curr >= 2:
            values["bowl_second"][0, 2:] = 1.0

        # DP loops bound to subgame
        for t in range(1, t_curr + 1):
            for k in range(k_curr + 1):
                if k <= 0:
                    values["bat_second"][t, k] = 1.0
                    values["bowl_second"][t, k] = 0.0
                    continue

                # bat_second
                opp_bowl_probs = _get_blended_policy(lookup, "bowl_second", t, k, T_total, M)
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

                # bowl_second
                opp_bat_probs = _get_blended_policy(lookup, "bat_second", t, k, T_total, M)
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

    else:
        # First innings subgame
        s_curr = min(state_curr, max_score)
        max_s_reachable = min(max_score, s_curr + t_curr * M)
        max_target = min(max_score, max_s_reachable + 1)

        # Solve second innings entirely for targets up to max_target
        values["bat_second"] = np.zeros((T_total + 1, max_target + 1))
        values["bowl_second"] = np.zeros((T_total + 1, max_target + 1))
        
        values["bat_second"][0, 0] = 1.0
        values["bowl_second"][0, 0] = 0.0
        if max_target >= 1:
            values["bat_second"][0, 1] = tie_value
            values["bowl_second"][0, 1] = tie_value
        if max_target >= 2:
            values["bowl_second"][0, 2:] = 1.0

        for t in range(1, T_total + 1):
            for k in range(max_target + 1):
                if k <= 0:
                    values["bat_second"][t, k] = 1.0
                    values["bowl_second"][t, k] = 0.0
                    continue

                opp_bowl_probs = _get_blended_policy(lookup, "bowl_second", t, k, T_total, M)
                wicket_payoff = tie_value if k == 1 else 0.0
                action_values = np.zeros(M)
                for action_idx in range(M):
                    new_k = k - (action_idx + 1)
                    continuation = 1.0 if new_k <= 0 else values["bat_second"][t - 1, new_k]
                    hit_prob = opp_bowl_probs[action_idx]
                    action_values[action_idx] = hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation
                probs, val = _argmax_distribution(action_values)
                values["bat_second"][t, k] = val

                opp_bat_probs = _get_blended_policy(lookup, "bat_second", t, k, T_total, M)
                wicket_payoff = tie_value if k == 1 else 1.0
                continuation = np.zeros(M)
                for bat_idx in range(M):
                    new_k = k - (bat_idx + 1)
                    continuation[bat_idx] = 0.0 if new_k <= 0 else values["bowl_second"][t - 1, new_k]
                base_value = float(np.dot(opp_bat_probs, continuation))
                action_values = base_value + opp_bat_probs * (wicket_payoff - continuation)
                probs, val = _argmax_distribution(action_values)
                values["bowl_second"][t, k] = val

        # Now solve first innings subgame
        # Note: Array is sized full (T_total+1, max_score+1) for simplicity of indexing,
        # but we only populate/compute up to t_curr and s_curr...max_s_reachable
        values["bat_first"] = np.zeros((T_total + 1, max_score + 1))
        values["bowl_first"] = np.zeros((T_total + 1, max_score + 1))
        strategies["bat_first"] = {}
        strategies["bowl_first"] = {}

        # Boundary conditions at t=0 (from second innings at T_total)
        for s in range(s_curr, max_s_reachable + 1):
            chase_k = min(s + 1, max_target)
            values["bat_first"][0, s] = values["bowl_second"][T_total, chase_k]
            values["bowl_first"][0, s] = values["bat_second"][T_total, chase_k]

        # First innings loops up to t_curr
        for t in range(1, t_curr + 1):
            # Only states reachable backward from t_curr
            s_min = max(0, s_curr)
            s_max = min(max_score, max_s_reachable - t * M)
            # Actually, to compute (t_curr, s_curr), we need (t_curr-1, s_curr...s_curr+M)
            # So s bounds: [s_curr, max_s_reachable]
            for s in range(s_min, max_s_reachable + 1):
                chase_k = min(s + 1, max_target)

                # bat_first
                opp_bowl_probs = _get_blended_policy(lookup, "bowl_first", t, s, T_total, M)
                wicket_payoff = values["bowl_second"][T_total, chase_k]
                action_values = np.zeros(M)
                for action_idx in range(M):
                    new_s = min(s + action_idx + 1, max_score)
                    continuation = values["bat_first"][t - 1, new_s]
                    hit_prob = opp_bowl_probs[action_idx]
                    action_values[action_idx] = hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation
                probs, val = _argmax_distribution(action_values)
                values["bat_first"][t, s] = val
                strategies["bat_first"][(t, s)] = probs

                # bowl_first
                opp_bat_probs = _get_blended_policy(lookup, "bat_first", t, s, T_total, M)
                wicket_payoff = values["bat_second"][T_total, chase_k]
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
