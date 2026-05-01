import numpy as np

from nash_full_game import phase_for_ball_count


FIRST_INNINGS_BUCKETS = ("early", "mid", "late")
PRESSURE_BUCKETS = ("low", "medium", "high")
ROLE_NAMES = ("bat_first", "bowl_first", "bat_second", "bowl_second")


def pressure_bucket(k, t, M):
    if t <= 0:
        return "high"
    required_rate = k / t
    normalized = required_rate / max(M, 1)
    if normalized <= 0.35:
        return "low"
    if normalized <= 0.65:
        return "medium"
    return "high"


def role_bucket_names(role):
    if role in ("bat_first", "bowl_first"):
        return FIRST_INNINGS_BUCKETS
    if role in ("bat_second", "bowl_second"):
        return PRESSURE_BUCKETS
    raise ValueError(f"Unknown role: {role}")


def get_bucket_for_role(role, t, state, T, M):
    if role in ("bat_first", "bowl_first"):
        return phase_for_ball_count(t, T)
    if role in ("bat_second", "bowl_second"):
        return pressure_bucket(state, t, M)
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
        for bucket in role_bucket_names(role):
            bucket_map[bucket] = _normalize_probs(supplied.get(bucket, uniform), M)
        lookup[role] = bucket_map

    return lookup


def solve_pressure_aware_best_response_full_game(
    T,
    M,
    max_score,
    opponent_bucket_policies,
    tie_value=0.5,
):
    lookup = _build_bucket_policy_lookup(M, opponent_bucket_policies)

    values = {
        "bat_second":  np.zeros((T + 1, max_score + 1)),
        "bowl_second": np.zeros((T + 1, max_score + 1)),
        "bat_first":   np.zeros((T + 1, max_score + 1)),
        "bowl_first":  np.zeros((T + 1, max_score + 1)),
    }
    strategies = {
        "bat_second":  {},
        "bowl_second": {},
        "bat_first":   {},
        "bowl_first":  {},
    }

    # ── second innings boundary conditions ───────────────────────────────────
    values["bat_second"][0, 0] = 1.0
    values["bowl_second"][0, 0] = 0.0
    if max_score >= 1:
        values["bat_second"][0, 1]  = tie_value
        values["bowl_second"][0, 1] = tie_value
    if max_score >= 2:
        values["bowl_second"][0, 2:] = 1.0

    # ── bat_second ────────────────────────────────────────────────────────────
    for t in range(1, T + 1):
        for k in range(max_score + 1):
            if k <= 0:
                values["bat_second"][t, k] = 1.0
                continue

            bucket = get_bucket_for_role("bowl_second", t, k, T, M)
            opp_bowl_probs = lookup["bowl_second"][bucket]
            wicket_payoff  = tie_value if k == 1 else 0.0
            action_values  = np.zeros(M)

            for action_idx in range(M):
                new_k        = k - (action_idx + 1)
                continuation = 1.0 if new_k <= 0 else values["bat_second"][t - 1, new_k]
                hit_prob     = opp_bowl_probs[action_idx]
                action_values[action_idx] = (
                    hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation
                )

            probs, val = _argmax_distribution(action_values)
            values["bat_second"][t, k]       = val
            strategies["bat_second"][(t, k)] = probs

    # ── bowl_second ───────────────────────────────────────────────────────────
    for t in range(1, T + 1):
        for k in range(max_score + 1):
            if k <= 0:
                values["bowl_second"][t, k] = 0.0
                continue

            bucket       = get_bucket_for_role("bat_second", t, k, T, M)
            opp_bat_probs = lookup["bat_second"][bucket]
            wicket_payoff = tie_value if k == 1 else 1.0
            continuation  = np.zeros(M)

            for bat_idx in range(M):
                new_k = k - (bat_idx + 1)
                continuation[bat_idx] = (
                    0.0 if new_k <= 0 else values["bowl_second"][t - 1, new_k]
                )

            base_value    = float(np.dot(opp_bat_probs, continuation))
            action_values = base_value + opp_bat_probs * (wicket_payoff - continuation)

            probs, val = _argmax_distribution(action_values)
            values["bowl_second"][t, k]       = val
            strategies["bowl_second"][(t, k)] = probs

    # ── first innings boundary conditions ────────────────────────────────────
    for s in range(max_score + 1):
        chase_k = min(s + 1, max_score)
        values["bat_first"][0, s]  = values["bowl_second"][T, chase_k]
        values["bowl_first"][0, s] = values["bat_second"][T, chase_k]

    # ── bat_first ─────────────────────────────────────────────────────────────
    # FIX: use actual score `s` in bucket lookup (was hardcoded to state=0)
    for t in range(1, T + 1):
        for s in range(max_score + 1):
            chase_k = min(s + 1, max_score)
            bucket          = get_bucket_for_role("bowl_first", t, s, T, M)  # <-- fixed
            opp_bowl_probs  = lookup["bowl_first"][bucket]
            wicket_payoff   = values["bowl_second"][T, chase_k]
            action_values   = np.zeros(M)

            for action_idx in range(M):
                new_s        = min(s + action_idx + 1, max_score)
                continuation = values["bat_first"][t - 1, new_s]
                hit_prob     = opp_bowl_probs[action_idx]
                action_values[action_idx] = (
                    hit_prob * wicket_payoff + (1.0 - hit_prob) * continuation
                )

            probs, val = _argmax_distribution(action_values)
            values["bat_first"][t, s]       = val
            strategies["bat_first"][(t, s)] = probs

    # ── bowl_first ────────────────────────────────────────────────────────────
    # FIX: use actual score `s` in bucket lookup (was hardcoded to state=0)
    for t in range(1, T + 1):
        for s in range(max_score + 1):
            chase_k       = min(s + 1, max_score)
            bucket        = get_bucket_for_role("bat_first", t, s, T, M)  # <-- fixed
            opp_bat_probs = lookup["bat_first"][bucket]
            wicket_payoff = values["bat_second"][T, chase_k]
            continuation  = np.zeros(M)

            for bat_idx in range(M):
                new_s = min(s + bat_idx + 1, max_score)
                continuation[bat_idx] = values["bowl_first"][t - 1, new_s]

            base_value    = float(np.dot(opp_bat_probs, continuation))
            action_values = base_value + opp_bat_probs * (wicket_payoff - continuation)

            probs, val = _argmax_distribution(action_values)
            values["bowl_first"][t, s]       = val
            strategies["bowl_first"][(t, s)] = probs

    return values, strategies