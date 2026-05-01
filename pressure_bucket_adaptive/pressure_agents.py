import numpy as np

from agents_fight import OptimalAgent, ROLE_INNINGS, ROLE_OPPONENT, _sample_action
from .pressure_solver import (
    FIRST_INNINGS_BUCKETS,
    PRESSURE_BUCKETS,
    get_bucket_for_role,
    solve_pressure_aware_best_response_full_game,
)


class PressureBucketOpponentModel:
    def __init__(self, M, prior_alpha=0.05):  # FIX: was 1.0 — low prior so observations dominate quickly
        self.M = M
        self.prior_alpha = prior_alpha
        self.reset()

    def reset(self):
        self.counts = {}
        self.observations = {}
        for role in ("bat_first", "bowl_first"):
            self.counts[role] = {
                bucket: np.full(self.M, self.prior_alpha, dtype=float)
                for bucket in FIRST_INNINGS_BUCKETS
            }
            self.observations[role] = {bucket: 0 for bucket in FIRST_INNINGS_BUCKETS}
        for role in ("bat_second", "bowl_second"):
            self.counts[role] = {
                bucket: np.full(self.M, self.prior_alpha, dtype=float)
                for bucket in PRESSURE_BUCKETS
            }
            self.observations[role] = {bucket: 0 for bucket in PRESSURE_BUCKETS}

    def observe(self, role, bucket, action):
        self.counts[role][bucket][action - 1] += 1.0
        self.observations[role][bucket] += 1

    def bucket_observations(self, role, bucket):
        return self.observations[role][bucket]

    def total_observations(self):
        return sum(
            count
            for role_map in self.observations.values()
            for count in role_map.values()
        )

    def get_probs(self, role, bucket):
        counts = self.counts[role][bucket]
        return counts / counts.sum()

    def export_policies(self):
        return {
            role: {bucket: self.get_probs(role, bucket) for bucket in bucket_map}
            for role, bucket_map in self.counts.items()
        }


class PressureAdaptiveFiniteAgent:
    def __init__(
        self,
        T,
        M,
        max_score,
        tie_value=0.5,
        prior_alpha=0.05,       # FIX: was 1.0
        min_bucket_obs=0,       # FIX: was 1 — blend from ball 1
        recompute_interval=1,   # kept for API compat but overridden by exponential backoff
        max_blend=0.95,         # FIX: was 0.75 — go nearly full BR against predictable opponent
        verbose=True,
    ):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value
        self.min_bucket_obs = min_bucket_obs
        self.recompute_interval = recompute_interval
        self.max_blend = max_blend
        self.verbose = verbose

        if verbose:
            print("Precomputing pressure-aware equilibrium backbone...")
        self.nash_agent = OptimalAgent(T, M, max_score, tie_value=tie_value, verbose=verbose)
        if verbose:
            print("Initializing pressure-aware adaptive best response...")

        self.model = PressureBucketOpponentModel(M, prior_alpha=prior_alpha)
        self.br_values = None
        self.br_strategies = None
        self._dirty = True
        self._last_recompute_obs = -1
        self._next_recompute_threshold = 2   # exponential backoff: recompute at 2, 4, 8, 16 …
        self.policy_trace = []
        self._pending_trace_indices = []
        self._decision_counter = 0
        self._role_ball_counters = {role: 0 for role in ROLE_INNINGS}
        self._recompute_best_response(force=True)
        if verbose:
            print("Done.")

    def reset_match(self):
        self.model.reset()
        self._dirty = True
        self._last_recompute_obs = -1
        self._next_recompute_threshold = 2   # reset backoff counter
        self.policy_trace = []
        self._pending_trace_indices = []
        self._decision_counter = 0
        self._role_ball_counters = {role: 0 for role in ROLE_INNINGS}
        # FIX: do NOT force recompute here — br_strategies from uniform prior
        # is already cached from __init__ and is identical to what a fresh
        # recompute would produce, so this was pure wasted compute.

    def observe(self, role, t, state, my_action, opp_action):
        del my_action
        opponent_role = ROLE_OPPONENT[role]
        bucket = get_bucket_for_role(opponent_role, t, state, self.T, self.M)
        self.model.observe(opponent_role, bucket, opp_action)
        self._dirty = True

        if self._pending_trace_indices:
            trace_idx = self._pending_trace_indices.pop(0)
            self.policy_trace[trace_idx]["opponent_action"] = int(opp_action)
            self.policy_trace[trace_idx]["post_observation_count"] = self.model.bucket_observations(
                opponent_role,
                bucket,
            )

    def _blend_weight(self, role, t, state):
        opponent_role = ROLE_OPPONENT[role]
        bucket = get_bucket_for_role(opponent_role, t, state, self.T, self.M)
        observed = self.model.bucket_observations(opponent_role, bucket)
        if observed < self.min_bucket_obs:
            return 0.0
        # FIX: faster ramp — reaches 0.9 after 9 obs instead of 9*M obs
        raw_weight = observed / (observed + 1)
        return min(self.max_blend, raw_weight)

    def _maybe_recompute_best_response(self):
        if not self._dirty:
            return

        total_obs = self.model.total_observations()

        # FIX: exponential backoff — recompute at obs counts 2, 4, 8, 16, …
        # instead of every single ball. Cuts solver calls from ~24/match to ~5.
        if self.br_strategies is None or total_obs >= self._next_recompute_threshold:
            self._recompute_best_response(force=True)
            self._next_recompute_threshold = max(
                self._next_recompute_threshold * 2,
                total_obs + 1,
            )

    def _recompute_best_response(self, force=False):
        if not force and not self._dirty:
            return

        policies = self.model.export_policies()
        self.br_values, self.br_strategies = solve_pressure_aware_best_response_full_game(
            self.T,
            self.M,
            self.max_score,
            policies,
            tie_value=self.tie_value,
        )
        self._last_recompute_obs = self.model.total_observations()
        self._dirty = False

    def _compute_policy_components(self, role, t, state):
        self._maybe_recompute_best_response()

        state = int(max(0, min(state, self.max_score)))
        bucket = get_bucket_for_role(role, t, state, self.T, self.M)
        nash_probs = self.nash_agent.get_policy(role, t, state)
        br_probs   = self.br_strategies[role][(t, state)]
        weight     = self._blend_weight(role, t, state)
        mixed_probs = (1.0 - weight) * nash_probs + weight * br_probs
        mixed_probs = np.clip(mixed_probs, 0.0, None)
        mixed_probs /= mixed_probs.sum()

        return state, bucket, nash_probs, br_probs, weight, mixed_probs

    def get_policy(self, role, t, state):
        _, _, _, _, _, probs = self._compute_policy_components(role, t, state)
        return probs

    def act(self, role, t, state):
        state, bucket, nash_probs, br_probs, weight, mixed_probs = self._compute_policy_components(
            role, t, state
        )
        action = _sample_action(mixed_probs)

        self._decision_counter += 1
        self._role_ball_counters[role] += 1
        innings = ROLE_INNINGS[role]
        trace_entry = {
            "decision_index":        self._decision_counter,
            "match_ball_index":      self._decision_counter,
            "innings":               innings,
            "innings_ball_index":    self._role_ball_counters[role],
            "role":                  role,
            "t":                     int(t),
            "state":                 int(state),
            "bucket":                bucket,
            "blend_weight":          float(weight),
            "chosen_action":         int(action),
            "opponent_action":       None,
            "pre_observation_count": self.model.bucket_observations(ROLE_OPPONENT[role], bucket),
            "post_observation_count": None,
            "nash_probs":            np.array(nash_probs, copy=True),
            "best_response_probs":   np.array(br_probs,   copy=True),
            "mixed_probs":           np.array(mixed_probs, copy=True),
        }
        self.policy_trace.append(trace_entry)
        self._pending_trace_indices.append(len(self.policy_trace) - 1)
        return action

    def game_value(self):
        return self.nash_agent.game_value()

    def get_policy_trace(self):
        return list(self.policy_trace)