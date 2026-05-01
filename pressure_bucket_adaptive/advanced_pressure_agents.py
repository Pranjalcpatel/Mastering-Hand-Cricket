import numpy as np

from agents_fight import OptimalAgent, ROLE_INNINGS, ROLE_OPPONENT, _sample_action
from pressure_bucket_adaptive.advanced_pressure_solver import (
    FIRST_INNINGS_BUCKETS,
    PRESSURE_BUCKETS,
    get_soft_bucket_for_role,
    solve_pressure_aware_best_response_subgame,
)


class AdvancedOpponentModel:
    def __init__(self, M, prior_alpha=0.05, gamma=0.95):
        self.M = M
        self.prior_alpha = prior_alpha
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.counts = {}
        self.observations = {}
        for role in ("bat_first", "bowl_first"):
            self.counts[role] = {
                bucket: np.full(self.M, self.prior_alpha, dtype=float)
                for bucket in FIRST_INNINGS_BUCKETS
            }
            self.observations[role] = {bucket: 0.0 for bucket in FIRST_INNINGS_BUCKETS}
        for role in ("bat_second", "bowl_second"):
            self.counts[role] = {
                bucket: np.full(self.M, self.prior_alpha, dtype=float)
                for bucket in PRESSURE_BUCKETS
            }
            self.observations[role] = {bucket: 0.0 for bucket in PRESSURE_BUCKETS}

    def observe(self, role, bucket_weights, action):
        for bucket, weight in bucket_weights.items():
            self.counts[role][bucket] *= self.gamma
            self.counts[role][bucket][action - 1] += weight
            self.observations[role][bucket] = (self.observations[role][bucket] * self.gamma) + weight

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


class AdvancedPressureAdaptiveAgent:
    def __init__(
        self,
        T,
        M,
        max_score,
        tie_value=0.5,
        prior_alpha=0.05,
        gamma=0.95,
        min_bucket_obs=0,
        max_blend=0.95,
        verbose=True,
    ):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value
        self.min_bucket_obs = min_bucket_obs
        self.max_blend = max_blend
        self.verbose = verbose

        if verbose:
            print("Precomputing pressure-aware equilibrium backbone...")
        self.nash_agent = OptimalAgent(T, M, max_score, tie_value=tie_value, verbose=verbose)
        if verbose:
            print("Initializing advanced pressure-aware adaptive best response...")

        self.model = AdvancedOpponentModel(M, prior_alpha=prior_alpha, gamma=gamma)
        self.br_values = None
        self.br_strategies = None
        
        self.policy_trace = []
        self._pending_trace_indices = []
        self._decision_counter = 0
        self._role_ball_counters = {role: 0 for role in ROLE_INNINGS}
        
        # We don't force a recompute in __init__ because subgame solving depends on (role, t, state).
        # It will recompute lazily on the first action.
        
        if verbose:
            print("Done.")

    def reset_match(self):
        self.model.reset()
        self.policy_trace = []
        self._pending_trace_indices = []
        self._decision_counter = 0
        self._role_ball_counters = {role: 0 for role in ROLE_INNINGS}
        self.br_values = None
        self.br_strategies = None

    def observe(self, role, t, state, my_action, opp_action):
        del my_action
        opponent_role = ROLE_OPPONENT[role]
        bucket_weights = get_soft_bucket_for_role(opponent_role, t, state, self.T, self.M)
        self.model.observe(opponent_role, bucket_weights, opp_action)

        if self._pending_trace_indices:
            trace_idx = self._pending_trace_indices.pop(0)
            self.policy_trace[trace_idx]["opponent_action"] = int(opp_action)
            # Log total observation across the primary bucket for simplicity in trace
            primary_bucket = max(bucket_weights, key=bucket_weights.get)
            self.policy_trace[trace_idx]["post_observation_count"] = self.model.bucket_observations(
                opponent_role,
                primary_bucket,
            )

    def _blend_weight(self, role, t, state):
        opponent_role = ROLE_OPPONENT[role]
        bucket_weights = get_soft_bucket_for_role(opponent_role, t, state, self.T, self.M)
        
        probs = np.zeros(self.M)
        total_obs = 0.0
        for bucket, w in bucket_weights.items():
            probs += w * self.model.get_probs(opponent_role, bucket)
            total_obs += w * self.model.bucket_observations(opponent_role, bucket)
            
        if total_obs < self.min_bucket_obs:
            return 0.0

        p_safe = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe))
        max_entropy = np.log(self.M)
        entropy_ratio = entropy / max_entropy
        
        confidence = 1.0 - entropy_ratio
        
        raw_weight = (total_obs / (total_obs + 1)) * confidence
        return min(self.max_blend, raw_weight)

    def _compute_policy_components(self, role, t, state):
        state = int(max(0, min(state, self.max_score)))
        
        # Subgame solving allows us to recompute every time effortlessly!
        policies = self.model.export_policies()
        self.br_values, self.br_strategies = solve_pressure_aware_best_response_subgame(
            role_curr=role,
            t_curr=int(t),
            state_curr=state,
            T_total=self.T,
            M=self.M,
            max_score=self.max_score,
            opponent_bucket_policies=policies,
            tie_value=self.tie_value,
        )

        bucket_weights = get_soft_bucket_for_role(role, t, state, self.T, self.M)
        primary_bucket = max(bucket_weights, key=bucket_weights.get)
        
        nash_probs = self.nash_agent.get_policy(role, t, state)
        br_probs   = self.br_strategies[role][(t, state)]
        weight     = self._blend_weight(role, t, state)
        mixed_probs = (1.0 - weight) * nash_probs + weight * br_probs
        mixed_probs = np.clip(mixed_probs, 0.0, None)
        mixed_probs /= mixed_probs.sum()

        return state, primary_bucket, nash_probs, br_probs, weight, mixed_probs

    def get_policy(self, role, t, state):
        _, _, _, _, _, probs = self._compute_policy_components(role, t, state)
        return probs

    def act(self, role, t, state):
        state, primary_bucket, nash_probs, br_probs, weight, mixed_probs = self._compute_policy_components(
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
            "bucket":                primary_bucket,
            "blend_weight":          float(weight),
            "chosen_action":         int(action),
            "opponent_action":       None,
            "pre_observation_count": self.model.bucket_observations(ROLE_OPPONENT[role], primary_bucket),
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
