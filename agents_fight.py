from nash_full_game import (
    PHASE_NAMES,
    phase_for_ball_count,
    solve_best_response_full_game,
    solve_full_game,
)
from nash_infinite_game import solve_infinite_game
import numpy as np


ROLE_OPPONENT = {
    "bat_first": "bowl_first",
    "bowl_first": "bat_first",
    "bat_second": "bowl_second",
    "bowl_second": "bat_second",
}
ROLE_INNINGS = {
    "bat_first": 1,
    "bowl_first": 1,
    "bat_second": 2,
    "bowl_second": 2,
}


def _sample_action(probs):
    return int(np.random.choice(np.arange(1, len(probs) + 1), p=probs))


def _uniform_probs(M):
    return np.full(M, 1.0 / M)


class RandomAgent:
    def __init__(self, M):
        self.M = M

    def reset_match(self):
        return None

    def observe(self, role, t, state, my_action, opp_action):
        return None

    def act(self, role, *state_args):
        del role, state_args
        return np.random.randint(1, self.M + 1)


class RequiredRunRateAgent:
    def __init__(self, T, M, max_score, par_fraction=0.5):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.par_fraction = par_fraction
        self.par_score = int(round(par_fraction * max_score))

    def reset_match(self):
        return None

    def observe(self, role, t, state, my_action, opp_action):
        return None

    def _threshold_support(self, threshold):
        threshold = int(max(1, min(self.M, threshold)))
        support = np.arange(threshold, self.M + 1)
        probs = np.full(len(support), 1.0 / len(support))
        return support, probs

    def _par_required_rate(self, t, score):
        if t <= 0:
            return self.M
        remaining = max(0, self.par_score - score)
        return max(1, int(np.ceil(remaining / t)))

    def act(self, role, t, state):
        if role == "bat_second":
            threshold = max(1, int(np.ceil(state / max(t, 1))))
        elif role == "bowl_second":
            threshold = max(1, int(np.ceil(state / max(t, 1))))
        elif role == "bat_first":
            threshold = self._par_required_rate(t, state)
        elif role == "bowl_first":
            threshold = self._par_required_rate(t, state)
        else:
            raise ValueError("Invalid role")

        support, probs = self._threshold_support(threshold)
        return int(np.random.choice(support, p=probs))


class OptimalAgent:
    def __init__(self, T, M, max_score, tie_value=0.5, verbose=True):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value

        if verbose:
            print("Precomputing equilibrium...")
        self.V, self.W, self.V_strat, self.W_strat = solve_full_game(
            T, M, max_score, tie_value=tie_value
        )
        if verbose:
            print("Done.")

    def reset_match(self):
        return None

    def observe(self, role, t, state, my_action, opp_action):
        return None

    def get_policy(self, role, t, state):
        state = int(max(0, min(state, self.max_score)))

        if role == "bat_first":
            p_opt, _ = self.W_strat[(t, state)]
            return p_opt
        if role == "bowl_first":
            _, q_opt = self.W_strat[(t, state)]
            return q_opt
        if role == "bat_second":
            p_opt, _ = self.V_strat[(t, state)]
            return p_opt
        if role == "bowl_second":
            _, q_opt = self.V_strat[(t, state)]
            return q_opt
        raise ValueError("Invalid role")

    def act(self, role, t, state):
        return _sample_action(self.get_policy(role, t, state))

    def game_value(self):
        return self.W[self.T][0]


class BestResponseFiniteAgent:
    def __init__(self, T, M, max_score, opponent_phase_policies=None, tie_value=0.5, verbose=True):
        self.T = T
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value
        self.opponent_phase_policies = opponent_phase_policies or {}
        self.verbose = verbose
        self.values, self.strategies = solve_best_response_full_game(
            T,
            M,
            max_score,
            self.opponent_phase_policies,
            tie_value=tie_value,
        )

    def reset_match(self):
        return None

    def observe(self, role, t, state, my_action, opp_action):
        return None

    def get_policy(self, role, t, state):
        state = int(max(0, min(state, self.max_score)))
        return self.strategies[role][(t, state)]

    def act(self, role, t, state):
        return _sample_action(self.get_policy(role, t, state))


class PhaseOpponentModel:
    def __init__(self, M, prior_alpha=1.0):
        self.M = M
        self.prior_alpha = prior_alpha
        self.reset()

    def reset(self):
        self.counts = {
            role: {phase: np.full(self.M, self.prior_alpha, dtype=float) for phase in PHASE_NAMES}
            for role in ROLE_OPPONENT.values()
        }
        self.observations = {
            role: {phase: 0 for phase in PHASE_NAMES}
            for role in ROLE_OPPONENT.values()
        }

    def observe(self, role, phase, action):
        self.counts[role][phase][action - 1] += 1.0
        self.observations[role][phase] += 1

    def get_probs(self, role, phase):
        counts = self.counts[role][phase]
        return counts / counts.sum()

    def export_policies(self):
        return {
            role: {phase: self.get_probs(role, phase) for phase in PHASE_NAMES}
            for role in self.counts
        }

    def bucket_observations(self, role, phase):
        return self.observations[role][phase]

    def total_observations(self):
        return sum(
            phase_count
            for role_map in self.observations.values()
            for phase_count in role_map.values()
        )


class AdaptiveFiniteAgent:
    def __init__(
        self,
        T,
        M,
        max_score,
        tie_value=0.5,
        prior_alpha=1.0,
        min_bucket_obs=1,
        recompute_interval=1,
        max_blend=0.75,
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
            print("Precomputing equilibrium backbone...")
        self.nash_agent = OptimalAgent(T, M, max_score, tie_value=tie_value, verbose=verbose)
        if verbose:
            print("Initializing adaptive best response...")

        self.model = PhaseOpponentModel(M, prior_alpha=prior_alpha)
        self.br_values = None
        self.br_strategies = None
        self._dirty = True
        self._last_recompute_obs = -1
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
        self.policy_trace = []
        self._pending_trace_indices = []
        self._decision_counter = 0
        self._role_ball_counters = {role: 0 for role in ROLE_INNINGS}
        self._recompute_best_response(force=True)

    def observe(self, role, t, state, my_action, opp_action):
        del state, my_action
        opponent_role = ROLE_OPPONENT[role]
        phase = phase_for_ball_count(t, self.T)
        self.model.observe(opponent_role, phase, opp_action)
        self._dirty = True

        if self._pending_trace_indices:
            trace_idx = self._pending_trace_indices.pop(0)
            self.policy_trace[trace_idx]["opponent_action"] = int(opp_action)
            self.policy_trace[trace_idx]["post_observation_count"] = self.model.bucket_observations(
                opponent_role,
                phase,
            )

    def _blend_weight(self, role, t):
        opponent_role = ROLE_OPPONENT[role]
        phase = phase_for_ball_count(t, self.T)
        observed = self.model.bucket_observations(opponent_role, phase)
        if observed < self.min_bucket_obs:
            return 0.0

        raw_weight = observed / (observed + self.M)
        return min(self.max_blend, raw_weight)

    def _maybe_recompute_best_response(self):
        if not self._dirty:
            return

        total_obs = self.model.total_observations()
        should_recompute = (
            self.br_strategies is None
            or total_obs == 0
            or total_obs - self._last_recompute_obs >= self.recompute_interval
        )

        if should_recompute:
            self._recompute_best_response(force=True)

    def _recompute_best_response(self, force=False):
        if not force and not self._dirty:
            return

        policies = self.model.export_policies()
        self.br_values, self.br_strategies = solve_best_response_full_game(
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
        phase = phase_for_ball_count(t, self.T)
        nash_probs = self.nash_agent.get_policy(role, t, state)
        br_probs = self.br_strategies[role][(t, state)]
        weight = self._blend_weight(role, t)
        mixed_probs = (1.0 - weight) * nash_probs + weight * br_probs
        mixed_probs = np.clip(mixed_probs, 0.0, None)
        mixed_probs /= mixed_probs.sum()

        return state, phase, nash_probs, br_probs, weight, mixed_probs

    def get_policy(self, role, t, state):
        _, _, _, _, _, probs = self._compute_policy_components(role, t, state)
        return probs

    def act(self, role, t, state):
        state, phase, nash_probs, br_probs, weight, mixed_probs = self._compute_policy_components(role, t, state)
        action = _sample_action(mixed_probs)

        self._decision_counter += 1
        self._role_ball_counters[role] += 1
        innings = ROLE_INNINGS[role]
        trace_entry = {
            "decision_index": self._decision_counter,
            "match_ball_index": self._decision_counter,
            "innings": innings,
            "innings_ball_index": self._role_ball_counters[role],
            "role": role,
            "t": int(t),
            "state": int(state),
            "phase": phase,
            "blend_weight": float(weight),
            "chosen_action": int(action),
            "opponent_action": None,
            "pre_observation_count": self.model.bucket_observations(ROLE_OPPONENT[role], phase),
            "post_observation_count": None,
            "nash_probs": np.array(nash_probs, copy=True),
            "best_response_probs": np.array(br_probs, copy=True),
            "mixed_probs": np.array(mixed_probs, copy=True),
        }
        self.policy_trace.append(trace_entry)
        self._pending_trace_indices.append(len(self.policy_trace) - 1)

        return action

    def game_value(self):
        return self.nash_agent.game_value()

    def get_policy_trace(self):
        return list(self.policy_trace)


class InfiniteOptimalAgent:
    def __init__(self, M, max_score, tie_value=0.5, verbose=True):
        self.M = M
        self.max_score = max_score
        self.tie_value = tie_value

        if verbose:
            print("Solving infinite-balls equilibrium...")
        self.V, self.W, self.V_strat, self.W_strat = solve_infinite_game(
            M, max_score, tie_value=tie_value
        )
        if verbose:
            print("Done.")

    def act(self, role, state):
        state = int(max(0, min(state, self.max_score)))

        if role == "bat_first":
            p_opt, _ = self.W_strat[state]
            return _sample_action(p_opt)
        if role == "bowl_first":
            _, q_opt = self.W_strat[state]
            return _sample_action(q_opt)
        if role == "bat_second":
            p_opt, _ = self.V_strat[state]
            return _sample_action(p_opt)
        if role == "bowl_second":
            _, q_opt = self.V_strat[state]
            return _sample_action(q_opt)
        raise ValueError("Invalid role")

    def game_value(self):
        return self.W[0]


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


def _reset_if_supported(agent):
    if hasattr(agent, "reset_match"):
        agent.reset_match()


def _observe_if_supported(agent, role, t, state, my_action, opp_action):
    if hasattr(agent, "observe"):
        agent.observe(role, t, state, my_action, opp_action)


def simulate_full_game(agent1, agent2, agent1_bats_first=True, T=None, max_score=None):
    T = _resolve_game_param(agent1, agent2, "T", T)
    max_score = _resolve_game_param(agent1, agent2, "max_score", max_score)

    _reset_if_supported(agent1)
    _reset_if_supported(agent2)

    # FIRST INNINGS
    t = T
    s = 0

    while t > 0:
        state = s

        if agent1_bats_first:
            bat_role, bowl_role = "bat_first", "bowl_first"
            bat = agent1.act(bat_role, t, state)
            bowl = agent2.act(bowl_role, t, state)
            _observe_if_supported(agent1, bat_role, t, state, bat, bowl)
            _observe_if_supported(agent2, bowl_role, t, state, bowl, bat)
        else:
            bat_role, bowl_role = "bat_first", "bowl_first"
            bat = agent2.act(bat_role, t, state)
            bowl = agent1.act(bowl_role, t, state)
            _observe_if_supported(agent2, bat_role, t, state, bat, bowl)
            _observe_if_supported(agent1, bowl_role, t, state, bowl, bat)

        if bat == bowl:
            break

        s = min(s + bat, max_score)
        t -= 1

    target = min(s + 1, max_score)

    # SECOND INNINGS
    t = T
    k = target

    while t > 0 and k > 0:
        state = k

        if agent1_bats_first:
            bat_role, bowl_role = "bat_second", "bowl_second"
            bat = agent2.act(bat_role, t, state)
            bowl = agent1.act(bowl_role, t, state)
            _observe_if_supported(agent2, bat_role, t, state, bat, bowl)
            _observe_if_supported(agent1, bowl_role, t, state, bowl, bat)
        else:
            bat_role, bowl_role = "bat_second", "bowl_second"
            bat = agent1.act(bat_role, t, state)
            bowl = agent2.act(bowl_role, t, state)
            _observe_if_supported(agent1, bat_role, t, state, bat, bowl)
            _observe_if_supported(agent2, bowl_role, t, state, bowl, bat)

        if bat == bowl:
            break

        k -= bat
        t -= 1

    if k <= 0:
        winner = 2 if agent1_bats_first else 1
    elif k == 1:
        winner = 0
    else:
        winner = 1 if agent1_bats_first else 2

    return winner


def simulate_infinite_game(agent1, agent2, agent1_bats_first=True, max_score=None):
    max_score = _resolve_game_param(agent1, agent2, "max_score", max_score)

    s = 0

    while True:
        if agent1_bats_first:
            bat = agent1.act("bat_first", s)
            bowl = agent2.act("bowl_first", s)
        else:
            bat = agent2.act("bat_first", s)
            bowl = agent1.act("bowl_first", s)

        if bat == bowl:
            break

        s = min(s + bat, max_score)

    target = min(s + 1, max_score)
    k = target

    while k > 0:
        if agent1_bats_first:
            bat = agent2.act("bat_second", k)
            bowl = agent1.act("bowl_second", k)
        else:
            bat = agent1.act("bat_second", k)
            bowl = agent2.act("bowl_second", k)

        if bat == bowl:
            break

        k -= bat

    if k <= 0:
        winner = 2 if agent1_bats_first else 1
    elif k == 1:
        winner = 0
    else:
        winner = 1 if agent1_bats_first else 2

    return winner


def compute_win_rate(T, M, trials=300, tie_value=0.5, adaptive=False):
    max_score = T * M
    if adaptive:
        agent_opt = AdaptiveFiniteAgent(T, M, max_score, tie_value=tie_value)
    else:
        agent_opt = OptimalAgent(T, M, max_score, tie_value=tie_value)
    agent_rand = RandomAgent(M)

    score = 0.0
    for _ in range(trials):
        winner = simulate_full_game(
            agent_opt,
            agent_rand,
            agent1_bats_first=np.random.rand() < 0.5,
        )
        if winner == 1:
            score += 1.0
        elif winner == 0:
            score += tie_value

    return score / trials


class PaperInfiniteAgent:
    def __init__(self, M):
        self.M = M
        self.strategy = self.compute_strategy()

    def compute_strategy(self):
        s = np.arange(1, self.M + 1)
        rho = 1.0

        for _ in range(1000):
            g = np.sum(s / (rho + s)) - 1
            dg = -np.sum(s / (rho + s) ** 2)
            rho -= g / dg

            if abs(g) < 1e-10:
                break

        p = 1 / (rho + s)
        p /= p.sum()
        return p

    def act(self, role, state):
        del role, state
        return _sample_action(self.strategy)


def main():
    M = 6
    max_score = 200
    trials = 3000

    opt_agent = InfiniteOptimalAgent(M, max_score)
    paper_agent = PaperInfiniteAgent(M)
    rand_agent = RandomAgent(M)

    print("\nOptimal strategy:")
    print(opt_agent.W_strat[0][0])

    print("\nPaper strategy:")
    print(paper_agent.strategy)

    p_opt = opt_agent.W_strat[0][0]
    p_paper = paper_agent.strategy

    l1 = np.sum(np.abs(p_opt - p_paper))
    kl = np.sum(p_opt * np.log(p_opt / p_paper))

    print("\nStrategy comparison")
    print("L1 distance:", l1)
    print("KL divergence:", kl)

    def run_match(agentA, agentB):
        score = 0

        for _ in range(trials):
            winner = simulate_infinite_game(
                agentA,
                agentB,
                agent1_bats_first=np.random.rand() < 0.5
            )

            if winner == 1:
                score += 1
            elif winner == 0:
                score += 0.5

        return score / trials

    print("\nWin rates")
    print("Optimal vs Random:", run_match(opt_agent, rand_agent))
    print("Paper vs Random:", run_match(paper_agent, rand_agent))
    print("Optimal vs Paper:", run_match(opt_agent, paper_agent))


if __name__ == "__main__":
    main()
