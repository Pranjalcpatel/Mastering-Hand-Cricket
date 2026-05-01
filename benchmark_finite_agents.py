import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents_fight import (
    AdaptiveFiniteAgent,
    BestResponseFiniteAgent,
    OptimalAgent,
    RandomAgent,
    simulate_full_game,
)
from nash_full_game import PHASE_NAMES


AGENT_ORDER = ("optimal", "adaptive", "best_response_uniform")
OPPONENT_ORDER = ("random", "optimal")
TOSS_MODES = ("mixed", "bat_first", "bat_second")
STATE_VALUE_SOURCES = ("nash", "best_response", "adaptive_best_response")


def parse_int_list(raw):
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw):
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_or_none(raw):
    text = raw.strip().lower()
    if text in {"none", "off", "0"}:
        return None
    return int(raw)


def uniform_phase_policy(M):
    uniform = np.full(M, 1.0 / M)
    return {
        role: {phase: uniform.copy() for phase in PHASE_NAMES}
        for role in ("bat_first", "bowl_first", "bat_second", "bowl_second")
    }


def build_agent(kind, T, M, max_score, tie_value):
    started = time.perf_counter()

    if kind == "optimal":
        agent = OptimalAgent(T, M, max_score, tie_value=tie_value, verbose=False)
    elif kind == "adaptive":
        agent = AdaptiveFiniteAgent(T, M, max_score, tie_value=tie_value, verbose=False)
    elif kind == "best_response_uniform":
        agent = BestResponseFiniteAgent(
            T,
            M,
            max_score,
            opponent_phase_policies=uniform_phase_policy(M),
            tie_value=tie_value,
            verbose=False,
        )
    elif kind == "random":
        agent = RandomAgent(M)
    else:
        raise ValueError(f"Unknown agent kind: {kind}")

    init_seconds = time.perf_counter() - started
    return agent, init_seconds


def run_match_block(
    agent_kind,
    opponent_kind,
    T,
    M,
    trials,
    toss_mode,
    tie_value,
    seed,
    agent_bundle,
    opponent_bundle,
    progress_log_interval,
):
    rng = np.random.default_rng(seed)
    max_score = T * M

    agent, agent_init_s = agent_bundle
    opponent, opp_init_s = opponent_bundle

    wins = 0
    losses = 0
    ties = 0

    started = time.perf_counter()

    for trial_idx in range(1, trials + 1):
        if toss_mode == "mixed":
            agent_bats_first = bool(rng.random() < 0.5)
        elif toss_mode == "bat_first":
            agent_bats_first = True
        elif toss_mode == "bat_second":
            agent_bats_first = False
        else:
            raise ValueError(f"Unknown toss mode: {toss_mode}")

        winner = simulate_full_game(
            agent,
            opponent,
            agent1_bats_first=agent_bats_first,
            T=T,
            max_score=max_score,
        )

        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            ties += 1

        if progress_log_interval and (trial_idx % progress_log_interval == 0 or trial_idx == trials):
            elapsed = time.perf_counter() - started
            score = wins + tie_value * ties
            print(
                "[match-progress] "
                f"agent={agent_kind} opponent={opponent_kind} T={T} M={M} "
                f"toss={toss_mode} trial={trial_idx}/{trials} "
                f"score_rate={score / trial_idx:.4f} elapsed_s={elapsed:.2f}"
            )

    play_seconds = time.perf_counter() - started
    score = wins + tie_value * ties

    return {
        "agent": agent_kind,
        "opponent": opponent_kind,
        "T": T,
        "M": M,
        "max_score": max_score,
        "trials": trials,
        "toss_mode": toss_mode,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "score_rate": score / trials,
        "win_rate": wins / trials,
        "tie_rate": ties / trials,
        "loss_rate": losses / trials,
        "agent_init_seconds": agent_init_s,
        "opponent_init_seconds": opp_init_s,
        "play_seconds": play_seconds,
        "seconds_per_match": play_seconds / trials,
    }


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def initialize_csv(path, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_rows(path, fieldnames, rows):
    if not rows:
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(rows)


def save_csv(rows, csv_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(rows, opponent, sweep_type, toss_mode):
    return [
        row for row in rows
        if row["opponent"] == opponent
        and row["sweep_type"] == sweep_type
        and row["toss_mode"] == toss_mode
    ]


def _plot_score_panel(ax, rows, opponent, sweep_type, x_key, title):
    panel_rows = _select_rows(rows, opponent, sweep_type, "mixed")
    if not panel_rows:
        ax.set_visible(False)
        return

    for agent in AGENT_ORDER:
        agent_rows = [row for row in panel_rows if row["agent"] == agent]
        agent_rows.sort(key=lambda row: row[x_key])
        xs = [row[x_key] for row in agent_rows]
        ys = [row["score_rate"] for row in agent_rows]
        ax.plot(xs, ys, marker="o", label=agent)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel("Score rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)


def _plot_runtime_panel(ax, rows, sweep_type, x_key, title):
    panel_rows = [row for row in rows if row["sweep_type"] == sweep_type and row["toss_mode"] == "mixed"]
    if not panel_rows:
        ax.set_visible(False)
        return

    for agent in AGENT_ORDER:
        agent_rows = [
            row for row in panel_rows
            if row["agent"] == agent and row["opponent"] == "random"
        ]
        agent_rows.sort(key=lambda row: row[x_key])
        xs = [row[x_key] for row in agent_rows]
        ys = [1000.0 * row["seconds_per_match"] for row in agent_rows]
        ax.plot(xs, ys, marker="o", label=agent)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel("ms / match")
    ax.grid(True, alpha=0.3)


def save_plots(rows, output_dir):
    score_fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_score_panel(
        axes[0, 0],
        rows,
        opponent="random",
        sweep_type="vary_T",
        x_key="T",
        title="Vs Random across T",
    )
    _plot_score_panel(
        axes[0, 1],
        rows,
        opponent="optimal",
        sweep_type="vary_T",
        x_key="T",
        title="Vs Optimal across T",
    )
    _plot_score_panel(
        axes[1, 0],
        rows,
        opponent="random",
        sweep_type="vary_M",
        x_key="M",
        title="Vs Random across M",
    )
    _plot_score_panel(
        axes[1, 1],
        rows,
        opponent="optimal",
        sweep_type="vary_M",
        x_key="M",
        title="Vs Optimal across M",
    )
    axes[0, 0].legend()
    score_fig.savefig(output_dir / "finite_agent_score_overview.png", dpi=160)
    plt.close(score_fig)

    runtime_fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    _plot_runtime_panel(axes[0], rows, sweep_type="vary_T", x_key="T", title="Runtime across T")
    _plot_runtime_panel(axes[1], rows, sweep_type="vary_M", x_key="M", title="Runtime across M")
    axes[0].legend()
    runtime_fig.savefig(output_dir / "finite_agent_runtime_overview.png", dpi=160)
    plt.close(runtime_fig)


def print_summary(rows):
    print("\nSummary: mixed toss score rate")
    print("agent,opponent,sweep,setting,score_rate,ms_per_match")

    for sweep_type, x_key in (("vary_T", "T"), ("vary_M", "M")):
        filtered = [row for row in rows if row["toss_mode"] == "mixed" and row["sweep_type"] == sweep_type]
        filtered.sort(key=lambda row: (row["opponent"], row["agent"], row[x_key]))
        for row in filtered:
            setting = f"{x_key}={row[x_key]}, M={row['M']}" if x_key == "T" else f"T={row['T']}, {x_key}={row[x_key]}"
            print(
                f"{row['agent']},{row['opponent']},{sweep_type},{setting},"
                f"{row['score_rate']:.4f},{1000.0 * row['seconds_per_match']:.3f}"
            )


def _clip_state(value, max_score):
    return int(max(0, min(value, max_score)))


def _nash_role_value(agent, role, t, state):
    state = _clip_state(state, agent.max_score)

    if role == "bat_first":
        return float(agent.W[t, state])
    if role == "bowl_first":
        return float(1.0 - agent.W[t, state])
    if role == "bat_second":
        return float(agent.V[t, state])
    if role == "bowl_second":
        return float(1.0 - agent.V[t, state])
    raise ValueError(f"Unknown role: {role}")


def _best_response_role_value(agent, role, t, state):
    state = _clip_state(state, agent.max_score)
    return float(agent.values[role][t, state])


def _adaptive_role_value(agent, source, role, t, state):
    state = _clip_state(state, agent.max_score)

    if source == "nash":
        return _nash_role_value(agent.nash_agent, role, t, state)
    if source == "adaptive_best_response":
        return float(agent.br_values[role][t, state])
    raise ValueError(f"Unknown adaptive source: {source}")


def tracked_state_specs(T, max_score, state_fracs):
    half_t = max(1, T // 2)
    specs = [
        ("bat_first", T, 0, "start"),
        ("bowl_first", T, 0, "start"),
        ("bat_first", half_t, 0, "mid"),
        ("bowl_first", half_t, 0, "mid"),
    ]

    for frac in state_fracs:
        needed = max(1, min(max_score, int(round(frac * max_score))))
        specs.append(("bat_second", T, needed, f"start_k_{frac:g}"))
        specs.append(("bowl_second", T, needed, f"start_k_{frac:g}"))
        specs.append(("bat_second", half_t, needed, f"mid_k_{frac:g}"))
        specs.append(("bowl_second", half_t, needed, f"mid_k_{frac:g}"))

    return specs


def collect_value_rows(
    side,
    entity_kind,
    entity,
    agent_kind,
    opponent_kind,
    sweep_type,
    toss_mode,
    T,
    M,
    trials,
    checkpoint,
    state_fracs,
):
    max_score = T * M
    specs = tracked_state_specs(T, max_score, state_fracs)
    rows = []

    if isinstance(entity, RandomAgent):
        return rows

    if isinstance(entity, OptimalAgent):
        sources = ("nash",)
    elif isinstance(entity, BestResponseFiniteAgent):
        sources = ("best_response",)
    elif isinstance(entity, AdaptiveFiniteAgent):
        sources = ("nash", "adaptive_best_response")
    else:
        return rows

    for source in sources:
        for role, t, state, state_label in specs:
            if isinstance(entity, OptimalAgent):
                value = _nash_role_value(entity, role, t, state)
            elif isinstance(entity, BestResponseFiniteAgent):
                value = _best_response_role_value(entity, role, t, state)
            else:
                value = _adaptive_role_value(entity, source, role, t, state)

            rows.append(
                {
                    "side": side,
                    "entity_kind": entity_kind,
                    "agent": agent_kind,
                    "opponent": opponent_kind,
                    "sweep_type": sweep_type,
                    "toss_mode": toss_mode,
                    "T": T,
                    "M": M,
                    "trials": trials,
                    "checkpoint": checkpoint,
                    "value_source": source,
                    "role": role,
                    "t": t,
                    "state": _clip_state(state, max_score),
                    "state_label": state_label,
                    "value": value,
                }
            )

    return rows


def needs_post_snapshot(entity):
    return isinstance(entity, AdaptiveFiniteAgent)


def benchmark_suite(args):
    rows = []
    state_rows = []
    seed = args.seed
    total_blocks = (
        len(args.T_values) * len(OPPONENT_ORDER) * len(AGENT_ORDER) * len(args.toss_modes)
        + len(args.M_values) * len(OPPONENT_ORDER) * len(AGENT_ORDER) * len(args.toss_modes)
    )
    block_index = 0
    benchmark_fieldnames = [
        "agent",
        "opponent",
        "T",
        "M",
        "max_score",
        "trials",
        "toss_mode",
        "wins",
        "losses",
        "ties",
        "score_rate",
        "win_rate",
        "tie_rate",
        "loss_rate",
        "agent_init_seconds",
        "opponent_init_seconds",
        "play_seconds",
        "seconds_per_match",
        "sweep_type",
    ]
    value_fieldnames = [
        "side",
        "entity_kind",
        "agent",
        "opponent",
        "sweep_type",
        "toss_mode",
        "T",
        "M",
        "trials",
        "checkpoint",
        "value_source",
        "role",
        "t",
        "state",
        "state_label",
        "value",
    ]

    benchmark_csv = args.output_dir / "finite_agent_benchmark_results.csv"
    value_csv = args.output_dir / "finite_agent_value_snapshots.csv"
    initialize_csv(benchmark_csv, benchmark_fieldnames)
    if args.save_value_snapshots:
        initialize_csv(value_csv, value_fieldnames)

    for T in args.T_values:
        max_score = T * args.fixed_M
        print(f"[setup] Building cached agents for vary_T with T={T}, M={args.fixed_M}")
        agent_cache = {
            kind: build_agent(kind, T, args.fixed_M, max_score, args.tie_value)
            for kind in set(AGENT_ORDER + OPPONENT_ORDER)
        }
        for opponent in OPPONENT_ORDER:
            for agent in AGENT_ORDER:
                for toss_mode in args.toss_modes:
                    block_index += 1
                    agent_obj, _ = agent_cache[agent]
                    opponent_obj, _ = agent_cache[opponent]
                    print(
                        "[block-start] "
                        f"{block_index}/{total_blocks} sweep=vary_T "
                        f"agent={agent} opponent={opponent} T={T} M={args.fixed_M} toss={toss_mode}"
                    )
                    value_batch = []

                    if args.save_value_snapshots:
                        pre_rows = []
                        pre_rows.extend(
                            collect_value_rows(
                                "agent",
                                agent,
                                agent_obj,
                                agent,
                                opponent,
                                "vary_T",
                                toss_mode,
                                T,
                                args.fixed_M,
                                args.trials,
                                "pre_matches",
                                args.state_fracs,
                            )
                        )
                        pre_rows.extend(
                            collect_value_rows(
                                "opponent",
                                opponent,
                                opponent_obj,
                                agent,
                                opponent,
                                "vary_T",
                                toss_mode,
                                T,
                                args.fixed_M,
                                args.trials,
                                "pre_matches",
                                args.state_fracs,
                            )
                        )
                        value_batch.extend(pre_rows)
                        state_rows.extend(pre_rows)

                    row = run_match_block(
                        agent_kind=agent,
                        opponent_kind=opponent,
                        T=T,
                        M=args.fixed_M,
                        trials=args.trials,
                        toss_mode=toss_mode,
                        tie_value=args.tie_value,
                        seed=seed,
                        agent_bundle=agent_cache[agent],
                        opponent_bundle=agent_cache[opponent],
                        progress_log_interval=args.progress_log_interval,
                    )
                    row["sweep_type"] = "vary_T"
                    rows.append(row)
                    append_csv_rows(benchmark_csv, benchmark_fieldnames, [row])
                    print(
                        "[block-done] "
                        f"{block_index}/{total_blocks} sweep=vary_T "
                        f"agent={agent} opponent={opponent} T={T} M={args.fixed_M} toss={toss_mode} "
                        f"score_rate={row['score_rate']:.4f} play_s={row['play_seconds']:.2f}"
                    )

                    if args.save_value_snapshots:
                        post_rows = []
                        if needs_post_snapshot(agent_obj):
                            post_rows.extend(
                                collect_value_rows(
                                    "agent",
                                    agent,
                                    agent_obj,
                                    agent,
                                    opponent,
                                    "vary_T",
                                    toss_mode,
                                    T,
                                    args.fixed_M,
                                    args.trials,
                                    "post_matches",
                                    args.state_fracs,
                                )
                            )
                        if needs_post_snapshot(opponent_obj):
                            post_rows.extend(
                                collect_value_rows(
                                    "opponent",
                                    opponent,
                                    opponent_obj,
                                    agent,
                                    opponent,
                                    "vary_T",
                                    toss_mode,
                                    T,
                                    args.fixed_M,
                                    args.trials,
                                    "post_matches",
                                    args.state_fracs,
                                )
                            )
                        value_batch.extend(post_rows)
                        state_rows.extend(post_rows)
                        append_csv_rows(value_csv, value_fieldnames, value_batch)
                    seed += 1

    for M in args.M_values:
        max_score = args.fixed_T * M
        print(f"[setup] Building cached agents for vary_M with T={args.fixed_T}, M={M}")
        agent_cache = {
            kind: build_agent(kind, args.fixed_T, M, max_score, args.tie_value)
            for kind in set(AGENT_ORDER + OPPONENT_ORDER)
        }
        for opponent in OPPONENT_ORDER:
            for agent in AGENT_ORDER:
                for toss_mode in args.toss_modes:
                    block_index += 1
                    agent_obj, _ = agent_cache[agent]
                    opponent_obj, _ = agent_cache[opponent]
                    print(
                        "[block-start] "
                        f"{block_index}/{total_blocks} sweep=vary_M "
                        f"agent={agent} opponent={opponent} T={args.fixed_T} M={M} toss={toss_mode}"
                    )
                    value_batch = []

                    if args.save_value_snapshots:
                        pre_rows = []
                        pre_rows.extend(
                            collect_value_rows(
                                "agent",
                                agent,
                                agent_obj,
                                agent,
                                opponent,
                                "vary_M",
                                toss_mode,
                                args.fixed_T,
                                M,
                                args.trials,
                                "pre_matches",
                                args.state_fracs,
                            )
                        )
                        pre_rows.extend(
                            collect_value_rows(
                                "opponent",
                                opponent,
                                opponent_obj,
                                agent,
                                opponent,
                                "vary_M",
                                toss_mode,
                                args.fixed_T,
                                M,
                                args.trials,
                                "pre_matches",
                                args.state_fracs,
                            )
                        )
                        value_batch.extend(pre_rows)
                        state_rows.extend(pre_rows)

                    row = run_match_block(
                        agent_kind=agent,
                        opponent_kind=opponent,
                        T=args.fixed_T,
                        M=M,
                        trials=args.trials,
                        toss_mode=toss_mode,
                        tie_value=args.tie_value,
                        seed=seed,
                        agent_bundle=agent_cache[agent],
                        opponent_bundle=agent_cache[opponent],
                        progress_log_interval=args.progress_log_interval,
                    )
                    row["sweep_type"] = "vary_M"
                    rows.append(row)
                    append_csv_rows(benchmark_csv, benchmark_fieldnames, [row])
                    print(
                        "[block-done] "
                        f"{block_index}/{total_blocks} sweep=vary_M "
                        f"agent={agent} opponent={opponent} T={args.fixed_T} M={M} toss={toss_mode} "
                        f"score_rate={row['score_rate']:.4f} play_s={row['play_seconds']:.2f}"
                    )

                    if args.save_value_snapshots:
                        post_rows = []
                        if needs_post_snapshot(agent_obj):
                            post_rows.extend(
                                collect_value_rows(
                                    "agent",
                                    agent,
                                    agent_obj,
                                    agent,
                                    opponent,
                                    "vary_M",
                                    toss_mode,
                                    args.fixed_T,
                                    M,
                                    args.trials,
                                    "post_matches",
                                    args.state_fracs,
                                )
                            )
                        if needs_post_snapshot(opponent_obj):
                            post_rows.extend(
                                collect_value_rows(
                                    "opponent",
                                    opponent,
                                    opponent_obj,
                                    agent,
                                    opponent,
                                    "vary_M",
                                    toss_mode,
                                    args.fixed_T,
                                    M,
                                    args.trials,
                                    "post_matches",
                                    args.state_fracs,
                                )
                            )
                        value_batch.extend(post_rows)
                        state_rows.extend(post_rows)
                        append_csv_rows(value_csv, value_fieldnames, value_batch)
                    seed += 1

    return rows, state_rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark finite-horizon hand-cricket agents."
    )
    parser.add_argument("--trials", type=int, default=100, help="Matches per configuration.")
    parser.add_argument(
        "--progress-log-interval",
        type=parse_int_or_none,
        default=25,
        help="Print per-match progress every N trials within a block. Use none/off/0 to disable.",
    )
    parser.add_argument("--fixed-T", type=int, default=12, help="T used when sweeping M.")
    parser.add_argument("--fixed-M", type=int, default=6, help="M used when sweeping T.")
    parser.add_argument("--T-values", type=parse_int_list, default=parse_int_list("4,8,12,16,20"))
    parser.add_argument("--M-values", type=parse_int_list, default=parse_int_list("3,4,5,6,8,10"))
    parser.add_argument("--tie-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--toss-modes",
        type=parse_str_list,
        default=list(TOSS_MODES),
        help="Comma-separated subset of mixed,bat_first,bat_second.",
    )
    parser.add_argument(
        "--state-fracs",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,0.75"),
        help="Second-innings state fractions to snapshot.",
    )
    parser.add_argument(
        "--save-value-snapshots",
        action="store_true",
        help="Store pre/post value snapshots for tracked states.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs") / "finite_agents",
        help="Directory for CSV and plots.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    invalid_toss_modes = [mode for mode in args.toss_modes if mode not in TOSS_MODES]
    if invalid_toss_modes:
        raise SystemExit(f"Invalid toss modes: {invalid_toss_modes}")

    ensure_dir(args.output_dir)
    rows, _state_rows = benchmark_suite(args)
    csv_path = args.output_dir / "finite_agent_benchmark_results.csv"
    save_plots(rows, args.output_dir)
    print_summary(rows)

    print(f"\nSaved results to {csv_path}")
    if args.save_value_snapshots:
        print(f"Saved value snapshots to {args.output_dir / 'finite_agent_value_snapshots.csv'}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
