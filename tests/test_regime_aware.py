"""Software tests for the regime-aware supervision code.

Run from the repository root:  python -m pytest tests -v
Test IDs (T-1 ...) match Table 5.1 of the thesis.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "MADRL")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_config                                            # noqa: E402
from regime_aware.controllers import (                                   # noqa: E402
    snap_to_regime, StaticCruiserSupervisor, RuleBasedSupervisor)
from regime_aware.prompt_builder import (                                # noqa: E402
    build_window_demand_features, build_window_echelon_features)
from regime_aware.statistical_tests import holm_bonferroni               # noqa: E402
from regime_aware.demand_loader import load_evaluation_demand_traces     # noqa: E402
from regime_aware.env_wrapper import RegimeAwareSerialEnv                 # noqa: E402

TEST_DATA = os.path.join(ROOT, "test_data")
RESULTS = os.path.join(ROOT, "benchmark_results")
SUITES = [("s1", "test_demand_merton", 100), ("s2", "test_demand_stress_shock", 30),
          ("s3", "test_demand_sustained_up", 120), ("s4", "test_demand_sustained_down", 30)]
GRID = {0.1, 0.375, 0.65}


def make_env():
    args = get_config().parse_known_args([])[0]
    args.num_agents, args.episode_length = 4, 200
    return RegimeAwareSerialEnv(args)


def echelons(ret_b=0.0, whs_b=0.0, avg_b=0.0):
    names = ["retailer", "warehouse", "distributor", "manufacturer"]
    e = {n: {"current_backlog": 0.0, "avg_backlog": avg_b} for n in names}
    e["retailer"]["current_backlog"] = ret_b
    e["warehouse"]["current_backlog"] = whs_b
    return {"echelons": e, "bullwhip_amplification_pct": 0.0}


CALM = {"demand_mean": 10.0, "demand_std": 1.0, "n_demand_spikes": 0}


# ----------------------------------------------------------------- unit
def test_T1_snap_to_regime():
    assert snap_to_regime(0.2) == 0.1
    assert snap_to_regime(0.5) == 0.375
    assert snap_to_regime(0.9) == 0.65
    assert snap_to_regime(-3.0) == 0.1
    assert snap_to_regime(0.375) == 0.375


def test_T2_static_supervisor_holds_cruiser():
    s = StaticCruiserSupervisor()
    for t in range(20, 200, 20):
        a, _ = s.decide_alphas(t, 20, CALM, echelons(ret_b=50), [0.1] * 4)
        assert a == [0.375] * 4


def test_T3_rule_supervisor_order_and_limits():
    s = RuleBasedSupervisor()
    high = {"demand_mean": 18.0, "demand_std": 1.0, "n_demand_spikes": 0}
    seq = [s.decide_alphas(t, 20, high, echelons(), [0.375] * 4)[0] for t in (20, 40, 60)]
    assert seq[0] == seq[1] == [0.65, 0.65, 0.1, 0.65]          # recovery, at most twice
    assert seq[2] != [0.65, 0.65, 0.1, 0.65]                     # third window not recovery
    s.reset_episode(0)
    spiky = {"demand_mean": 10.0, "demand_std": 5.0, "n_demand_spikes": 3}
    assert s.decide_alphas(20, 20, spiky, echelons(), [0.375] * 4)[0] == [0.1] * 4
    s.reset_episode(0)
    assert s.decide_alphas(20, 20, CALM, echelons(ret_b=6.0), [0.375] * 4)[0] == [0.65, 0.1, 0.1, 0.1]
    s.reset_episode(0)
    assert s.decide_alphas(20, 20, CALM, echelons(), [0.375] * 4)[0] == [0.375] * 4


def test_T4_window_demand_features():
    d = [10] * 19 + [30]
    f = build_window_demand_features(d)
    assert f["demand_mean"] == pytest.approx(11.0)
    assert f["demand_std"] == pytest.approx(round(float(np.std(d)), 2))
    assert f["n_demand_spikes"] == 1                             # only the 30 exceeds mean + 1.8 sd
    assert f["demand_max"] == 30.0


def test_T5_window_echelon_features():
    orders = [[10.0] * 20, [10.0] * 20, [10.0] * 20, [0.0, 20.0] * 10]
    inv = [[5.0] * 20, [0.0] * 20, [0.0] * 20, [0.0] * 20]
    back = [[1.0] * 20, [0.0] * 20, [0.0] * 20, [0.0] * 20]
    e = build_window_echelon_features(orders, inv, back, [10.0] * 20, [0.375] * 4)
    assert e["per_period_cost"] == pytest.approx(6.0)            # (5 + 1) per period at the retailer
    assert e["service_level_pct"] == pytest.approx(90.0)         # 20 backlog units over 200 demanded
    assert e["bullwhip_amplification_pct"] > 0                   # volatile manufacturer, flat retailer


def test_T6_holm_correction():
    assert holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]


def test_T7_reward_formula():
    env = make_env()
    env.reset(train=True, fixed_alphas=[0.1, 0.375, 0.65, 0.375])
    env.train = True
    raw = [-4.0, -8.0, -2.0, -6.0]
    got = [r[0] for r in env.get_processed_rewards(raw)]
    mean = sum(raw) / 4.0
    want = [a * c + (1 - a) * mean for a, c in zip([0.1, 0.375, 0.65, 0.375], raw)]
    assert got == pytest.approx(want)


def test_T8_curriculum_sampling():
    env = make_env()
    np.random.seed(0)
    draws = []
    for _ in range(4000):
        env.reset(train=True)
        assert len(set(env.alphas)) == 1                         # one regime for every tier
        draws.append(env.alphas[0])
    share = {a: draws.count(a) / len(draws) for a in (0.375, 0.1, 0.65)}
    assert share[0.375] == pytest.approx(0.70, abs=0.03)
    assert share[0.1] == pytest.approx(0.15, abs=0.03)
    assert share[0.65] == pytest.approx(0.15, abs=0.03)


def test_T9_observation_carries_alpha_critic_does_not():
    env = make_env()
    obs = env.reset(train=False, fixed_alphas=[0.1, 0.375, 0.65, 0.1])
    assert all(len(o) == 8 for o in obs)
    assert [round(float(o[-1]), 3) for o in obs] == [0.1, 0.375, 0.65, 0.1]
    from regime_aware.env_wrapper import RegimeAwareSubprocVecEnv
    args = get_config().parse_known_args([])[0]
    args.num_agents, args.episode_length, args.n_rollout_threads = 4, 200, 1
    vec = RegimeAwareSubprocVecEnv(args)
    assert vec.share_observation_space[0].shape[0] == 28         # 4 tiers x 7 features, no alpha


# ----------------------------------------------------------------- integration
def test_T10_trace_loader_counts():
    for _, folder, n in SUITES:
        traces = load_evaluation_demand_traces(os.path.join(TEST_DATA, folder))
        assert len(traces) >= n
        assert all(len(t["demand"]) >= 200 for t in traces[:n])


@pytest.fixture(scope="module")
def runners():
    from regime_aware.evaluate_benchmark import BenchmarkRunner
    return {m: BenchmarkRunner(mode=m) for m in ("base_specialist", "fixed_cruiser", "rule_based")}


def test_T11_supervisor_called_every_20_periods(runners):
    r = runners["rule_based"]
    calls = []
    orig = r.controller.decide_alphas

    def spy(**kw):
        calls.append(kw["timestep"])
        return orig(**kw)
    r.controller.decide_alphas = spy
    tr = load_evaluation_demand_traces(os.path.join(TEST_DATA, "test_demand_stress_shock"))[0]
    r.run_trace(tr["trace_idx"], tr["demand"])
    r.controller.decide_alphas = orig
    assert calls == list(range(20, 200, 20))


def test_T12_llm_reply_snapping_and_fallback(tmp_path):
    sys.path.insert(0, ROOT)
    from evaluate_llm_unguided import UnguidedLLMSupervisor
    sup = UnguidedLLMSupervisor(api_key="test", log_dir=str(tmp_path))
    d = build_window_demand_features([10] * 20)
    e = build_window_echelon_features([[10.0] * 20] * 4, [[5.0] * 20] * 4, [[0.0] * 20] * 4, [10] * 20, [0.375] * 4)
    replies = {
        "valid": '```json\n{"window_diagnosis": "x", "target_mode": "HYBRID", "assigned_alphas": '
                 '{"retailer": 0.7, "warehouse": 0.12, "distributor": 0.1, "manufacturer": 0.4}, '
                 '"operational_rationale": "y"}\n```',
        "malformed": "not json at all",
        "empty": None,
    }
    out = {}
    for k, rep in replies.items():
        sup._query_openai = lambda prompt, rep=rep: rep
        sup._query_gemini = lambda prompt, rep=rep: rep
        alphas, meta = sup.decide_alphas(20, 20, d, e, [0.375] * 4)
        out[k] = (alphas, meta["fallback_triggered"])
    assert out["valid"] == ([0.65, 0.1, 0.1, 0.375], False)     # off-grid values snapped
    assert out["malformed"][1] is True and set(out["malformed"][0]) <= GRID
    assert out["empty"][1] is True and set(out["empty"][0]) <= GRID
    assert len(glob.glob(os.path.join(str(tmp_path), "*.jsonl"))) == 1


# ----------------------------------------------------------------- system
@pytest.mark.parametrize("mode", ["base_specialist", "fixed_cruiser", "rule_based"])
def test_T13_rerun_reproduces_reported_costs(runners, mode):
    ref = json.load(open(os.path.join(RESULTS, f"results_{mode}_280.json")))
    r = runners[mode]
    for key, folder, n in SUITES:
        traces = load_evaluation_demand_traces(os.path.join(TEST_DATA, folder))[:n]
        got = [r.run_trace(t["trace_idx"], t["demand"])["per_period_cost"] for t in traces]
        assert got == pytest.approx(ref[key]["trace_costs"], abs=1e-9), f"{mode} {key}"


def test_T14_evaluation_is_deterministic(runners):
    tr = load_evaluation_demand_traces(os.path.join(TEST_DATA, "test_demand_merton"))[0]
    a = runners["rule_based"].run_trace(tr["trace_idx"], tr["demand"])
    b = runners["rule_based"].run_trace(tr["trace_idx"], tr["demand"])
    assert a == b


def test_T15_llm_decision_log_audit():
    files = glob.glob(os.path.join(RESULTS, "logs", "llm_unguided_logs_280", "*.jsonl"))
    rows = [json.loads(l) for f in files for l in open(f)]
    assert len(files) == 280 and len(rows) == 2520
    assert all(set(r["assigned_alphas"]) <= GRID for r in rows)
    assert sum(bool(r["fallback_triggered"]) for r in rows) == 0


# ----------------------------------------------------------------- performance
def test_T16_trace_runtime(runners):
    tr = load_evaluation_demand_traces(os.path.join(TEST_DATA, "test_demand_merton"))[:20]
    t0 = time.perf_counter()
    for t in tr:
        runners["fixed_cruiser"].run_trace(t["trace_idx"], t["demand"])
    per_trace = (time.perf_counter() - t0) / len(tr)
    print(f"\nmean wall-clock per 200-period trace (CPU): {per_trace:.3f} s")
    assert per_trace < 1.0


# ----------------------------------------------------------------- robustness
def test_T17_empty_and_degenerate_windows():
    f = build_window_demand_features([])
    assert f["demand_mean"] == 0.0 and f["n_demand_spikes"] == 0
    g = build_window_demand_features([0] * 20)                   # zero demand, zero spread
    assert g["n_demand_spikes"] == 0
    e = build_window_echelon_features([[]] * 4, [[]] * 4, [[]] * 4, [], [0.375] * 4)
    assert e["service_level_pct"] >= 0.0


def test_T18_demand_above_order_cap(runners):
    res = runners["fixed_cruiser"].run_trace(0, [45] * 200)      # every period above the cap of 20
    assert np.isfinite(res["per_period_cost"]) and res["customer_fill_rate"] <= 100.0
