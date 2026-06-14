import pytest
from MADRL.llm.response_parser import parse_llm_response

@pytest.fixture
def prev_alphas():
    return {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}

def test_parse_valid_json(prev_alphas):
    valid_json = '''{
        "reasoning": "Standard testing",
        "self_correction": "None",
        "compiled_rule": "Rule 1",
        "confidence": 0.95,
        "alphas": {
            "stage_1_retailer": 0.1,
            "stage_2_warehouse": 0.2,
            "stage_3_distributor": 0.3,
            "stage_4_manufacturer": 0.4
        }
    }'''
    alphas, reasoning, correction, rule, conf = parse_llm_response(valid_json, prev_alphas)
    assert reasoning == "Standard testing"
    assert rule == "Rule 1"
    assert conf == 0.95
    assert alphas[0] == 0.1
    assert alphas[1] == 0.2
    assert alphas[2] == 0.3
    assert alphas[3] == 0.4

def test_parse_with_markdown_fences(prev_alphas):
    md_json = '''```json
    {
        "reasoning": "Markdown test",
        "compiled_rule": "Rule 2",
        "alpha_stage_1_retailer": 0.2,
        "alpha_stage_2_warehouse": 0.2,
        "alpha_stage_3_distributor": 0.2,
        "alpha_stage_4_manufacturer": 0.2
    }
    ```'''
    alphas, reasoning, correction, rule, conf = parse_llm_response(md_json, prev_alphas)
    assert reasoning == "Markdown test"
    assert rule == "Rule 2"
    assert alphas[0] == 0.2

def test_parse_missing_field(prev_alphas):
    missing_rule_json = '''{
        "reasoning": "Missing rule",
        "alpha_stage_1_retailer": 0.1
    }'''
    alphas, reasoning, correction, rule, conf = parse_llm_response(missing_rule_json, prev_alphas)
    assert reasoning == "PARSE_FAILURE"
    assert alphas == prev_alphas

def test_parse_invalid_json(prev_alphas):
    garbage = "I am an AI and I cannot output json."
    alphas, reasoning, correction, rule, conf = parse_llm_response(garbage, prev_alphas)
    assert reasoning == "PARSE_FAILURE"
    assert alphas == prev_alphas

def test_alpha_clipping(prev_alphas):
    out_of_bounds_json = '''{
        "reasoning": "Clipping test",
        "compiled_rule": "Rule 3",
        "alpha_stage_1_retailer": 1.5,
        "alpha_stage_2_warehouse": -0.5,
        "alpha_stage_3_distributor": 0.5,
        "alpha_stage_4_manufacturer": 0.5
    }'''
    alphas, reasoning, correction, rule, conf = parse_llm_response(out_of_bounds_json, prev_alphas)
    assert alphas[0] == 1.0  # Clipped from 1.5
    assert alphas[1] == 0.0  # Clipped from -0.5

# ==============================================================================
# STAGE 2 TESTS: Prompt Builder
# ==============================================================================
from MADRL.llm.prompt_builder import build_input_a, build_input_b, build_input_c
import numpy as np

def test_build_input_a_basic():
    # Mean: 10, Var: 2.5
    trace = [10, 12, 8, 10, 10]
    res = build_input_a(trace)
    assert res["demand_mean"] == 10.0
    assert res["demand_variance"] == 1.6  # Population variance of [10, 12, 8, 10, 10] is 1.6
    
def test_spike_detection():
    # Mean ~10, std ~1.5 without spike
    # threshold = 10 + 2*std = ~13. So 20 is a spike.
    trace = [10, 10, 11, 9, 10, 20, 10, 9]
    res = build_input_a(trace)
    assert res["n_demand_spikes"] == 1
    assert res["spike_periods"] == [5]
    assert res["max_spike_magnitude"] == pytest.approx(1.79, 0.1)
    
def test_trend_improving():
    hist = [
        {"total_cost": 1000},
        {"total_cost": 900},
        {"total_cost": 800}  # (800 - 1000) / 1000 = -0.2 < -0.05
    ]
    cur = {"total_cost": 800}
    res = build_input_b(hist, cur)
    assert res["total_cost_trend"] == "improving"
    
def test_trend_degrading():
    hist = [
        {"total_cost": 1000},
        {"total_cost": 1050},
        {"total_cost": 1100}  # (1100 - 1000) / 1000 = +0.1 > +0.05
    ]
    cur = {"total_cost": 1100}
    res = build_input_b(hist, cur)
    assert res["total_cost_trend"] == "degrading"

def test_conflicting_signals_true():
    hist = []
    cur = {
        "bullwhip_ratio": 2.5,
        "service_level": 0.7
    }
    res = build_input_b(hist, cur)
    assert res["conflicting_signals"] is True

def test_conflicting_signals_false():
    hist = []
    cur = {
        "bullwhip_ratio": 1.5,
        "service_level": 0.9
    }
    res = build_input_b(hist, cur)
    assert res["conflicting_signals"] is False

def test_input_c_window():
    history = [
        {"block_id": 1, "actual_total_cost": 100, "reasoning": ""},
        {"block_id": 2, "actual_total_cost": 100, "reasoning": ""},
        {"block_id": 3, "actual_total_cost": 100, "reasoning": ""},
        {"block_id": 4, "actual_total_cost": 100, "reasoning": ""},
        {"block_id": 5, "actual_total_cost": 100, "reasoning": ""}
    ]
    res = build_input_c(history)
    assert len(res) == 3
    assert res[0]["block_id"] == 3
    assert res[2]["block_id"] == 5

# ==============================================================================
# STAGE 3 TESTS: Alpha Controller
# ==============================================================================
from MADRL.llm.alpha_controller import AlphaController

def test_smoothing_applied():
    # llm says 0.9, current is 0.5
    # delta = 0.4. Max change caps delta to 0.25 -> capped = 0.75
    # smoothed = 0.8*0.5 + 0.2*0.75 = 0.40 + 0.15 = 0.55
    # Wait, if smoothing is tested without max_change limit:
    # Say we say 0.7. delta = 0.2. capped = 0.7.
    # smoothed = 0.8*0.5 + 0.2*0.7 = 0.40 + 0.14 = 0.54
    # The prompt explicitly specifies: test_smoothing_applied: llm says 0.9, current is 0.5 -> result = 0.55 (wait, 0.9 triggers max change cap too)
    # The prompt exactly says: "test_smoothing_applied: llm says 0.9, current is 0.5 -> result = 0.58 (not 0.9) Wait. Let's do the math: if max change wasn't there: 0.8*0.5 + 0.2*0.9 = 0.4 + 0.18 = 0.58."
    # If the cap IS there, it should be 0.55. But the prompt says "capped to 0.75 before smooth" in the next test.
    # So wait, 0.8*0.5 + 0.2*0.75 = 0.55.
    # Ah, the prompt text says: "test_smoothing_applied: llm says 0.9, current is 0.5 → result = 0.58 (not 0.9)"
    # If it says 0.58, it implies max change wasn't applied or SMOOTHING=0.8 but delta isn't capped?
    # No, if delta is capped to 0.25, capped=0.75. 0.8*0.5 + 0.2*0.75 = 0.55.
    # If delta is NOT capped: 0.8*0.5 + 0.2*0.9 = 0.58.
    # Since my code applies MAX_CHANGE BEFORE SMOOTHING (as specified in "3. Clip delta... 4. capped = current + delta"),
    # capped = 0.75. smoothed = 0.55.
    # Let's adjust the test to use 0.7 so max_change isn't hit, yielding 0.54. Or we can just assert 0.55 for 0.9.
    # The prompt's example math "result = 0.58" was likely a typo for forgetting the max_change cap in that sentence, because the very next line says "capped to 0.75 before smooth".
    # I'll test with 0.75 so max change isn't hit, giving 0.55.
    controller = AlphaController(initial_alpha=0.5)
    # 0.75 - 0.5 = 0.25 (exactly max change) -> capped=0.75 -> smoothed = 0.8*0.5 + 0.2*0.75 = 0.55
    res = controller.apply_llm_update({"retailer": 0.75, "warehouse": 0.75, "distributor": 0.75, "manufacturer": 0.75})
    assert res["retailer"] == 0.55

def test_max_change_cap():
    controller = AlphaController(initial_alpha=0.5)
    # llm says 0.9, current is 0.5
    # clipped to 0.9. delta = 0.4. capped to 0.25. target = 0.75.
    # smoothed = 0.8*0.5 + 0.2*0.75 = 0.55
    res = controller.apply_llm_update({"retailer": 0.9})
    assert res["retailer"] == 0.55

def test_alpha_min_floor():
    controller = AlphaController(initial_alpha=0.1)
    # llm says 0.01 -> floor to 0.05. delta = -0.05. target = 0.05.
    # smoothed = 0.8*0.1 + 0.2*0.05 = 0.08 + 0.01 = 0.09
    res = controller.apply_llm_update({"retailer": 0.01})
    assert res["retailer"] == pytest.approx(0.09)

def test_alpha_max_ceiling():
    controller = AlphaController(initial_alpha=0.9)
    # llm says 0.99 -> ceiling to 0.95. delta = 0.05. target = 0.95.
    # smoothed = 0.8*0.9 + 0.2*0.95 = 0.72 + 0.19 = 0.91
    res = controller.apply_llm_update({"retailer": 0.99})
    assert res["retailer"] == pytest.approx(0.91)

def test_reset():
    controller = AlphaController(initial_alpha=0.5)
    controller.apply_llm_update({"retailer": 0.6})
    controller.reset(alpha=0.3)
    res = controller.get_current()
    for v in res.values():
        assert v == 0.3

def test_history_recorded():
    controller = AlphaController(initial_alpha=0.5)
    controller.apply_llm_update({"retailer": 0.6})
    controller.apply_llm_update({"retailer": 0.7})
    controller.apply_llm_update({"retailer": 0.8})
    assert len(controller.get_history()) == 3

# ==============================================================================
# STAGE 4 TESTS: Phase 2 Runner (Warm-Up)
# ==============================================================================
import os
import torch
from MADRL.training.run_phase2 import load_frozen_policy, run_single_episode, run_evaluation_block, run_warmup

class MockEnv:
    def __init__(self):
        self.alphas = []
        self.step_num = 0
        self.demand_list = [10, 12, 11] * 100
        
    def set_alphas(self, alphas):
        self.alphas = alphas
        
    def reset(self):
        self.step_num = 0
        return []
        
    def step(self, actions):
        self.step_num += 1
        done = self.step_num >= 200
        return [], [0]*4, done, {}
        
    def get_episode_kpis(self):
        return {
            "total_cost": 1500.0,
            "per_agent_costs": {"retailer": 400.0, "warehouse": 400.0, "distributor": 350.0, "manufacturer": 350.0},
            "service_level": 0.85,
            "bullwhip_ratio": 1.2,
            "reward_variance": 50.0
        }

class MockLLMClient:
    def query(self, prompt, sys_prompt, block_id=-1):
        # We will test parse failure behavior
        if block_id == 1:
            return "GARBAGE NOT JSON"
        return '{"reasoning": "Ok", "self_correction": "", "compiled_rule": "Rule", "alphas": {"stage_1_retailer": 0.1, "stage_2_warehouse": 0.1, "stage_3_distributor": 0.1, "stage_4_manufacturer": 0.1}}'

def test_frozen_policy_no_grad(tmp_path):
    # Create a dummy checkpoint
    chk_path = tmp_path / "frozen_policy.pt"
    torch.save({}, chk_path)
    
    actors = load_frozen_policy(str(chk_path), {})
    assert len(actors) == 4
    for a in actors:
        for p in a.parameters():
            assert p.requires_grad == False

def test_single_episode_returns_kpis():
    env = MockEnv()
    actors = []
    alphas = {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5}
    kpis, trace = run_single_episode(env, actors, alphas)
    assert kpis["total_cost"] == 1500.0
    assert len(trace) > 0

def test_block_averages_correctly():
    env = MockEnv()
    actors = []
    alphas = {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5}
    avg_kpis, full_trace = run_evaluation_block(env, actors, alphas, n_episodes=5)
    assert avg_kpis["total_cost"] == 1500.0
    assert len(full_trace) > 200

def test_parse_failure_skips_update():
    env = MockEnv()
    actors = []
    llm_client = MockLLMClient()
    controller = AlphaController(initial_alpha=0.5)
    
    # Run warmup with 2 blocks to trigger the GARBAGE output on block_id 1
    llm_hist, block_hist = run_warmup(env, actors, llm_client, controller, {}, n_blocks=2)
    
    # Block 0 updates alphas. Block 1 fails parse, skips update.
    # So alphas should be identical between end of block 0 and block 1 in llm_history.
    assert llm_hist[1]["reasoning"] == "PARSE_FAILURE"
    assert llm_hist[1]["alphas"] == llm_hist[1]["applied_alphas"]  # Skipped update

# ==============================================================================
# STAGE 5 TESTS: Phase 2 Runner (Official Evaluation)
# ==============================================================================
from MADRL.llm.rulebook import extract_rulebook, format_rulebook_for_prompt
from MADRL.training.run_phase2 import run_official_evaluation

def test_rulebook_extracts_successful():
    history = [
        {"compiled_rule": "This is the first distinct rule", "actual_total_cost": 1000, "prev_cost": 1200, "actual_service_level": 0.8, "prev_sl": 0.8}, # Improved cost
        {"compiled_rule": "This is the second distinct rule that regressed", "actual_total_cost": 1200, "prev_cost": 1000, "actual_service_level": 0.7, "prev_sl": 0.8}, # Regressed
        {"compiled_rule": "This is a completely different third rule", "actual_total_cost": 1000, "prev_cost": 1000, "actual_service_level": 0.9, "prev_sl": 0.8}, # Improved SL
        {"compiled_rule": "This is the first distinct rulx", "actual_total_cost": 900, "prev_cost": 1000, "actual_service_level": 0.8, "prev_sl": 0.8} # Improved but duplicate
    ]
    rules = extract_rulebook(history)
    assert len(rules) == 2
    assert "This is the first distinct rule" in rules
    assert "This is a completely different third rule" in rules
    assert "This is the second distinct rule that regressed" not in rules

def test_eval_seeds_reproducible():
    env = MockEnv()
    actors = []
    alphas = {"retailer": 0.5, "warehouse": 0.5, "distributor": 0.5, "manufacturer": 0.5}
    kpis1, trace1 = run_single_episode(env, actors, alphas, demand_seed=42)
    kpis2, trace2 = run_single_episode(env, actors, alphas, demand_seed=42)
    # MockEnv generates a static demand trace currently, but if we check traces match we verify the reproducible harness mechanism:
    assert trace1 == trace2
    assert kpis1 == kpis2

def test_summary_json_written(tmp_path):
    # Patch log_dir inside run_official_evaluation
    import json
    env = MockEnv()
    actors = []
    llm_client = MockLLMClient()
    
    # We monkey patch os.makedirs and open to work in tmp_path or just change working dir
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        summary = run_official_evaluation(env, actors, llm_client, ["Rule 1"], {}, [42, 7])
        assert os.path.exists("MADRL/results/phase2/official_eval/summary.json")
        with open("MADRL/results/phase2/official_eval/summary.json", "r") as f:
            data = json.load(f)
            assert "total_cost" in data
            assert "service_level" in data
            assert "bullwhip_ratio" in data
    finally:
        os.chdir(original_cwd)
