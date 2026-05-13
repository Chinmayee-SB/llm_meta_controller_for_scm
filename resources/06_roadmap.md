# Project Roadmap (Final v2)

---

## Pre-Registered Config (Lock Before Coding — Commit to Git First)

```yaml
# config.yaml — ALL values sourced from MADRL paper or annotated as our addition

# ── Environment ──────────────────────────────────────────────────────────────
environment:
  n_agents: 4
  episode_length: 200            # Table G.1: T=200
  lead_time: 4                   # Table 1 "Typical": L=4
  O_max: 20                      # repo: ACTION_DIM-1
  initial_inventory: 0           # repo default
  holding_cost: 1                # Table 1: h^i=1 uniform ALL agents
  backlog_cost: 1                # Table 1: b^i=1 uniform ALL agents

# ── Demand ────────────────────────────────────────────────────────────────────
demand:
  model: merton_jd
  mu: 5.0e-5                     # Table G.2 Serial
  sigma: 1.0e-2                  # Table G.2 Serial
  lambda: 15                     # Table G.2 Serial
  l1: 10                         # Table G.2 Serial
  l2: 0                          # Table G.2 Serial
  l3: 0.01                       # Table G.2 Serial
  clip_min: 1
  clip_max: 40

# ── HAPPO ─────────────────────────────────────────────────────────────────────
happo:
  gru_num_layers: 2              # Table G.1
  gru_hidden: 128                # Table G.1
  fc_hidden: 128
  gamma: 0.95                    # Table G.1
  gae_lambda: 0.95               # repo config
  lr: 1.0e-4                     # Table G.1 (actor and critic same)
  ppo_clip: 0.2                  # Table G.1
  entropy_coeff: 0.01            # Table G.1
  ppo_epoch: 15                  # repo config
  data_chunk_length: 10          # repo config
  max_grad_norm: 0.5             # repo config
  weight_init: orthogonal
  num_env_steps: 3000000         # repo config: 3e6 steps
  n_rollout_threads: 5           # repo config

# ── Alpha (our addition) ──────────────────────────────────────────────────────
alpha:
  sampling: per_agent_independent_uniform   # NOT shared across agents
  min: 0.0
  max: 1.0
  in_actor_obs: true
  in_critic_obs: false

# ── Stability Gate ────────────────────────────────────────────────────────────
stability_gate:
  n_random_traces: 50
  n_traces_per_alpha_value: 10
  alpha_test_values: [0.0, 0.25, 0.5, 0.75, 1.0]
  min_service_level_avg: 0.80
  max_bullwhip_ratio_avg: 3.0
  max_cost_vs_basestock_ratio: 1.20
  max_backlog_per_agent: 500
  min_random_traces_passing: 45
  min_service_level_at_any_alpha: 0.75

# ── Phase 2 ───────────────────────────────────────────────────────────────────
phase2:
  block_size: 5
  warmup_blocks: 10              # 10 blocks × 5 episodes = 50 warm-up episodes
  rules_frozen_after_warmup: true
  official_eval_traces: 20
  alpha_smoothing: 0.8
  alpha_max_change_per_block: 0.25
  alpha_min: 0.05
  alpha_max: 0.95
  llm_model: claude-sonnet-4-6
  llm_temperature: 0.3
  llm_max_tokens: 500
  history_window: 3
  block_size_ablation: [1, 5, 10]

# ── Final Evaluation ──────────────────────────────────────────────────────────
evaluation:
  n_final_eval_traces: 50
  n_training_repeats: 5
  seeds: [0, 1, 2, 3, 4]
```

---

## Repository Structure

```
llm-marl-supply-chain/
├── config/
│   └── default_config.yaml          # PRE-REGISTERED — commit before training
├── env/
│   ├── supply_chain_env.py          # Core Dec-POMDP (adapted from MADRL repo)
│   ├── demand_models.py             # Merton JD + Poisson
│   └── disruption_wrapper.py
├── agents/
│   ├── networks.py                  # 2-layer GRU Actor + Centralized Critic
│   └── happo.py                     # HAPPO training loop + GAE
├── llm/
│   ├── prompt_builder.py
│   ├── llm_client.py
│   ├── response_parser.py
│   ├── alpha_controller.py
│   └── rulebook.py                  # Warm-up rule extraction and freezing
├── baselines/
│   ├── centralized_lp.py
│   ├── base_stock.py
│   ├── rule_based_alpha.py
│   └── random_alpha.py
├── training/
│   ├── train_phase1.py
│   ├── stability_gate.py
│   ├── run_phase2.py
│   └── evaluate.py
├── analysis/
│   ├── metrics.py
│   ├── visualize.py
│   └── llm_trace_analysis.py
└── results/
    ├── phase1_gate/
    ├── phase2_warmup/
    ├── phase2_rulebook/
    └── final_evaluation/
```

---

## Phase 0: Setup (Week 1, Days 1–2)

1. Clone MADRL repo: `https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management`
2. Write `config/default_config.yaml` with all pre-registered values above
3. Commit config — `git commit -m "pre-register config — do not modify after this commit"`
4. Write `config_lock.md` documenting why each value was chosen and its source
5. Set up W&B project

---

## Phase 1: Adapt MADRL Baseline + Environment (Week 1 Days 3–5)

**Changes to MADRL repo (`envs/serial.py`):**

| Change | Original | Ours | Reason |
|--------|----------|------|--------|
| LEVEL_NUM | 3 | 4 | Explicit Manufacturer agent |
| H, B costs | [1,1,1,1,1] | [1,1,1,1] | Same values, 4 agents |
| LEAD_TIME | 4 | 4 | No change needed |
| EPISODE_LEN | 200 | 200 | No change needed |
| ALPHA | fixed 0.5 | per-agent U(0,1) | Domain randomization |
| OBS_DIM | 7 | 8 (add α) | Actor conditions on α |
| get_processed_rewards | fixed α | per-agent α | Equation 7 |
| KPI tracking | minimal | full 6 KPIs | Phase 2 needs them |

**Changes to `config.py`:**

| Param | Original | Ours | Reason |
|-------|----------|------|--------|
| num_agents | 3 | 4 | 4 agents |
| All other params | as-is | as-is | Match paper exactly |

**Unit tests (must all pass before training):**
- mass conservation test
- backlog update correctness
- pipeline vector = lead_time length
- α in actor obs only (not critic)
- per-agent independent α at reset
- KPI tracking accumulates correctly

---

## Phase 2: Baselines (Week 2)

1. Centralized LP (cvxpy) — full information upper bound
2. Non-stationary base-stock — classical heuristic
3. Fixed α=0.5 HAPPO — run MADRL baseline unchanged except 4 agents
4. Rule-based α — `if bullwhip>2: upstream α-=0.1`
5. Random α — sanity check

Run all on 20 evaluation traces. Record benchmark numbers before Phase 1 training.

---

## Phase 3: Phase 1 Training (Weeks 3–4)

1. Implement 2-layer GRU actor and centralized critic
2. Implement GAE + HAPPO sequential update
3. Train with `num_env_steps=3e6`, 5 parallel rollout threads
4. Monitor convergence: expect ~327±35 episodes based on paper
5. Once training curve stable, initiate stability gate

**Debug checklist:**
- [ ] Gradient norms stable (not exploding)
- [ ] Entropy not collapsing (policy still exploring)
- [ ] Total cost decreasing over episodes
- [ ] Different α values produce visibly different ordering behaviors

---

## Phase 4: Stability Gate (Week 4 final days)

Run `stability_gate.py` with pre-registered thresholds. Must pass before Phase 2.

---

## Phase 5: Phase 2 LLM Warm-Up (Week 5)

1. Implement prompt_builder, llm_client, response_parser, alpha_controller
2. Run 10 warm-up blocks (50 episodes)
3. Extract and freeze rulebook
4. Save: `results/phase2_rulebook/llm_rulebook.json`

---

## Phase 6: Phase 2 Official Evaluation (Week 5–6)

1. Run 20-trace official evaluation with frozen rulebook
2. Compare to all 5 baselines
3. Log all LLM reasoning traces

---

## Phase 7: Disruption Experiments (Week 7)

Re-run all methods under:
- No disruption (baseline)
- Mild: prob=0.05, severity=0.5, duration=3 periods
- Severe: guaranteed at period 100, severity=0.9, duration=8 periods

---

## Phase 8: Ablations (Week 8)

| Ablation | Change | Tests |
|----------|--------|-------|
| Fixed α=0.5 | No LLM, α frozen | Value of any adaptation |
| Rule-based α | Single-signal | Value of multi-signal LLM |
| Random α | Random each block | Direction matters? |
| Block size=1 | Single episode | Noise sensitivity |
| Block size=10 | 10 episodes | Stability vs responsiveness |
| No Input C | Remove LLM history | Value of self-correction |
| No Input A | Remove demand stats | Value of demand context |

---

## Phase 9: Final Evaluation (Week 8)

- 50 hold-out traces per method
- 5 training repeats → mean ± 95% CI
- Paired t-test: LLM+MARL vs rule-based on total cost (p<0.05 for significance)

---

## Phase 10: Paper Writing (Weeks 9–10)

Sections: Introduction → Related Work → Problem Formulation → Methodology → Experiments → Discussion → Conclusion

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup + Env adaptation | Working env, unit tests pass |
| 2 | Baselines | Benchmark numbers |
| 3–4 | Phase 1 training | `gate_results.json` passed |
| 5 | Phase 2 warm-up + eval | `llm_rulebook.json` + 20-trace results |
| 6 | Phase 2 complete | All LLM traces logged |
| 7 | Disruption experiments | Recovery curves |
| 8 | Ablations + final eval | `main_table.csv` |
| 9–10 | Paper | Submitted draft |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| HAPPO doesn't converge with domain-randomized α | Medium | Start with fixed α=0.5 first, confirm convergence, then add randomization |
| Universal policy degenerate at α extremes | Low-Medium | α-coverage gate catches this |
| LLM gives invalid JSON | Low | Fallback to prev α; robust parser |
| LLM does not beat rule-based | Medium | Ablation design accounts for this; negative result is publishable |
| Phase 2 oscillates | Medium | Smoothing=0.8 + max_change=0.25 prevent this |
