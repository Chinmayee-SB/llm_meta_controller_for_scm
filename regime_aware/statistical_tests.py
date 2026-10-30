"""
statistical_tests.py

Statistical Analysis and Hypothesis Testing for the 280-Trace Benchmark.
Reproduces Table I (Headline Metrics) and Table III (Paired Differences & p-values),
plus 95% Confidence Intervals. If 'results_llm_unguided_280.json' is present,
it also evaluates the unguided LLM supervisor against Rules, Static, and Original LLM.
"""

import os
import sys
import json
import numpy as np
from scipy import stats

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark_results")
UNGUIDED_LOGS_DIR = os.path.join(RESULTS_DIR, "logs", "llm_unguided_logs_280")


def load_results():
    base_file = os.path.join(RESULTS_DIR, "results_base_specialist_280.json")
    cruiser_file = os.path.join(RESULTS_DIR, "results_fixed_cruiser_280.json")
    rule_file = os.path.join(RESULTS_DIR, "results_rule_based_280.json")
    llm_file = os.path.join(RESULTS_DIR, "results_llm_280.json")
    unguided_file = os.path.join(RESULTS_DIR, "results_llm_unguided_280.json")

    with open(base_file, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(cruiser_file, "r", encoding="utf-8") as f:
        cruiser = json.load(f)
    with open(rule_file, "r", encoding="utf-8") as f:
        rule = json.load(f)
    with open(llm_file, "r", encoding="utf-8") as f:
        llm = json.load(f)

    unguided = None
    if os.path.exists(unguided_file):
        with open(unguided_file, "r", encoding="utf-8") as f:
            unguided = json.load(f)

    return base, cruiser, rule, llm, unguided


def holm_bonferroni(p_values: list, alpha: float = 0.05) -> list:
    """Applies Holm-Bonferroni correction over a family of hypothesis tests."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    is_significant = [False] * m

    for rank, (orig_idx, p_val) in enumerate(indexed):
        threshold = alpha / float(m - rank)
        if p_val <= threshold:
            is_significant[orig_idx] = True
        else:
            break
    return is_significant


def print_table_1(base, cruiser, rule, llm, unguided=None):
    suites = [
        ("s1", "S1: Merton (In-Distribution)", 100),
        ("s2", "S2: Pulse Shock (Transient)", 30),
        ("s3", "S3: Sustained Step-Up (+90.5%)", 120),
        ("s4", "S4: Sustained Step-Down (-71.7%)", 30),
    ]

    has_ung = unguided is not None
    extra_col = " | {'LLM Unguided':<16}" if has_ung else ""
    line_len = 135 if has_ung else 115

    print("=" * line_len)
    print(" TABLE I: HEADLINE METRICS OVER 280 FROZEN TRACES")
    print("=" * line_len)
    header = f"{'Suite (N)':<34} | {'Metric':<14} | {'Base Specialist':<16} | {'Static Cruiser':<16} | {'Rule-Based':<16} | {'Zero-Shot LLM':<16}"
    if has_ung:
        header += f" | {'LLM Unguided':<16}"
    print(header)
    print("-" * line_len)

    for k, name, n in suites:
        # Cost row
        b_c = f"{base[k]['cost_mean']:.2f}"
        c_c = f"{cruiser[k]['cost_mean']:.2f}"
        r_c = f"{rule[k]['cost_mean']:.2f}"
        l_c = f"{llm[k]['cost_mean']:.2f}"
        row = f"{name + f' ({n})':<34} | {'Cost':<14} | {b_c:<16} | {c_c:<16} | {r_c:<16} | {l_c:<16}"
        if has_ung:
            u_c = f"{unguided[k]['cost_mean']:.2f}"
            row += f" | {u_c:<16}"
        print(row)

        # CFR row
        b_cfr = f"{base[k]['cfr_mean']:.1f}%"
        c_cfr = f"{cruiser[k]['cfr_mean']:.1f}%"
        r_cfr = f"{rule[k]['cfr_mean']:.1f}%"
        l_cfr = f"{llm[k]['cfr_mean']:.1f}%"
        row_cfr = f"{'':<34} | {'Fill Rate':<14} | {b_cfr:<16} | {c_cfr:<16} | {r_cfr:<16} | {l_cfr:<16}"
        if has_ung:
            u_cfr = f"{unguided[k]['cfr_mean']:.1f}%"
            row_cfr += f" | {u_cfr:<16}"
        print(row_cfr)

        # Bullwhip Amplification row
        b_amp = f"{base[k]['amp_mean']:+.1f}%"
        c_amp = f"{cruiser[k]['amp_mean']:+.1f}%"
        r_amp = f"{rule[k]['amp_mean']:+.1f}%"
        l_amp = f"{llm[k]['amp_mean']:+.1f}%"
        row_amp = f"{'':<34} | {'Bullwhip A':<14} | {b_amp:<16} | {c_amp:<16} | {r_amp:<16} | {l_amp:<16}"
        if has_ung:
            u_amp = f"{unguided[k]['amp_mean']:+.1f}%"
            row_amp += f" | {u_amp:<16}"
        print(row_amp)
        print("-" * line_len)


def print_table_3(base, cruiser, rule, llm, unguided=None):
    suites = [
        ("s1", "S1 (100)", 100),
        ("s2", "S2 (30)", 30),
        ("s3", "S3 (120)", 120),
        ("s4", "S4 (30)", 30),
    ]

    contrasts = [
        ("Static - Base", cruiser, base),
        ("Rules - Static", rule, cruiser),
        ("LLM - Static", llm, cruiser),
        ("LLM - Rules", llm, rule),
    ]

    if unguided is not None:
        contrasts.extend([
            ("Unguided - Rules", unguided, rule),
            ("Unguided - Static", unguided, cruiser),
            ("Unguided - LLM", unguided, llm),
        ])

    print("\n" + "=" * 115)
    print(" TABLE III: PAIRED DIFFERENCES ON SHARED TRACES (Delta with p-value)")
    print(" Negative favors first-named model. Asterisk (*) marks significance surviving Holm correction.")
    print("=" * 115)

    all_cost_pvals = []
    contrast_order = []
    for c_label, m1, m2 in contrasts:
        for k, s_label, n in suites:
            c1 = np.array(m1[k]["trace_costs"])
            c2 = np.array(m2[k]["trace_costs"])
            p_c = 1.0 if np.allclose(c1, c2) else float(stats.ttest_rel(c1, c2)[1])
            all_cost_pvals.append(p_c)
            contrast_order.append((c_label, k))

    cost_sig = holm_bonferroni(all_cost_pvals, alpha=0.05)
    sig_map = {contrast_order[i]: cost_sig[i] for i in range(len(contrast_order))}

    print("\n[PER-PERIOD COST CONTRASTS]")
    header = f"{'Contrast':<20} | {'S1 (100)':<22} | {'S2 (30)':<22} | {'S3 (120)':<22} | {'S4 (30)':<22}"
    print(header)
    print("-" * len(header))

    for c_label, m1, m2 in contrasts:
        row_str = f"{c_label:<20} | "
        for k, s_label, n in suites:
            c1 = np.array(m1[k]["trace_costs"])
            c2 = np.array(m2[k]["trace_costs"])
            diff_mean = float(np.mean(c1 - c2))
            p_c = 1.0 if np.allclose(c1, c2) else float(stats.ttest_rel(c1, c2)[1])
            star = "*" if sig_map.get((c_label, k), False) else ""
            cell = f"{diff_mean:+5.2f}{star} (p={p_c:.2e})"
            row_str += f"{cell:<22} | "
        print(row_str[:-2])

    print("\n[CUSTOMER FILL RATE CONTRASTS (Percentage Points)]")
    print(header)
    print("-" * len(header))
    cfr_contrasts = contrasts[:1]
    if unguided is not None:
        cfr_contrasts = [
            ("Static - Base", cruiser, base),
            ("Unguided - Rules", unguided, rule),
            ("Unguided - Static", unguided, cruiser),
            ("Unguided - LLM", unguided, llm),
        ]
    for c_label, m1, m2 in cfr_contrasts:
        row_str = f"{c_label:<20} | "
        for k, s_label, n in suites:
            cfr1 = np.array(m1[k]["trace_cfrs"])
            cfr2 = np.array(m2[k]["trace_cfrs"])
            diff_mean = float(np.mean(cfr1 - cfr2))
            p_cfr = 1.0 if np.allclose(cfr1, cfr2) else float(stats.ttest_rel(cfr1, cfr2)[1])
            cell = f"{diff_mean:+5.2f} (p={p_cfr:.2e})"
            row_str += f"{cell:<22} | "
        print(row_str[:-2])

    print("\n[PAIRED 95% CONFIDENCE INTERVALS: COST (scipy.stats.t.interval)]")
    header_ci = f"{'Contrast':<20} | {'S1 (100)':<24} | {'S2 (30)':<24} | {'S3 (120)':<24} | {'S4 (30)':<24}"
    print(header_ci)
    print("-" * len(header_ci))
    for c_label, m1, m2 in contrasts:
        row_str = f"{c_label:<20} | "
        for k, s_label, n in suites:
            c1 = np.array(m1[k]["trace_costs"])
            c2 = np.array(m2[k]["trace_costs"])
            diff = c1 - c2
            m = float(np.mean(diff))
            se = float(stats.sem(diff))
            ci = (m, m) if se < 1e-12 else stats.t.interval(0.95, len(diff) - 1, loc=m, scale=se)
            cell = f"[{ci[0]:+6.2f}, {ci[1]:+6.2f}]"
            row_str += f"{cell:<24} | "
        print(row_str[:-2])

    print("\n[CUSTOMER FILL RATE 95% CONFIDENCE INTERVALS]")
    print(header_ci)
    print("-" * len(header_ci))
    for c_label, m1, m2 in cfr_contrasts:
        cfr_row = f"{c_label:<20} | "
        for k, s_label, n in suites:
            c1 = np.array(m1[k]["trace_cfrs"])
            c2 = np.array(m2[k]["trace_cfrs"])
            diff = c1 - c2
            m = float(np.mean(diff))
            se = float(stats.sem(diff))
            ci = (m, m) if se < 1e-12 else stats.t.interval(0.95, len(diff) - 1, loc=m, scale=se)
            cell = f"[{ci[0]:+6.2f}, {ci[1]:+6.2f}]"
            cfr_row += f"{cell:<24} | "
        print(cfr_row[:-2])

    if unguided is not None and os.path.exists(UNGUIDED_LOGS_DIR):
        total_dec = 0
        fb_dec = 0
        for f in os.listdir(UNGUIDED_LOGS_DIR):
            if f.endswith(".jsonl"):
                with open(os.path.join(UNGUIDED_LOGS_DIR, f), encoding="utf-8") as fp:
                    for line in fp:
                        total_dec += 1
                        d = json.loads(line)
                        if d.get("fallback_triggered", False):
                            fb_dec += 1
        print("\n" + "=" * 115)
        print(" LLM UNGUIDED FALLBACK AUDIT")
        print("=" * 115)
        print(f"Total Decisions: {total_dec} across 280 traces")
        print(f"Total Fallbacks: {fb_dec} ({fb_dec / max(1, total_dec) * 100:.2f}%)")

    print("=" * 115)


def main():
    base, cruiser, rule, llm, unguided = load_results()
    print_table_1(base, cruiser, rule, llm, unguided)
    print_table_3(base, cruiser, rule, llm, unguided)


if __name__ == "__main__":
    main()
