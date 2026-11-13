"""
verify.py

Checks that the demo engine reproduces benchmark_results exactly, per trace.
Run with: python demo/verify.py
"""

import time

import numpy as np

from engine import SUITES, Engine


def main():
    engine = Engine()
    started = time.time()
    worst = 0.0
    print(f"{'arm':<8}{'suite':<7}{'traces':<8}{'published':<12}{'live':<12}{'max |diff|':<10}")
    for arm in ("base", "static", "rules", "llm"):
        for suite in SUITES:
            published = engine.published[arm][suite]["trace_costs"]
            live = []
            for trace in range(len(published)):
                demand = engine.demand(suite, trace)
                if arm == "base":
                    run = engine.simulate("base", demand)
                elif arm == "static":
                    run = engine.simulate("curriculum", demand)
                elif arm == "rules":
                    run = engine.simulate("curriculum", demand, rules=True)
                else:
                    run = engine.simulate("curriculum", demand, engine.logged_schedule(suite, trace))
                live.append(run["cost"])
            diff = float(np.max(np.abs(np.array(live) - np.array(published))))
            worst = max(worst, diff)
            print(f"{arm:<8}{suite:<7}{len(live):<8}{np.mean(published):<12.2f}{np.mean(live):<12.2f}{diff:<10.4f}")
    print(f"\nWorst per-trace difference: {worst:.4f} cost units ({time.time() - started:.0f} s)")
    if worst > 1e-6:
        raise SystemExit("MISMATCH: the demo engine does not reproduce the published results.")
    print("PASS: every trace reproduces the published result.")


if __name__ == "__main__":
    main()
