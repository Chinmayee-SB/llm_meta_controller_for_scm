import json

with open("MADRL/results/phase1_gate/gate_results.json") as f:
    data = json.load(f)

print(f"Traces passing: {data['random_traces']['traces_passing']}")
print("Failing traces:")
for t in data['random_traces']['per_trace_results']:
    if not t['passed']:
        print(f"Trace {t['trace_idx']} ({t['trace_type']}) SL: {t['service_level']:.3f}, BW: {t['bullwhip_ratio']:.3f}, Backlog: {t['max_backlog']} | Failures: {t['failures']}")
