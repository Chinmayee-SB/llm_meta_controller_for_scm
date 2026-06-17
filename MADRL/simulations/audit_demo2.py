import json

with open("MADRL/simulations/output/demo2_alpha_impact.json", "r") as f:
    data = json.load(f)

print("Smoothing Config:", data["config"]["smoothing"])
for block in data["blocks"]:
    print(f"\nBlock {block['block_id']}:")
    print("Recommended:", block["llm_output"]["alpha_recommended"])
    print("Applied:", block["alpha_after_update"])
    print("Cost:", block["block_kpis"]["total_cost"])
    print("SL:", block["block_kpis"]["service_level"])
    print("BW:", block["block_kpis"]["bullwhip_ratio"])
