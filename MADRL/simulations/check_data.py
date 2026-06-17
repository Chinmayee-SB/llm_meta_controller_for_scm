import json

d = json.load(open('MADRL/simulations/output/demo1_disruption.json'))
ts = d['timesteps']

print("t\tM_Inv\tM_Back\tM_Ord\tSL")
for x in ts:
    if x['t'] >= 80 and x['t'] % 5 == 0:
        inv = x['inventory']['manufacturer']
        back = x['backlog']['manufacturer']
        ord_q = x['order_qty']['manufacturer']
        sl = x.get('service_level_rolling', 0)
        print(f"{x['t']}\t{inv}\t{back}\t{ord_q}\t{sl:.2f}")
