import numpy as np, random as pyrandom, json, time, os, sys
os.makedirs("results/benchmark", exist_ok=True)

with open("channel_model/benchmark_ablation_fixed.py", encoding="utf-8") as f:
    src = f.read()
exec(src.split("if __name__")[0])

oma = compute_oma_throughput(DQL_OPT_POS, NEAR_DEVS, FAR_DEVS, 50, 99)
print("OMA: " + str(round(oma,2)) + " Mbps")

ddqn_r = run_ddqn(128, 2000, label="DDQN fixed")
a2c_r  = run_a2c(128, 2000)
dql_r  = run_dql(128, 2000, label="DQL proposed fixed")

print("=== FINAL RESULTS ===")
print("DDQN: " + str(round(ddqn_r["final_thr"],2)) + " Mbps  pos=" + str([round(x,1) for x in ddqn_r["best_pos"]]))
print("A2C:  " + str(round(a2c_r["final_thr"],2))  + " Mbps  pos=" + str([round(x,1) for x in a2c_r["best_pos"]]))
print("DQL:  " + str(round(dql_r["final_thr"],2))  + " Mbps  pos=" + str([round(x,1) for x in dql_r["best_pos"]]))