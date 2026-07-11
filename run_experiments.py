import json
import numpy as np

from Simulator import run_all_scenarios

N_RUNS = 30

fixed_runs = []
adaptive_runs = []

for i in range(N_RUNS):

    print(f"Run {i+1}/{N_RUNS}")

    fixed = run_all_scenarios(
        light_mode="fixed",
        seed=i
    )

    adaptive = run_all_scenarios(
        light_mode="adaptive",
        seed=i
    )

    fixed_runs.append(fixed)
    adaptive_runs.append(adaptive)

with open("fixed_runs.json","w") as f:
    json.dump(fixed_runs,f,indent=4)

with open("adaptive_runs.json","w") as f:
    json.dump(adaptive_runs,f,indent=4)

print("Finished.")
