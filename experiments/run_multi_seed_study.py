"""
Multi-seed study runner.

Run this as a NEW CELL, AFTER the main simulation cell has already executed
(same pattern as validation_tests.py) - it reuses SCENARIOS, SCENARIO_OVERRIDES,
run_scenario_headless, Env etc. which are already defined in the notebook's
global namespace. Does NOT modify the simulation source.

Produces (written to the Colab working directory - use the Colab file browser,
or !ls, to find/download them):
  1. raw_results.csv        - one row per (scenario, light_mode, seed) run - the
                               full raw dataset (420 rows: 7 scenarios x 2 modes x 30 seeds)
  2. aggregated_results.csv - one row per (scenario, light_mode) with mean/std/
                               95% CI (t-distribution, df=N-1) for every metric,
                               computed over the FULL 30-seed sample
  3. convergence_results.csv - for each (scenario, light_mode, N in {5,10,15,20,25,30}):
                               mean and 95% CI half-width, computed on the FIRST N
                               seeds of the same run (cumulative subsets) - this is
                               what justifies "why N=30 seeds is enough"
"""
import time
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = list(range(1000, 1030))          # 30 seeds, disjoint from validation-test seeds (2000s)
CONVERGENCE_NS = [5, 10, 15, 20, 25, 30]
LIGHT_MODES = ('fixed', 'adaptive')

METRIC_KEYS = [
    'avg_travel', 'min_travel', 'max_travel',
    'avg_wait', 'min_wait', 'max_wait',
    'avg_reservation_wait', 'min_reservation_wait', 'max_reservation_wait',
    'avg_utility', 'min_utility', 'max_utility',
    'collisions', 'completed_trips', 'vehicles_spawned',
]

# -----------------------------------------------------
# 1. Raw runs
# -----------------------------------------------------
t0 = time.time()
rows = []
for mode in LIGHT_MODES:
    for sid in SCENARIOS:
        for s in SEEDS:
            r = run_scenario_headless(sid, light_mode=mode, seed=s)
            r['light_mode'] = mode
            r['seed'] = s
            rows.append(r)
print(f'Ran {len(rows)} scenario instances in {time.time()-t0:.1f}s')

raw_df = pd.DataFrame(rows)
raw_df = raw_df[['scenario', 'label', 'light_mode', 'seed'] + METRIC_KEYS]
raw_df.to_csv('raw_results.csv', index=False)
print('Wrote raw_results.csv:', raw_df.shape)

# -----------------------------------------------------
# 2. Aggregated results (full N=30 sample) with 95% CI (t-distribution)
# -----------------------------------------------------
def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = values.mean()
    if n < 2:
        return m, 0.0, 0.0
    se = values.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    half_width = tcrit * se
    return m, values.std(ddof=1), half_width


agg_rows = []
for (sid, mode), grp in raw_df.groupby(['scenario', 'light_mode']):
    row = {'scenario': sid, 'label': grp['label'].iloc[0], 'light_mode': mode, 'n_seeds': len(grp)}
    for k in METRIC_KEYS:
        m, sd, ci = mean_ci95(grp[k].values)
        row[f'{k}_mean'] = m
        row[f'{k}_std'] = sd
        row[f'{k}_ci95'] = ci
    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows).sort_values(['scenario', 'light_mode']).reset_index(drop=True)
agg_df.to_csv('aggregated_results.csv', index=False)
print('Wrote aggregated_results.csv:', agg_df.shape)

# -----------------------------------------------------
# 3. Convergence analysis: cumulative subsets of the SAME 30-seed run
# -----------------------------------------------------
conv_rows = []
for (sid, mode), grp in raw_df.groupby(['scenario', 'light_mode']):
    grp_sorted = grp.sort_values('seed').reset_index(drop=True)
    for n in CONVERGENCE_NS:
        subset = grp_sorted.iloc[:n]
        row = {'scenario': sid, 'light_mode': mode, 'n_seeds': n}
        for k in ('avg_travel', 'avg_wait', 'avg_reservation_wait', 'avg_utility'):
            m, sd, ci = mean_ci95(subset[k].values)
            row[f'{k}_mean'] = m
            row[f'{k}_ci95'] = ci
        conv_rows.append(row)

conv_df = pd.DataFrame(conv_rows)
conv_df.to_csv('convergence_results.csv', index=False)
print('Wrote convergence_results.csv:', conv_df.shape)

print('\nDone in', round(time.time() - t0, 1), 's total')
