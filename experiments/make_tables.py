"""
Table generation for Chapter 6.

Table 1: 14 rows (7 scenarios x 2 light modes), a SINGLE representative run
         per (scenario, light_mode) - seed=42, fixed and reproducible.
         This is a "raw baseline" table, not statistically averaged.

Table 2: 14 rows, computed over the FIRST N=20 seeds (1000-1019, a subset of
         the same 30-seed study) - mean, std, and 96% CI per metric.
         NOTE: 96% CI (not the more conventional 95%) per explicit request -
         uses the t-distribution critical value at alpha=0.04 (two-tailed),
         df = N-1 = 19.
"""
import pandas as pd
import numpy as np
from scipy import stats

METRIC_KEYS = [
    'avg_travel', 'min_travel', 'max_travel',
    'avg_wait', 'min_wait', 'max_wait',
    'avg_reservation_wait', 'min_reservation_wait', 'max_reservation_wait',
    'avg_utility', 'min_utility', 'max_utility',
    'collisions', 'completed_trips', 'vehicles_spawned',
]

# =====================================================
# Table 1 - single representative run (seed=42), 14 rows
# =====================================================
rows = []
for mode in ('fixed', 'adaptive'):
    for sid in SCENARIOS:
        r = run_scenario_headless(sid, light_mode=mode, seed=42)
        r['light_mode'] = mode
        rows.append(r)

table1 = pd.DataFrame(rows)[['scenario', 'label', 'light_mode'] + METRIC_KEYS]
table1 = table1.sort_values(['scenario', 'light_mode']).reset_index(drop=True)
table1.to_csv('table1_all_14_scenarios_single_run_seed42.csv', index=False)
print('Table 1 (single run, seed=42):')
print(table1[['scenario', 'light_mode', 'avg_travel', 'avg_wait', 'avg_reservation_wait',
               'avg_utility', 'collisions']].to_string(index=False))


# =====================================================
# Table 2 - N=20 seeds (first 20 of the 30-seed study), mean/std/96% CI
# =====================================================
N = 20
SEEDS_20 = list(range(1000, 1000 + N))

rows2 = []
for mode in ('fixed', 'adaptive'):
    for sid in SCENARIOS:
        for s in SEEDS_20:
            r = run_scenario_headless(sid, light_mode=mode, seed=s)
            r['light_mode'] = mode
            r['seed'] = s
            rows2.append(r)

raw20 = pd.DataFrame(rows2)

def mean_std_ci96(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = values.mean()
    sd = values.std(ddof=1) if n > 1 else 0.0
    if n < 2:
        return m, sd, 0.0
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.98, df=n - 1)   # 96% two-tailed -> alpha/2 = 0.02 -> quantile 0.98
    return m, sd, tcrit * se

table2_rows = []
for (sid, mode), grp in raw20.groupby(['scenario', 'light_mode']):
    row = {'scenario': sid, 'label': grp['label'].iloc[0], 'light_mode': mode, 'n_seeds': len(grp)}
    for k in METRIC_KEYS:
        m, sd, ci = mean_std_ci96(grp[k].values)
        row[f'{k}_mean'] = m
        row[f'{k}_std'] = sd
        row[f'{k}_ci96'] = ci
    table2_rows.append(row)

table2 = pd.DataFrame(table2_rows).sort_values(['scenario', 'light_mode']).reset_index(drop=True)
table2.to_csv('table2_20seeds_mean_std_ci96.csv', index=False)
print('\nTable 2 (N=20 seeds, mean / std / 96% CI) - key columns:')
show_cols = ['scenario', 'light_mode', 'n_seeds',
             'avg_travel_mean', 'avg_travel_std', 'avg_travel_ci96',
             'avg_wait_mean', 'avg_wait_std', 'avg_wait_ci96']
print(table2[show_cols].round(2).to_string(index=False))

print('\nWrote table1_all_14_scenarios_single_run_seed42.csv and table2_20seeds_mean_std_ci96.csv')
