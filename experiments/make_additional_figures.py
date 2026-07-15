"""
Additional figures & tables requested for Chapter 6:
  - standalone Average Travel Time / Average Wait Time figures (split from fig2)
  - Average Reservation Waiting Time figure (new)
  - Travel Time Improvement (%) figure - PAIRED analysis (same seed, fixed vs adaptive)
  - Error-bars methodology figure (raw per-seed scatter + mean + 95% CI)
  - Utility vs. traffic load figure
  - Table 1: 14 rows (7 scenarios x 2 light modes), single representative run (seed=42)
  - Table 2: 14 rows, first N=20 seeds, mean / std / 96% CI per metric
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ---- palette (same as before: no green, black/gray/blue/red/white only) ----
C_FIXED = '#4D4D4D'
C_ADAPTIVE = '#1F4E99'
C_IMPROVE = '#1F4E99'    # blue = adaptive better
C_REGRESS = '#B22222'    # red = adaptive worse
C_GRID = '#DDDDDD'
C_TEXT = '#1A1A1A'
C_POINT = '#808080'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': C_TEXT, 'text.color': C_TEXT,
    'xtick.color': C_TEXT, 'ytick.color': C_TEXT, 'axes.grid': True,
    'grid.color': C_GRID, 'grid.linewidth': 0.6, 'font.size': 11,
    'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
})

raw = pd.read_csv('raw_results.csv')
agg = pd.read_csv('aggregated_results.csv')

SCENARIO_ORDER = ['VS-01', 'VS-03', 'VS-13', 'VS-02', 'VS-14', 'VS-12', 'VS-11']
SCENARIO_LABELS = {
    'VS-01': 'VS-01\n(12 veh)', 'VS-02': 'VS-02\n(52 veh)', 'VS-03': 'VS-03\n(32 veh)',
    'VS-11': 'VS-11\n(102 veh)', 'VS-12': 'VS-12\n(77 veh)', 'VS-13': 'VS-13\n(32 veh)',
    'VS-14': 'VS-14\n(55 veh)',
}
width = 0.36
x = np.arange(len(SCENARIO_ORDER))


def grouped_bar(metric, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fixed_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')][f'{metric}_mean'].values[0] for s in SCENARIO_ORDER]
    fixed_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')][f'{metric}_ci95'].values[0] for s in SCENARIO_ORDER]
    adap_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')][f'{metric}_mean'].values[0] for s in SCENARIO_ORDER]
    adap_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')][f'{metric}_ci95'].values[0] for s in SCENARIO_ORDER]

    ax.bar(x - width/2, fixed_vals, width, yerr=fixed_ci, capsize=3, color=C_FIXED,
           label='Fixed timing', edgecolor='black', linewidth=0.4)
    ax.bar(x + width/2, adap_vals, width, yerr=adap_ci, capsize=3, color=C_ADAPTIVE,
           label='Adaptive timing', edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'figures/{filename}', dpi=200)
    plt.close(fig)
    print(f'{filename} done')


# =====================================================
# 1. Average Travel Time (standalone)
# =====================================================
grouped_bar('avg_travel', 'Seconds', 'Average Travel Time (mean ± 95% CI, N=30 seeds)',
            'fig_avg_travel_time.png')

# =====================================================
# 2. Average Wait Time (standalone)
# =====================================================
grouped_bar('avg_wait', 'Seconds', 'Average Wait Time (mean ± 95% CI, N=30 seeds)',
            'fig_avg_wait_time.png')

# =====================================================
# 3. Average Reservation Waiting Time (NEW)
# =====================================================
grouped_bar('avg_reservation_wait', 'Seconds',
            'Average Reservation Waiting Time (mean ± 95% CI, N=30 seeds)\n'
            '(subset of total wait: time spent at a stop line trying to enter an intersection)',
            'fig_avg_reservation_wait.png')


# =====================================================
# 4. Travel Time Improvement (%) - PAIRED analysis (same seed, both modes)
# =====================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
pivot = raw.pivot_table(index=['scenario', 'seed'], columns='light_mode', values='avg_travel').reset_index()
pivot['pct_improvement'] = (pivot['fixed'] - pivot['adaptive']) / pivot['fixed'] * 100

means, cis = [], []
for s in SCENARIO_ORDER:
    vals = pivot[pivot.scenario == s]['pct_improvement'].values
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    ci = stats.t.ppf(0.975, df=len(vals) - 1) * se
    means.append(m)
    cis.append(ci)

colors = [C_IMPROVE if m >= 0 else C_REGRESS for m in means]
ax.bar(x, means, yerr=cis, capsize=4, color=colors, edgecolor='black', linewidth=0.4)
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
ax.set_ylabel('Travel time improvement (%)\n[positive = adaptive faster than fixed]')
ax.set_title('Travel Time Improvement: Adaptive vs. Fixed timing\n'
             '(paired per-seed comparison, mean ± 95% CI, N=30 seeds)')

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=C_IMPROVE, edgecolor='black', label='Adaptive faster (improvement)'),
                Patch(facecolor=C_REGRESS, edgecolor='black', label='Adaptive slower (regression)')]
ax.legend(handles=legend_elems, frameon=False, loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig('figures/fig_travel_time_improvement_pct.png', dpi=200)
plt.close(fig)
print('fig_travel_time_improvement_pct.png done')

# save the paired improvement numbers too, useful for the report text
imp_summary = pd.DataFrame({
    'scenario': SCENARIO_ORDER,
    'mean_pct_improvement': means,
    'ci95_pct_improvement': cis,
})
imp_summary.to_csv('travel_time_improvement_summary.csv', index=False)


# =====================================================
# 5. Error-bars methodology illustration (raw seeds + mean + CI, one scenario)
# =====================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
DEMO_SCENARIO = 'VS-02'
for i, (mode, color, xpos) in enumerate([('fixed', C_FIXED, 0), ('adaptive', C_ADAPTIVE, 1)]):
    vals = raw[(raw.scenario == DEMO_SCENARIO) & (raw.light_mode == mode)]['avg_wait'].values
    jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(np.full(len(vals), xpos) + jitter, vals, color=C_POINT, alpha=0.6, s=28,
               zorder=2, label='Individual seed result' if i == 0 else None)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    ci = stats.t.ppf(0.975, df=len(vals) - 1) * se
    ax.errorbar(xpos, m, yerr=ci, fmt='D', color=color, markersize=10, capsize=6,
                elinewidth=2.2, zorder=3,
                label=f'{mode.capitalize()} mean ± 95% CI' if True else None)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Fixed timing\n(30 seeds)', 'Adaptive timing\n(30 seeds)'])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel('Average wait time (s)')
ax.set_title(f'What the error bars represent ({DEMO_SCENARIO}):\n'
             'each gray dot = one full 300s run with a different random seed')
ax.legend(frameon=False, loc='upper right', fontsize=8.5)
fig.tight_layout()
fig.savefig('figures/fig_error_bars_demo.png', dpi=200)
plt.close(fig)
print('fig_error_bars_demo.png done')


# =====================================================
# 6. Utility vs. traffic load (load = vehicles spawned)
# =====================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
density_order = sorted(SCENARIO_ORDER, key=lambda s: agg[agg.scenario == s]['vehicles_spawned_mean'].mean())
for mode, color, marker in [('fixed', C_FIXED, 'o'), ('adaptive', C_ADAPTIVE, 's')]:
    xs, ys, cis = [], [], []
    for s in density_order:
        row = agg[(agg.scenario == s) & (agg.light_mode == mode)].iloc[0]
        xs.append(row['vehicles_spawned_mean'])
        ys.append(row['avg_utility_mean'])
        cis.append(row['avg_utility_ci95'])
    xs, ys, cis = np.array(xs), np.array(ys), np.array(cis)
    ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=1.8,
             label='Fixed timing' if mode == 'fixed' else 'Adaptive timing')
    ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.15)

# linear trend line (pooled, both modes) to make the downward trend explicit
all_x = np.concatenate([xs for _ in [0]])  # placeholder, recompute pooled below
pooled = agg.copy()
pooled_x = pooled['vehicles_spawned_mean'].values
pooled_y = pooled['avg_utility_mean'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(pooled_x, pooled_y)
xs_line = np.array([min(pooled_x), max(pooled_x)])
ax.plot(xs_line, slope * xs_line + intercept, color=C_REGRESS, linestyle='--', linewidth=1.5,
        label=f'Linear trend (both modes pooled): r={r_value:.2f}, p={p_value:.4f}')

ax.set_xlabel('Vehicles spawned per 300s run (traffic load)')
ax.set_ylabel('Average per-vehicle utility (0–1)')
ax.set_title('Utility declines as traffic load increases\n(mean ± 95% CI band, N=30 seeds)')
ax.legend(frameon=False, loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig('figures/fig_utility_vs_load.png', dpi=200)
plt.close(fig)
print('fig_utility_vs_load.png done')
print(f'  pooled linear regression: slope={slope:.5f}, r={r_value:.3f}, p={p_value:.5f}')

print('\nAll additional figures written to figures/')
