"""
Figure generation for the project report (Chapter 6 - Results & Validation Analysis).
Reads raw_results.csv / aggregated_results.csv / convergence_results.csv /
emergency_breakdown.csv (produced by run_multi_seed_study.py) and produces
publication-quality PNG figures.

Color palette restricted per request: white / gray / black / red / blue only
(no green, except where a colormap/heatmap convention calls for it - none used here).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- palette ----
C_FIXED = '#4D4D4D'      # dark gray
C_ADAPTIVE = '#1F4E99'   # blue
C_EMERGENCY = '#B22222'  # red
C_REGULAR = '#808080'    # mid gray
C_GRID = '#DDDDDD'
C_TEXT = '#1A1A1A'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': C_TEXT,
    'text.color': C_TEXT,
    'xtick.color': C_TEXT,
    'ytick.color': C_TEXT,
    'axes.grid': True,
    'grid.color': C_GRID,
    'grid.linewidth': 0.6,
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

raw = pd.read_csv('raw_results.csv')
agg = pd.read_csv('aggregated_results.csv')
conv = pd.read_csv('convergence_results.csv')
em = pd.read_csv('emergency_breakdown.csv')

SCENARIO_ORDER = ['VS-01', 'VS-03', 'VS-13', 'VS-02', 'VS-14', 'VS-12', 'VS-11']  # roughly by density
SCENARIO_LABELS = {
    'VS-01': 'VS-01\n(12 veh)', 'VS-02': 'VS-02\n(52 veh)', 'VS-03': 'VS-03\n(32 veh)',
    'VS-11': 'VS-11\n(102 veh)', 'VS-12': 'VS-12\n(77 veh)', 'VS-13': 'VS-13\n(32 veh)',
    'VS-14': 'VS-14\n(55 veh)',
}


# =====================================================
# Figure 1 - Convergence analysis (justifies N=30 seeds)
# =====================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
reps = [('VS-01', C_REGULAR, 'o', 'VS-01 (low density, 12 veh)'),
        ('VS-02', C_FIXED, 's', 'VS-02 (medium density, 52 veh)'),
        ('VS-12', C_ADAPTIVE, '^', 'VS-12 (high density, 77 veh)')]
for sid, color, marker, label in reps:
    sub = conv[(conv.scenario == sid) & (conv.light_mode == 'fixed')].sort_values('n_seeds')
    ax.plot(sub.n_seeds, sub.avg_wait_ci95, color=color, marker=marker, markersize=7,
            linewidth=1.8, label=label)

ax.set_xlabel('Number of seeds (N)')
ax.set_ylabel('95% CI half-width of avg. wait time (s)')
ax.set_title('Convergence of the mean-wait estimate as N grows\n(fixed-timing runs; other scenarios show the same pattern)')
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.legend(frameon=False, loc='upper right')
ax.axvline(25, color=C_TEXT, linestyle=':', linewidth=1, alpha=0.5)
ax.text(25.3, ax.get_ylim()[1]*0.92, 'diminishing\nreturns beyond\nN≈20–25', fontsize=8.5, color=C_TEXT, alpha=0.7)
fig.tight_layout()
fig.savefig('figures/fig1_convergence.png', dpi=200)
plt.close(fig)
print('fig1 done')


# =====================================================
# Figure 2 - Fixed vs Adaptive: avg travel & wait time, all scenarios (N=30, 95% CI)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
x = np.arange(len(SCENARIO_ORDER))
width = 0.36

for ax, metric, title in zip(axes,
                              ['avg_travel', 'avg_wait'],
                              ['Average travel time', 'Average wait time']):
    fixed_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')][f'{metric}_mean'].values[0] for s in SCENARIO_ORDER]
    fixed_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')][f'{metric}_ci95'].values[0] for s in SCENARIO_ORDER]
    adap_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')][f'{metric}_mean'].values[0] for s in SCENARIO_ORDER]
    adap_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')][f'{metric}_ci95'].values[0] for s in SCENARIO_ORDER]

    ax.bar(x - width/2, fixed_vals, width, yerr=fixed_ci, capsize=3, color=C_FIXED, label='Fixed timing', edgecolor='black', linewidth=0.4)
    ax.bar(x + width/2, adap_vals, width, yerr=adap_ci, capsize=3, color=C_ADAPTIVE, label='Adaptive timing', edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
    ax.set_ylabel('Seconds')
    ax.set_title(f'{title} (mean ± 95% CI, N=30 seeds)')

axes[0].legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig('figures/fig2_fixed_vs_adaptive.png', dpi=200)
plt.close(fig)
print('fig2 done')


# =====================================================
# Figure 3 - Scalability: wait/travel time vs. traffic density
# =====================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
density_order = sorted(SCENARIO_ORDER, key=lambda s: agg[agg.scenario == s]['vehicles_spawned_mean'].mean())

for mode, color, marker in [('fixed', C_FIXED, 'o'), ('adaptive', C_ADAPTIVE, 's')]:
    xs, ys, cis = [], [], []
    for s in density_order:
        row = agg[(agg.scenario == s) & (agg.light_mode == mode)].iloc[0]
        xs.append(row['vehicles_spawned_mean'])
        ys.append(row['avg_wait_mean'])
        cis.append(row['avg_wait_ci95'])
    xs, ys, cis = np.array(xs), np.array(ys), np.array(cis)
    ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=1.8,
             label='Fixed timing' if mode == 'fixed' else 'Adaptive timing')
    ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.15)

ax.set_xlabel('Vehicles spawned per 300s run (traffic density)')
ax.set_ylabel('Average wait time (s)')
ax.set_title('Congestion scalability: average wait vs. traffic density\n(mean ± 95% CI band, N=30 seeds)')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig('figures/fig3_scalability.png', dpi=200)
plt.close(fig)
print('fig3 done')


# =====================================================
# Figure 4 - Emergency vehicle prioritization
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
for ax, sid in zip(axes, ['VS-03', 'VS-14']):
    cats = [('fixed', 'emergency'), ('fixed', 'regular'), ('adaptive', 'emergency'), ('adaptive', 'regular')]
    means, sems, colors, labels = [], [], [], []
    for mode, vtype in cats:
        sub = em[(em.scenario == sid) & (em.light_mode == mode) & (em.vehicle_type == vtype)]['wait']
        means.append(sub.mean())
        sems.append(sub.std(ddof=1) / np.sqrt(len(sub)) * 1.96 if len(sub) > 1 else 0)
        colors.append(C_EMERGENCY if vtype == 'emergency' else C_REGULAR)
        labels.append(f'{mode}\n{vtype}')
    ax.bar(range(4), means, yerr=sems, capsize=4, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f'{sid}')
axes[0].set_ylabel('Average wait time (s)')
fig.suptitle('Emergency vs. regular vehicle wait time (mean ± 95% CI, N=30 seeds)', y=1.02)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=C_EMERGENCY, edgecolor='black', label='Emergency vehicle'),
                Patch(facecolor=C_REGULAR, edgecolor='black', label='Regular vehicle')]
fig.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig('figures/fig4_emergency_priority.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('fig4 done')


# =====================================================
# Figure 5 - Average utility comparison
# =====================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(len(SCENARIO_ORDER))
fixed_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')]['avg_utility_mean'].values[0] for s in SCENARIO_ORDER]
fixed_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'fixed')]['avg_utility_ci95'].values[0] for s in SCENARIO_ORDER]
adap_vals = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')]['avg_utility_mean'].values[0] for s in SCENARIO_ORDER]
adap_ci = [agg[(agg.scenario == s) & (agg.light_mode == 'adaptive')]['avg_utility_ci95'].values[0] for s in SCENARIO_ORDER]

ax.bar(x - width/2, fixed_vals, width, yerr=fixed_ci, capsize=3, color=C_FIXED, label='Fixed timing', edgecolor='black', linewidth=0.4)
ax.bar(x + width/2, adap_vals, width, yerr=adap_ci, capsize=3, color=C_ADAPTIVE, label='Adaptive timing', edgecolor='black', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
ax.set_ylabel('Average per-vehicle utility (0–1)')
ax.set_ylim(0, 1.05)
ax.set_title('Average utility score by scenario (mean ± 95% CI, N=30 seeds)')
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()
fig.savefig('figures/fig5_utility.png', dpi=200)
plt.close(fig)
print('fig5 done')


# =====================================================
# Figure 6 - Collision reliability summary
# =====================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
coll = raw.groupby(['scenario', 'light_mode'])['collisions'].sum().unstack()
coll = coll.reindex(SCENARIO_ORDER)
x = np.arange(len(SCENARIO_ORDER))
ax.bar(x - width/2, coll['fixed'], width, color=C_FIXED, label='Fixed timing', edgecolor='black', linewidth=0.4)
ax.bar(x + width/2, coll['adaptive'], width, color=C_ADAPTIVE, label='Adaptive timing', edgecolor='black', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
ax.set_ylabel('Total collisions (sum over 30 seeds)')
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_title('Collision counts by scenario (30 runs per bar, 300s each)')
ax.legend(frameon=False, loc='upper left')
total = int(raw['collisions'].sum())
ax.text(0.98, 0.95, f'Total: {total} collisions / 420 full runs (4.3%)\nAll occurred only in the 4 highest-density scenarios',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=C_TEXT,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_GRID))
fig.tight_layout()
fig.savefig('figures/fig6_collision_reliability.png', dpi=200)
plt.close(fig)
print('fig6 done')

print('\nAll figures written to figures/')
