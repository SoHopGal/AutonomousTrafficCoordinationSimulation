"""
Validation & Testing Suite - Multi-Agent Traffic Simulation
=============================================================
Run this as a NEW CELL, AFTER the main simulation cell has already executed
(it reuses the classes/functions already defined there: TrafficLight,
IntersectionLights, ReservationManager, Vehicle, Env, DIR_VEC, LANE_OFFSET,
SCENARIOS, SCENARIO_OVERRIDES, run_scenario_headless).

This file does NOT modify the simulation source in any way - it only reads
public state (e.g. inter.manager.slots, vehicle.wait_time) to check that the
system behaves according to its own design rules (SDD) and the goals stated
in each VS-xx scenario. Two layers of checks:

  1. UNIT TESTS      - deterministic, isolated checks of individual classes
                        (traffic light timing, reservation mutual exclusion,
                        turn logic, lane geometry, collision math, utility bounds)
  2. SCENARIO TESTS   - full 300s runs across multiple seeds, checking safety
                        and design invariants that must hold for ANY random
                        outcome, not just one lucky run
"""

import numpy as np


# =====================================================
# Minimal test-report collector
# =====================================================
class ValidationReport:
    def __init__(self):
        self.results = []

    def check(self, name, passed, detail=''):
        self.results.append((name, bool(passed), detail))
        status = 'PASS' if passed else 'FAIL'
        line = f'[{status}] {name}'
        if detail:
            line += f' - {detail}'
        print(line)

    def summary(self):
        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        print()
        print(f'=== Validation Summary: {passed}/{total} checks passed ===')
        if passed < total:
            print('Failed checks:')
            for name, p, detail in self.results:
                if not p:
                    print(f'  - {name}: {detail}')
        return passed == total


report = ValidationReport()


# =====================================================
# PART 1 - Unit tests (deterministic, no full scenario run)
# =====================================================
print('=== Unit tests ===')

# ---- TrafficLight phase-cycle timing ----
tl = TrafficLight(start_green=True)
report.check('TrafficLight starts in the requested phase (green)', tl.phase == 'green')
for _ in range(101):        # >= 10.0s of green (green_duration); +1 step guards against
    tl.step(0.1)             # float accumulation of 0.1 landing just under 10.0
report.check('TrafficLight switches green->yellow after green_duration',
             tl.phase == 'yellow', f'phase={tl.phase}')
for _ in range(21):          # >= 2.0s of yellow (yellow_duration), same float-safety margin
    tl.step(0.1)
report.check('TrafficLight switches yellow->red after yellow_duration',
             tl.phase == 'red', f'phase={tl.phase}')

# ---- IntersectionLights: H and V must never both be green (fixed mode) ----
il = IntersectionLights()
both_green = False
for _ in range(500):
    il.step_fixed(0.1)
    if il.horizontal.is_green() and il.vertical.is_green():
        both_green = True
        break
report.check('IntersectionLights (fixed mode): H and V axes are never green simultaneously',
             not both_green)

# ---- IntersectionLights: same check for adaptive mode, under varying queue pressure ----
il2 = IntersectionLights()
both_green_adaptive = False
for i in range(2000):
    h_wait = 5 if i % 2 == 0 else 0
    v_wait = 0 if i % 2 == 0 else 5
    il2.step_adaptive(0.1, h_wait, v_wait)
    if il2.horizontal.is_green() and il2.vertical.is_green():
        both_green_adaptive = True
        break
report.check('IntersectionLights (adaptive mode): H and V axes are never green simultaneously',
             not both_green_adaptive)


# ---- ReservationManager: strict mutual exclusion ----
class _FakeVehicle:
    """Minimal stand-in - ReservationManager only ever reads .vehicle_id."""
    def __init__(self, vid):
        self.vehicle_id = vid


rm = ReservationManager()
v1, v2 = _FakeVehicle(1), _FakeVehicle(2)
granted_1 = rm.request(v1)
granted_2 = rm.request(v2)
report.check('ReservationManager: a second vehicle is refused while the slot is occupied',
             granted_1 is True and granted_2 is False)
rm.release(v1)
granted_2b = rm.request(v2)
report.check('ReservationManager: request succeeds once the holder releases',
             granted_2b is True)

# ---- ReservationManager.force_grant: occupant-inside-box protection (SDD 3.7) ----
rm2 = ReservationManager()
holder = _FakeVehicle(10)
rm2.request(holder)
emergency_veh = _FakeVehicle(11)
refused = rm2.force_grant(emergency_veh, occupant_inside_box=True)
report.check('force_grant: refused while the current holder is still physically inside the box',
             refused is False)
granted = rm2.force_grant(emergency_veh, occupant_inside_box=False)
report.check('force_grant: succeeds once the holder is no longer inside the box',
             granted is True and 11 in rm2.slots)

# ---- Vehicle.choose_turn respects lane_side (SDD lane-discipline rule) ----
np.random.seed(0)
violations = 0
for _ in range(3000):
    veh = Vehicle(1, 0.0, 0.0, 'right', 'right')   # right-lane: straight or right-turn only
    d = veh.choose_turn()
    if d not in ('right', 'down'):                  # from 'right': straight->'right', right-turn->'down'
        violations += 1
report.check('choose_turn: right-lane vehicles only go straight or turn right',
             violations == 0, f'{violations}/3000 violations')

violations = 0
for _ in range(3000):
    veh = Vehicle(1, 0.0, 0.0, 'right', 'left')     # left-lane: straight or left-turn only
    d = veh.choose_turn()
    if d not in ('right', 'up'):                     # from 'right': straight->'right', left-turn->'up'
        violations += 1
report.check('choose_turn: left-lane vehicles only go straight or turn left',
             violations == 0, f'{violations}/3000 violations')

# ---- LANE_OFFSET completeness ----
missing = [(d, s) for d in DIR_VEC for s in ('left', 'right') if (d, s) not in LANE_OFFSET]
report.check('LANE_OFFSET defines an offset for every (direction, lane_side) pair',
             len(missing) == 0, str(missing))

# ---- Collision rectangle-overlap geometry (spot-check true/false positives) ----
def _rects_overlap(a, b):
    dx = abs(a.pos[0] - b.pos[0])
    dy = abs(a.pos[1] - b.pos[1])
    return dx < (a.w + b.w) / 2.0 and dy < (a.h + b.h) / 2.0

close_same_lane_a = Vehicle(1, 0.0, 0.0, 'right', 'right')
close_same_lane_b = Vehicle(2, 1.0, 0.0, 'right', 'right')   # 1 unit apart, body width 3.2
report.check('Collision geometry: overlapping same-lane vehicles ARE flagged',
             _rects_overlap(close_same_lane_a, close_same_lane_b))

far_diff_lane_a = Vehicle(3, 0.0, 0.0, 'right', 'right')
far_diff_lane_b = Vehicle(4, 0.0, 10.0, 'down', 'right')     # 10 units apart on a perpendicular road
report.check('Collision geometry: distant vehicles on different roads are NOT flagged',
             not _rects_overlap(far_diff_lane_a, far_diff_lane_b))

# ---- utility(): bounded and directionally sensible ----
moving = Vehicle(1, 0.0, 0.0, 'right', 'right')
u_moving = moving.utility(light_green=True, dist_front=10)
waiting = Vehicle(2, 0.0, 0.0, 'right', 'right')
waiting.waiting = True
waiting.v = np.zeros(2)
u_waiting = waiting.utility(light_green=False, dist_front=None)
report.check('utility(): score is bounded in [0, 1]',
             0.0 <= u_moving <= 1.0 and 0.0 <= u_waiting <= 1.0,
             f'moving={u_moving:.3f} waiting={u_waiting:.3f}')
report.check('utility(): a moving vehicle at a green light scores higher than a stopped one at red',
             u_moving > u_waiting, f'moving={u_moving:.3f} waiting={u_waiting:.3f}')


# =====================================================
# PART 2 - Scenario-level validation (multi-seed, full 300s runs)
# =====================================================
print()
print('=== Scenario-level validation (multi-seed) ===')


def run_scenario_with_breakdown(scenario_id, light_mode='fixed', seed=None):
    """Read-only instrumentation around Env - does NOT modify the simulation
    source. Tracks emergency-vs-regular wait/travel times and the maximum
    number of simultaneous reservation-manager slots observed at any single
    intersection at any tick, purely by reading public attributes."""
    if seed is not None:
        np.random.seed(seed)
    cfg = SCENARIOS[scenario_id]
    overrides = SCENARIO_OVERRIDES.get(scenario_id, {})
    e = Env(light_mode=light_mode, total_vehicles=cfg['vehicles'],
            total_emergency=cfg['emergency'], **overrides)
    steps = int(cfg['sim_time'] / e.dt)

    registry = {}
    emergency_wait, regular_wait = [], []
    max_slots_seen = 0

    for _ in range(steps):
        ids_before = {v.vehicle_id for v in e.vehicles}
        for v in e.vehicles:
            registry[v.vehicle_id] = v
        e.step()
        for inter in e.intersections.values():
            max_slots_seen = max(max_slots_seen, len(inter.manager.slots))
        ids_after = {v.vehicle_id for v in e.vehicles}
        for fid in ids_before - ids_after:
            v = registry[fid]
            (emergency_wait if v.is_emergency else regular_wait).append(v.wait_time)

    return dict(
        emergency_wait=emergency_wait, regular_wait=regular_wait,
        max_reservation_slots=max_slots_seen, collisions=e.collision_count,
        avg_wait=e.average_wait_time(), avg_reservation_wait=e.average_reservation_wait_time(),
    )


SAFETY_SEEDS = list(range(2000, 2010))   # 10 seeds per scenario per light mode

total_collisions = 0
max_slots_global = 0
reservation_subset_violations = 0
n_runs = 0

for sid in SCENARIOS:
    for mode in ('fixed', 'adaptive'):
        for s in SAFETY_SEEDS:
            res = run_scenario_with_breakdown(sid, light_mode=mode, seed=s)
            n_runs += 1
            total_collisions += res['collisions']
            max_slots_global = max(max_slots_global, res['max_reservation_slots'])
            if res['avg_reservation_wait'] > res['avg_wait'] + 1e-9:
                reservation_subset_violations += 1

report.check(f'Zero collisions across all {len(SCENARIOS)} scenarios x 2 light modes x '
             f'{len(SAFETY_SEEDS)} seeds ({n_runs} runs)',
             total_collisions == 0, f'{total_collisions} collisions detected')

report.check('ReservationManager mutual exclusion holds throughout every run '
             '(never more than 1 slot held per intersection at any tick)',
             max_slots_global <= 1, f'max observed = {max_slots_global}')

report.check(f'avg_reservation_wait never exceeds avg_wait, across all {n_runs} runs '
             '(reservation wait must be a true subset of total wait)',
             reservation_subset_violations == 0,
             f'{reservation_subset_violations}/{n_runs} runs violated this')

# ---- Emergency prioritization (VS-03, VS-14 - the scenarios with meaningful emergency traffic) ----
for sid in ('VS-03', 'VS-14'):
    em_waits, reg_waits = [], []
    for s in SAFETY_SEEDS:
        res = run_scenario_with_breakdown(sid, light_mode='fixed', seed=s)
        em_waits += res['emergency_wait']
        reg_waits += res['regular_wait']
    if em_waits and reg_waits:
        em_mean, reg_mean = float(np.mean(em_waits)), float(np.mean(reg_waits))
        report.check(f'{sid}: emergency vehicles wait less than regular vehicles on average',
                     em_mean < reg_mean,
                     f'emergency={em_mean:.2f}s (n={len(em_waits)}) '
                     f'regular={reg_mean:.2f}s (n={len(reg_waits)})')
    else:
        report.check(f'{sid}: emergency prioritization check', False,
                     'no completed emergency or regular vehicles collected - increase SAFETY_SEEDS')

# =====================================================
all_passed = report.summary()
