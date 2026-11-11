"""
rl_agent.py: a reinforcement learning (RL) agent that learns when to go
fast, normal or slow, instead of following a hand-written rule.

Plain-language summary
  * The agent sees a small summary of its situation (the "state"):
      - what it thinks is below it now (sand/rock, healthy, bleached),
      - how many metres ago it last saw bleaching (1, 2, ... 6 or more),
      - whether bleaching was reported next to this spot on the previous lane.
  * It chooses one of three actions: fast, normal or detail.
  * After every metre it gets a score (the "reward"), Equation 6 of the
    interim report:
        R = alpha * A - beta * E - gamma * S
    A = mapping value of that metre (bleached coral photographed in detail,
        or healthy coral photographed at the normal rate)
    E = energy used in that metre (kJ)
    S = photos taken in that metre
  * Q-learning: the agent keeps a table Q[state, action] of how good each
    action has been in each state, tries actions (sometimes at random, to
    explore), and updates the table from the rewards it gets.
  * Note: the vehicle's choice does not change what it sees next, so future
    rewards do not depend on today's action. A discount of 0 is therefore
    used, which makes each update a running average of the reward.
"""
import numpy as np
import reefsim as rs

SINCE_MAX = 6                     # "6 or more metres since bleaching was seen"
CHECKPOINTS = {0, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000}   # early training steps to check
ACTIONS = [rs.FAST, rs.NORMAL, rs.DETAIL]
REWARD = dict(alpha=1.0, beta=3.0, gamma=0.01, value_bleached=1.0, value_healthy=0.2)


def n_states():
    return 2 * (1 + 2 * SINCE_MAX)    # (bleached now) or (sand/healthy x since), times neighbour 0/1


def state_index(seen, since, nb):
    if seen == rs.BLEACHED:
        s = 0
    else:
        s = 1 + (0 if seen == rs.SAND else SINCE_MAX) + (min(since, SINCE_MAX) - 1)
    return s + int(nb) * (1 + 2 * SINCE_MAX)


def describe_state(s):
    nb = s // (1 + 2 * SINCE_MAX); s = s % (1 + 2 * SINCE_MAX)
    side = "bleaching on previous lane nearby" if nb else "no bleaching on previous lane nearby"
    if s == 0:
        return "Sees bleached coral", side
    seen = "sand/rock" if s <= SINCE_MAX else "healthy coral"
    since = (s - 1) % SINCE_MAX + 1
    since_txt = f"{since} m since bleaching" if since < SINCE_MAX else f"{SINCE_MAX}+ m since bleaching"
    return f"Sees {seen}, {since_txt}", side


def reward(true_cls, photos, energy_j, w=REWARD):
    A = 0.0
    if true_cls == rs.BLEACHED and photos >= rs.DETAIL_PHOTOS:
        A = w["value_bleached"]
    elif true_cls == rs.HEALTHY and photos >= rs.NORMAL_PHOTOS:
        A = w["value_healthy"]
    return w["alpha"] * A - w["beta"] * energy_j / 1000 - w["gamma"] * photos


class QPolicy:
    adaptive = True
    def __init__(self, Q, epsilon=0.0, rng=None):
        self.Q, self.epsilon, self.rng = Q, epsilon, rng or np.random.default_rng(0)
    def reset(self):
        self.since = SINCE_MAX
    def state(self, seen, nb):
        return state_index(seen, self.since, nb)
    def act(self, seen, nb=0):
        s = state_index(seen, self.since, nb)
        self.last_state = s
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(3))
        else:
            a = int(np.argmax(self.Q[s]))
        self.since = 0 if seen == rs.BLEACHED else self.since
        self.since = min(self.since + 1, SINCE_MAX)
        self.last_action = a
        return ACTIONS[a]


def train(n_reefs=120, sand_fractions=(0.3, 0.5, 0.7), clf=None, seed=12345, w=REWARD,
          eps_start=0.5, eps_end=0.02, progress=None, perception="Measured classifier", unet_M=None):
    """Q-learning on random training reefs (different seeds from the test reefs).
    `perception` chooses how the vehicle sees (classifier + sensor, or Tiny U-Net camera)."""
    def look(cells, obs_seed, reef, rows, cols):
        r = np.random.default_rng(obs_seed)
        if clf is not None:
            return rs.observe_path(cells, r, clf)
        return rs.perceive(cells, r, perception, unet_M,
                           ctx=dict(reef=reef, rows=rows, cols=cols, seed=obs_seed - 7))
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states(), 3))
    N = np.zeros((n_states(), 3))
    curve = []
    # fixed validation reefs (one per sand level) to track progress fairly
    val = []
    vrng = np.random.default_rng(seed + 1)
    for sf in sand_fractions:
        reef = rs.make_reef(np.random.default_rng(int(vrng.integers(1 << 31))), sf)
        rows, cols, cells = rs.lawnmower_path(reef)
        seen = look(cells, int(vrng.integers(1 << 31)), reef, rows, cols)
        val.append((cells, seen, rs.neighbour_feature(rows, cols, seen)))

    def validate():
        tot, n = 0.0, 0
        for cells, seen, nb in val:
            log = rs.run_survey(cells, seen, QPolicy(Q), nb)
            tot += total_reward(cells, log, w); n += len(cells)
        return tot / n
    for ep in range(n_reefs):
        eps = eps_start + (eps_end - eps_start) * ep / max(n_reefs - 1, 1)
        sf = sand_fractions[ep % len(sand_fractions)]
        reef = rs.make_reef(np.random.default_rng(int(rng.integers(1 << 31))), sf)
        rows, cols, cells = rs.lawnmower_path(reef)
        seen = look(cells, int(rng.integers(1 << 31)), reef, rows, cols)
        nb = rs.neighbour_feature(rows, cols, seen)
        pol = QPolicy(Q, epsilon=eps, rng=rng); pol.reset()
        d, total = 0.0, 0.0
        for i in range(len(cells)):
            step = ep * len(cells) + i
            if step in CHECKPOINTS:
                curve.append(dict(steps_trained=step, reefs_trained=round(step / len(cells), 3), epsilon=round(eps, 3),
                                  validation_reward_per_m=round(validate(), 4)))
            mode = pol.act(seen[i], nb[i])
            n, t, e, d = rs.cell_cost(mode, d)
            e += rs.ROBOT["look_energy_j"]
            r = reward(cells[i], n, e, w)
            s, a = pol.last_state, pol.last_action
            N[s, a] += 1
            Q[s, a] += (r - Q[s, a]) / N[s, a]       # discount 0: running average of reward
            total += r
        if (ep + 1) % 20 == 0:
            curve.append(dict(steps_trained=(ep + 1) * len(cells), reefs_trained=ep + 1, epsilon=round(eps, 3),
                              validation_reward_per_m=round(validate(), 4)))
        if progress:
            progress(ep + 1, n_reefs)
    return Q, N, curve


def total_reward(cells, log, w=REWARD):
    return float(sum(reward(c, p, e, w) for c, p, e in zip(cells, log["photos"], log["energy_j"])))


def policy_table(Q, N):
    rows = []
    for s in range(n_states()):
        what, side = describe_state(s)
        rows.append(dict(situation=what, previous_lane=side,
                         learned_action=rs.MODE_NAMES[int(np.argmax(Q[s]))] if N[s].sum() else "(never seen)",
                         times_seen=int(N[s].sum()),
                         q_fast=round(float(Q[s, 0]), 3), q_normal=round(float(Q[s, 1]), 3), q_detail=round(float(Q[s, 2]), 3)))
    return rows
