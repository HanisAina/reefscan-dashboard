"""
fusion.py: Gap 2, combining depth, colour (camera) and position in one decision.

Plain-language summary
  * Depth: the reef gets a height map. Coral stands 0.2-1.0 m above the
    seabed, sand is flat, and rock patches also stand up (so depth alone
    cannot tell rock from coral). A depth sensor reads the height with noise.
  * Colour: the Tiny U-Net camera's photo decision, with its measured
    error rates (the same draws as the camera-only robot).
  * Position: reef types come in patches, so what the robot already decided
    for the previous metre (1 m away) and the same spot on the previous lane
    (2 m away) tells it something about the current metre.
  * The three are combined with Bayes' rule:
        P(class | all) ∝ P(class | position) × P(depth | class) × P(camera | class)
    and the most likely class is chosen.

All sensor settings are assumptions and are listed in DEPTH below.
"""
import math
import numpy as np
import reefsim as rs

DEPTH = dict(sand_mean=0.0, sand_sd=0.03,     # flat sand (m)
             rock_share=0.30, rock_mean=0.40, rock_sd=0.12,   # share of sand/rock cells that are rock
             coral_mean=0.60, coral_sd=0.20,  # coral height above the seabed (m)
             sensor_sd=0.08)                  # depth sensor noise (m)
POSITION_WEIGHT = 0.6   # damping of neighbour evidence (0 = ignore neighbours, 1 = full); tuned on practice reefs

FUSED_NAME = rs.FUSED_NAME
VARIANTS = ["Camera only", "Depth only", "Camera + depth", "Camera + depth + position"]


# ---------------------------------------------------------------- the reef's shape
def make_relief(reef, rng, d=DEPTH):
    sand = reef == rs.SAND
    f = rs.smooth_noise(rng, reef.shape, 3.0)
    rock = sand & (f > np.quantile(f[sand], 1 - d["rock_share"])) if sand.any() else np.zeros_like(sand)
    h = np.where(reef == rs.SAND, rng.normal(d["sand_mean"], d["sand_sd"], reef.shape),
                 rng.normal(d["coral_mean"], d["coral_sd"], reef.shape))
    h = np.where(rock, rng.normal(d["rock_mean"], d["rock_sd"], reef.shape), h)
    return np.clip(h, 0, None)


def _norm(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def depth_likelihood(z, d=DEPTH):
    """P(depth reading | sand/rock, healthy, bleached)."""
    se = d["sensor_sd"]
    s_sand = math.hypot(d["sand_sd"], se); s_rock = math.hypot(d["rock_sd"], se); s_coral = math.hypot(d["coral_sd"], se)
    sand = (1 - d["rock_share"]) * _norm(z, d["sand_mean"], s_sand) + d["rock_share"] * _norm(z, d["rock_mean"], s_rock)
    coral = _norm(z, d["coral_mean"], s_coral)
    return np.array([sand, coral, coral]) + 1e-12


# ---------------------------------------------------------------- how patchy reefs are
_T = None


def neighbour_tables(n_reefs=12, seed=777):
    """P(this metre's class | neighbour's class), learned once from practice reefs
    (separate seeds from the test reefs). Returns (along-lane, across-lane, overall share)."""
    global _T
    if _T is not None:
        return _T
    along = np.ones((3, 3)); across = np.ones((3, 3)); share = np.ones(3)
    rng = np.random.default_rng(seed)
    for k in range(n_reefs):
        reef = rs.make_reef(np.random.default_rng(int(rng.integers(1 << 31))), [0.3, 0.5, 0.7][k % 3])
        a, b = reef[:, 1:].ravel(), reef[:, :-1].ravel()
        np.add.at(along, (b, a), 1)
        a, b = reef[1:, :].ravel(), reef[:-1, :].ravel()
        np.add.at(across, (b, a), 1)
        share += np.bincount(reef.ravel(), minlength=3)
    _T = (along / along.sum(1, keepdims=True), across / across.sum(1, keepdims=True), share / share.sum())
    return _T


# ---------------------------------------------------------------- one survey's perception
def observe_fused(reef, rows, cols, cells, cam_rng, depth_rng, unet_M, use_camera=True, use_depth=True,
                  use_position=True, d=DEPTH, return_posterior=False, weight=None):
    """What the robot believes in each metre, combining the chosen sensors."""
    relief = make_relief(reef, np.random.default_rng(depth_rng.integers(1 << 31)), d)
    cam = rs.observe_confusion(cells, cam_rng, unet_M)              # same draws as camera-only
    M = np.asarray(unet_M, float); M = M / M.sum(1, keepdims=True)  # P(seen | true)
    along, across, share = neighbour_tables()
    post = np.zeros(reef.shape + (3,))
    seen = np.empty(len(cells), dtype=np.int8)
    P = np.zeros((len(cells), 3))
    w = POSITION_WEIGHT if weight is None else weight
    for i, (r, c) in enumerate(zip(rows, cols)):
        prior = share.copy()
        if use_position:
            step = 1 if r % 2 == 0 else -1
            if 0 <= c - step < reef.shape[1] and post[r, c - step].sum() > 0:     # previous metre on this lane
                prior *= ((post[r, c - step] @ along) / share) ** w
            if r > 0:                                                         # same spot, previous lane
                prior *= ((post[r - 1, c] @ across) / share) ** w
            prior = prior / prior.sum()
        p = prior.copy()
        if use_depth:
            p *= depth_likelihood(relief[r, c] + depth_rng.normal(0, d["sensor_sd"]), d)
        if use_camera:
            p *= M[:, cam[i]]
        p = p / p.sum()
        post[r, c] = p
        P[i] = p
        seen[i] = int(np.argmax(p))
    return (seen, P) if return_posterior else seen


def perceive_fused(reef, rows, cols, cells, seed, unet_M, variant="Camera + depth + position"):
    if variant == "Camera only":            # the raw camera reading, exactly as the camera-only robot uses it
        return rs.observe_confusion(cells, np.random.default_rng(seed + 7), unet_M)
    flags = {"Camera only": (True, False, False), "Depth only": (False, True, False),
             "Camera + depth": (True, True, False), "Camera + depth + position": (True, True, True)}[variant]
    return observe_fused(reef, rows, cols, cells, np.random.default_rng(seed + 7), np.random.default_rng(seed + 11),
                         unet_M, *flags)


def perception_scores(cells, seen):
    """Per-metre accuracy of what the robot believed."""
    bl_true, bl_seen = cells == rs.BLEACHED, seen == rs.BLEACHED
    coral_true, coral_seen = cells != rs.SAND, seen != rs.SAND
    return dict(accuracy=float((cells == seen).mean()),
                coral_vs_sand_accuracy=float((coral_true == coral_seen).mean()),
                bleached_recall=float((bl_true & bl_seen).sum() / max(bl_true.sum(), 1)),
                bleached_precision=float((bl_true & bl_seen).sum() / max(bl_seen.sum(), 1)),
                false_alarms_per_100m=float(100 * (bl_seen & ~bl_true).sum() / len(cells)))
