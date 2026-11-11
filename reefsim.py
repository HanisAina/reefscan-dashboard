"""
reefsim.py: core of the reef survey simulation.

Shared by the experiments, the knowledge graph, the RL agent and the
Streamlit dashboard. Needs only numpy.

Plain-language summary
  * A reef is a grid: each row is one survey lane, each cell is 1 m of track.
    Every cell is sand/rock, healthy coral or bleached coral.
  * The vehicle visits the cells lane by lane, back and forth ("lawnmower").
  * In each cell it "observes" what is below it. A seabed sensor says coral or
    not, then the classifier says healthy or bleached. Both can be wrong, at
    the rates given in the settings.
  * A policy turns observations into a mode (fast / normal / detail), which
    sets the speed and the distance between photos.
  * Time, energy and photos are added up, and coverage of bleached coral is
    measured.
"""
import math
import numpy as np

SAND, HEALTHY, BLEACHED = 0, 1, 2
CLASS_NAMES = {SAND: "Sand/rock", HEALTHY: "Healthy coral", BLEACHED: "Bleached coral"}
MODE_NAMES = ["fast", "normal", "detail"]
FAST, NORMAL, DETAIL = 0, 1, 2

# ---------------------------------------------------------------- settings
MAP = dict(lanes=40, lane_cells=100, cell_len_m=1.0, lane_width_m=2.0,
           clump_size=4.0, bleach_clump=3.0, bleached_share=0.30)
ROBOT = dict(hotel_power_w=40.0, drag_coeff=12.0, photo_energy_j=10.0,
             look_energy_j=0.2, photo_mb=4.0)
MODES = {FAST: dict(speed=1.5, spacing=2.00),
         NORMAL: dict(speed=1.0, spacing=0.50),
         DETAIL: dict(speed=0.5, spacing=0.25)}
DETAIL_PHOTOS = 4          # photos per metre in detail mode
NORMAL_PHOTOS = 2          # photos per metre in normal mode
SENSOR = dict(sand_correct=0.97, coral_correct=0.97)
CLASSIFIERS = {
    "Perfect classifier": dict(p_bleached_found=1.0, p_healthy_called_bleached=0.0),
    # measured on the independent Hugging Face dataset: 388/508 and 31/661
    "Measured classifier": dict(p_bleached_found=388 / 508, p_healthy_called_bleached=31 / 661),
}


# ---------------------------------------------------------------- reefs
def smooth_noise(rng, shape, scale):
    """Random field with blobs about `scale` cells wide (Gaussian blur via FFT)."""
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    blur = np.exp(-2 * (math.pi * scale) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * blur))


def make_reef(rng, sand_fraction, m=MAP):
    shape = (m["lanes"], m["lane_cells"])
    f = smooth_noise(rng, shape, m["clump_size"])
    coral = f > np.quantile(f, sand_fraction)
    reef = np.full(shape, SAND, dtype=np.int8)
    reef[coral] = HEALTHY
    g = smooth_noise(rng, shape, m["bleach_clump"])
    cut = np.quantile(g[coral], 1 - m["bleached_share"])
    reef[coral & (g > cut)] = BLEACHED
    return reef


def lawnmower_path(reef):
    """Row and column of every cell, in the order the vehicle visits them."""
    lanes, n = reef.shape
    rows, cols = [], []
    for i in range(lanes):
        cs = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        for c in cs:
            rows.append(i); cols.append(c)
    rows, cols = np.array(rows), np.array(cols)
    return rows, cols, reef[rows, cols]


# ---------------------------------------------------------------- observations
def observe_path(cells, rng, clf, sensor=SENSOR):
    """What the vehicle *thinks* is in each cell (same draws for every policy)."""
    seen = np.empty(len(cells), dtype=np.int8)
    for i, c in enumerate(cells):
        if c == SAND:
            looks_like_coral = rng.random() > sensor["sand_correct"]
        else:
            looks_like_coral = rng.random() < sensor["coral_correct"]
        if not looks_like_coral:
            seen[i] = SAND
        elif c == BLEACHED:
            seen[i] = BLEACHED if rng.random() < clf["p_bleached_found"] else HEALTHY
        else:
            seen[i] = BLEACHED if rng.random() < clf["p_healthy_called_bleached"] else HEALTHY
    return seen


def observe_confusion(cells, rng, M):
    """Camera-only perception (Tiny U-Net). M[true][seen] = probability that a
    cell of class `true` is seen as class `seen` (rows: sand, healthy, bleached)."""
    M = np.asarray(M, dtype=float)
    M = M / M.sum(1, keepdims=True)
    cum = np.cumsum(M, 1)
    u = np.array([rng.random() for _ in range(len(cells))])
    seen = np.empty(len(cells), dtype=np.int8)
    for i, c in enumerate(cells):
        seen[i] = int(np.searchsorted(cum[c], u[i], side="right"))
    return np.minimum(seen, 2).astype(np.int8)


def load_unet_perception(path):
    """Reads the photo-level confusion matrix saved by colab/train_unet.py."""
    import json, os
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    return r.get("photo_confusion")


def perceive(cells, rng, name, unet_M=None, sensor=None, ctx=None):
    """One entry point for every kind of perception.
    The fused option also needs ctx = dict(reef, rows, cols, seed); it draws its camera
    readings from seed + 7, the same draws as the camera-only robot."""
    if name in CLASSIFIERS:
        return observe_path(cells, rng, CLASSIFIERS[name], sensor or SENSOR)
    if name == UNET_NAME and unet_M is not None:
        return observe_confusion(cells, rng, unet_M)
    if name == FUSED_NAME and unet_M is not None and ctx is not None:
        import fusion
        return fusion.perceive_fused(ctx["reef"], ctx["rows"], ctx["cols"], cells, ctx["seed"], unet_M)
    raise ValueError(name)


UNET_NAME = "Tiny U-Net camera only"
FUSED_NAME = "Fused: camera + depth + position"


# ---------------------------------------------------------------- policies
class FixedPolicy:
    adaptive = False
    def __init__(self, mode): self.mode = mode
    def reset(self): pass
    def act(self, seen, nb=0): return self.mode


class RulePolicy:
    """Hand-written rule: fast over sand, normal over healthy, detail over
    bleached, and stay in detail for `linger` more cells after bleaching."""
    adaptive = True
    def __init__(self, linger=3): self.linger = linger
    def reset(self): self.left = 0
    def act(self, seen, nb=0):
        if seen == BLEACHED:
            self.left = self.linger
            return DETAIL
        if self.left > 0:
            self.left -= 1
            return DETAIL
        return FAST if seen == SAND else NORMAL


# ---------------------------------------------------------------- one survey
def cell_cost(mode, dist_since_photo, m=MAP, r=ROBOT, modes=None):
    """Photos, time and energy for one cell in one mode (photo spacing carries over)."""
    modes = MODES if modes is None else modes
    sp, v, L = modes[mode]["spacing"], modes[mode]["speed"], m["cell_len_m"]
    travelled, n = 0.0, 0
    while travelled + (sp - dist_since_photo) <= L + 1e-9:
        travelled += sp - dist_since_photo
        dist_since_photo = 0.0
        n += 1
    dist_since_photo += L - travelled
    t = L / v
    e = (r["hotel_power_w"] + r["drag_coeff"] * v ** 3) * t + n * r["photo_energy_j"]
    return n, t, e, dist_since_photo


def neighbour_feature(rows, cols, seen):
    """1 if bleaching was reported on the previous lane within one cell of
    this position (the previous lane is 2 m away and was surveyed earlier)."""
    lanes = rows.max() + 1
    n = cols.max() + 1
    grid = np.zeros((lanes, n), dtype=bool)
    grid[rows, cols] = seen == BLEACHED
    nb = np.zeros(len(rows), dtype=np.int8)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if r > 0:
            nb[i] = grid[r - 1, max(c - 1, 0):c + 2].any()
    return nb


def run_survey(cells, seen, policy, nb=None, mode_settings=None, robot=None):
    """Runs a policy over a path. Returns per-cell arrays.
    `mode_settings` and `robot` can override the default speeds and power settings."""
    robot = robot or ROBOT
    policy.reset()
    if nb is None:
        nb = np.zeros(len(cells), dtype=np.int8)
    N = len(cells)
    modes = np.empty(N, dtype=np.int8)
    photos = np.empty(N, dtype=np.int16)
    times = np.empty(N); energy = np.empty(N)
    d = 0.0
    for i in range(N):
        mode = policy.act(seen[i], nb[i])
        n, t, e, d = cell_cost(mode, d, r=robot, modes=mode_settings)
        if policy.adaptive:
            e += robot["look_energy_j"]
        modes[i], photos[i], times[i], energy[i] = mode, n, t, e
    return dict(modes=modes, photos=photos, time_s=times, energy_j=energy)


def metrics(cells, log, robot=None):
    ph = log["photos"]
    bl = cells == BLEACHED
    total = ph.sum()
    return dict(
        time_min=log["time_s"].sum() / 60,
        energy_kj=log["energy_j"].sum() / 1000,
        photos=int(total),
        storage_gb=total * (robot or ROBOT)["photo_mb"] / 1000,
        bleached_detail_pct=100 * (ph[bl] >= DETAIL_PHOTOS).mean() if bl.any() else float("nan"),
        photos_per_bleached_cell=ph[bl].mean() if bl.any() else float("nan"),
        photos_on_sand_pct=100 * ph[cells == SAND].sum() / max(total, 1),
    )


def strategies(linger=3):
    return {
        "Fixed standard": FixedPolicy(NORMAL),
        "Fixed detailed": FixedPolicy(DETAIL),
        "Adaptive": RulePolicy(linger),
        "Adaptive (no linger)": RulePolicy(0),
    }


def map_seeds(sand_fractions, n_maps, seed=2026):
    """Same reef seeds as the original experiments."""
    rng = np.random.default_rng(seed)
    out = []
    for sf in sand_fractions:
        for k in range(n_maps):
            out.append((sf, k, int(rng.integers(1 << 31))))
    return out
