"""
photo_lab.py: train a quick reef photo classifier from uploaded photos (used by researcher.py).

Runs on a normal CPU with no PyTorch, so it works on the free Streamlit host:
  * every photo is reduced to 54 simple colour and texture numbers (features),
  * a scikit-learn model (logistic regression or random forest) learns from them,
  * near-copies are found with a perceptual hash (dHash), exactly as in the main study,
  * accuracy is measured twice: with a random split and with near-copies kept together,
    so the shortcut effect from the study can be seen on any dataset.
"""
import io, zipfile, pickle
import numpy as np
from PIL import Image, ImageOps

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLIT_NAMES = {"train", "training", "test", "testing", "valid", "validation", "val", "images", "image", "data",
               "dataset", "photos", "all", "__macosx"}
FEATURE_SIZE = 128
MAX_PHOTOS = 3000


# ------------------------------------------------------------------ reading photos
def class_from_path(path):
    """Class = the nearest folder name that is not a split name (train/test/valid...)."""
    parts = [p for p in path.replace("\\", "/").split("/")[:-1] if p and not p.startswith(".")]
    for p in reversed(parts):
        if p.lower() not in SPLIT_NAMES:
            return p.strip()
    return None


def iter_zip(data, max_photos=MAX_PHOTOS):
    """Yields (class, name, raw bytes) from a zip with one folder per class."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(IMG_EXT) and "__macosx" not in n.lower()]
        for n in names[:max_photos]:
            yield class_from_path(n), n, z.read(n)


def read_zip(data, max_photos=MAX_PHOTOS):
    """Returns a list of (class, name, PIL image); fine for small sets and tests."""
    out, skipped = [], 0
    for cls, n, raw in iter_zip(data, max_photos):
        try:
            if cls is None:
                raise ValueError
            out.append((cls, n, prepare(Image.open(io.BytesIO(raw)))))
        except Exception:
            skipped += 1
    return out, skipped


def process(items, thumbs_per_class=6):
    """One pass over (class, name, raw bytes): features, hashes and a few thumbnails, no full images kept.
    Returns dict(X, y, labels, names, hashes, thumbs, skipped)."""
    X, cls_list, names, hashes, thumbs, skipped = [], [], [], [], {}, 0
    for cls, n, raw in items:
        try:
            if not cls:
                raise ValueError("no class")
            im = prepare(Image.open(io.BytesIO(raw)))
            X.append(features(im)); hashes.append(dhash_variants(im)); cls_list.append(cls); names.append(n)
            t = thumbs.setdefault(cls, [])
            if len(t) < thumbs_per_class:
                th = im.copy(); th.thumbnail((160, 160)); t.append(th)
        except Exception:
            skipped += 1
    labels = sorted(set(cls_list))
    y = np.array([labels.index(c) for c in cls_list], dtype=int)
    return dict(X=np.vstack(X) if X else np.zeros((0, len(FEATURE_NAMES)), np.float32), y=y, labels=labels,
                names=names, hashes=hashes, thumbs=thumbs, skipped=skipped)


def prepare(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((FEATURE_SIZE * 2, FEATURE_SIZE * 2))   # small copy only: keeps memory low on the free host
    return im


# ------------------------------------------------------------------ near-copy check (same method as the study)
def dhash(im, size=8):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash_variants(im):
    ims = [im, ImageOps.mirror(im), ImageOps.flip(im)]
    ims += [im.rotate(a, expand=True) for a in (90, 180, 270)]
    ims += [ImageOps.mirror(im.rotate(90, expand=True)), ImageOps.flip(im.rotate(90, expand=True))]
    return [dhash(x) for x in ims]


def _bits(h):
    return np.array([(h >> (63 - i)) & 1 for i in range(64)], dtype=np.float32)


def near_copy_groups(images=None, threshold=6, hashes=None):
    """Union-find over photos whose hashes (any of 8 flips/rotations) differ by <= threshold bits.
    Hamming distances are computed for all pairs at once with matrix products.
    Pass PIL images, or `hashes` = the dhash_variants of each photo."""
    var = hashes if hashes is not None else [dhash_variants(im) for im in images]   # n x 8, first = the photo itself
    n = len(var)
    if n == 0:
        return np.array([], dtype=int)
    B = np.vstack([_bits(v[0]) for v in var])                     # n x 64
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in range(8):
        V = np.vstack([_bits(v[k]) for v in var])
        for s0 in range(0, n, 1000):                              # chunks keep memory small
            Bc = B[s0:s0 + 1000]
            D = Bc @ (1 - V).T + (1 - Bc) @ V.T                   # Hamming distance, chunk x n
            ii, jj = np.where(D <= threshold)
            for i, j in zip(ii + s0, jj):
                if i != j:
                    ra, rb = find(int(i)), find(int(j))
                    if ra != rb:
                        parent[ra] = rb
    return np.array([find(i) for i in range(n)])


# ------------------------------------------------------------------ features
def features(im):
    """54 numbers per photo: colour histograms, colour statistics, 'pale' share and texture."""
    im = im.resize((FEATURE_SIZE, FEATURE_SIZE))
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f = []
    for ch, bins in ((h, 12), (s, 8), (v, 8)):
        hist, _ = np.histogram(ch, bins=bins, range=(0, 1)); f += list(hist / ch.size)
    for c in range(3):
        f += [rgb[..., c].mean(), rgb[..., c].std()]
    f += [s.mean(), s.std(), v.mean(), v.std()]
    pale = (v > 0.7) & (s < 0.25)                       # white-ish, the look of bleached coral
    f += [pale.mean(), ((v > 0.85) & (s < 0.15)).mean()]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    f += [(b - r).mean(), (g - r).mean(), (b - g).mean()]
    grey = rgb.mean(2)
    gx = np.abs(np.diff(grey, axis=1)); gy = np.abs(np.diff(grey, axis=0))
    mag = np.concatenate([gx.ravel(), gy.ravel()])
    hist, _ = np.histogram(mag, bins=8, range=(0, 0.5)); f += list(hist / mag.size)
    f += [mag.mean(), mag.std(), float(np.mean(mag > 0.1))]
    return np.array(f, dtype=np.float32)


FEATURE_NAMES = ([f"hue_bin_{i}" for i in range(12)] + [f"sat_bin_{i}" for i in range(8)] + [f"val_bin_{i}" for i in range(8)]
                 + ["red_mean", "red_sd", "green_mean", "green_sd", "blue_mean", "blue_sd", "sat_mean", "sat_sd",
                    "val_mean", "val_sd", "pale_share", "very_pale_share", "blue_minus_red", "green_minus_red",
                    "blue_minus_green"] + [f"edge_bin_{i}" for i in range(8)] + ["edge_mean", "edge_sd", "edge_share"])


def feature_matrix(images):
    return np.vstack([features(im) for im in images])


# ------------------------------------------------------------------ models and evaluation
def make_model(kind, seed=0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "Random forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))


def cross_validate(X, y, groups, kind, folds=5, seed=0):
    """Returns (random-split predictions, group-aware predictions); each is an array aligned with y."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    n_min = int(np.bincount(y).min())
    k = max(2, min(folds, n_min))
    out = {}
    splitters = {"Random split": StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)}
    n_groups = len(np.unique(groups))
    if n_groups >= k:
        splitters["Near-copies kept together"] = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for name, sp in splitters.items():
        pred = np.full(len(y), -1)
        prob = np.zeros((len(y), int(y.max()) + 1))
        for tr, te in (sp.split(X, y, groups) if "Group" in type(sp).__name__ else sp.split(X, y)):
            if len(np.unique(y[tr])) < len(np.unique(y)):
                continue
            m = make_model(kind, seed).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te]); prob[te] = m.predict_proba(X[te])
        out[name] = (pred, prob)
    return out, k


def confusion(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, pred):
        if b >= 0:
            cm[a, b] += 1
    return cm


def per_class(cm, labels):
    rows = []
    for i, l in enumerate(labels):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
        rows.append(dict(Class=l, Precision=round(p, 3), Recall=round(r, 3),
                         F1=round(2 * p * r / (p + r), 3) if p + r else float("nan"), Photos=int(cm[i].sum())))
    return rows


def train_final(X, y, kind, seed=0):
    return make_model(kind, seed).fit(X, y)


def export(model, labels, info):
    return pickle.dumps(dict(model=model, labels=labels, feature_names=FEATURE_NAMES, feature_size=FEATURE_SIZE,
                             info=info, how_to_use="features = photo_lab.features(PIL image); model.predict_proba([features])"))


def sim_rates(cm, labels):
    """Recall of the bleached class and the share of healthy photos called bleached, for the survey simulation."""
    low = [l.lower() for l in labels]
    bi = next((i for i, l in enumerate(low) if "bleach" in l or "unhealthy" in l), None)
    hi = next((i for i, l in enumerate(low) if "healthy" in l and "un" not in l), None)
    if bi is None or hi is None:
        return None
    rec = cm[bi, bi] / cm[bi].sum() if cm[bi].sum() else None
    fa = cm[hi, bi] / cm[hi].sum() if cm[hi].sum() else None
    return None if rec is None or fa is None else (float(rec), float(fa))
