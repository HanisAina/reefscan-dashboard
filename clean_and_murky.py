# ------------------------------------------------------------
# Step 4: close Gap 4 (shortcut photos) and test Problem 3 (murky water)
# Paste this whole file into ONE new Colab cell and run it.
# Runtime: T4 GPU (Runtime > Change runtime type > T4 GPU).
# Time: about 15-25 minutes.
#
# Part A (Gap 4): photos that are near-copies of each other (frames from
#   the same video) are grouped together. The data is re-split so that
#   no group is in both training and testing, and the classifier is
#   retrained on this clean split.
# Part B (Problem 3): both models are tested on photos made progressively
#   murkier (red light lost, haze, less light, blur, noise), with and
#   without a simple colour-correction step.
#
# Needs these files in Google Drive (from earlier steps):
#   reef_fyp/archive.zip                      Kaggle dataset
#   reef_fyp/results/best_model.pt            original classifier
#   reef_fyp/unet/unet_best.pt                Tiny U-Net        (optional)
#   reef_fyp/unet/coralscapes_256x512.npz     U-Net test photos (optional)
#
# Saves to reef_fyp/results/:
#   clean_model.pt, clean_retrain.json, robustness.json,
#   murky_examples.png, murky_curves.png
# ------------------------------------------------------------
import os, json, time, zipfile, random, urllib.request
import numpy as np
from PIL import Image, ImageFilter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/reef_fyp"
RES = os.path.join(ROOT, "results")
UNET_DIR = os.path.join(ROOT, "unet")
os.makedirs(RES, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise SystemExit("No GPU in this session. Go to Runtime > Change runtime type > T4 GPU, then run again.")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

for name, p in [("Kaggle zip", os.path.join(ROOT, "archive.zip")),
                ("Original classifier", os.path.join(RES, "best_model.pt"))]:
    if not os.path.exists(p):
        raise SystemExit(f"Missing {name}: {p}")
HAVE_UNET = os.path.exists(os.path.join(UNET_DIR, "unet_best.pt")) and \
            os.path.exists(os.path.join(UNET_DIR, "coralscapes_256x512.npz"))
print("Device:", DEVICE, "| U-Net part:", "yes" if HAVE_UNET else "skipped (files not found)")

LOCAL = "/content/coral_local"
BASE = os.path.join(LOCAL, "Coral Reef Images")
if not os.path.isdir(BASE):
    print("Unzipping Kaggle dataset...")
    with zipfile.ZipFile(os.path.join(ROOT, "archive.zip")) as z:
        z.extractall(LOCAL)

CLASSES = ["Bleached", "Healthy"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_classifier():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m


# ================================================================ PART A
print("\n=== Part A: grouping near-copy photos ===")
items = []                                   # (path, label, original split)
for split in ["train", "valid", "test"]:
    for y, c in enumerate(CLASSES):
        d = os.path.join(BASE, split, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, f), y, split))
N = len(items)
print("Photos:", N)

TABLE16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def popcount(x):
    x = x.astype(np.uint64)
    m = np.uint64(0xFFFF)
    return (TABLE16[(x & m).astype(np.int64)].astype(np.int32) +
            TABLE16[((x >> np.uint64(16)) & m).astype(np.int64)] +
            TABLE16[((x >> np.uint64(32)) & m).astype(np.int64)] +
            TABLE16[((x >> np.uint64(48)) & m).astype(np.int64)])


def dhash(g):
    a = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(">u8")[0]


def versions(g):
    out = []
    for b in (g, g.transpose(Image.FLIP_LEFT_RIGHT)):
        for r in (0, 90, 180, 270):
            out.append(b.rotate(r, expand=True))
    return out


t0 = time.time()
smalls = []
for i, (p, _, _) in enumerate(items):
    im = Image.open(p)
    try:
        im.draft("L", (160, 160))           # fast JPEG decoding at low size
    except Exception:
        pass
    smalls.append(im.convert("L").resize((96, 72), Image.BILINEAR))
    if i % 2000 == 0:
        print(f"  read {i}/{N}")
H = np.array([dhash(s) for s in smalls], dtype=np.uint64)

parent = list(range(N))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


THRESH = 6
for i, s in enumerate(smalls):
    for v in versions(s):
        d = popcount(H ^ np.uint64(dhash(v)))
        for j in np.where(d <= THRESH)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    if i % 2000 == 0:
        print(f"  compared {i}/{N}")
roots = np.array([find(i) for i in range(N)])
groups = {}
for i, r in enumerate(roots):
    groups.setdefault(int(r), []).append(i)
sizes = np.array(sorted((len(g) for g in groups.values()), reverse=True))
print(f"Groups of near-copies: {len(groups)} (largest {sizes[0]}, median {int(np.median(sizes))}), {time.time() - t0:.0f}s")

# how many original test photos shared a group with an original training photo?
grp_splits = {r: {items[i][2] for i in g} for r, g in groups.items()}
leak_test = sum(1 for i, it in enumerate(items) if it[2] == "test" and "train" in grp_splits[int(roots[i])])
leak_valid = sum(1 for i, it in enumerate(items) if it[2] == "valid" and "train" in grp_splits[int(roots[i])])
n_test = sum(1 for it in items if it[2] == "test"); n_valid = sum(1 for it in items if it[2] == "valid")
print(f"Original test photos sharing a group with training: {leak_test}/{n_test}; validation: {leak_valid}/{n_valid}")

# group-aware split: 80 / 10 / 10 of each class, whole groups only
rng = random.Random(SEED)
split_of = {}
for y in range(len(CLASSES)):
    gs = [r for r, g in groups.items() if np.bincount([items[i][1] for i in g], minlength=2).argmax() == y]
    rng.shuffle(gs)
    total = sum(len(groups[r]) for r in gs)
    filled = {"test": 0, "valid": 0}
    for r in gs:
        if filled["test"] < 0.10 * total:
            s = "test"
        elif filled["valid"] < 0.10 * total:
            s = "valid"
        else:
            s = "train"
        split_of[r] = s
        if s in filled:
            filled[s] += len(groups[r])
new = {"train": [], "valid": [], "test": []}
for i, (p, y, _) in enumerate(items):
    new[split_of[int(roots[i])]].append((p, y))
for s in new:
    print(f"  clean {s}: {len(new[s])} photos ({np.bincount([y for _, y in new[s]], minlength=2).tolist()} Bleached/Healthy)")


class ListDS(Dataset):
    def __init__(self, it, tf):
        self.it, self.tf = it, tf
    def __len__(self):
        return len(self.it)
    def __getitem__(self, k):
        p, y = self.it[k]
        return self.tf(Image.open(p).convert("RGB")), y


train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
dl = lambda it, tf, sh: DataLoader(ListDS(it, tf), 64, shuffle=sh, num_workers=2)

print("\n=== Part A: retraining on the clean split ===")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
model = model.to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
EPOCHS = 6
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()


def predict(m, loader):
    m.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P += m(x.to(DEVICE)).argmax(1).cpu().tolist(); Y += list(y)
    return np.array(P), np.array(Y)


def report(P, Y):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(Y, P):
        cm[int(t), int(p)] += 1
    out = dict(accuracy=round(float(np.trace(cm) / cm.sum()), 4), confusion_matrix=cm.tolist(), per_class={})
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        out["per_class"][c] = dict(precision=round(float(tp / max(tp + fp, 1)), 4),
                                   recall=round(float(tp / max(tp + fn, 1)), 4), support=int(cm[i].sum()))
    return out


train_dl, valid_dl, test_dl = dl(new["train"], train_tf, True), dl(new["valid"], eval_tf, False), dl(new["test"], eval_tf, False)
best, hist = -1, []
clean_path = os.path.join(RES, "clean_model.pt")
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); seen = correct = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); out = model(x); loss = loss_fn(out, y); loss.backward(); opt.step()
        seen += len(y); correct += (out.argmax(1) == y).sum().item()
    sched.step()
    P, Y = predict(model, valid_dl); va = float((P == Y).mean())
    hist.append(dict(epoch=ep, train_acc=correct / seen, valid_acc=va))
    print(f"Epoch {ep}: train {correct / seen:.3f} | valid {va:.3f} | {time.time() - t0:.0f}s")
    if va > best:
        best = va
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, clean_path)
model.load_state_dict(torch.load(clean_path, map_location=DEVICE)["state_dict"])
clean_test = report(*predict(model, test_dl))
print("Clean test accuracy:", clean_test["accuracy"])

# independent dataset (Hugging Face), same as check_model.py
HF_DIR = "/content/hf_coral"
if not os.path.isdir(HF_DIR):
    print("Downloading independent dataset...")
    urllib.request.urlretrieve("https://huggingface.co/datasets/esahit/coral-health-classification/resolve/main/total_dataset.zip",
                               "/content/hf_coral.zip")
    with zipfile.ZipFile("/content/hf_coral.zip") as z:
        z.extractall(HF_DIR)
MAP = {"healthy": "Healthy", "unhealthy": "Bleached", "bleached": "Bleached"}
hf = []
for r, _, fs in os.walk(HF_DIR):
    if "__MACOSX" in r:
        continue
    for f in sorted(fs):
        lab = f.split("_")[0].lower()
        if f.lower().endswith(IMG_EXT) and not f.startswith("._") and lab in MAP:
            hf.append((os.path.join(r, f), CLASSES.index(MAP[lab])))
print("Independent photos:", len(hf))

orig = build_classifier()
orig.load_state_dict(torch.load(os.path.join(RES, "best_model.pt"), map_location="cpu")["state_dict"])
orig.to(DEVICE)
hf_dl = dl(hf, eval_tf, False)
ind_clean = report(*predict(model, hf_dl))
ind_orig = report(*predict(orig, hf_dl))
print(f"Independent accuracy: original {ind_orig['accuracy']:.4f} | clean {ind_clean['accuracy']:.4f}")

clean_res = dict(
    method="Near-copy groups (64-bit difference hash, 8 flips/rotations, <= 6 bits) joined into groups; "
           "whole groups assigned 80/10/10 per class so no group crosses splits.",
    photos=N, groups=len(groups), largest_group=int(sizes[0]), median_group=int(np.median(sizes)),
    groups_of_one=int((sizes == 1).sum()),
    original_test_sharing_group_with_train=dict(count=leak_test, total=n_test, percent=round(100 * leak_test / n_test, 1)),
    original_valid_sharing_group_with_train=dict(count=leak_valid, total=n_valid, percent=round(100 * leak_valid / n_valid, 1)),
    clean_split={s: len(v) for s, v in new.items()},
    clean_best_valid_acc=round(best, 4), history=hist,
    clean_test=clean_test, independent_clean_model=ind_clean, independent_original_model=ind_orig)
json.dump(clean_res, open(os.path.join(RES, "clean_retrain.json"), "w"), indent=2)

# ================================================================ PART B
print("\n=== Part B: murky water test ===")
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEVEL_NAMES = ["Clear", "Slightly murky", "Murky", "Very murky", "Extremely murky"]
VEIL = np.array([30, 105, 115], np.float32)       # blue-green water colour


def murky(img, t, seed):
    """Simulated murky water: red light lost, haze, less light, sensor noise, blur."""
    if t <= 0:
        return img
    a = np.asarray(img, np.float32).copy()
    a[..., 0] *= (1 - 0.6 * t)
    a = a * (1 - 0.55 * t) + VEIL * 0.55 * t
    a *= (1 - 0.35 * t)
    a += np.random.default_rng(seed).normal(0, 10 * t, a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im.filter(ImageFilter.GaussianBlur(radius=2.5 * t * im.width / 512))


def colour_correct(img):
    """Grey-world white balance + contrast stretch (a common simple underwater fix)."""
    a = np.asarray(img, np.float32)
    m = a.reshape(-1, 3).mean(0) + 1e-6
    a = a * np.clip(m.mean() / m, 0.5, 4.0)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = (a - lo) / max(hi - lo, 1e-6) * 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


hf_imgs = [Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR) for p, _ in hf]
hf_y = np.array([y for _, y in hf])
to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def clf_preds(m, imgs):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 128):
            x = torch.stack([to_t(im) for im in imgs[i:i + 128]]).to(DEVICE)
            out += m(x).argmax(1).cpu().tolist()
    return np.array(out)


rob = dict(levels=LEVELS, level_names=LEVEL_NAMES, classifier={}, unet={})
for name, m in [("original", orig), ("clean", model)]:
    rob["classifier"][name] = {"raw": [], "corrected": []}
for li, t in enumerate(LEVELS):
    deg = [murky(im, t, seed=k) for k, im in enumerate(hf_imgs)]
    cor = [colour_correct(im) for im in deg]
    for name, m in [("original", orig), ("clean", model)]:
        for key, ims in [("raw", deg), ("corrected", cor)]:
            r = report(clf_preds(m, ims), hf_y)
            r["level"] = t
            rob["classifier"][name][key].append(r)
    print(f"  classifier {LEVEL_NAMES[li]:16s} original {rob['classifier']['original']['raw'][-1]['accuracy']:.3f} "
          f"(corrected {rob['classifier']['original']['corrected'][-1]['accuracy']:.3f}) | clean "
          f"{rob['classifier']['clean']['raw'][-1]['accuracy']:.3f}")

example = None
if HAVE_UNET:
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class TinyUNet(nn.Module):
        def __init__(self, n_cls=4, c=(16, 32, 64, 128)):
            super().__init__()
            self.e1, self.e2, self.e3, self.b = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
            self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2); self.d3 = block(c[3], c[2])
            self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2); self.d2 = block(c[2], c[1])
            self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2); self.d1 = block(c[1], c[0])
            self.head = nn.Conv2d(c[0], n_cls, 1)
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
            b = self.b(F.max_pool2d(e3, 2))
            d3 = self.d3(torch.cat([self.u3(b), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)

    print("Loading U-Net test photos...")
    z = np.load(os.path.join(UNET_DIR, "coralscapes_256x512.npz"))
    Xt, Yt = z["X_test"], z["Y_test"]
    un = TinyUNet().to(DEVICE)
    un.load_state_dict(torch.load(os.path.join(UNET_DIR, "unet_best.pt"), map_location=DEVICE)["state_dict"]); un.eval()
    M_ = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1); S_ = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)
    IGNORE = 255

    def unet_predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                x = torch.from_numpy(X[i:i + 16]).to(DEVICE).permute(0, 3, 1, 2).float() / 255
                out.append(un((x - M_) / S_).argmax(1).cpu().numpy().astype(np.uint8))
        return np.concatenate(out)

    def photo_label(m):
        sand, healthy, bleached = [(m == c).sum() for c in (1, 2, 3)]
        seabed = sand + healthy + bleached
        if seabed == 0:
            return 0
        if bleached / seabed >= 0.05:
            return 2
        if (healthy + bleached) / seabed >= 0.15:
            return 1
        return 0

    true_img = np.array([photo_label(np.where(y == IGNORE, 0, y)) for y in Yt])
    rob["unet"] = {"raw": [], "corrected": []}
    for li, t in enumerate(LEVELS):
        deg = np.stack([np.asarray(murky(Image.fromarray(x), t, seed=k)) for k, x in enumerate(Xt)])
        cor = np.stack([np.asarray(colour_correct(Image.fromarray(x))) for x in deg])
        for key, X in [("raw", deg), ("corrected", cor)]:
            P = unet_predict(X)
            msk = Yt != IGNORE
            cm = np.bincount(Yt[msk].astype(int) * 4 + P[msk].astype(int), minlength=16).reshape(4, 4)
            tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
            iou = tp / np.maximum(tp + fp + fn, 1); dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            pc = np.zeros((3, 3), dtype=int)
            for a, b in zip(true_img, [photo_label(p) for p in P]):
                pc[a, b] += 1
            rob["unet"][key].append(dict(level=t, mean_iou=round(float(iou.mean()), 4), bleached_dice=round(float(dice[3]), 4),
                                         pixel_accuracy=round(float(tp.sum() / cm.sum()), 4),
                                         photo_confusion=pc.tolist(), photo_accuracy=round(float(np.trace(pc) / pc.sum()), 4)))
        print(f"  U-Net {LEVEL_NAMES[li]:16s} mIoU {rob['unet']['raw'][-1]['mean_iou']:.3f} "
              f"(corrected {rob['unet']['corrected'][-1]['mean_iou']:.3f})")
    # example strip from one Coralscapes test photo (Apache-2.0)
    k = int(np.argmax([(y == 3).mean() for y in Yt]))
    example = Image.fromarray(Xt[k])

json.dump(rob, open(os.path.join(RES, "robustness.json"), "w"), indent=2)

# ---------------- pictures
if example is not None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 4.6))
    for i, t in enumerate(LEVELS):
        d = murky(example, t, seed=1)
        axes[0, i].imshow(d); axes[0, i].set_title(LEVEL_NAMES[i], fontsize=10)
        axes[1, i].imshow(colour_correct(d)); axes[1, i].set_title("after colour correction", fontsize=9)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "murky_examples.png"), dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = [100 * t for t in LEVELS]
for name, ls in [("original", "-"), ("clean", "--")]:
    for key, col in [("raw", "#e8384f"), ("corrected", "#1259ff")]:
        r = rob["classifier"][name][key]
        ax[0].plot(x, [v["accuracy"] for v in r], ls, marker="o", color=col, label=f"{name}, {key}")
ax[0].set_title("Classifier accuracy (independent dataset)"); ax[0].set_xlabel("Murkiness (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
if rob["unet"]:
    for key, col in [("raw", "#e8384f"), ("corrected", "#1259ff")]:
        ax[1].plot(x, [v["mean_iou"] for v in rob["unet"][key]], marker="o", color=col, label=f"mean IoU, {key}")
        ax[1].plot(x, [v["bleached_dice"] for v in rob["unet"][key]], ":", marker="s", color=col, label=f"bleached Dice, {key}")
    ax[1].set_title("Tiny U-Net (Coralscapes test)"); ax[1].set_xlabel("Murkiness (%)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "murky_curves.png"), dpi=200)

# ---------------- summary to paste into the chat
short = dict(
    clean={k: clean_res[k] for k in ["photos", "groups", "largest_group", "median_group", "groups_of_one",
                                     "original_test_sharing_group_with_train", "original_valid_sharing_group_with_train",
                                     "clean_split", "clean_best_valid_acc"]},
    clean_test=clean_test, independent_clean_model=ind_clean, independent_original_model=ind_orig,
    murky_classifier={n: {k: [[v["accuracy"], v["per_class"]["Bleached"]["recall"], v["confusion_matrix"]] for v in rob["classifier"][n][k]]
                          for k in ["raw", "corrected"]} for n in ["original", "clean"]},
    murky_unet={k: [[v["mean_iou"], v["bleached_dice"], v["photo_accuracy"], v["photo_confusion"]] for v in rob["unet"][k]]
                for k in (["raw", "corrected"] if rob["unet"] else [])},
)
print("\n===== COPY EVERYTHING BELOW INTO THE CHAT =====")
print(json.dumps(short, separators=(",", ":")))
