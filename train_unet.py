# ------------------------------------------------------------
# Tiny U-Net: outline healthy coral, bleached coral and sand/rock
# in every pixel of an underwater photo.
#
# Paste this whole file into ONE new Colab cell and run it.
# Runtime: T4 GPU strongly recommended (Runtime > Change runtime type).
# Time: about 5 min download + 5 min preparing + 15-25 min training.
#
# Data: Coralscapes (Sauder et al., 2025), 2,075 Red Sea reef photos
#       with pixel labels, Apache-2.0 licence, from Hugging Face
#       EPFL-ECEO/coralscapes.
#
# Saves to Google Drive reef_fyp/unet/:
#   unet_best.pt                  trained model
#   unet_results.json             scores (copy the printout into the chat)
#   unet_examples.png             photo / true outline / predicted outline
#   unet_training_curve.png       learning curve
# ------------------------------------------------------------
import os, json, time, subprocess, sys
try:
    import datasets  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets import load_dataset
from PIL import Image

from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/reef_fyp/unet"
CACHE = "/content/coralscapes_256x512.npz"                    # fast local copy
DRIVE_CACHE = os.path.join(OUT, "coralscapes_256x512.npz")    # kept on Drive, survives restarts
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 256, 512            # photos are 1024 x 2048; shrunk by 4 to fit a small model
EPOCHS = 30
BATCH = 8
LR = 2e-3
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
print("Device:", DEVICE, "| epochs:", EPOCHS)

# ---------- 1. group the 39 Coralscapes labels into 4 classes ----------
CLASSES = ["Other", "Sand/rock", "Healthy coral", "Bleached coral"]
IGNORE = 255
SAND_ROCK = [5, 18, 12, 10, 3, 20, 23, 32, 37]          # sand, rubble, hard substrate, algae-covered, dead corals
HEALTHY = [6, 17, 21, 22, 25, 27, 28, 31, 34, 36]        # all "alive" corals, millepora, turbinaria
BLEACHED = [4, 16, 19, 33]                               # all "bleached" corals
LUT = np.zeros(256, dtype=np.uint8)                      # everything else -> Other (0)
LUT[0] = IGNORE                                          # 0 = unlabelled pixel
for i in SAND_ROCK: LUT[i] = 1
for i in HEALTHY: LUT[i] = 2
for i in BLEACHED: LUT[i] = 3

# ---------- 2. download and shrink (cached after the first run) ----------
if not os.path.exists(CACHE) and os.path.exists(DRIVE_CACHE):
    print("Copying prepared data from Google Drive...")
    subprocess.run(["cp", DRIVE_CACHE, CACHE], check=True)
if os.path.exists(CACHE):
    z = np.load(CACHE); data = {k: z[k] for k in z.files}
    print("Loaded prepared data (no download needed)")
else:
    print("Downloading Coralscapes (about 6 GB)...")
    ds = load_dataset("EPFL-ECEO/coralscapes")
    data = {}
    for split in ["train", "validation", "test"]:
        n = len(ds[split])
        X = np.zeros((n, H, W, 3), dtype=np.uint8)
        Y = np.zeros((n, H, W), dtype=np.uint8)
        for i, ex in enumerate(ds[split]):
            X[i] = np.asarray(ex["image"].convert("RGB").resize((W, H), Image.BILINEAR))
            lab = np.asarray(ex["label"].resize((W, H), Image.NEAREST))
            if lab.ndim == 3:
                lab = lab[..., 0]
            if i == 0 and split == "train":
                print("  label values in first image:", np.unique(lab)[:20])
                assert lab.max() <= 39, "Labels look like colours, not class ids. Send this message to Claude."
            Y[i] = LUT[lab]
            if i % 200 == 0:
                print(f"  {split}: {i}/{n}")
        data[f"X_{split}"], data[f"Y_{split}"] = X, Y
    np.savez(CACHE, **data)
if not os.path.exists(DRIVE_CACHE):
    print("Saving prepared data to Google Drive for next time...")
    subprocess.run(["cp", CACHE, DRIVE_CACHE], check=True)
if DEVICE == "cpu":
    raise SystemExit("\nNo GPU in this session. The data is saved, so nothing is lost.\n"
                     "Go to Runtime > Change runtime type > T4 GPU, then run this cell again.\n"
                     "Training on CPU takes about 30 minutes per round and does not learn bleached coral.")
for split in ["train", "validation", "test"]:
    print(split, data[f"X_{split}"].shape)

# pixel share of each class in training (used to weight the loss)
counts = np.bincount(data["Y_train"].ravel(), minlength=256)[:4].astype(float)
freq = counts / counts.sum()
print("Training pixel share:", {c: f"{100 * f:.1f}%" for c, f in zip(CLASSES, freq)})
class_w = torch.tensor((1 / np.maximum(freq, 1e-4)) ** 0.5, dtype=torch.float32)
class_w = class_w / class_w.mean()

# photos containing bleached coral are rare, so they are drawn 4 times as often during training
has_b = (data["Y_train"] == 3).reshape(len(data["Y_train"]), -1).mean(1) > 0.001
sample_p = np.where(has_b, 4.0, 1.0); sample_p = sample_p / sample_p.sum()
print(f"Training photos with bleached coral: {has_b.sum()} of {len(has_b)} (drawn 4x as often)")

# ---------- 3. the model ----------
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class TinyUNet(nn.Module):
    """A small U-Net: 4 levels, 16-32-64-128 channels."""
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

model = TinyUNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Tiny U-Net parameters: {n_params:,}")

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
def to_input(xb):
    x = torch.from_numpy(xb).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD

def dice_loss(logits, y, n_cls=4):
    """1 - mean Dice over classes (Equation 2 of the interim report), ignoring unlabelled pixels."""
    valid = (y != IGNORE).unsqueeze(1).float()
    y0 = y.clone(); y0[y0 == IGNORE] = 0
    p = logits.softmax(1) * valid
    oh = F.one_hot(y0.long(), n_cls).permute(0, 3, 1, 2).float() * valid
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1)).mean()

ce = nn.CrossEntropyLoss(weight=class_w.to(DEVICE), ignore_index=IGNORE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = int(np.ceil(len(data["X_train"]) / BATCH))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch)

def predict(X, bs=16):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(to_input(X[i:i + bs])).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)

def pixel_confusion(Y, P, k=4):
    m = Y != IGNORE
    return np.bincount(Y[m].astype(int) * k + P[m].astype(int), minlength=k * k).reshape(k, k)

def scores(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1); dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return iou, dice, tp.sum() / max(cm.sum(), 1)

# ---------- 4. train ----------
best_miou, history = -1, []
t_start = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); perm = np.random.choice(len(data["X_train"]), len(data["X_train"]), p=sample_p); tot = 0.0; t0 = time.time()
    for s in range(steps_per_epoch):
        idx = np.sort(perm[s * BATCH:(s + 1) * BATCH])
        x = to_input(data["X_train"][idx]); y = torch.from_numpy(data["Y_train"][idx]).to(DEVICE).long()
        if np.random.rand() < 0.5:                      # random left-right flip
            x, y = x.flip(3), y.flip(2)
        x = x * (1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=DEVICE) - 0.5)) \
            + 0.2 * (torch.rand(len(x), 1, 1, 1, device=DEVICE) - 0.5)   # brightness/contrast jitter
        logits = model(x)
        loss = ce(logits, y) + dice_loss(logits, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    Pv = predict(data["X_validation"])
    iou, dice, acc = scores(pixel_confusion(data["Y_validation"], Pv))
    history.append(dict(epoch=ep, loss=tot / steps_per_epoch, val_miou=float(iou.mean()), val_bleached_iou=float(iou[3])))
    print(f"Epoch {ep:2d}  loss {tot / steps_per_epoch:.3f}  val mIoU {iou.mean():.3f}  "
          f"bleached IoU {iou[3]:.3f}  ({time.time() - t0:.0f}s)")
    if iou.mean() > best_miou:
        best_miou = iou.mean()
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "size": [H, W]}, os.path.join(OUT, "unet_best.pt"))
train_minutes = (time.time() - t_start) / 60

# ---------- 5. test ----------
model.load_state_dict(torch.load(os.path.join(OUT, "unet_best.pt"), map_location=DEVICE)["state_dict"])
Pt = predict(data["X_test"]); Yt = data["Y_test"]
cm = pixel_confusion(Yt, Pt)
iou, dice, acc = scores(cm)

# one decision per photo, for the robot: bleached / healthy / sand-rock
BLEACH_MIN, CORAL_MIN = 0.05, 0.15     # share of seabed pixels (sand/rock + coral)
def photo_label(m):
    sand, healthy, bleached = [(m == c).sum() for c in (1, 2, 3)]
    seabed = sand + healthy + bleached
    if seabed == 0:
        return 0
    if bleached / seabed >= BLEACH_MIN:
        return 2
    if (healthy + bleached) / seabed >= CORAL_MIN:
        return 1
    return 0
true_img = np.array([photo_label(np.where(y == IGNORE, 0, y)) for y in Yt])
pred_img = np.array([photo_label(p) for p in Pt])
img_cm = np.zeros((3, 3), dtype=int)
for a, b in zip(true_img, pred_img):
    img_cm[a, b] += 1

# speed
model.eval(); x1 = to_input(data["X_test"][:1])
with torch.no_grad():
    for _ in range(5): model(x1)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): model(x1)
    if DEVICE == "cuda": torch.cuda.synchronize()
ms_gpu = (time.time() - t0) / 30 * 1000
cpu_model = TinyUNet(); cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}); cpu_model.eval()
xc = x1.cpu()
with torch.no_grad():
    cpu_model(xc); t0 = time.time()
    for _ in range(5): cpu_model(xc)
ms_cpu = (time.time() - t0) / 5 * 1000
size_mb = os.path.getsize(os.path.join(OUT, "unet_best.pt")) / 1e6

res = dict(
    dataset="Coralscapes (EPFL-ECEO/coralscapes), 4 grouped classes",
    classes=CLASSES, input_size=[H, W], parameters=int(n_params), model_size_mb=round(size_mb, 2),
    device=DEVICE, epochs=EPOCHS, train_minutes=round(train_minutes, 1),
    images={s: int(len(data[f"X_{s}"])) for s in ["train", "validation", "test"]},
    train_pixel_share={c: round(float(f), 4) for c, f in zip(CLASSES, freq)},
    test_pixel_accuracy=round(float(acc), 4),
    test_iou={c: round(float(v), 4) for c, v in zip(CLASSES, iou)},
    test_dice={c: round(float(v), 4) for c, v in zip(CLASSES, dice)},
    test_mean_iou=round(float(iou.mean()), 4),
    pixel_confusion=cm.tolist(),
    photo_rule=dict(bleached_min_share=BLEACH_MIN, coral_min_share=CORAL_MIN),
    photo_classes=["Sand/rock", "Healthy coral", "Bleached coral"],
    photo_confusion=img_cm.tolist(),          # rows = true photo label, columns = predicted
    photo_accuracy=round(float(np.trace(img_cm) / img_cm.sum()), 4),
    ms_per_image_gpu=round(ms_gpu, 2), ms_per_image_cpu=round(ms_cpu, 1),
    history=history,
)
res["headline"] = {"Mean IoU (test)": f"{100 * res['test_mean_iou']:.1f}%",
                   "Bleached coral Dice (test)": f"{res['test_dice']['Bleached coral']:.3f}",
                   "Model size": f"{res['model_size_mb']} MB, {n_params / 1e6:.2f} M parameters"}
json.dump(res, open(os.path.join(OUT, "unet_results.json"), "w"), indent=2)

# ---------- 6. pictures ----------
cmap = ListedColormap(["#4d4d4d", "#e8d8a8", "#5a9e6f", "#ffffff"])
has_bleach = [i for i in range(len(Yt)) if (Yt[i] == 3).mean() > 0.03]
pick = (has_bleach[:3] + [i for i in range(len(Yt)) if i not in has_bleach][:1])[:4]
fig, axes = plt.subplots(len(pick), 3, figsize=(12, 2.2 * len(pick)), squeeze=False)
for r, i in enumerate(pick):
    y = Yt[i].astype(float); y[Yt[i] == IGNORE] = np.nan
    axes[r, 0].imshow(data["X_test"][i]); axes[r, 1].imshow(np.ma.masked_invalid(y), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[r, 2].imshow(Pt[i], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    for a in axes[r]: a.axis("off")
axes[0, 0].set_title("Photo"); axes[0, 1].set_title("True outline"); axes[0, 2].set_title("Tiny U-Net outline")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=cmap(i), ec="#999", label=c) for i, c in enumerate(CLASSES)], loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "unet_examples.png"), dpi=150)
fig, ax = plt.subplots(figsize=(5, 3.3))
ax.plot([h["epoch"] for h in history], [h["val_miou"] for h in history], "o-", label="Validation mean IoU")
ax.plot([h["epoch"] for h in history], [h["val_bleached_iou"] for h in history], "s-", label="Validation bleached IoU")
ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "unet_training_curve.png"), dpi=200)

print("\n===== COPY EVERYTHING BELOW INTO THE CHAT =====")
print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
