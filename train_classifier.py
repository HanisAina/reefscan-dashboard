# ------------------------------------------------------------
# Step 2: train the coral health classifier (MobileNetV3-Small)
# Paste this whole file into ONE new Colab cell and run it.
# Works on GPU (fast) or CPU (slower, fewer epochs).
#
# Reads:  /content/drive/MyDrive/reef_fyp/archive.zip
# Saves:  /content/drive/MyDrive/reef_fyp/results/
#         - best_model.pt          the trained model
#         - results.json           accuracy, per-class scores, timings
#         - confusion_matrix.png   picture for the paper
#         - training_curve.png     picture for the paper
# ------------------------------------------------------------
import os, json, time, zipfile, hashlib, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- settings ----------
IN_COLAB = os.path.isdir("/content")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ZIP     = "/content/drive/MyDrive/reef_fyp/archive.zip"
    LOCAL   = "/content/coral_local"          # fast local copy
    RESULTS = "/content/drive/MyDrive/reef_fyp/results"
else:                                          # for testing outside Colab
    ZIP     = os.environ.get("CORAL_ZIP", "archive.zip")
    LOCAL   = os.environ.get("CORAL_LOCAL", "coral_local")
    RESULTS = os.environ.get("CORAL_RESULTS", "results")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 6 if DEVICE == "cuda" else 3      # fewer on CPU to save time
BATCH      = 64
LR         = 3e-4
SEED       = 42
PRETRAINED = os.environ.get("CORAL_PRETRAINED", "1") == "1"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(RESULTS, exist_ok=True)
print("Running on:", DEVICE, "| epochs:", EPOCHS)

# ---------- 1. unzip to the local disk (reading from Drive is very slow) ----------
BASE = os.path.join(LOCAL, "Coral Reef Images")
if not os.path.isdir(BASE):
    print("Unzipping to local disk...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(LOCAL)

# ---------- 2. check for identical photos shared between splits ----------
def file_hashes(split):
    out = {}
    for root, _, files in os.walk(os.path.join(BASE, split)):
        for f in files:
            with open(os.path.join(root, f), "rb") as fh:
                out[hashlib.md5(fh.read()).hexdigest()] = f
    return out
h = {s: file_hashes(s) for s in ["train", "valid", "test"]}
overlap = {
    "train_vs_valid": len(h["train"].keys() & h["valid"].keys()),
    "train_vs_test":  len(h["train"].keys() & h["test"].keys()),
    "valid_vs_test":  len(h["valid"].keys() & h["test"].keys()),
}
print("Identical photos shared between splits:", overlap)

# ---------- 3. load the photos ----------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

train_ds = datasets.ImageFolder(os.path.join(BASE, "train"), train_tf)
valid_ds = datasets.ImageFolder(os.path.join(BASE, "valid"), eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(BASE, "test"),  eval_tf)
CLASSES  = train_ds.classes
print("Classes:", CLASSES)
print("Images  train:", len(train_ds), "valid:", len(valid_ds), "test:", len(test_ds))

workers = 2 if IN_COLAB else 0
train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=workers)
valid_dl = DataLoader(valid_ds, BATCH, shuffle=False, num_workers=workers)
test_dl  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=workers)

# ---------- 4. the model: MobileNetV3-Small, pretrained on ImageNet ----------
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if PRETRAINED else None
model = models.mobilenet_v3_small(weights=weights)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
model = model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def run_eval(dl):
    model.eval(); preds, labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            preds += out.argmax(1).tolist(); labels += y.tolist()
    preds, labels = np.array(preds), np.array(labels)
    return total_loss / len(labels), (preds == labels).mean(), preds, labels

# ---------- 5. train ----------
history, best_acc = [], -1
best_path = os.path.join(RESULTS, "best_model.pt")
t_start = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); out = model(x); loss = loss_fn(out, y)
        loss.backward(); opt.step()
        run_loss += loss.item() * len(y); seen += len(y)
        correct += (out.argmax(1) == y).sum().item()
        if i % 25 == 0:
            print(f"  epoch {ep} batch {i}/{len(train_dl)}", end="\r")
    sched.step()
    v_loss, v_acc, _, _ = run_eval(valid_dl)
    history.append(dict(epoch=ep, train_loss=run_loss/seen, train_acc=correct/seen,
                         valid_loss=v_loss, valid_acc=float(v_acc)))
    print(f"Epoch {ep}: train acc {correct/seen:.3f} | valid acc {v_acc:.3f} "
          f"| {time.time()-t0:.0f}s")
    if v_acc > best_acc:
        best_acc = v_acc
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, best_path)
train_minutes = (time.time() - t_start) / 60

# ---------- 6. test on photos the model has never seen ----------
model.load_state_dict(torch.load(best_path, map_location=DEVICE)["state_dict"])
_, t_acc, preds, labels = run_eval(test_dl)
k = len(CLASSES)
cm = np.zeros((k, k), dtype=int)
for t, p in zip(labels, preds):
    cm[t, p] += 1
per_class = {}
for i, c in enumerate(CLASSES):
    tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_class[c] = dict(precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), support=int(cm[i].sum()))

# how fast is one photo? (matters for the robot)
model.eval(); x1 = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    for _ in range(5): model(x1)
    t0 = time.time()
    for _ in range(50): model(x1)
ms_per_image = (time.time() - t0) / 50 * 1000

results = dict(
    dataset="Kaggle: asfarhossainsitab/coral-reefs-images",
    classes=CLASSES, device=DEVICE, epochs=EPOCHS, pretrained=PRETRAINED,
    images=dict(train=len(train_ds), valid=len(valid_ds), test=len(test_ds)),
    split_overlap=overlap,
    best_valid_acc=round(float(best_acc), 4),
    test_acc=round(float(t_acc), 4),
    per_class=per_class,
    confusion_matrix=cm.tolist(),   # rows = true class, columns = predicted
    ms_per_image=round(ms_per_image, 2),
    train_minutes=round(train_minutes, 1),
    model_size_mb=round(os.path.getsize(best_path) / 1e6, 2),
    history=history)
with open(os.path.join(RESULTS, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)

# ---------- 7. pictures for the paper ----------
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(k)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(k)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(k):
    for j in range(k):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title(f"Test set, accuracy {t_acc:.1%}")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=200)

fig, ax = plt.subplots(figsize=(5, 3.5))
ep_ = [h_["epoch"] for h_ in history]
ax.plot(ep_, [h_["train_acc"] for h_ in history], "o-", label="Training")
ax.plot(ep_, [h_["valid_acc"] for h_ in history], "s-", label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "training_curve.png"), dpi=200)

# ---------- 8. summary to copy into the chat ----------
print("\n===== COPY EVERYTHING BELOW INTO THE CHAT =====")
print(json.dumps({k_: v for k_, v in results.items() if k_ != "history"}, indent=1))
