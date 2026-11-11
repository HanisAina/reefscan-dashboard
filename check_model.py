# ------------------------------------------------------------
# Step 3: honesty checks for the trained model
# Paste into ONE new Colab cell AFTER train_classifier.py has run
# (same session is fine; a new session also works).
#
# Check A: near-copies. Are test/valid photos just flipped, rotated
#          or brightened versions of training photos?
# Check B: independent test. Accuracy on a different dataset
#          (Hugging Face esahit/coral-health-classification).
# Saves:   results/check_results.json, results/near_copies.png,
#          results/confusion_matrix_independent.png
# ------------------------------------------------------------
import os, json, zipfile, urllib.request
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")
ROOT    = "/content/drive/MyDrive/reef_fyp"
RESULTS = os.path.join(ROOT, "results")
LOCAL   = "/content/coral_local"
BASE    = os.path.join(LOCAL, "Coral Reef Images")
if not os.path.isdir(BASE):
    with zipfile.ZipFile(os.path.join(ROOT, "archive.zip")) as z:
        z.extractall(LOCAL)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXT)]
    return sorted(out)

# ================= Check A: near-copies =================
def dhash(img):
    """64-bit fingerprint that survives resizing and brightness changes."""
    a = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]

def all_versions(img):
    """The photo plus its flips and rotations (8 versions)."""
    vs = []
    for base in (img, img.transpose(Image.FLIP_LEFT_RIGHT)):
        for r in (0, 90, 180, 270):
            vs.append(base.rotate(r, expand=True))
    return vs

def popcount64(x):
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)

print("Fingerprinting training photos...")
train_files = list_images(os.path.join(BASE, "train"))
train_h = np.array([dhash(Image.open(f)) for f in train_files], dtype=np.uint64)

THRESH = 6   # out of 64 bits; 0 = identical, under ~6 = visually the same photo
near = {}
examples = []
for split in ["valid", "test"]:
    files = list_images(os.path.join(BASE, split))
    count = 0
    for f in files:
        best_d, best_j = 64, -1
        for v in all_versions(Image.open(f)):
            d = popcount64(train_h ^ np.uint64(dhash(v)))
            j = int(d.argmin())
            if d[j] < best_d:
                best_d, best_j = int(d[j]), j
        if best_d <= THRESH:
            count += 1
            if len(examples) < 6:
                examples.append((f, train_files[best_j], best_d))
    near[split] = dict(total=len(files), near_copies=count,
                       percent=round(100 * count / len(files), 1))
    print(f"{split}: {count} of {len(files)} photos have a near-copy in training")

if examples:
    fig, axes = plt.subplots(2, len(examples), figsize=(2.2 * len(examples), 4.6))
    axes = np.array(axes).reshape(2, -1)
    for i, (a, b, d) in enumerate(examples):
        axes[0, i].imshow(Image.open(a)); axes[0, i].set_title(f"test/valid\ndiff {d}", fontsize=8)
        axes[1, i].imshow(Image.open(b)); axes[1, i].set_title("training", fontsize=8)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "near_copies.png"), dpi=150)
    plt.show()

# ================= Check B: independent dataset =================
URL = ("https://huggingface.co/datasets/esahit/coral-health-classification"
       "/resolve/main/total_dataset.zip")
HZIP, HDIR = "/content/hf_coral.zip", "/content/hf_coral"
if not os.path.isdir(HDIR):
    print("Downloading independent dataset (about 109 MB)...")
    urllib.request.urlretrieve(URL, HZIP)
    with zipfile.ZipFile(HZIP) as z:
        z.extractall(HDIR)

ckpt = torch.load(os.path.join(RESULTS, "best_model.pt"), map_location=DEVICE)
CLASSES = ckpt["classes"]                      # ['Bleached', 'Healthy']
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
model.load_state_dict(ckpt["state_dict"]); model.to(DEVICE).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# file names look like "healthy_BHD_12.png"; "unhealthy" means bleached; skip "dead"
NAME_TO_CLASS = {"healthy": "Healthy", "unhealthy": "Bleached", "bleached": "Bleached"}
items = []
for f in list_images(HDIR):
    lab = os.path.basename(f).split("_")[0].lower()
    if lab in NAME_TO_CLASS and "__MACOSX" not in f:
        items.append((f, CLASSES.index(NAME_TO_CLASS[lab]), os.path.basename(f).split("_")[1]))
print("Independent photos used:", len(items))

k = len(CLASSES); cm = np.zeros((k, k), dtype=int)
by_source = {}
with torch.no_grad():
    for i in range(0, len(items), 64):
        chunk = items[i:i + 64]
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f, _, _ in chunk]).to(DEVICE)
        p = model(x).argmax(1).cpu().tolist()
        for (f, y, src), pr in zip(chunk, p):
            cm[y, pr] += 1
            s = by_source.setdefault(src, [0, 0]); s[0] += int(y == pr); s[1] += 1

ind_acc = np.trace(cm) / cm.sum()
per_class = {}
for i, c in enumerate(CLASSES):
    tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    per_class[c] = dict(precision=round(prec, 4), recall=round(rec, 4), support=int(cm[i].sum()))

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(k)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(k)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for a in range(k):
    for b in range(k):
        ax.text(b, a, cm[a, b], ha="center", va="center",
                color="white" if cm[a, b] > cm.max() / 2 else "black")
ax.set_title(f"Independent dataset, accuracy {ind_acc:.1%}")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "confusion_matrix_independent.png"), dpi=200)
plt.show()

out = dict(near_copy_check=dict(threshold_bits=THRESH, **near),
           independent_test=dict(
               dataset="Hugging Face: esahit/coral-health-classification (healthy + unhealthy only)",
               accuracy=round(float(ind_acc), 4), per_class=per_class,
               confusion_matrix=cm.tolist(),
               accuracy_by_source={s: round(c / n, 4) for s, (c, n) in by_source.items()}))
with open(os.path.join(RESULTS, "check_results.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\n===== COPY EVERYTHING BELOW INTO THE CHAT =====")
print(json.dumps(out, indent=1))
