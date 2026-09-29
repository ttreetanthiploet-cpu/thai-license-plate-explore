#!/usr/bin/env python3
"""
train_letter_cnn.py - train the small CNN that reads ONE Thai plate letter.

Why: the letters on a plate are isolated characters with no word context, so a
per-glyph classifier is a natural fit. EasyOCR's Thai model is weakest exactly
on the rare consonants (ฒ ฌ ฎ ฏ ฐ ฑ ฬ ...); this model sees every class equally.

Training data (all free, offline once downloaded):
  * 42 Thai consonants + 10 digits rendered from ~90 free OFL font files
    (Google Fonts) plus any Thai fonts you point it to
  * heavy augmentation: stroke weight, rotation, shear, width, crop jitter,
    low resolution, blur, noise, shading, JPEG
  * optional: your own labelled plate glyphs (folders named by the character),
    e.g. exported with  thai_plate_reader.py --save-glyphs glyphs/
    -> this is the best way to adapt the model to the real plate typeface

Usage
  python train_letter_cnn.py --download-fonts fonts/            # once (~13 MB, GitHub)
  python train_letter_cnn.py --fonts fonts/ --out thai_letter_cnn.pt
  python train_letter_cnn.py --fonts fonts/ --real glyphs/ --out thai_letter_cnn.pt
  python train_letter_cnn.py --fonts fonts/ --holdout Niramit,K2D  # measure generalisation

Needs: torch, opencv-python, numpy, Pillow (all already installed with EasyOCR).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from thai_plate_reader import (LETTER_CNN_CLASSES, LETTER_CNN_SIZE, build_letter_cnn,
                               glyph_to_square, imread)

# free OFL fonts with Thai glyphs (Google Fonts repository on GitHub)
GOOGLE_FONTS = {
    "sarabun": "Sarabun", "kanit": "Kanit", "prompt": "Prompt", "niramit": "Niramit",
    "krub": "Krub", "k2d": "K2D", "koho": "KoHo", "kodchasan": "Kodchasan",
    "thasadith": "Thasadith", "baijamjuree": "BaiJamjuree", "chakrapetch": "ChakraPetch",
    "mitr": "Mitr", "athiti": "Athiti", "pridi": "Pridi", "taviraj": "Taviraj",
    "trirong": "Trirong", "maitree": "Maitree", "fahkwang": "Fahkwang",
    "ibmplexsansthai": "IBMPlexSansThai", "ibmplexsansthailooped": "IBMPlexSansThaiLooped",
    "mali": "Mali", "chonburi": "Chonburi",
}
VARIABLE_FONTS = {"notosansthai": "NotoSansThai[wdth,wght].ttf",
                  "notosansthailooped": "NotoSansThaiLooped[wdth,wght].ttf",
                  "notoserifthai": "NotoSerifThai[wdth,wght].ttf",
                  "anuphan": "Anuphan[wght].ttf"}
STYLES = ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/"


def download_fonts(folder):
    os.makedirs(folder, exist_ok=True)
    jobs = [(d, f"{p}-{s}.ttf") for d, p in GOOGLE_FONTS.items() for s in STYLES]
    jobs += list(VARIABLE_FONTS.items())
    n = 0
    for d, f in jobs:
        out = os.path.join(folder, f.replace("[", "_").replace("]", "_").replace(",", "_"))
        if os.path.exists(out):
            n += 1
            continue
        try:
            urllib.request.urlretrieve(RAW + d + "/" + urllib.parse.quote(f), out)
            n += 1
        except Exception:
            pass  # not every family has every weight
    print(f"{n} font files in {folder}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def font_instances(path, size=96):
    """Yield (name, ImageFont) - several weights for variable fonts."""
    base = os.path.splitext(os.path.basename(path))[0]
    font = ImageFont.truetype(path, size)
    try:
        axes = font.get_variation_axes()
    except Exception:
        axes = []
    if not axes:
        yield base, font
        return
    for wght in (400, 600, 800):
        f = ImageFont.truetype(path, size)
        values = []
        for a in axes:
            name = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
            v = wght if name.lower().startswith("weight") else a["default"]
            values.append(min(max(v, a["minimum"]), a["maximum"]))
        f.set_variation_by_axes(values)
        yield f"{base}-w{wght}", f


def render_tight(font, ch):
    img = Image.new("L", (220, 220), 255)
    ImageDraw.Draw(img).text((50, 30), ch, font=font, fill=0)
    a = np.array(img)
    ys, xs = np.where(a < 128)
    if len(ys) < 30:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def render_bases(font_files, classes):
    """-> list of (image, class_index, font_name)"""
    bases = []
    for path in font_files:
        for name, font in font_instances(path):
            missing = render_tight(font, "")  # .notdef box for a missing glyph
            for ci, ch in enumerate(classes):
                g = render_tight(font, ch)
                if g is None or (missing is not None and g.shape == missing.shape
                                 and np.array_equal(g, missing)):
                    continue
                bases.append((g, ci, name))
    return bases


def load_real(folder, classes):
    """Real glyph crops stored as <folder>/<char>/*.png"""
    bases = []
    for ci, ch in enumerate(classes):
        for p in glob.glob(os.path.join(folder, ch, "*")):
            g = imread(p)
            if g is not None:
                bases.append((cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), ci, "real"))
    return bases


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------
def augment(glyph, rng, size=LETTER_CNN_SIZE):
    g = glyph
    h, w = g.shape
    p = int(0.35 * max(h, w)) + 4
    g = cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_CONSTANT, value=255)

    # stroke weight: thicker (erode dark-on-white) / thinner (dilate)
    r = int(round(h * rng.uniform(0.01, 0.04)))
    mode = rng.integers(0, 3)
    if r >= 1 and mode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        g = cv2.erode(g, k) if mode == 1 else cv2.dilate(g, k)

    # rotation, shear, width change
    H, W = g.shape
    ang = np.deg2rad(rng.uniform(-6, 6))
    sh, sx = rng.uniform(-0.18, 0.18), rng.uniform(0.82, 1.18)
    A = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ \
        np.array([[sx, sh], [0, 1.0]])
    c = np.array([W / 2, H / 2])
    M = np.hstack([A, (c - A @ c)[:, None]])
    g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)

    # tight crop with jitter (segmentation is never pixel perfect)
    ys, xs = np.where(g < 128)
    if len(ys) == 0:
        return glyph_to_square(glyph, size)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hh = y1 - y0
    j = lambda: int(round(hh * rng.uniform(-0.02, 0.06)))
    y0, y1 = max(0, y0 - j()), min(H, y1 + j())
    x0, x1 = max(0, x0 - j()), min(W, x1 + j())
    g = g[y0:y1, x0:x1]

    # low resolution + blur
    gh = rng.uniform(12, 70)
    s = gh / g.shape[0]
    small = cv2.resize(g, (max(2, int(g.shape[1] * s)), max(2, int(gh))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 1.0))
    g = small.astype(np.float32)

    # ink / paper levels, shading, noise
    ink, paper = rng.uniform(0, 120), rng.uniform(170, 255)
    g = ink + (paper - ink) * (g / 255.0)
    ramp = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(1.0, 1.1), g.shape[1])[None, :]
    g = g * (ramp if rng.random() < 0.5 else 1.0)
    g = g + rng.normal(0, rng.uniform(0, 10), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        q = int(rng.integers(30, 90))
        g = cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 0)
    bg = int(np.percentile(g, 90))
    return glyph_to_square(g, size, bg=bg)


def make_set(bases, rng, per_base, real_repeat=1):
    items = [(g, y) for g, y, name in bases
             for _ in range(per_base * (real_repeat if name == "real" else 1))]
    X = np.empty((len(items), 1, LETTER_CNN_SIZE, LETTER_CNN_SIZE), np.uint8)
    Y = np.empty(len(items), np.int64)
    for i, (g, y) in enumerate(items):
        X[i, 0] = augment(g, rng)
        Y[i] = y
    return X, Y


def to_tensor(X):
    import torch
    return (torch.from_numpy(X).float() / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(bases, val_bases, out, epochs=12, per_base=6, real_repeat=20, seed=0):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_letter_cnn(len(LETTER_CNN_CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xv = Yv = None
    if val_bases:
        Xv, Yv = make_set(val_bases, np.random.default_rng(123), 4)
    steps_per_epoch = None
    for ep in range(epochs):
        t = time.time()
        X, Y = make_set(bases, rng, per_base, real_repeat)
        order = rng.permutation(len(X))
        if steps_per_epoch is None:
            steps_per_epoch = int(np.ceil(len(X) / 256))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs,
                                                        steps_per_epoch=steps_per_epoch)
        model.train()
        tot = n = 0
        for s in range(steps_per_epoch):
            idx = order[s * 256:(s + 1) * 256]
            if len(idx) == 0:
                break
            xb, yb = to_tensor(X[idx]), torch.from_numpy(Y[idx])
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        msg = f"epoch {ep + 1:2d}/{epochs}  loss {tot / n:.3f}  ({time.time() - t:.0f}s, {len(X)} samples)"
        if Xv is not None:
            msg += f"  held-out fonts acc {evaluate(model, Xv, Yv):.3f}"
        print(msg, flush=True)
    torch.save({"classes": LETTER_CNN_CLASSES, "size": LETTER_CNN_SIZE,
                "state_dict": model.state_dict(), "version": 1}, out)
    print("saved", out)
    return model


def evaluate(model, X, Y):
    import torch
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(to_tensor(X[i:i + 512])).argmax(1)
                          for i in range(0, len(X), 512)]).numpy()
    return float((pred == Y).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-fonts", metavar="DIR", help="download the free fonts and exit")
    ap.add_argument("--fonts", nargs="+", help="font folders or files (.ttf/.otf)")
    ap.add_argument("--real", help="folder of real glyph crops: <folder>/<char>/*.png")
    ap.add_argument("--holdout", default="", help="comma-separated font name prefixes kept for validation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-glyph", type=int, default=6, help="augmented samples per rendered glyph per epoch")
    ap.add_argument("--out", default="thai_letter_cnn.pt")
    args = ap.parse_args()

    if args.download_fonts:
        download_fonts(args.download_fonts)
        return
    files = []
    for f in args.fonts or []:
        files += sorted(glob.glob(os.path.join(f, "*.[ot]tf"))) if os.path.isdir(f) else [f]
    if not files:
        ap.error("no fonts given (use --download-fonts first, then --fonts DIR)")
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    t = time.time()
    bases = render_bases(files, LETTER_CNN_CLASSES)
    hold = tuple(h.strip().lower() for h in args.holdout.split(",") if h.strip())
    val = [b for b in bases if hold and b[2].lower().startswith(hold)]
    bases = [b for b in bases if not (hold and b[2].lower().startswith(hold))]
    if args.real:
        real = load_real(args.real, LETTER_CNN_CLASSES)
        print(f"{len(real)} real glyphs")
        bases += real
    print(f"{len(bases)} training glyph renders, {len(val)} held-out "
          f"({len({b[2] for b in bases})} font styles) in {time.time() - t:.0f}s")
    train(bases, val, args.out, epochs=args.epochs, per_base=args.per_glyph)


if __name__ == "__main__":
    main()
