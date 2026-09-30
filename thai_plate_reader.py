#!/usr/bin/env python3
"""
thai_plate_reader.py - offline Thai licence-plate reader built on EasyOCR.

Layouts (plain or graphic background), told apart by the plate's shape:

    car, 34 x 15 cm, 2 lines
        +----------------------------+
        |  5กข       2662             |   <- top line : letters | number
        |       กรุงเทพมหานคร     (o)  |   <- bottom   : province (+ emblem)
        +----------------------------+

    motorcycle, 22 x 17.2 cm, 3 lines
        +------------------+
        |      7ขห     (o)  |   <- letters (+ emblem)
        |  กรุงเทพมหานคร     |   <- province
        |      6769        |   <- number
        +------------------+

Pipeline
  1. plate_candidates() / warp_plate()
                      find the plate's light rectangle and warp it flat
  2. segment_plate()  connected components -> crops
                        letters  : cut into single glyphs  [5] [ก] [ข]
                        number   : 1-4 digits
                        province : its own line, security emblem removed
  3. letters          each glyph is read ON ITS OWN by a letter engine
                      (EasyOCR / small CNN / Tesseract / PaddleOCR, or an
                      average of several), then the glyphs are combined and
                      checked against the plate format [digit?][consonant x1-2]
                      (motorcycles issued before ~2013: 3 consonants, กขค 123)
  4. number/province  EasyOCR *recogniser only* (the CRAFT detector is never
                      loaded) with per-field allow-lists; the province is picked
                      by scoring all official names with the CTC likelihood

CLI
  python thai_plate_reader.py car.jpg --debug debug.png
  python thai_plate_reader.py car.jpg --letters cnn          # or easyocr, easyocr+cnn ...
  python thai_plate_reader.py bike.jpg --layout motorcycle   # default: auto

Library
  from thai_plate_reader import ThaiPlateReader
  reader = ThaiPlateReader()          # load ONCE at start-up (a few seconds)
  result = reader.read("car.jpg")     # then ~1 s per plate on a 2-core CPU
  print(result["plate"], result["province"], result["needs_review"])

Not covered: wide CCTV scenes (crop the plate with a detector first, then
call read()).
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Character sets and the official province list
# ---------------------------------------------------------------------------
# the 42 modern Thai consonants (obsolete ฃ / ฅ are not in EasyOCR's charset)
THAI_CONSONANTS = "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
DIGITS = "0123456789"

PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น",
    "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย",
    "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน",
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี",
    "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก",
    "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี",
    "เบตง",  # Betong district issues its own plates
]
# Only characters that actually occur in province names are allowed there
PROVINCE_CHARS = "".join(sorted(set("".join(PROVINCES))))

# Top line: [optional digit] + 1-2 consonants,  then 1-4 digits (no leading 0)
LETTERS_RE = re.compile(rf"^[1-9]?[{THAI_CONSONANTS}]{{1,2}}$")
# motorcycles: the same, or 3 consonants on plates issued before ~2013 (กขค 123)
MOTO_LETTERS_RE = re.compile(rf"^(?:[1-9]?[{THAI_CONSONANTS}]{{1,2}}|[{THAI_CONSONANTS}]{{3}})$")
NUMBER_RE = re.compile(r"^[1-9][0-9]{0,3}$")

PLATE_H = 220  # height (px) of the rectified car plate; ratios below are of the plate height

# Plate layouts, tried in this order - the motorcycle one first because it needs
# two lines of large characters, so it cannot mistake a car plate for one:
#   motorcycle  22 x 17.2 cm, 3 lines: letters | province | number
#   car         34 x 15 cm,   2 lines: letters + number | province
# aspect: width / height of the plate face (frames hide part of it, a photo taken
# from above shortens it); height: px of the rectified plate - the motorcycle
# province line is small, so that plate is warped larger
LAYOUTS = {
    "motorcycle": dict(aspect=(1.1, 2.0), height=300, letters_re=MOTO_LETTERS_RE),
    "car": dict(aspect=(1.5, 4.0), height=PLATE_H, letters_re=LETTERS_RE),
}

# Small CNN that classifies ONE letter glyph (see train_letter_cnn.py)
LETTER_CNN_CLASSES = THAI_CONSONANTS + DIGITS
LETTER_CNN_SIZE = 40


def glyph_to_square(gray, size=LETTER_CNN_SIZE, bg=255):
    """Pad a tight glyph crop to a square (keeping its aspect ratio), resize.
    Shared by training and inference so both see identical inputs."""
    h, w = gray.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(gray, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=bg)
    return cv2.resize(sq, (size, size),
                      interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def build_letter_cnn(n_classes=len(LETTER_CNN_CLASSES)):
    """~130k-parameter CNN, 40x40 grey input -> class logits (a few ms on CPU)."""
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)]
    return nn.Sequential(
        *block(1, 16), *block(16, 16), nn.MaxPool2d(2),
        *block(16, 32), *block(32, 32), nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
        *block(64, 96), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(96, n_classes))


# ---------------------------------------------------------------------------
# Small geometry helper
# ---------------------------------------------------------------------------
@dataclass
class Box:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    ids: list = field(default_factory=list)  # connected-component labels

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0
    @property
    def cy(self): return (self.y0 + self.y1) / 2

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1),
                   self.ids + other.ids)

    def as_tuple(self):
        return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))


def _union_all(boxes):
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


def _merge_columns(boxes):
    """Merge boxes that overlap horizontally (pieces of the same glyph)."""
    merged = []
    for b in sorted(boxes, key=lambda b: b.x0):
        if merged and b.x0 < merged[-1].x1:
            merged[-1] = merged[-1].union(b)
        else:
            merged.append(b)
    return merged


class PlateNotFound(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 1. Plate localisation + rectification
# ---------------------------------------------------------------------------
def _order_corners(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])  # tl, tr, br, bl


def _contour_quad(c):
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return _order_corners(approx)
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(c)))


def _fitting_layouts(aspect, layouts):
    return [name for name in layouts
            if LAYOUTS[name]["aspect"][0] <= aspect <= LAYOUTS[name]["aspect"][1]]


def plate_candidates(bgr, max_candidates=4, layouts=tuple(LAYOUTS)):
    """Quadrilaterals that look like a plate face (light, rectangular, shaped
    like a plate of one of the layouts), most plate-like first, at most
    max_candidates per layout; the whole image is always the last one.

    Good for plate-centred photos (a phone shot of the plate / rear of vehicle).
    For wide CCTV-style scenes crop the plate with a detector (e.g. YOLO) first.
    """
    H, W = bgr.shape[:2]
    blur = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    t1, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    brighter = blur[blur > t1].reshape(-1, 1)
    # second, higher threshold separates the plate from a white car body
    t2 = cv2.threshold(brighter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] \
        if brighter.size > 100 else t1

    found = []
    for t in sorted({t1, t2}):
        light = (blur > t).astype(np.uint8) * 255
        contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if area < 0.005 * H * W or min(rw, rh) < 20:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            rectangularity = area / (rw * rh)
            fits = _fitting_layouts(aspect, layouts)
            if fits and rectangularity >= 0.8:
                found.append((area * rectangularity, _contour_quad(c), fits))

    quads, count = [], dict.fromkeys(layouts, 0)
    for _, q, fits in sorted(found, key=lambda f: -f[0]):
        if all(count[name] >= max_candidates for name in fits):
            continue
        if all(np.abs(q - p).max() > 10 for p in quads):   # skip duplicates
            quads.append(q)
            for name in fits:
                count[name] += 1
    quads.append(np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return quads


def plate_layouts(quad, layouts=tuple(LAYOUTS), whole_image=False):
    """Layouts to try on a plate candidate: the ones whose shape fits the quad;
    all of them for the whole-image fallback, or when none fits."""
    tl, tr, br, bl = quad
    width = np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
    height = np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)
    fits = _fitting_layouts(width / max(height, 1), layouts)
    return list(layouts) if whole_image or not fits else fits


def warp_plate(bgr, quad, out_h=PLATE_H):
    """Perspective-correct the plate so its height is out_h pixels."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    out_w = int(round(out_h * width / max(height, 1)))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 2. Segmentation into letters | number | province
# ---------------------------------------------------------------------------
def _otsu_separability(ch):
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x = ch.ravel().astype(np.float32)
    lo, hi = x[x <= t], x[x > t]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0 = lo.size / x.size
    return w0 * (1 - w0) * (lo.mean() - hi.mean()) ** 2 / (x.var() + 1e-6)


def normalise_plate(plate_bgr):
    """Return (norm, ink):
    norm - grey image, dark text on a flat white background (fed to the OCR)
    ink  - binary mask of text pixels (used for segmentation)

    Plates come in several ink colours (black, green, blue, red ...), so we pick
    the colour channel where text and background separate best, then divide
    out uneven lighting (shadows, glare gradients) before thresholding.
    """
    H = plate_bgr.shape[0]
    channels = [cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)] + list(cv2.split(plate_bgr))
    ch = max(channels, key=_otsu_separability)

    k = max(15, int(0.15 * H)) | 1  # wider than a stroke, so closing = background

    # polarity: text should be the minority (darker) class - judged with the
    # shading removed, as a strong shadow can make most of the plate "dark"
    flat = ch.astype(np.float32) - cv2.GaussianBlur(ch, (0, 0), k).astype(np.float32)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (flat <= t).mean() > 0.5:
        ch = 255 - ch

    background = cv2.morphologyEx(ch, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(ch, background, scale=255)
    _, ink = cv2.threshold(cv2.GaussianBlur(norm, (3, 3), 0), 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return norm, ink


def _is_ring(mask):
    """True for the round security emblem printed next to the province."""
    h, w = mask.shape
    if not 0.75 <= w / max(h, 1) <= 1.33:
        return False
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    hole = int((filled > 0).sum()) - int((mask > 0).sum())
    return hole / float(w * h) > 0.35


def _components(ink, speckle=0.0004):
    """Connected components of the ink that can be characters -> (boxes, labels).
    speckle: smallest component kept, as a fraction of the plate area."""
    H, W = ink.shape
    # erase long straight lines (printed border, holder edge) so that glyphs
    # touching them are not thrown away together with the frame
    lines = (cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3 * W), 1))) |
             cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.7 * H)))))
    ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < speckle * H * W:                     # speckle
            continue
        if w > 0.5 * W or h > 0.8 * H:                 # frame / plate holder
            continue
        elongated = w > 6 * h or h > 6 * w
        near_edge = (x < 0.08 * W or y < 0.08 * H or
                     x + w > 0.92 * W or y + h > 0.92 * H)
        if elongated and near_edge:                    # printed border line
            continue
        comps.append(Box(x, y, x + w, y + h, [i]))
    return comps, labels


def _glyph_row(comps, tall):
    """The tall components of one line -> glyph boxes, left to right, with the
    small pieces inside the line's band (detached tails, broken strokes) added."""
    y0, y1 = min(c.y0 for c in tall), max(c.y1 for c in tall)
    glyphs = _merge_columns(tall)
    tall_ids = {i for g in glyphs for i in g.ids}
    for c in comps:
        if c.ids[0] in tall_ids or not (y0 <= c.cy <= y1):
            continue
        for gi, g in enumerate(glyphs):
            if c.x0 < g.x1 and c.x1 > g.x0:
                glyphs[gi] = g.union(c)
                break
    return glyphs


def _province_box(low, labels):
    """The province word among the small components of its line."""
    cols = _merge_columns(low)
    ph = np.percentile([c.h for c in low], 75)          # typical glyph height
    clusters = [[cols[0]]]
    for c in cols[1:]:
        if c.x0 - clusters[-1][-1].x1 > 0.6 * ph:
            clusters.append([c])
        else:
            clusters[-1].append(c)
    # province names have no spaces -> the widest cluster is the province;
    # this drops the serial number / emblem when they stand apart
    words = max(clusters, key=lambda cl: cl[-1].x1 - cl[0].x0)
    # emblem glued to the end of the word? drop ring-shaped end pieces
    for end in (-1, 0):
        if len(words) > 3:
            b = words[end]
            mask = np.isin(labels[b.y0:b.y1, b.x0:b.x1], b.ids).astype(np.uint8) * 255
            if _is_ring(mask):
                words.pop(end)
    return _union_all(words)


def _segment_car(comps, labels, H, W):
    """2 lines: letters + number on top, province below."""
    # ---- top line: the tall glyphs ------------------------------------------
    tall = [c for c in comps if 0.25 * H <= c.h <= 0.8 * H and c.w <= 0.35 * W]
    if len(tall) < 2:
        raise PlateNotFound("could not find the large characters of the top line")
    cy = np.median([c.cy for c in tall])
    ch = np.median([c.h for c in tall])
    if cy > 0.65 * H:
        raise PlateNotFound("large characters are not in the upper part of the plate")
    tall = [c for c in tall if abs(c.cy - cy) < 0.35 * ch]
    glyphs = _glyph_row(comps, tall) if tall else []
    if len(glyphs) < 2:
        raise PlateNotFound("top line has fewer than two glyphs")

    # letters | number are separated by the widest gap; prefer splits that
    # respect the format (1-3 glyphs on the left, 1-4 on the right)
    gaps = [glyphs[i + 1].x0 - glyphs[i].x1 for i in range(len(glyphs) - 1)]
    valid = [i for i in range(len(gaps))
             if 1 <= i + 1 <= 3 and 1 <= len(glyphs) - (i + 1) <= 4]
    split = max(valid or range(len(gaps)), key=lambda i: gaps[i])

    # ---- bottom line: province ------------------------------------------------
    top_y1 = max(c.y1 for c in tall)
    used = {i for g in glyphs for i in g.ids}
    low = [c for c in comps
           if c.ids[0] not in used and c.y0 >= top_y1 - 0.03 * H and c.h < 0.6 * ch]
    if not low:
        raise PlateNotFound("no province line found under the registration number")
    return glyphs[:split + 1], glyphs[split + 1:], _province_box(low, labels)


def _segment_motorcycle(comps, labels, H, W):
    """3 lines: letters, province, number."""
    # ---- letters (top) and number (bottom): two lines of large glyphs --------
    big = [c for c in comps if 0.15 * H <= c.h <= 0.45 * H and c.w <= 0.3 * W]
    if len(big) < 2:
        raise PlateNotFound("could not find the large characters")
    ch = np.median([c.h for c in big])
    big = sorted((c for c in big if c.h >= 0.75 * ch), key=lambda c: c.cy)
    # the province line lies between them -> they are split by the widest gap
    gaps = [b.cy - a.cy for a, b in zip(big, big[1:])]
    if not gaps or max(gaps) < 1.2 * ch:
        raise PlateNotFound("large characters do not form two lines")
    k = int(np.argmax(gaps)) + 1
    rows = []
    for row in (big[:k], big[k:]):
        cy = np.median([c.cy for c in row])
        rows.append([c for c in row if abs(c.cy - cy) < 0.35 * ch])
    top, bottom = rows
    if not (top and bottom and
            np.median([c.cy for c in top]) < 0.5 * H < np.median([c.cy for c in bottom])):
        raise PlateNotFound("the lines of large characters are not above and below the middle")
    ratio = np.median([c.h for c in top]) / np.median([c.h for c in bottom])
    if not 0.75 <= ratio <= 1.33:
        raise PlateNotFound("letters and number lines differ in size")
    letter_glyphs, number_glyphs = _glyph_row(comps, top), _glyph_row(comps, bottom)

    # ---- middle line: province -----------------------------------------------
    top_y1 = max(c.y1 for c in top)
    bottom_y0 = min(c.y0 for c in bottom)
    used = {i for g in letter_glyphs + number_glyphs for i in g.ids}
    low = [c for c in comps
           if c.ids[0] not in used and c.h < 0.7 * ch
           and c.y0 >= top_y1 - 0.03 * H and c.y1 <= bottom_y0 + 0.03 * H]
    if not low:
        raise PlateNotFound("no province line found between the letters and the number")
    return letter_glyphs, number_glyphs, _province_box(low, labels)


def segment_plate(ink, layout="car"):
    """Split the binary text mask into letters / number / province boxes.
    layout: "car" (2 lines) or "motorcycle" (3 lines), see LAYOUTS.

    Returns dict(letters=Box, number=Box, province=Box, labels=label_image,
    letter_glyphs=[Box], number_glyphs=[Box], n_letter_glyphs=int,
    n_number_glyphs=int, layout=str).
    """
    H, W = ink.shape
    if layout == "car":
        comps, labels = _components(ink)
        letter_glyphs, number_glyphs, province = _segment_car(comps, labels, H, W)
    elif layout == "motorcycle":
        # the province line is small: keep its tiny vowel marks
        comps, labels = _components(ink, speckle=0.0002)
        letter_glyphs, number_glyphs, province = _segment_motorcycle(comps, labels, H, W)
    else:
        raise ValueError(f"unknown layout {layout!r}; choose from {tuple(LAYOUTS)}")
    return dict(letters=_union_all(letter_glyphs), number=_union_all(number_glyphs),
                province=province, labels=labels,
                letter_glyphs=letter_glyphs, number_glyphs=number_glyphs,
                n_letter_glyphs=len(letter_glyphs), n_number_glyphs=len(number_glyphs),
                layout=layout)


def field_crop(norm, ink, labels, box, pad_frac=0.12, pad_x_frac=None):
    """Crop a field (or a single glyph) from the normalised plate with a white
    margin, erasing ink that belongs to anything else (neighbouring glyphs,
    emblem, border lines). Margins are fractions of the box height."""
    img = norm.copy()
    foreign = (ink > 0) & ~np.isin(labels, box.ids)
    foreign = cv2.dilate(foreign.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img[foreign] = 255
    py = max(4, int(round(pad_frac * box.h)))
    px = py if pad_x_frac is None else max(4, int(round(pad_x_frac * box.h)))
    img = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=255)
    return img[box.y0:box.y1 + 2 * py, box.x0:box.x1 + 2 * px]


def glyph_crop(norm, ink, labels, box):
    """Tight crop of ONE letter glyph, with ink of neighbouring glyphs erased."""
    img = norm[box.y0:box.y1, box.x0:box.x1].copy()
    foreign = (ink[box.y0:box.y1, box.x0:box.x1] > 0) & \
        ~np.isin(labels[box.y0:box.y1, box.x0:box.x1], box.ids)
    img[cv2.dilate(foreign.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 255
    return img


# ---------------------------------------------------------------------------
# 3. Letter engines - each one reads ONE glyph at a time
# ---------------------------------------------------------------------------
# Every engine has  predict(glyphs, allowed) -> [probability vector per glyph],
# where glyphs are tight grey crops (dark text on white) and allowed[i] is the
# string of characters permitted at position i. The per-glyph results are then
# combined into the letter string (see combine_letters).

def letter_allowed_sets(n, layout="car"):
    """Allowed characters per letter position, from the plate format
    [optional digit 1-9] + 1-2 consonants (motorcycles: or 3 consonants)."""
    if n >= 3:
        first = (THAI_CONSONANTS if layout == "motorcycle" else "") + "123456789"
        return [first] + [THAI_CONSONANTS] * (n - 1)
    if n == 2:
        return [THAI_CONSONANTS + "123456789", THAI_CONSONANTS]
    return [THAI_CONSONANTS] * n


def _pad_white(g, frac=0.3):
    p = max(4, int(round(frac * g.shape[0])))
    return cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_CONSTANT, value=255)


def _single_char_dist(text, score, allowed):
    """Turn an engine's (text, score) into a probability vector over allowed: the
    character read gets `score` on top of an even share of the rest, so it stays
    the most likely one even when the engine is unsure (score < 1/len(allowed))."""
    ch = next((c for c in text if c in allowed), None)
    q = np.full(len(allowed), 1.0 / len(allowed))
    if ch is not None:
        score = float(np.clip(score, 0.0, 0.99))
        q *= 1 - score
        q[allowed.index(ch)] += score
    return q


class EasyOCRLetters:
    """EasyOCR, one glyph at a time. The glyph is centred on a wide white canvas
    (3x its height - like a lone character in a line of text) and every allowed
    character is scored with the CTC likelihood, so exactly one is returned."""
    name = "easyocr"

    def __init__(self, plate_reader, canvas_ratio=3.0):
        self.pr, self.k = plate_reader, canvas_ratio

    def canvas(self, g):
        h, w = g.shape
        py = max(2, int(round(0.12 * h)))
        W = max(int(round(self.k * h)), w + 2)
        out = np.full((h + 2 * py, W), 255, np.uint8)
        x0 = (W - w) // 2
        out[py:py + h, x0:x0 + w] = g
        return out

    def predict(self, glyphs, allowed):
        P = self.pr.frame_probs_batch([self.canvas(g) for g in glyphs])
        out = []
        for Pi, al in zip(P, allowed):
            p = self.pr._restrict(Pi, al)
            ll = self.pr._ctc_loglik(p, al, list(al))
            q = np.exp(ll - ll.max())
            out.append(q / q.sum())
        return out


class CNNLetters:
    """Small CNN trained on 42 consonants + 10 digits rendered from free fonts
    (train_letter_cnn.py). ~130k parameters, a few ms per plate on CPU."""
    name = "cnn"

    def __init__(self, weights=None):
        import torch
        self._torch = torch
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "thai_letter_cnn.pt")
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:  # very old torch
            ckpt = torch.load(weights, map_location="cpu")
        self.classes, self.size = ckpt["classes"], ckpt["size"]
        self.model = build_letter_cnn(len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.index = {c: i for i, c in enumerate(self.classes)}

    def predict(self, glyphs, allowed):
        torch = self._torch
        x = np.stack([glyph_to_square(g, self.size) for g in glyphs])[:, None]
        x = (torch.from_numpy(x).float() / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).numpy()
        out = []
        for p, al in zip(probs, allowed):
            q = p[[self.index[c] for c in al]]
            out.append(q / q.sum())
        return out


class TesseractLetters:
    """Tesseract 5 (LSTM) with the Thai model, one glyph at a time. Each glyph is
    read as a raw text line (--psm 13): single-character mode (--psm 10) returned
    nothing or a wrong letter for about a third of the plate glyphs in testing.
    Needs the tesseract program + Thai data (apt install tesseract-ocr-tha,
    conda install -c conda-forge tesseract, or tha.traineddata from
    github.com/tesseract-ocr/tessdata_best). A tessdata/ folder next to this file
    holding tha.traineddata is used when present, and when tesseract is not on the
    PATH it is also looked for next to the Python executable (venv / conda env)."""
    name = "tesseract"

    def __init__(self, tessdata_dir=None):
        import pytesseract
        self.pt = pytesseract
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
        if tessdata_dir is None and os.path.exists(os.path.join(local, "tha.traineddata")):
            tessdata_dir = local
        if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
            found = shutil.which("tesseract", path=os.path.dirname(sys.executable))
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
        tessdata = f'--tessdata-dir "{tessdata_dir}"' if tessdata_dir else ""
        if "tha" not in pytesseract.get_languages(config=tessdata):  # fail now, not at the first read
            raise RuntimeError("Tesseract has no Thai model: install tesseract-ocr-tha or put "
                               "tha.traineddata in a folder and pass tessdata_dir")
        self.cfg = f"{tessdata} --oem 1 --psm 13"

    def predict(self, glyphs, allowed):
        out = []
        for g, al in zip(glyphs, allowed):
            d = self.pt.image_to_data(_pad_white(g), lang="tha", output_type=self.pt.Output.DICT,
                                      config=f"{self.cfg} -c tessedit_char_whitelist={al}")
            words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if str(t).strip()]
            text = "".join(t for t, _ in words)
            score = max((c for _, c in words), default=0.0) / 100.0
            out.append(_single_char_dist(text, score, al))
        return out


class PaddleLetters:
    """PaddleOCR's Thai recogniser (th_PP-OCRv5_mobile_rec, Apache-2.0), one glyph
    at a time, run directly on paddlepaddle with PaddleOCR's own preprocessing
    (same output as PaddleOCR 3.7's TextRecognition). The paddleocr package is not
    needed: it pins opencv-contrib-python, which overwrites the headless OpenCV.
    As for EasyOCR, every allowed character is scored with the CTC likelihood.
    pip install paddlepaddle ; the model (8 MB) downloads from Hugging Face on
    first use into PaddleOCR's cache, ~/.paddlex/official_models/<model_name>."""
    name = "paddle"
    FILES = ("inference.json", "inference.pdiparams", "inference.yml")

    def __init__(self, plate_reader, model_dir=None, model_name="th_PP-OCRv5_mobile_rec"):
        import yaml
        with warnings.catch_warnings():  # "No ccache found" from paddle's C++ extension helper
            warnings.simplefilter("ignore")
            from paddle import inference
        self.pr = plate_reader
        model_dir = model_dir or os.path.join(os.path.expanduser("~"), ".paddlex",
                                              "official_models", model_name)
        for f in self.FILES:
            path = os.path.join(model_dir, f)
            if not os.path.exists(path):
                os.makedirs(model_dir, exist_ok=True)
                urllib.request.urlretrieve(f"https://huggingface.co/PaddlePaddle/{model_name}"
                                           f"/resolve/main/{f}", path + ".part")
                os.replace(path + ".part", path)
        with open(os.path.join(model_dir, "inference.yml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        # output columns: CTC blank, the model's dictionary, then a space
        self.index = {c: i + 1 for i, c in enumerate(cfg["PostProcess"]["character_dict"])}
        self.shape = next(op["RecResizeImg"]["image_shape"]            # [3, 48, 320]
                          for op in cfg["PreProcess"]["transform_ops"] if "RecResizeImg" in op)

        config = inference.Config(os.path.join(model_dir, "inference.json"),
                                  os.path.join(model_dir, "inference.pdiparams"))
        config.disable_gpu()
        config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(min(4, os.cpu_count() or 1))
        config.enable_new_ir(True)
        config.enable_new_executor()
        config.set_optimization_level(3)
        config.enable_memory_optim()
        config.disable_glog_info()
        self.predictor = inference.create_predictor(config)
        self.frame_probs([np.full((60, 40), 255, np.uint8)])  # warm-up

    def frame_probs(self, glyphs):
        """Per-frame softmax (N, T, C); PaddleOCR's resize, normalisation and padding."""
        C, H, W0 = self.shape
        imgs = [cv2.cvtColor(_pad_white(g), cv2.COLOR_GRAY2BGR) for g in glyphs]
        widths = [int(H * max(W0 / H, im.shape[1] / im.shape[0])) for im in imgs]
        x = np.zeros((len(imgs), C, H, max(widths)), np.float32)
        for i, (im, W) in enumerate(zip(imgs, widths)):
            w = min(math.ceil(H * im.shape[1] / im.shape[0]), W)
            im = cv2.resize(im, (w, H)).astype(np.float32).transpose(2, 0, 1) / 255
            x[i, :, :, :w] = (im - 0.5) / 0.5
        inp = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        inp.reshape(x.shape)
        inp.copy_from_cpu(x)
        self.predictor.run()
        return self.predictor.get_output_handle(self.predictor.get_output_names()[0]).copy_to_cpu()

    def predict(self, glyphs, allowed):
        out = []
        for g, al in zip(glyphs, allowed):
            P = self.frame_probs([g])[0]   # one at a time: faster than a batch on CPU
            p = P[:, [0] + [self.index[c] for c in al]]
            p = p / p.sum(1, keepdims=True)
            ll = self.pr._ctc_loglik(p, al, list(al))
            q = np.exp(ll - ll.max())
            out.append(q / q.sum())
        return out


class EnsembleLetters:
    """Weighted average of several engines' per-glyph probabilities."""

    def __init__(self, engines, weights=None):
        self.engines = list(engines)
        self.weights = list(weights or [1.0] * len(self.engines))
        self.name = "+".join(e.name for e in self.engines)
        self.last = {}

    def predict(self, glyphs, allowed):
        per = {e.name: e.predict(glyphs, allowed) for e in self.engines}
        self.last = per
        total = sum(self.weights)
        return [sum(w * per[e.name][i] for e, w in zip(self.engines, self.weights)) / total
                for i in range(len(glyphs))]


def combine_letters(dists, allowed, pattern=LETTERS_RE, k=3, top=5):
    """Per-glyph probabilities -> best string, joint probability, alternatives."""
    options = [[(al[j], float(p[j])) for j in np.argsort(-p)[:k]] for p, al in zip(dists, allowed)]
    cands = []
    for combo in itertools.product(*options):
        s = "".join(c for c, _ in combo)
        if pattern.match(s):
            cands.append((s, float(np.prod([q for _, q in combo]))))
    cands.sort(key=lambda c: -c[1])
    best = cands[0] if cands else ("".join(o[0][0] for o in options), 0.0)
    # an engine that saw nothing returns a flat distribution -> show '?', not a guess
    unread = [i for i, (p, al) in enumerate(zip(dists, allowed)) if p.max() <= 1.5 / len(al)]
    if unread:
        best = ("".join("?" if i in unread else c for i, c in enumerate(best[0])), 0.0)
    per_glyph = [[(c, round(q, 3)) for c, q in o] for o in options]
    return dict(text=best[0], conf=best[1], candidates=cands[:top], per_glyph=per_glyph,
                format_ok=bool(pattern.match(best[0])))


# ---------------------------------------------------------------------------
# 4. The reader: EasyOCR for number + province, a letter engine for the letters
# ---------------------------------------------------------------------------
LETTER_ENGINES = ("easyocr", "cnn", "tesseract", "paddle")
# weight of each engine when several are averaged (the CNN was the most
# accurate in testing; Tesseract's low weight dates from before its --psm fix)
ENGINE_WEIGHTS = {"cnn": 2.0, "easyocr": 1.0, "paddle": 1.0, "tesseract": 0.5}


class ThaiPlateReader:
    """Load once, call .read() many times.

    letters_engine - how the letters (e.g. 5กข) are read, one glyph at a time:
        "easyocr"            EasyOCR on each letter separately
        "cnn"                small font-trained CNN (thai_letter_cnn.pt) - most
                             accurate in testing and ~5 ms
        "tesseract"          Tesseract 5 LSTM (optional install, see TesseractLetters)
        "paddle"             PaddleOCR's Thai model on paddlepaddle (optional install)
        "easyocr+cnn" / ["easyocr", "cnn", ...]   weighted average of engines
                             (ENGINE_WEIGHTS); disagreement sets needs_review
        "group"              old behaviour: EasyOCR reads the letters crop in one go
        "auto" (default)     "easyocr+cnn" if the CNN weights exist, else "easyocr"
    layout - "auto" (default): car or motorcycle plate, told apart by the plate's
        shape; "car" or "motorcycle" to accept only that layout
    reader_kwargs are passed to easyocr.Reader - e.g. a fine-tuned model:
        ThaiPlateReader(recog_network="thai_plate", user_network_directory="models/",
                        model_storage_directory="models/")
    """

    def __init__(self, gpu=False, langs=("th",), letters_engine="auto", cnn_weights=None,
                 tessdata_dir=None, layout="auto", **reader_kwargs):
        if layout != "auto" and layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; choose from auto, {', '.join(LAYOUTS)}")
        self.layouts = tuple(LAYOUTS) if layout == "auto" else (layout,)
        import easyocr
        import easyocr.easyocr as _eo
        import torch

        self._torch = torch
        with warnings.catch_warnings():  # torch deprecation noise from EasyOCR's CPU quantisation
            warnings.simplefilter("ignore")
            # detector=False: we segment ourselves, so CRAFT is never loaded
            self.reader = easyocr.Reader(list(langs), gpu=gpu, detector=False,
                                         verbose=False, **reader_kwargs)
        self._imgH = _eo.imgH
        self._index = {c: i + 1 for i, c in enumerate(self.reader.character)}  # 0 = CTC blank
        missing = set(THAI_CONSONANTS + DIGITS + PROVINCE_CHARS) - set(self._index)
        if missing:
            raise ValueError(f"recognition model lacks characters: {''.join(sorted(missing))}")
        self.frame_probs(np.full((64, 256), 255, np.uint8))  # warm-up: first real call is not slower

        self.cnn_weights = cnn_weights or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "thai_letter_cnn.pt")
        self.tessdata_dir = tessdata_dir
        self._engines = {}
        if letters_engine == "auto":
            letters_engine = "easyocr+cnn" if os.path.exists(self.cnn_weights) else "easyocr"
        self.letters_engine = self.make_letters_engine(letters_engine)

    def make_letters_engine(self, spec):
        """'easyocr' | 'cnn' | 'tesseract' | 'paddle' | 'a+b' | ['a', 'b'] | 'group' | engine
        Each engine is created once and reused, so switching specs reloads no model."""
        if spec == "group" or hasattr(spec, "predict"):
            return spec
        names = spec.split("+") if isinstance(spec, str) else list(spec)
        for name in names:
            if name in self._engines:
                continue
            if name == "easyocr":
                self._engines[name] = EasyOCRLetters(self)
            elif name == "cnn":
                self._engines[name] = CNNLetters(self.cnn_weights)
            elif name == "tesseract":
                self._engines[name] = TesseractLetters(self.tessdata_dir)
            elif name == "paddle":
                self._engines[name] = PaddleLetters(self)
            else:
                raise ValueError(f"unknown letters engine {name!r}; choose from {LETTER_ENGINES}")
        engines = [self._engines[name] for name in names]
        if len(engines) == 1:
            return engines[0]
        return EnsembleLetters(engines, [ENGINE_WEIGHTS[e.name] for e in engines])

    # -- low level --------------------------------------------------------------
    def frame_probs_batch(self, grays):
        """frame_probs for several images in one forward pass -> (N, T, C)."""
        from PIL import Image
        from easyocr.recognition import AlignCollate

        imgH = self._imgH
        imgW = max(imgH, max(math.ceil(imgH * g.shape[1] / g.shape[0]) for g in grays))
        batch = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)(
            [Image.fromarray(g) for g in grays]).to(self.reader.device)
        with self._torch.no_grad():
            logits = self.reader.recognizer(batch, None)
        return self._torch.softmax(logits, dim=2).cpu().numpy()

    def frame_probs(self, gray):
        """Per-frame softmax over [blank] + model charset, shape (T, C).
        Same preprocessing as easyocr.Reader.recognize()."""
        from PIL import Image
        from easyocr.recognition import AlignCollate
        from easyocr.utils import get_image_list

        h, w = gray.shape[:2]
        image_list, max_width = get_image_list([[0, w, 0, h]], [], gray,
                                               model_height=self._imgH)
        collate = AlignCollate(imgH=self._imgH, imgW=int(max_width), keep_ratio_with_pad=True)
        batch = collate([Image.fromarray(image_list[0][1])]).to(self.reader.device)
        with self._torch.no_grad():
            logits = self.reader.recognizer(batch, None)
        return self._torch.softmax(logits, dim=2)[0].cpu().numpy()

    def _restrict(self, probs, allowed):
        """Keep blank + allowed chars, renormalise (what EasyOCR's allowlist does).
        Returns probs over a reduced alphabet ['' + allowed]."""
        cols = [0] + [self._index[c] for c in allowed]
        p = probs[:, cols]
        return p / p.sum(1, keepdims=True)

    @staticmethod
    def _greedy(p, alphabet):
        """Best-path CTC decode -> (text, easyocr-style confidence, per-char dists)."""
        best = p.argmax(1)
        text, spans, t = [], [], 0
        while t < len(best):
            k = best[t]
            if k == 0:
                t += 1
                continue
            s = t
            while t < len(best) and best[t] == k:
                t += 1
            text.append(alphabet[k - 1])
            spans.append((s, t))
        maxp = p.max(1)[best != 0]
        conf = float(maxp.prod() ** (2.0 / np.sqrt(len(maxp)))) if len(maxp) else 0.0
        dists = []
        for s, e in spans:
            d = p[s:e, 1:].mean(0)
            dists.append(d / d.sum())
        return "".join(text), conf, dists

    def _ctc_loglik(self, p, alphabet, words):
        """log P(word | image) for every candidate word (CTC forward algorithm)."""
        torch = self._torch
        pos = {c: i + 1 for i, c in enumerate(alphabet)}
        T = p.shape[0]
        logp = torch.from_numpy(np.log(np.clip(p, 1e-12, 1.0))).float()
        logp = logp.unsqueeze(1).expand(T, len(words), p.shape[1])
        targets = torch.tensor([pos[c] for w in words for c in w], dtype=torch.long)
        target_len = torch.tensor([len(w) for w in words], dtype=torch.long)
        input_len = torch.full((len(words),), T, dtype=torch.long)
        nll = torch.nn.functional.ctc_loss(logp, targets, input_len, target_len,
                                           blank=0, reduction="none")
        return -nll.numpy()

    def _rank(self, p, alphabet, words, top=5, min_post=1e-3):
        """Candidates sorted by posterior (softmax of CTC log-likelihoods)."""
        ll = self._ctc_loglik(p, alphabet, words)
        post = np.exp(ll - ll.max())
        post /= post.sum()
        order = np.argsort(-post)[:top]
        return [(words[i], float(post[i])) for n, i in enumerate(order)
                if n == 0 or post[i] >= min_post]

    # -- field readers ------------------------------------------------------------
    def read_code(self, gray, allowed, pattern, k_alt=4):
        """Letters or number: greedy decode, then re-rank the combinations of the
        top-k alternatives per character with the CTC likelihood, keeping only
        strings that match the plate format."""
        p = self._restrict(self.frame_probs(gray), allowed)
        raw, conf, dists = self._greedy(p, allowed)
        cands = []
        if dists:
            options = [[allowed[j] for j in np.argsort(-d)[:k_alt]] for d in dists]
            words = ["".join(w) for w in itertools.product(*options)]
            words = [w for w in words if pattern.match(w)] or words
            cands = self._rank(p, allowed, words)
        text = cands[0][0] if cands else raw
        return dict(text=text, raw=raw, conf=conf, candidates=cands,
                    format_ok=bool(pattern.match(text)))

    def read_province(self, gray):
        p = self._restrict(self.frame_probs(gray), PROVINCE_CHARS)
        raw, conf, _ = self._greedy(p, PROVINCE_CHARS)
        cands = self._rank(p, PROVINCE_CHARS, PROVINCES)
        best = cands[0][0]
        similarity = difflib.SequenceMatcher(None, raw, best).ratio()
        return dict(text=best, raw=raw, conf=conf, candidates=cands,
                    posterior=cands[0][1], similarity=similarity)

    # -- full pipeline ------------------------------------------------------------
    def read(self, image, return_debug=False):
        """image: file path or BGR numpy array. Returns a JSON-friendly dict."""
        t0 = time.perf_counter()
        bgr = imread(image) if isinstance(image, (str, os.PathLike)) else image
        if bgr is None:
            raise FileNotFoundError(image)

        # try plate-like regions in turn, each with the layouts its shape fits;
        # keep the first that segments into a valid layout
        error = None
        quads = plate_candidates(bgr, layouts=self.layouts)
        attempts = [(quad, layout) for n, quad in enumerate(quads)
                    for layout in plate_layouts(quad, self.layouts,
                                                whole_image=n == len(quads) - 1)]
        for quad, layout in attempts:
            plate = warp_plate(bgr, quad, LAYOUTS[layout]["height"])
            norm, ink = normalise_plate(plate)
            try:
                seg = segment_plate(ink, layout)
                break
            except PlateNotFound as e:
                error = e
        else:
            raise error
        crops = {name: field_crop(norm, ink, seg["labels"], seg[name])
                 for name in ("letters", "number", "province")}
        # one tight crop per letter glyph, e.g. [5] [ก] [ข]
        boxes = seg["letter_glyphs"]
        glyphs = [glyph_crop(norm, ink, seg["labels"], b) for b in boxes]
        t1 = time.perf_counter()

        # letters: read glyph by glyph, then combine; fall back to the whole crop
        # when the glyphs look merged (touching letters) or the count is odd
        per_letter = (self.letters_engine != "group" and 1 <= len(boxes) <= 3
                      and all(b.w <= 1.3 * b.h for b in boxes))
        if per_letter:
            letters = self.read_letters(glyphs, layout=layout)
            mode = f"per-letter ({letters['engine']})"
        else:
            letters = self.read_code(crops["letters"], THAI_CONSONANTS + DIGITS,
                                     LAYOUTS[layout]["letters_re"])
            mode = "whole crop (easyocr)" + ("" if self.letters_engine == "group" else ", fallback")
        t2 = time.perf_counter()
        number = self.read_code(crops["number"], DIGITS, NUMBER_RE)
        province = self.read_province(crops["province"])
        t3 = time.perf_counter()

        checks = {
            "letters_format": letters["format_ok"],
            "number_format": number["format_ok"],
            "letters_glyph_count": len(letters["text"]) == seg["n_letter_glyphs"],
            "number_glyph_count": len(number["text"]) == seg["n_number_glyphs"],
        }
        by_engine = letters.get("by_engine")
        if by_engine:
            checks["letter_engines_agree"] = len(set(by_engine.values())) == 1
        needs_review = (not all(checks.values()) or
                        letters["conf"] < (0.80 if per_letter else 0.90) or
                        number["conf"] < 0.90 or province["posterior"] < 0.90 or
                        province["similarity"] < 0.6)
        result = {
            "plate": f"{letters['text']} {number['text']}",
            "layout": layout,
            "letters": letters["text"],
            "number": number["text"],
            "province": province["text"],
            "confidence": {
                "letters": round(letters["conf"], 3),
                "number": round(number["conf"], 3),
                "province": round(province["posterior"], 3),
                "province_vs_raw_ocr": round(province["similarity"], 3),
            },
            "alternatives": {
                "letters": [(w, round(s, 3)) for w, s in letters["candidates"]],
                "number": [(w, round(s, 3)) for w, s in number["candidates"]],
                "province": [(w, round(s, 3)) for w, s in province["candidates"][:3]],
            },
            "letters_detail": {
                "mode": mode,
                "per_letter_top3": letters.get("per_glyph"),
                "by_engine": by_engine,
            },
            "raw_ocr": {"letters": letters.get("raw", letters["text"]), "number": number["raw"],
                        "province": province["raw"]},
            "checks": checks,
            "needs_review": bool(needs_review),
            "timing_ms": {"segment": round(1000 * (t1 - t0)),
                          "letters": round(1000 * (t2 - t1)),
                          "number_province": round(1000 * (t3 - t2)),
                          "total": round(1000 * (t3 - t0))},
        }
        if return_debug:
            result["_debug"] = dict(image=bgr, quad=quad, plate=plate, norm=norm,
                                    ink=ink, seg=seg, crops=crops, glyphs=glyphs)
        return result

    def read_letters(self, glyphs, engine=None, layout="car"):
        """Read letter glyphs ONE AT A TIME with a letters engine, then combine."""
        engine = engine or self.letters_engine
        allowed = letter_allowed_sets(len(glyphs), layout)
        dists = engine.predict(glyphs, allowed)
        out = combine_letters(dists, allowed, LAYOUTS[layout]["letters_re"])
        out["engine"] = engine.name
        if isinstance(engine, EnsembleLetters):
            out["by_engine"] = {name: "".join(al[int(np.argmax(p))] for p, al in zip(ds, allowed))
                                for name, ds in engine.last.items()}
        return out


# ---------------------------------------------------------------------------
# Utilities: robust imread, debug sheet, CLI
# ---------------------------------------------------------------------------
def imread(path):
    """cv2.imread that also works with Thai / non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def debug_sheet(dbg):
    """One image showing the detected plate, the three boxes and the crops."""
    img = dbg["image"].copy()
    cv2.polylines(img, [dbg["quad"].astype(np.int32)], True, (0, 0, 255), 2)
    plate = dbg["plate"].copy()
    colours = {"letters": (255, 0, 0), "number": (0, 160, 0), "province": (0, 0, 255)}
    for name, col in colours.items():
        x0, y0, x1, y1 = dbg["seg"][name].as_tuple()
        cv2.rectangle(plate, (x0, y0), (x1 - 1, y1 - 1), col, 2)
        cv2.putText(plate, name, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    for g in dbg["seg"].get("letter_glyphs", []):          # one yellow box per letter
        cv2.rectangle(plate, (g.x0 + 3, g.y0 + 3), (g.x1 - 4, g.y1 - 4), (0, 200, 255), 1)
    ink = 255 - dbg["ink"]
    # letters shown glyph by glyph (what the letter engines read), then number, province
    crops = [_pad_white(g, 0.15) for g in dbg.get("glyphs", [])] or [dbg["crops"]["letters"]]
    crops += [dbg["crops"]["number"], dbg["crops"]["province"]]

    def bgr(im):
        return im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    def row(images, h, gap=10):
        """Resize every image to height h and place them side by side."""
        out = []
        for im in images:
            im = bgr(im)
            out += [cv2.resize(im, (max(1, round(im.shape[1] * h / im.shape[0])), h)),
                    np.full((h, gap, 3), 255, np.uint8)]
        return np.hstack(out[:-1])

    rows = [row([img, ink], 200), row([plate], 260), row(crops, 90)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 6, 6, 6, 6 + width - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(255, 255, 255)) for r in rows]
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read a Thai licence plate offline with EasyOCR")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--letters", default="auto",
                    help="letters engine: easyocr | cnn | tesseract | paddle | group | "
                         "easyocr+cnn (any '+' combination) | auto (default)")
    ap.add_argument("--layout", default="auto", choices=("auto",) + tuple(LAYOUTS),
                    help="plate layout: car (2 lines), motorcycle (3 lines) or auto (default)")
    ap.add_argument("--tessdata-dir", help="folder with tha.traineddata (tesseract engine; "
                                           "default: tessdata/ next to this file, if present)")
    ap.add_argument("--debug", help="write a debug sheet (single image) or a folder for many")
    ap.add_argument("--save-crops", help="folder to save the 3 field crops (dataset building)")
    ap.add_argument("--save-glyphs", help="folder to save each letter glyph as <folder>/<char>/*.png "
                                          "(fix wrong folders, then: train_letter_cnn.py --real)")
    ap.add_argument("--json", action="store_true", help="print full JSON result")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Thai on Windows consoles

    t = time.perf_counter()
    reader = ThaiPlateReader(gpu=args.gpu, letters_engine=args.letters,
                             tessdata_dir=args.tessdata_dir, layout=args.layout)
    print(f"model loaded in {time.perf_counter() - t:.1f}s", file=sys.stderr)

    for path in args.images:
        try:
            res = reader.read(path, return_debug=True)
        except PlateNotFound as e:
            print(f"{path}: plate not found ({e})")
            continue
        dbg = res.pop("_debug")
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.debug:
            out = (os.path.join(args.debug, f"{stem}_debug.png")
                   if len(args.images) > 1 or os.path.isdir(args.debug) else args.debug)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            imwrite(out, debug_sheet(dbg))
        if args.save_crops:
            os.makedirs(args.save_crops, exist_ok=True)
            for k, im in dbg["crops"].items():
                imwrite(os.path.join(args.save_crops, f"{stem}_{k}.png"), im)
        if args.save_glyphs and len(dbg["glyphs"]) == len(res["letters"]) and "?" not in res["letters"]:
            for i, (ch, g) in enumerate(zip(res["letters"], dbg["glyphs"])):
                os.makedirs(os.path.join(args.save_glyphs, ch), exist_ok=True)
                imwrite(os.path.join(args.save_glyphs, ch, f"{stem}_{i}.png"), g)
        if args.json:
            print(json.dumps({"image": path, **res}, ensure_ascii=False, indent=2))
        else:
            flag = "  [review]" if res["needs_review"] else ""
            print(f"{path}: {res['plate']}  {res['province']}  [{res['layout']}]  "
                  f"(conf L={res['confidence']['letters']:.2f} "
                  f"N={res['confidence']['number']:.2f} "
                  f"P={res['confidence']['province']:.2f}, "
                  f"{res['timing_ms']['total']} ms){flag}")
            detail = res["letters_detail"]
            print(f"    letters: {detail['mode']}"
                  + (f"  {detail['by_engine']}" if detail["by_engine"] else ""))
            if res["needs_review"]:
                alts = ", ".join(f"{w} {s:.2f}" for w, s in res["alternatives"]["letters"])
                print(f"    letter candidates: {alts}")


if __name__ == "__main__":
    main()
