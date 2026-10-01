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

Pipeline (ThaiPlateReader.read runs the four steps; each is also a function)
  1. classify_plate()  car or motorcycle plate. Plate-like regions come from the
                       light plate face (grey or brightest colour channel) and
                       from rows of large characters (also in turned copies of
                       the photo, for tilted plates); each is segmented with the
                       layouts its shape fits, and the layout of the most
                       plausible segmentation (segmentation_score) wins
  2. locate_plate()    the most plausible region with that layout, warped flat
  3. locate_fields()   connected components -> boxes and crops
                         letters  : one box per character  [5] [ก] [ข]
                         number   : one box per digit (1-4)
                         province : the band under the top line (motorcycle:
                                    between the letters and the number)
  4. read_fields()     letters: each character read ON ITS OWN by a letter
                       engine (small CNN / EasyOCR / Tesseract / PaddleOCR, or an
                       average of several), combined with a reading of the whole
                       crop and checked against the plate format
                       [digit?][consonant x1-2] (motorcycles before ~2013: 3
                       consonants, กขค 123; buses and trucks: 2-3 digits, 30-1234);
                       the whole crop alone when the characters look merged.
                       number / province: a text-line recogniser - PaddleOCR's
                       Thai model when installed, else EasyOCR's (its CRAFT
                       detector is never loaded) - with per-field allow-lists;
                       the province is picked by scoring all official names

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
import importlib.util
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

# Top line: [optional digit] + 1-2 consonants, or the 2-3 digit group of a
# bus / truck plate (30-1234, 700-1234); then 1-4 digits
LETTERS_RE = re.compile(rf"^(?:[1-9]?[{THAI_CONSONANTS}]{{1,2}}|[1-9][0-9]{{1,2}})$")
# motorcycles: the same, or 3 consonants on plates issued before ~2013 (กขค 123)
MOTO_LETTERS_RE = re.compile(
    rf"^(?:[1-9]?[{THAI_CONSONANTS}]{{1,2}}|[{THAI_CONSONANTS}]{{3}}|[1-9][0-9]{{1,2}})$")
# no leading 0, except in a 4-digit number (bus / truck 30-0638, red dealer plate ก-0327)
NUMBER_RE = re.compile(r"^(?:[1-9][0-9]{0,3}|0[0-9]{3})$")

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
# weights: trained on fonts + real plate glyphs if present, else on fonts only
CNN_WEIGHTS = ("thai_letter_cnn_plates.pt", "thai_letter_cnn.pt")


def default_cnn_weights():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, f) for f in CNN_WEIGHTS]
    return next((p for p in paths if os.path.exists(p)), paths[-1])


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


QUAD_MARGIN = 0.04
WIDE_MARGIN = 0.08   # enlarged copies of the light-rectangle candidates


def plate_candidates(bgr, max_candidates=4, layouts=tuple(LAYOUTS)):
    """Quadrilaterals that look like a plate face (light, rectangular, shaped
    like a plate of one of the layouts), most plate-like first, at most
    max_candidates per layout; the whole image is always the last one.

    Good for plate-centred photos (a phone shot of the plate / rear of vehicle).
    For wide CCTV-style scenes crop the plate with a detector (e.g. YOLO) first.
    """
    H, W = bgr.shape[:2]
    found = []
    # white plates are light in grey; red (dealer) and yellow (taxi) plates only
    # in their brightest colour channel
    for ch in (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr.max(axis=2)):
        blur = cv2.GaussianBlur(ch, (5, 5), 0)
        t1, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brighter = blur[blur > t1].reshape(-1, 1)
        # second, higher threshold separates the plate from a white car body
        t2 = cv2.threshold(brighter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] \
            if brighter.size > 100 else t1
        for t in sorted({t1, t2}):
            light = (blur > t).astype(np.uint8) * 255
            contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                if area < 0.005 * H * W or min(rw, rh) < 20:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                fits = _fitting_layouts(aspect, layouts)
                if not fits:
                    continue
                rectangularity = area / (rw * rh)
                # large characters touching the edge of the plate face (red plates)
                # bite notches into its outline: judged on the convex hull, such a
                # face is still a candidate, tried after the clean rectangles
                notched = cv2.contourArea(cv2.convexHull(c)) / (rw * rh)
                # a margin, so a glyph at the very edge of the face is not cut
                quad = _contour_quad(c)
                quad = quad.mean(0) + (quad - quad.mean(0)) * (1 + QUAD_MARGIN)
                if rectangularity >= 0.8:
                    found.append((0, -area * rectangularity, quad, fits))
                elif notched >= 0.85:
                    found.append((1, -area * notched, quad, fits))

    quads, count = [], dict.fromkeys(layouts, 0)
    for _, _, q, fits in sorted(found, key=lambda f: f[:2]):
        if all(count[name] >= max_candidates for name in fits):
            continue
        if all(np.abs(q - p).max() > 10 for p in quads):   # skip duplicates
            quads.append(q)
            for name in fits:
                count[name] += 1
    # the same faces enlarged: a character at the very edge of the face (often
    # under the frame) is cut off otherwise; the candidate selector chooses
    for q in list(quads):
        tl, tr, br, bl = q
        top, bottom = (tr - tl) * WIDE_MARGIN, (br - bl) * WIDE_MARGIN
        left, right = (bl - tl) * WIDE_MARGIN, (br - tr) * WIDE_MARGIN
        quads.append(np.float32([tl - top - left, tr + top - right,
                                 br + bottom + right, bl - bottom + left]))
    quads += [q for q in text_line_quads(bgr, layouts)
              if all(np.abs(q - p).max() > 10 for p in quads)]
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


def _text_lines(ink, min_h, max_h):
    """Rows of >= 2 ink components of similar height, each left to right."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = sorted((Box(x, y, x + w, y + h) for x, y, w, h, a in stats[1:]
                    if min_h <= h <= max_h and 0.08 * h <= w <= 2.5 * h and a >= 0.12 * w * h),
                   key=lambda b: b.x0)
    lines = []
    for b in boxes:
        best = None
        for line in lines:
            last, hm = line[-1], np.median([c.h for c in line])
            gap = b.x0 - last.x1
            if (abs(b.cy - last.cy) < 0.3 * hm and 0.7 <= b.h / hm <= 1.4
                    and -0.2 * hm <= gap <= 1.8 * hm and (best is None or gap < best[0])):
                best = (gap, line)
        if best:
            best[1].append(b)
        else:
            lines.append([b])
    # letters are wider than a third of their height; a row of thin bars is a grille
    return [line for line in lines if len(line) >= 2 and
            any(b.w >= 0.3 * b.h for b in line) and sum(b.w < 0.3 * b.h for b in line) <= 5]


def _line_frame(line):
    """(centre x, centre y, unit vector along the line, unit normal, glyph height,
    half length) of a row of glyph boxes; the slope is fitted to the glyph centres."""
    xs = np.array([(b.x0 + b.x1) / 2 for b in line], np.float32)
    ys = np.array([b.cy for b in line], np.float32)
    slope = float(np.clip(np.polyfit(xs, ys, 1)[0], -0.35, 0.35)) if len(line) >= 3 else 0.0
    x0, x1 = min(b.x0 for b in line), max(b.x1 for b in line)
    cx = (x0 + x1) / 2
    cy = float(ys.mean() + slope * (cx - xs.mean()))
    u = np.float32([1, slope]) / math.hypot(1, slope)
    v = np.float32([-u[1], u[0]])  # points down
    return cx, cy, u, v, float(np.median([b.h for b in line])), (x1 - x0) / 2 * float(u[0])


def _quad(c, u, v, half_w, top, bottom):
    return np.float32([c + a * u + b * v for a, b in
                       ((-half_w, top), (half_w, top), (half_w, bottom), (-half_w, bottom))])


# rows of glyphs are looked for in the image turned by these angles (degrees), so
# plates tilted by up to ~30 degrees (motorcycles, photos taken from the side)
# still form rows; each row tolerates a few degrees of slant itself
TEXT_ROW_ANGLES = (0, -12, 12, -24, 24)


def _rotation(shape, angle):
    """Affine matrix turning an image of this shape by angle degrees onto a canvas
    large enough to hold it, and the canvas size (w, h)."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    W2, H2 = int(np.ceil(h * s_ + w * c)), int(np.ceil(h * c + w * s_))
    M[:, 2] += (W2 - w) / 2, (H2 - h) / 2
    return M, (W2, H2)


def text_line_quads(bgr, layouts=tuple(LAYOUTS), max_quads=6):
    """Plate quads found from the characters rather than the plate colour: the
    largest row of similar-height glyphs is the top line of a car plate (the
    province lies below it); two such rows one above the other are the letters
    and number lines of a motorcycle plate. The quad follows the row's slant.
    Finds red, yellow and green plates, plates whose frame hides the edge, and
    tilted plates (rows are also looked for in turned copies, TEXT_ROW_ANGLES)."""
    H, W = bgr.shape[:2]
    # worked at 300 px high: small crops are upscaled for stable morphology, large
    # photos shrunk (at full size the closing below takes seconds per image)
    s = 300 / H
    img = cv2.resize(bgr, None, fx=s, fy=s,
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    h = img.shape[0]
    k = max(15, int(0.15 * h)) | 1   # wider than a stroke, so closing = background
    inks = []
    for ch in (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img.max(axis=2)):
        for x in (ch, 255 - ch):     # dark text on a light plate, and the reverse
            bg = cv2.morphologyEx(x, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            norm = cv2.divide(x, bg, scale=255)
            _, ink = cv2.threshold(cv2.GaussianBlur(norm, (3, 3), 0), 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            inks.append(ink)

    found = []   # (score, quad in the 300-px image, layout)
    for angle in TEXT_ROW_ANGLES:
        if angle:
            M, size = _rotation(img.shape, angle)
            back = cv2.invertAffineTransform(M)
        rows = [r for ink in inks for r in _text_lines(
            cv2.warpAffine(ink, M, size, flags=cv2.INTER_NEAREST) if angle else ink,
            0.08 * h, 0.6 * h)]
        for score, q, layout in _row_quads(rows, layouts):
            if angle:   # back to the unturned image
                q = (np.hstack([q, np.ones((4, 1), np.float32)]) @ back.T).astype(np.float32)
            found.append((score, q, layout))

    quads = []   # one quad per plate: turned copies find the same plate again
    for _, q, layout in sorted(found, key=lambda f: -f[0]):
        q = q / s
        centre, size = q.mean(0), np.linalg.norm(q[3] - q[0])
        if all(np.linalg.norm(centre - p.mean(0)) > 0.3 * size or
               not 0.75 <= size / np.linalg.norm(p[3] - p[0]) <= 1.33 or layout != pl
               for p, pl in quads):
            quads.append((q, layout))
        if len(quads) >= max_quads:
            break
    return [q for q, _ in quads]


def _row_quads(rows, layouts):
    """(score, quad, layout) for the plates suggested by rows of glyphs."""
    found = []
    frames = [_line_frame(r) for r in rows]
    for r, (cx, cy, u, v, gh, half) in zip(rows, frames):
        c = np.float32([cx, cy])
        # motorcycle: another row of the same size 1.5-3.2 glyph heights below
        if "motorcycle" in layouts:
            for r2, (cx2, cy2, _, _, gh2, half2) in zip(rows, frames):
                d = float(np.dot(np.float32([cx2, cy2]) - c, v))
                if (r2 is not r and 0.75 <= gh2 / gh <= 1.33 and 1.5 * gh <= d <= 3.2 * gh
                        and abs(cx2 - cx) < max(half, half2) and len(r) + len(r2) >= 4):
                    hw = max(half, half2) + 0.35 * gh
                    top, bottom = -0.8 * gh, d + 0.8 * gh
                    hw = max(hw, 0.6 * (bottom - top))
                    found.append(((len(r) + len(r2)) * gh, _quad(c, u, v, hw, top, bottom),
                                  "motorcycle"))
            # a row whose partner was not found (it touches the frame or the
            # province) may still be either line of a motorcycle plate
            if len(r) >= 2:
                for top, bottom in ((-0.8 * gh, 3.1 * gh), (-3.1 * gh, 0.8 * gh)):
                    hw = max(half + 0.35 * gh, 0.6 * (bottom - top))
                    found.append((0.5 * len(r) * gh, _quad(c, u, v, hw, top, bottom),
                                  "motorcycle"))
        if "car" in layouts and len(r) >= 3:
            hw = max(half + 0.4 * gh, 2.6 * gh)   # the province can be wider than the row
            found.append((len(r) * gh, _quad(c, u, v, hw, -0.8 * gh, 1.65 * gh), "car"))
    return found


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


def normalise_plate(plate_bgr, polarity="auto"):
    """Return (norm, ink):
    norm - grey image, dark text on a flat white background (fed to the OCR)
    ink  - binary mask of text pixels (used for segmentation)

    Plates come in several ink colours (black, green, blue, red ...), so we pick
    the colour channel where text and background separate best, then divide
    out uneven lighting (shadows, glare gradients) before thresholding.
    polarity - "dark" text on a lighter plate (every plate but the green rental
    ones), "light" text on a darker plate, or "auto": the minority class is text.
    """
    H, W = plate_bgr.shape[:2]
    k = max(15, int(0.15 * H)) | 1  # wider than a stroke, so closing = background

    def flat(c):
        """Shading removed: strokes stay, large regions (plate vs frame, shadows) go.
        The wide blur is done at 1/4 scale (same result, ~15x faster)."""
        small = cv2.resize(c, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
        blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), k / 4), (W, H),
                          interpolation=cv2.INTER_LINEAR)
        f = c.astype(np.float32) - blur.astype(np.float32)
        return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # channel and polarity are judged on the plate's centre with the shading
    # removed: the frame and car around it separate well from the plate in a
    # channel where the text does not (blue channel of a red or yellow plate)
    centre = (slice(int(0.15 * H), int(0.85 * H)), slice(int(0.1 * W), int(0.9 * W)))
    channels = ([cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY), plate_bgr.max(axis=2)]
                + list(cv2.split(plate_bgr)))
    flats = [flat(c)[centre] for c in channels]
    best = max(range(len(channels)), key=lambda i: _otsu_separability(flats[i]))
    ch, fc = channels[best], flats[best]

    def binarise(c):
        background = cv2.morphologyEx(c, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        norm = cv2.divide(c, background, scale=255)
        _, ink = cv2.threshold(cv2.GaussianBlur(norm, (3, 3), 0), 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return norm, ink

    if polarity == "auto":   # text should be the minority class
        t, _ = cv2.threshold(fc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        polarity = "light" if (fc <= t).mean() > 0.5 else "dark"
    return binarise(255 - ch if polarity == "light" else ch)


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
        if x == 0 or x + w == W:                       # cut by a side: frame, holder, car
            continue
        elongated = w > 6 * h or h > 6 * w
        near_edge = (x < 0.08 * W or y < 0.08 * H or
                     x + w > 0.92 * W or y + h > 0.92 * H)
        if elongated and near_edge:                    # printed border line
            continue
        comps.append(Box(x, y, x + w, y + h, [i]))
    return comps, labels


def _glyph_row(comps, tall, labels):
    """The tall components of one line -> glyph boxes, left to right, with the
    small pieces inside the line's band (detached tails, broken strokes) added,
    dashes trimmed off and touching glyphs cut apart."""
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
    h = np.median([g.h for g in glyphs])
    glyphs = [p for g in glyphs for p in _split_glyph(g, labels, h)]
    # a thin bar at either end, taller or shorter than the characters, is the
    # edge of the frame (a "1" has the height of the other characters)
    while len(glyphs) > 1 and glyphs[0].w < 0.25 * h and abs(glyphs[0].h - h) > 0.2 * h:
        glyphs.pop(0)
    while len(glyphs) > 1 and glyphs[-1].w < 0.25 * h and abs(glyphs[-1].h - h) > 0.2 * h:
        glyphs.pop()
    return glyphs


def _split_glyph(g, labels, h, depth=0):
    """One glyph box of a line of large characters (h: the line's glyph height)
    -> boxes. A dash stuck to its side ("ก-" / "-9" on red and bus plates) is
    trimmed off; two touching glyphs are cut at the thinnest column when only one
    stroke crosses it. The boxes keep the component ids: crops use the box."""
    mask = np.isin(labels[g.y0:g.y1, g.x0:g.x1], g.ids)
    prof = mask.sum(0)
    thin = prof < 0.2 * h
    a, b = 0, len(prof)
    while a < b and thin[a]:
        a += 1
    while b > a and thin[b - 1]:
        b -= 1

    def dash(c0, c1):
        """a thin run long enough and at mid-height: a dash, not a serif or the flag of a 1"""
        rows = np.flatnonzero(mask[:, c0:c1].any(1))
        return (c1 - c0 >= 0.2 * h and rows.size > 0 and
                rows[0] >= 0.25 * g.h and rows[-1] < 0.75 * g.h)

    if b - a < 0.1 * h or not dash(0, a):
        a = 0
    if b - a < 0.1 * h or not dash(b, len(prof)):
        b = len(prof)
    pieces = [(a, b)]
    if b - a > 1.1 * h and depth < 3:   # wider than any single character
        lo, hi = a + int(0.3 * (b - a)), a + int(0.7 * (b - a))
        c = lo + int(np.argmin(prof[lo:hi]))
        if prof[c] < 0.18 * h:
            pieces = [(a, c), (c + 1, b)]
    out = []
    for p0, p1 in pieces:
        rows = np.flatnonzero(mask[:, p0:p1].any(1))
        if rows.size == 0 or rows[-1] + 1 - rows[0] < 0.5 * h:
            continue
        box = Box(g.x0 + p0, g.y0 + int(rows[0]), g.x0 + p1, g.y0 + int(rows[-1]) + 1, g.ids)
        out += [box] if len(pieces) == 1 else _split_glyph(box, labels, h, depth + 1)
    return out


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
    glyphs = _glyph_row(comps, tall, labels) if tall else []
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
    # the province is read from the band under the top line rather than from its
    # components: a blurred word can merge into one blob that is dropped as a
    # frame, and the text of a dealer sticker under the plate would join it
    gh = np.median([g.h for g in glyphs])
    band = Box(int(max(0, min(g.x0 for g in glyphs) - 0.3 * gh)), int(top_y1 + 0.02 * H),
               int(min(W, max(g.x1 for g in glyphs) + 0.3 * gh)), int(min(H, top_y1 + 0.85 * gh)),
               [i for c in low for i in c.ids])
    province = band if band.h >= 6 else _province_box(low, labels)
    return glyphs[:split + 1], glyphs[split + 1:], province


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
    letter_glyphs, number_glyphs = _glyph_row(comps, top, labels), _glyph_row(comps, bottom, labels)
    if not (letter_glyphs and number_glyphs):
        raise PlateNotFound("a line of large characters holds only dashes")

    # ---- middle line: province -----------------------------------------------
    top_y1 = max(c.y1 for c in top)
    bottom_y0 = min(c.y0 for c in bottom)
    used = {i for g in letter_glyphs + number_glyphs for i in g.ids}
    low = [c for c in comps
           if c.ids[0] not in used and c.h < 0.7 * ch
           and c.y0 >= top_y1 - 0.03 * H and c.y1 <= bottom_y0 + 0.03 * H]
    if not low:
        raise PlateNotFound("no province line found between the letters and the number")
    # read from the band between the two lines (see _segment_car)
    glyphs = letter_glyphs + number_glyphs
    gh = np.median([g.h for g in glyphs])
    band = Box(int(max(0, min(g.x0 for g in glyphs) - 0.5 * gh)), max(g.y1 for g in letter_glyphs),
               int(min(W, max(g.x1 for g in glyphs) + 0.5 * gh)), min(g.y0 for g in number_glyphs),
               [i for c in low for i in c.ids])
    province = band if band.h >= 6 else _province_box(low, labels)
    return letter_glyphs, number_glyphs, province


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


# Weights of segmentation_score: a softmax over each photo's segmenting candidates,
# fitted on half of final_data (6,344 plate crops, split by a hash of the file
# name) and checked on the other half - both glyph counts right for 70% of the
# plates, against 56% when the first candidate that segments is kept. Features:
# whole - the whole-image fallback; order - how many candidates segmented before;
# nl* / nn* - number of letter / number glyphs; gh - glyph height / plate height;
# h_cv - spread of the glyph heights; wmin / wmax - narrowest / widest glyph
# (width / height); span - glyph rows' width / plate width; prov_w, prov_h -
# province box width / plate width and height / glyph height; gap - widest gap
# of the top line / the next widest (car)
SEG_SCORE_WEIGHTS = dict(
    whole=0.42, moto=-0.27, log_order=-1.26, gh_car=2.99, gh2_car=-2.59, gh_moto=2.81,
    gh2_moto=19.84, h_cv=-11.45, wmax=-2.69, wmin=4.34, span=-1.83, span2=-0.7, prov_w=3.92,
    prov_h=1.12, log_gap=0.57, nl1=0.69, nl2=2.64, nl3=2.27, nn1=-2.19, nn2=-1.26, nn3=1.16,
    nn4=3.98)


def segmentation_score(seg, shape, order=0, whole=False):
    """How plausible a segmentation looks (higher is better), from its geometry
    alone; used to choose among the plate candidates that segment."""
    H, W = shape
    row = seg["letter_glyphs"] + seg["number_glyphs"]
    hs = np.array([g.h for g in row], float)
    ratio = np.array([g.w for g in row], float) / hs
    gh = float(np.median(hs))
    car = seg["layout"] == "car"
    span = (max(g.x1 for g in row) - min(g.x0 for g in row)) / W
    gaps = sorted(b.x0 - a.x1 for a, b in zip(row, row[1:])) if car else []
    f = dict(whole=float(whole), moto=float(not car), log_order=math.log1p(order),
             gh_car=car * gh / H, gh2_car=car * (gh / H) ** 2,
             gh_moto=(not car) * gh / H, gh2_moto=(not car) * (gh / H) ** 2,
             h_cv=float(hs.std() / hs.mean()), wmax=min(3.0, ratio.max()),
             wmin=min(3.0, ratio.min()), span=span, span2=span ** 2,
             prov_w=seg["province"].w / W, prov_h=min(3.0, seg["province"].h / gh),
             log_gap=math.log(np.clip(gaps[-1] / max(1.0, gaps[-2]), 0.2, 20))
             if len(gaps) >= 2 else 0.0)
    f[f"nl{seg['n_letter_glyphs']}"] = 1.0
    f[f"nn{seg['n_number_glyphs']}"] = 1.0
    return sum(SEG_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in f.items())


POLARITIES = ("auto", "dark", "light")


# ---------------------------------------------------------------------------
# The pipeline in four steps (ThaiPlateReader.read runs them in this order)
#   1. classify_plate(bgr)              car or motorcycle plate
#   2. locate_plate(bgr, layout)        the plate's quad and the rectified plate
#   3. locate_fields(plate, layout)     letters / number / province boxes, and one
#                                       box per character of the letters and number
#   4. ThaiPlateReader.read_fields(..)  read the three fields
# A plate-like region is judged by how well its fields can be located, so steps
# 1-3 share one search over the regions (search_plates).
# ---------------------------------------------------------------------------
def search_plates(bgr, layouts=tuple(LAYOUTS)):
    """Every plate-like region (plate_candidates), warped with each layout its
    shape fits, whose fields can be located; each scored by how plausible its
    segmentation looks (segmentation_score). The regions are tried with the text
    polarity judged from the pixels; only if none segments are they tried again
    with dark, then light text forced. -> list of dict(score, quad, layout,
    plate, norm, ink, seg), best first; raises PlateNotFound."""
    error = None
    quads = plate_candidates(bgr, layouts=layouts)
    attempts = [(n, quad, layout) for n, quad in enumerate(quads)
                for layout in plate_layouts(quad, layouts, whole_image=n == len(quads) - 1)]
    plates = {}
    for polarity in POLARITIES:
        found = []
        for i, (n, quad, layout) in enumerate(attempts):
            if i not in plates:
                plates[i] = warp_plate(bgr, quad, LAYOUTS[layout]["height"])
            norm, ink = normalise_plate(plates[i], polarity)
            try:
                seg = segment_plate(ink, layout)
            except PlateNotFound as e:
                error = e
                continue
            score = segmentation_score(seg, ink.shape, len(found), whole=n == len(quads) - 1)
            found.append(dict(score=score, quad=quad, layout=layout, plate=plates[i],
                              norm=norm, ink=ink, seg=seg))
        if found:
            return sorted(found, key=lambda f: -f["score"])
    raise error


def classify_plate(bgr, layouts=tuple(LAYOUTS), found=None):
    """Step 1 - car or motorcycle plate: the layout of the most plausible
    segmentation over all plate-like regions. found: search_plates() output, if
    already computed. -> dict(layout, scores={layout: best score}, found)"""
    found = search_plates(bgr, layouts) if found is None else found
    scores = {}
    for f in found:
        scores[f["layout"]] = max(scores.get(f["layout"], -math.inf), f["score"])
    return dict(layout=max(scores, key=scores.get), scores=scores, found=found)


def locate_plate(bgr, layout, found=None):
    """Step 2 - the plate, given its layout: of the plate-like regions whose
    fields can be located with that layout, the most plausible one.
    -> dict(quad, plate, layout, score, and the step-3 norm, ink, seg)"""
    found = search_plates(bgr, (layout,)) if found is None else found
    mine = [f for f in found if f["layout"] == layout]
    if not mine:
        raise PlateNotFound(f"no {layout} plate found")
    return max(mine, key=lambda f: f["score"])


def locate_fields(plate, layout, polarity="auto"):
    """Step 3 - the fields of a rectified plate (warp_plate output): boxes of the
    letters, number and province, one box per character of the letters and the
    number (seg["letter_glyphs"], seg["number_glyphs"]), and the crops the
    readers use. -> dict(layout, norm, ink, seg, crops, letter_glyphs, number_glyphs)"""
    norm, ink = normalise_plate(plate, polarity)
    return field_crops(dict(layout=layout, norm=norm, ink=ink, seg=segment_plate(ink, layout)))


def field_crops(fields):
    """Add the crops to located fields: letters and number with foreign ink erased,
    the province band, and one tight crop per character."""
    norm, ink, seg = fields["norm"], fields["ink"], fields["seg"]
    crops = {name: field_crop(norm, ink, seg["labels"], seg[name]) for name in ("letters", "number")}
    crops["province"] = band_crop(norm, seg["province"])
    return dict(fields, crops=crops,
                letter_glyphs=[glyph_crop(norm, ink, seg["labels"], b) for b in seg["letter_glyphs"]],
                number_glyphs=[glyph_crop(norm, ink, seg["labels"], b) for b in seg["number_glyphs"]])


def locate_and_segment(bgr, layouts=tuple(LAYOUTS)):
    """Steps 1-3 in one call. Returns (quad, layout, plate, norm, ink, seg)."""
    cls = classify_plate(bgr, layouts)
    p = locate_plate(bgr, cls["layout"], cls["found"])
    return p["quad"], p["layout"], p["plate"], p["norm"], p["ink"], p["seg"]


def band_crop(norm, box, pad=4):
    """Plain crop of a region (no ink erased), its edge pixels repeated as a margin."""
    return cv2.copyMakeBorder(norm[box.y0:box.y1, box.x0:box.x1], pad, pad, pad, pad,
                              cv2.BORDER_REPLICATE)


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
    [optional digit 1-9] + 1-2 consonants (motorcycles: or 3 consonants), or
    2-3 digits (bus / truck). Which strings are valid is checked afterwards."""
    if n >= 3:
        first = (THAI_CONSONANTS if layout == "motorcycle" else "") + "123456789"
        return [first] + [THAI_CONSONANTS + DIGITS] * (n - 1)
    if n == 2:
        return [THAI_CONSONANTS + "123456789", THAI_CONSONANTS + DIGITS]
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
    """Small CNN for 42 consonants + 10 digits (train_letter_cnn.py), trained on
    renders of free fonts - and, in thai_letter_cnn_plates.pt, also on real plate
    glyphs. ~130k parameters, a few ms per plate on CPU."""
    name = "cnn"

    def __init__(self, weights=None):
        import torch
        self._torch = torch
        weights = weights or default_cnn_weights()
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


def combine_letters(dists, allowed, pattern=LETTERS_RE, k=3, top=5, rescore=None):
    """Per-glyph probabilities -> best string, joint probability, alternatives.
    rescore(strings) -> extra log-scores (a line recogniser reading the whole
    letters crop) added to each candidate's log joint probability; the
    probabilities are then the posterior over the candidates."""
    options = [[(al[j], float(p[j])) for j in np.argsort(-p)[:k]] for p, al in zip(dists, allowed)]
    cands = []
    for combo in itertools.product(*options):
        s = "".join(c for c, _ in combo)
        if pattern.match(s):
            cands.append((s, float(np.prod([q for _, q in combo]))))
    if rescore is not None and cands:
        score = np.log([max(q, 1e-12) for _, q in cands]) + rescore([s for s, _ in cands])
        post = np.exp(score - score.max())
        cands = [(s, float(q)) for (s, _), q in zip(cands, post / post.sum())]
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
# 4. The reader: a line recogniser for number + province, a letter engine for the letters
# ---------------------------------------------------------------------------
# a reading needs review when a format or glyph-count check fails or a confidence
# is below its threshold. Fitted on the dev half of final_data: on the held-out
# half a third of the plates are left unflagged, 94% of them fully right.
REVIEW_THRESHOLDS = dict(letters=0.70, letters_whole_crop=0.50, number=0.80, province=0.99,
                         province_vs_raw_ocr=0.6)
# checks that must pass; "letter_engines_agree" is reported too, but flagging it
# caught no extra errors once the thresholds above are applied
REVIEW_CHECKS = ("letters_format", "number_format", "letters_glyph_count", "number_glyph_count")


def review_needed(checks, confidence, per_letter, thresholds=REVIEW_THRESHOLDS):
    """checks: {name: passed}; confidence: the "confidence" dict of read()."""
    t = thresholds
    return bool(not all(v for k, v in checks.items() if k in REVIEW_CHECKS) or
                confidence["letters"] < t["letters" if per_letter else "letters_whole_crop"] or
                confidence["number"] < t["number"] or confidence["province"] < t["province"] or
                confidence["province_vs_raw_ocr"] < t["province_vs_raw_ocr"])


LETTER_ENGINES = ("easyocr", "cnn", "tesseract", "paddle")
# weight of each engine when several are averaged (the CNN was the most
# accurate in testing; Tesseract's low weight dates from before its --psm fix)
ENGINE_WEIGHTS = {"cnn": 2.0, "easyocr": 1.0, "paddle": 1.0, "tesseract": 0.5}
# weight of the field engine's reading of the whole letters crop (CTC
# log-likelihood) when it is combined with the per-glyph reading
LINE_WEIGHT = 0.1
# added to a province name's CTC log-likelihood per character, against the
# short-name bias on blurred crops. Chosen on the dev half of final_data as the
# value that reads the provinces other than Bangkok best (so it does not just
# favour the most common, longest name): on held-out images 61% -> 69% right
PROVINCE_CHAR_BONUS = 3.0
PROVINCE_BONUS = PROVINCE_CHAR_BONUS * np.array([len(p) for p in PROVINCES], float)


class ThaiPlateReader:
    """Load once, call .read() many times.

    letters_engine - how the letters (e.g. 5กข) are read, one glyph at a time:
        "easyocr"            EasyOCR on each letter separately
        "cnn"                small CNN (thai_letter_cnn_plates.pt: fonts + real plate
                             glyphs; else thai_letter_cnn.pt: fonts only), ~5 ms
        "tesseract"          Tesseract 5 LSTM (optional install, see TesseractLetters)
        "paddle"             PaddleOCR's Thai model on paddlepaddle (optional install)
        "easyocr+cnn" / ["easyocr", "cnn", ...]   weighted average of engines
                             (ENGINE_WEIGHTS); disagreement is reported in "checks"
        "group"              old behaviour: the field engine reads the letters crop in one go
        "auto" (default)     "easyocr+cnn" if the CNN weights exist, else "easyocr"
    field_engine - the recogniser for the number, the province and the letters
        crop when it cannot be cut into glyphs (read as a line, CTC-scored):
        "paddle"             PaddleOCR's Thai model (optional install) - on 650
                             plate photos: number 71% right vs 61% with EasyOCR,
                             province 48% vs 42%
        "easyocr"            EasyOCR's Thai recogniser
        "auto" (default)     "paddle" if paddlepaddle is installed, else "easyocr"
    layout - "auto" (default): car or motorcycle plate, told apart by the plate's
        shape; "car" or "motorcycle" to accept only that layout
    reader_kwargs are passed to easyocr.Reader - e.g. a fine-tuned model:
        ThaiPlateReader(recog_network="thai_plate", user_network_directory="models/",
                        model_storage_directory="models/")
    """

    def __init__(self, gpu=False, langs=("th",), letters_engine="auto", cnn_weights=None,
                 tessdata_dir=None, layout="auto", field_engine="auto", **reader_kwargs):
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

        self.cnn_weights = cnn_weights or default_cnn_weights()
        self.tessdata_dir = tessdata_dir
        self._engines = {}
        if letters_engine == "auto":
            letters_engine = "easyocr+cnn" if os.path.exists(self.cnn_weights) else "easyocr"
        self.letters_engine = self.make_letters_engine(letters_engine)
        if os.path.exists(self.cnn_weights):
            self.make_letters_engine("cnn")   # refine_letter_boxes reads with it

        if field_engine == "auto":
            field_engine = "paddle" if importlib.util.find_spec("paddle") else "easyocr"
        if field_engine not in ("easyocr", "paddle"):
            raise ValueError(f"unknown field engine {field_engine!r}; choose from auto, easyocr, paddle")
        self.field_engine = field_engine
        if field_engine == "paddle":
            self.make_letters_engine("paddle")  # the same model instance serves both

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

    def _rank(self, p, alphabet, words, top=5, min_post=1e-3, bonus=None):
        """Candidates sorted by posterior (softmax of CTC log-likelihoods, plus
        bonus: an optional score per word)."""
        ll = self._ctc_loglik(p, alphabet, words)
        if bonus is not None:
            ll = ll + bonus
        post = np.exp(ll - ll.max())
        post /= post.sum()
        order = np.argsort(-post)[:top]
        return [(words[i], float(post[i])) for n, i in enumerate(order)
                if n == 0 or post[i] >= min_post]

    # -- field readers ------------------------------------------------------------
    def line_probs(self, gray, allowed):
        """Per-frame probabilities over [blank] + allowed for a line of text, from
        the field engine: EasyOCR's or PaddleOCR's recogniser."""
        if self.field_engine == "paddle":
            eng = self._engines["paddle"]
            p = eng.frame_probs([gray])[0][:, [0] + [eng.index[c] for c in allowed]]
            return p / p.sum(1, keepdims=True)
        return self._restrict(self.frame_probs(gray), allowed)

    def read_code(self, gray, allowed, pattern, k_alt=4, length=None):
        """Letters or number: greedy decode, then re-rank the combinations of the
        top-k alternatives per character with the CTC likelihood, keeping only
        strings that match the plate format. length: the number of glyphs that
        segmentation counted; every valid string of that length is ranked too
        (the greedy decode can drop a character, e.g. one of two equal digits)."""
        p = self.line_probs(gray, allowed)
        raw, conf, dists = self._greedy(p, allowed)
        cands = []
        if dists:
            # at most ~2000 combinations: a bad crop can decode to a dozen
            # characters, and 4^12 strings would exhaust the memory
            k = max(1, min(k_alt, int(2000 ** (1 / len(dists)))))
            options = [[allowed[j] for j in np.argsort(-d)[:k]] for d in dists]
            words = ["".join(w) for w in itertools.product(*options)]
            valid = [w for w in words if pattern.match(w)]
            if not valid and len(dists) <= 12:
                # more characters read than the format allows (a dash or the frame
                # read as a digit): try dropping some; the CTC score picks which
                best = [o[0] for o in options]
                shorter = {"".join(best[i] for i in keep) for n in range(1, len(best))
                           for keep in itertools.combinations(range(len(best)), n)}
                valid = sorted(w for w in shorter if pattern.match(w))
            if length and len(allowed) ** length <= 10000:
                valid = sorted(set(valid) | {w for w in map("".join, itertools.product(
                    allowed, repeat=length)) if pattern.match(w)})
            cands = self._rank(p, allowed, valid or words)
        text = cands[0][0] if cands else raw
        return dict(text=text, raw=raw, conf=conf, candidates=cands,
                    format_ok=bool(pattern.match(text)))

    def read_province(self, gray):
        """Every official name scored with the CTC likelihood, plus
        PROVINCE_CHAR_BONUS per character: on a blurred crop the likelihood
        favours short names (เลย, ตาก ...) whatever the text."""
        p = self.line_probs(gray, PROVINCE_CHARS)
        raw, conf, _ = self._greedy(p, PROVINCE_CHARS)
        cands = self._rank(p, PROVINCE_CHARS, PROVINCES, bonus=PROVINCE_BONUS)
        best = cands[0][0]
        similarity = difflib.SequenceMatcher(None, raw, best).ratio()
        return dict(text=best, raw=raw, conf=conf, candidates=cands,
                    posterior=cands[0][1], similarity=similarity)

    # -- full pipeline ------------------------------------------------------------
    def read(self, image, return_debug=False):
        """image: file path or BGR numpy array. Returns a JSON-friendly dict.
        Runs the four steps: classify_plate, locate_plate, locate_fields and
        read_fields; result["steps"] holds what each step found."""
        t0 = time.perf_counter()
        bgr = imread(image) if isinstance(image, (str, os.PathLike)) else image
        if bgr is None:
            raise FileNotFoundError(image)

        cls = classify_plate(bgr, self.layouts)                      # 1. car or motorcycle
        layout = cls["layout"]
        t1 = time.perf_counter()
        plate = locate_plate(bgr, layout, cls["found"])              # 2. the plate
        fields = self.refine_letter_boxes(                           # 3. its fields
            field_crops(dict(plate, layout=layout)))
        seg = fields["seg"]
        t2 = time.perf_counter()
        letters, number, province, mode = self.read_fields(fields)  # 4. read them
        t3 = time.perf_counter()

        per_letter = mode.startswith("per-letter")
        checks = {
            "letters_format": letters["format_ok"],
            "number_format": number["format_ok"],
            "letters_glyph_count": len(letters["text"]) == seg["n_letter_glyphs"],
            "number_glyph_count": len(number["text"]) == seg["n_number_glyphs"],
        }
        by_engine = letters.get("by_engine")
        if by_engine:
            checks["letter_engines_agree"] = len(set(by_engine.values())) == 1
        confidence = {"letters": letters["conf"], "number": number["conf"],
                      "province": province["posterior"],
                      "province_vs_raw_ocr": province["similarity"]}
        needs_review = review_needed(checks, confidence, per_letter)
        boxes = lambda bs: [list(b.as_tuple()) for b in bs]
        result = {
            "plate": f"{letters['text']} {number['text']}",
            "layout": layout,
            "letters": letters["text"],
            "number": number["text"],
            "province": province["text"],
            "steps": {
                "1_plate_type": layout,
                "1_plate_type_scores": {k: round(float(v), 2) for k, v in cls["scores"].items()},
                "2_plate_quad": [[round(float(x), 1), round(float(y), 1)] for x, y in plate["quad"]],
                "3_fields": {name: list(seg[name].as_tuple())
                             for name in ("letters", "number", "province")},
                "3_letter_boxes": boxes(seg["letter_glyphs"]),
                "3_number_boxes": boxes(seg["number_glyphs"]),
            },
            "confidence": {k: round(v, 3) for k, v in confidence.items()},
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
            "needs_review": needs_review,
            "timing_ms": {"classify_locate": round(1000 * (t1 - t0)),
                          "fields": round(1000 * (t2 - t1)),
                          "read": round(1000 * (t3 - t2)),
                          "total": round(1000 * (t3 - t0))},
        }
        if return_debug:
            result["_debug"] = dict(image=bgr, quad=plate["quad"], plate=plate["plate"],
                                    norm=fields["norm"], ink=fields["ink"], seg=seg,
                                    crops=fields["crops"], glyphs=fields["letter_glyphs"],
                                    number_glyphs=fields["number_glyphs"])
        return result

    def refine_letter_boxes(self, fields):
        """Step 3, last part - choose the letter boxes by reading them. A consonant
        broken into two pieces gives one box too many, and so does the frame edge
        next to the letters; so two close neighbouring boxes merged, and a thin
        first box dropped, are read too (by the letter CNN), and the boxes whose
        best valid string has the highest mean log-probability per character are
        kept. (On held-out images: 8 plates fixed, none broken, of 399.)"""
        cnn = self._engines.get("cnn")
        seg, layout = fields["seg"], fields["layout"]
        boxes = seg["letter_glyphs"]
        if (cnn is None or not 2 <= len(boxes) <= 3
                or not all(b.w <= 1.3 * b.h for b in boxes)):
            return fields
        h = float(np.median([b.h for b in boxes + seg["number_glyphs"]]))
        options = [boxes]
        for i in range(len(boxes) - 1):
            a, b = boxes[i], boxes[i + 1]
            if b.x0 - a.x1 <= 0.3 * h and b.x1 - a.x0 <= 1.15 * h:
                options.append(boxes[:i] + [a.union(b)] + boxes[i + 2:])
        if boxes[0].w < 0.3 * boxes[0].h:
            options.append(boxes[1:])
        if len(options) == 1:
            return fields
        norm, ink, labels = fields["norm"], fields["ink"], seg["labels"]
        best = None
        for bs in options:
            glyphs = [glyph_crop(norm, ink, labels, b) for b in bs]
            allowed = letter_allowed_sets(len(glyphs), layout)
            dists = cnn.predict(glyphs, allowed)
            out = combine_letters(dists, allowed, LAYOUTS[layout]["letters_re"])
            if "?" in out["text"] or not out["format_ok"]:
                continue
            score = np.mean([np.log(max(d[al.index(c)], 1e-9))
                             for d, al, c in zip(dists, allowed, out["text"])])
            if best is None or score > best[0]:
                best = (score, bs, glyphs)
        if best is None or best[1] is boxes:
            return fields
        _, bs, glyphs = best
        seg = dict(seg, letter_glyphs=bs, n_letter_glyphs=len(bs), letters=_union_all(bs))
        crops = dict(fields["crops"], letters=field_crop(norm, ink, labels, seg["letters"]))
        return dict(fields, seg=seg, crops=crops, letter_glyphs=glyphs)

    def read_fields(self, fields):
        """Step 4 - read located fields (locate_fields / field_crops output).
        Letters: one character at a time with the letters engine (combined with
        the field engine's reading of the whole letters crop), or the whole crop
        at once when the characters look merged or there are more than 3.
        Number: the field engine reads the crop as a group, ranking the strings
        of the counted length too. Province: every official name is scored.
        -> (letters, number, province, letters_mode)"""
        layout, seg, crops = fields["layout"], fields["seg"], fields["crops"]
        boxes = seg["letter_glyphs"]
        per_letter = (self.letters_engine != "group" and 1 <= len(boxes) <= 3
                      and all(b.w <= 1.3 * b.h for b in boxes))
        if per_letter:
            letters = self.read_letters(fields["letter_glyphs"], layout=layout,
                                        line_crop=crops["letters"])
            mode = f"per-letter ({letters['engine']})"
        else:
            letters = self.read_code(crops["letters"], THAI_CONSONANTS + DIGITS,
                                     LAYOUTS[layout]["letters_re"])
            mode = (f"whole crop ({self.field_engine})"
                    + ("" if self.letters_engine == "group" else ", fallback"))
        n = seg["n_number_glyphs"]
        number = self.read_code(crops["number"], DIGITS, NUMBER_RE,
                                length=n if 1 <= n <= 4 else None)
        province = self.read_province(crops["province"])
        return letters, number, province, mode

    def read_letters(self, glyphs, engine=None, layout="car", line_crop=None):
        """Read letter glyphs ONE AT A TIME with a letters engine, then combine.
        line_crop: the whole letters crop; the field engine's reading of it is
        added to the score of every candidate (LINE_WEIGHT)."""
        engine = engine or self.letters_engine
        allowed = letter_allowed_sets(len(glyphs), layout)
        dists = engine.predict(glyphs, allowed)
        rescore = None
        if line_crop is not None and LINE_WEIGHT:
            alphabet = THAI_CONSONANTS + DIGITS
            p = self.line_probs(line_crop, alphabet)
            rescore = lambda words: LINE_WEIGHT * self._ctc_loglik(p, alphabet, words)
        out = combine_letters(dists, allowed, LAYOUTS[layout]["letters_re"], rescore=rescore)
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
    ap.add_argument("--fields", default="auto", choices=("auto", "easyocr", "paddle"),
                    help="recogniser for number and province: paddle (if installed) | easyocr")
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
                             tessdata_dir=args.tessdata_dir, layout=args.layout,
                             field_engine=args.fields)
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
