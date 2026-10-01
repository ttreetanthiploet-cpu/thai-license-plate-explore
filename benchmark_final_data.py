#!/usr/bin/env python3
"""
benchmark_final_data.py - run every letters engine on the final_data plates.

Each image is segmented ONCE (ThaiPlateReader.read), the number and province
are read ONCE (they do not depend on the letters engine), then the same letter
glyphs are given to every engine. Ensembles are built from the engines' cached
per-glyph probabilities with EnsembleLetters, so they cost nothing extra.

Output: one CSV row per (image, method) - the predictions only; scoring against
solution.csv is done in final_data_benchmark.ipynb.

    python benchmark_final_data.py --data ../get_license_data/data/final_data
    python benchmark_final_data.py --split test --sample 200   # 200 held-out images
    python benchmark_final_data.py --split test --min-height 200 --sample 300 \
        --out benchmark/final_data_predictions_200px.csv       # large photos only
    python benchmark_final_data.py --limit 200            # the first 200 images
    python benchmark_final_data.py --methods cnn,easyocr+cnn

Splits: "dev" = md5(file name) % 2 == 0 (used to develop and tune the reader and
to train the letter CNN), "test" = the other half, never trained or tuned on.

Resumable: images that already have a row for every method are skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

DEFAULT_DATA = "/Users/tanut/Desktop/work/get_license_data/data/final_data"
DEFAULT_METHODS = ("group,easyocr,cnn,tesseract,paddle,"
                   "easyocr+cnn,cnn+paddle,easyocr+cnn+paddle,easyocr+cnn+tesseract+paddle")
COLUMNS = ["image", "method", "letters_pred", "letters_conf", "letters_format_ok",
           "letters_mode", "by_engine", "number_pred", "number_conf", "province_pred",
           "province_posterior", "province_similarity", "layout", "n_letter_glyphs",
           "n_number_glyphs", "needs_review", "error", "classify_locate_ms", "fields_ms", "read_ms",
           "letters_ms"]


class CachedEngine:
    """A letters engine that returns probabilities computed earlier."""

    def __init__(self, name, dists):
        self.name, self.dists = name, dists

    def predict(self, glyphs, allowed):
        return self.dists


def init_worker(methods, threads):
    global READER, METHODS, TPR
    warnings.simplefilter("ignore")
    import cv2
    import torch
    cv2.setNumThreads(1)   # OpenCV starts a thread per core in every worker otherwise
    torch.set_num_threads(threads)
    import thai_plate_reader as tpr
    TPR, METHODS = tpr, methods
    # read() only needs a letters engine for its own letters: the cheap CNN
    READER = tpr.ThaiPlateReader(letters_engine="cnn")
    for name in sorted({n for m in methods for n in m.split("+")} - {"group"}):
        READER.make_letters_engine(name)


def needs_review(res, letters, per_letter):
    """ThaiPlateReader.read()'s review rule, for another letters result."""
    checks = dict(res["checks"])
    checks["letters_format"] = letters["format_ok"]
    checks["letters_glyph_count"] = len(letters["text"]) == res["_n_letter_glyphs"]
    checks.pop("letter_engines_agree", None)
    if letters.get("by_engine"):
        checks["letter_engines_agree"] = len(set(letters["by_engine"].values())) == 1
    return TPR.review_needed(checks, dict(res["confidence"], letters=letters["conf"]), per_letter)


def process(path):
    tpr = TPR
    image = os.path.basename(path)
    try:
        res = READER.read(path, return_debug=True)
    except Exception as e:  # PlateNotFound, or an unreadable image
        err = f"{type(e).__name__}: {e}"
        return [dict(image=image, method=m, error=err) for m in METHODS]

    dbg = res.pop("_debug")
    layout, glyphs = res["layout"], dbg["glyphs"]
    res["_n_letter_glyphs"] = dbg["seg"]["n_letter_glyphs"]
    per_letter = res["letters_detail"]["mode"].startswith("per-letter")
    allowed = tpr.letter_allowed_sets(len(glyphs), layout)

    # each base engine reads the glyphs once
    dists, ms = {}, {}
    if per_letter:
        for name in sorted({n for m in METHODS for n in m.split("+")} - {"group"}):
            t = time.perf_counter()
            dists[name] = READER._engines[name].predict(glyphs, allowed)
            ms[name] = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    group = READER.read_code(dbg["crops"]["letters"], tpr.THAI_CONSONANTS + tpr.DIGITS,
                             tpr.LAYOUTS[layout]["letters_re"])
    ms["group"] = 1000 * (time.perf_counter() - t)

    rows = []
    for m in METHODS:
        names = m.split("+")
        if m == "group" or not per_letter:        # read() falls back to the whole crop
            letters, mode = group, "whole crop" + ("" if m == "group" else ", fallback")
        else:
            engines = [CachedEngine(n, dists[n]) for n in names]
            engine = engines[0] if len(engines) == 1 else tpr.EnsembleLetters(
                engines, [tpr.ENGINE_WEIGHTS[n] for n in names])
            letters = READER.read_letters(glyphs, engine=engine, layout=layout,
                                          line_crop=dbg["crops"]["letters"])
            mode = "per-letter"
        letters_ms = ms["group"] if mode.startswith("whole") else sum(ms[n] for n in names)
        rows.append(dict(
            image=image, method=m, letters_pred=letters["text"],
            letters_conf=round(letters["conf"], 4), letters_format_ok=letters["format_ok"],
            letters_mode=mode, by_engine="|".join(f"{k}={v}" for k, v in
                                                  (letters.get("by_engine") or {}).items()),
            number_pred=res["number"], number_conf=res["confidence"]["number"],
            province_pred=res["province"], province_posterior=res["confidence"]["province"],
            province_similarity=res["confidence"]["province_vs_raw_ocr"], layout=layout,
            n_letter_glyphs=dbg["seg"]["n_letter_glyphs"],
            n_number_glyphs=dbg["seg"]["n_number_glyphs"],
            needs_review=needs_review(res, letters, per_letter), error="",
            classify_locate_ms=res["timing_ms"]["classify_locate"],
            fields_ms=res["timing_ms"]["fields"], read_ms=res["timing_ms"]["read"],
            letters_ms=round(letters_ms, 1)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=DEFAULT_DATA, help="final_data folder (images/, solution.csv)")
    ap.add_argument("--out", default="benchmark/final_data_predictions.csv")
    ap.add_argument("--methods", default=DEFAULT_METHODS,
                    help="comma-separated letters engines / '+' ensembles / group")
    ap.add_argument("--limit", type=int, help="only the first N images of solution.csv")
    ap.add_argument("--split", choices=("all", "dev", "test"), default="all",
                    help="dev: md5(image) %% 2 == 0, test: the held-out other half")
    ap.add_argument("--min-height", type=int, help="only images at least this many pixels tall")
    ap.add_argument("--sample", type=int, help="a fixed random sample of N images (seed 0)")
    ap.add_argument("--workers", type=int, default=3,
                    help="worker processes; each holds ~1 GB of models (8 GB RAM: 3)")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    args = ap.parse_args(argv)
    methods = args.methods.split(",")

    with open(os.path.join(args.data, "solution.csv"), encoding="utf-8-sig") as fh:
        images = [r["image"] for r in csv.DictReader(fh)][:args.limit]
    if args.split != "all":
        want = ("dev", "test").index(args.split)
        images = [f for f in images if int(hashlib.md5(f.encode()).hexdigest(), 16) % 2 == want]
    if args.min_height:
        from PIL import Image
        images = [f for f in images
                  if Image.open(os.path.join(args.data, "images", f)).size[1] >= args.min_height]
    if args.sample:
        images = sorted(random.Random(0).sample(images, min(args.sample, len(images))))
    # resume: keep the images that have a row for every method (a killed run can
    # leave one image half written), then append the rest
    kept = []
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8", newline="") as fh:
            old = list(csv.DictReader(fh))
        per_image = {}
        for r in old:
            per_image.setdefault(r["image"], set()).add(r["method"])
        kept = [r for r in old if per_image[r["image"]] >= set(methods)]
    done = {r["image"] for r in kept}
    todo = [os.path.join(args.data, "images", f) for f in images if f not in done]
    print(f"{len(images)} images, {len(done)} already done, {len(todo)} to read; "
          f"methods: {', '.join(methods)}", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    # ProcessPoolExecutor, not Pool: if a worker dies (e.g. out of memory) the run
    # stops with an error instead of waiting forever; rerun to resume
    with open(args.out, "w", newline="", encoding="utf-8") as fh, \
            ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                initializer=init_worker, initargs=(methods, args.threads)) as pool:
        writer = csv.DictWriter(fh, COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
        fh.flush()
        for n, rows in enumerate(pool.map(process, todo, chunksize=4), 1):
            writer.writerows(rows)
            if n % 100 == 0 or n == len(todo):
                fh.flush()
                rate = n / (time.perf_counter() - t0)
                print(f"{n}/{len(todo)}  {rate:.1f} img/s  "
                      f"~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
