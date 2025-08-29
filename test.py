#!/usr/bin/env python3
"""
Domino pip counter (robust version)

Usage:
  python domino_pip_counter_fixed.py /path/to/image.jpg --save result.jpg --debug_dir ./debug
  python domino_pip_counter_fixed.py /path/to/image.jpg --show   # shows windows; press any key to exit

Key ideas:
- White-tile mask using LAB + HSV (avoids background clutter)
- Per-tile perspective warp to a canonical 200x400 (scale-invariant)
- Illumination normalization + CLAHE for highlights
- Midline (separator) suppression before pip detection
- SimpleBlobDetector on binarized, cleaned pips (with contour fallback)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ----------------------- Tunables -----------------------
MAX_SIDE = 1600

# White mask thresholds (LAB + HSV). Conservative but robust.
LAB_L_MIN = 180         # white brightness
LAB_A_DEV = 16          # |a-128| <= LAB_A_DEV
LAB_B_DEV = 20          # |b-128| <= LAB_B_DEV
HSV_V_MIN = 170         # high value
HSV_S_MAX = 80          # low saturation

MIN_TILE_AREA = 6_000   # ignore tiny contours
ASPECT_MIN = 1.55       # domino ~2:1
ASPECT_MAX = 2.9

# Canonical tile size after warp (HxW must be 2:1)
TILE_H = 400
TILE_W = 200

# Separator band thickness (fraction of tile height)
SEP_BAND_FRAC = 0.09    # ~9% of height

# Pip morphology (relative to TILE_H)
PIP_R_MIN_FRAC = 0.025  # ~10 px at H=400
PIP_R_MAX_FRAC = 0.075  # ~30 px at H=400

# Blob detector thresholds
BLOB_MIN_THRESHOLD = 5
BLOB_MAX_THRESHOLD = 200
BLOB_THRESHOLD_STEP = 5

# --------------------------------------------------------


@dataclass
class TileDetection:
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]  # ((cx,cy),(w,h),angle)
    contour: np.ndarray                                           # Nx1x2
    M: np.ndarray                                                 # 3x3 perspective (src->dst)
    Minv: np.ndarray                                              # 3x3 inverse (dst->src)


def resize_max(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def save_dbg(img: np.ndarray, path: Path, show: bool, win_name: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    if show:
        cv2.imshow(win_name or path.stem, img)


def white_tile_mask(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)

    m1 = cv2.inRange(L, LAB_L_MIN, 255)
    m2 = cv2.inRange(A, 128 - LAB_A_DEV, 128 + LAB_A_DEV)
    m3 = cv2.inRange(B, 128 - LAB_B_DEV, 128 + LAB_B_DEV)
    m4 = cv2.inRange(Vv, HSV_V_MIN, 255)
    m5 = cv2.inRange(Ss, 0, HSV_S_MAX)

    mask = cv2.bitwise_and(m1, m2)
    mask = cv2.bitwise_and(mask, m3)
    mask = cv2.bitwise_and(mask, m4)
    mask = cv2.bitwise_and(mask, m5)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl (clockwise)."""
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_tile(bgr: np.ndarray, rect) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = cv2.boxPoints(rect).astype(np.float32)  # (4,2)
    (cx, cy), (w, h), ang = rect
    # Ensure height >= width (vertical orientation)
    H, W = (TILE_H, TILE_W) if h >= w else (TILE_H, TILE_W)  # we still warp to 200x400
    src = order_box_points(box)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)
    return warp, M, Minv


def illumination_normalize(gray: np.ndarray) -> np.ndarray:
    # Divide by a big blur to flatten shading, then CLAHE
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    blur = np.clip(blur, 1, 255)
    norm = cv2.divide(gray, blur, scale=128)  # 0..255ish
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def pip_binary_from_tile(tile_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    H, W = tile_bgr.shape[:2]
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    norm = illumination_normalize(gray)
    inv = cv2.bitwise_not(norm)

    # Adaptive threshold on the inverted image -> pips become white
    pip_bin = cv2.adaptiveThreshold(
        inv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -5
    )

    # Close small gaps in ring-like pips, then open to remove specks
    disk_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    disk_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_bin = cv2.morphologyEx(pip_bin, cv2.MORPH_CLOSE, disk_close, iterations=1)
    pip_bin = cv2.morphologyEx(pip_bin, cv2.MORPH_OPEN,  disk_open,  iterations=1)

    # Fill holes (turn crescents / rings into solid blobs)
    flood = pip_bin.copy()
    h, w = flood.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)   # fill background
    flood_inv = cv2.bitwise_not(flood)
    pip_filled = pip_bin | flood_inv

    return pip_filled, H, W


def suppress_separator(pip_bin: np.ndarray, center_zone: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Remove the dark midline by zeroing a horizontal band near its maximum darkness.
    center_zone: optional (y0,y1) band to search. If None, search 35%-65% of height.
    """
    H, W = pip_bin.shape[:2]
    y0 = int(0.35 * H) if center_zone is None else center_zone[0]
    y1 = int(0.65 * H) if center_zone is None else center_zone[1]
    band = pip_bin[y0:y1, :]
    # Look for densest horizontal line (row with max white pixels)
    sums = band.sum(axis=1)  # white=255
    row_rel = int(np.argmax(sums))
    y_mid = y0 + row_rel
    band_half = max(2, int(SEP_BAND_FRAC * H / 2))
    pip_bin[max(0, y_mid - band_half):min(H, y_mid + band_half), :] = 0
    return pip_bin


def blob_params_for_tile(H: int) -> cv2.SimpleBlobDetector_Params:
    rmin = max(4, int(PIP_R_MIN_FRAC * H))
    rmax = max(rmin + 2, int(PIP_R_MAX_FRAC * H))
    min_area = int(np.pi * (rmin ** 2))
    max_area = int(np.pi * (rmax ** 2))

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = BLOB_MIN_THRESHOLD
    params.maxThreshold = BLOB_MAX_THRESHOLD
    params.thresholdStep = BLOB_THRESHOLD_STEP

    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)

    params.filterByCircularity = True
    params.minCircularity = 0.5

    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    params.filterByConvexity = True
    params.minConvexity = 0.5

    params.filterByColor = True
    params.blobColor = 255  # we detect on binary with white pips

    return params


def detect_pips(tile_bgr: np.ndarray, dbg_prefix: Path | None = None, show: bool = False) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    pip_bin, H, W = pip_binary_from_tile(tile_bgr)
    pip_bin = suppress_separator(pip_bin)

    if dbg_prefix is not None:
        save_dbg(tile_bgr, dbg_prefix.with_name(dbg_prefix.stem + "_tile_rgb.png"), show, "tile")
        save_dbg(pip_bin, dbg_prefix.with_name(dbg_prefix.stem + "_pip_bin.png"), show, "pip_bin")

    # Blob detector
    params = blob_params_for_tile(H)
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(pip_bin)

    centers = [(kp.pt[0], kp.pt[1]) for kp in kps]

    # Fallback: contour filtering if blob count looks too low
    if len(centers) == 0:
        cnts, _ = cv2.findContours(pip_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers_fallback = []
        min_area = params.minArea
        max_area = params.maxArea
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            per = cv2.arcLength(c, True) + 1e-6
            circ = 4 * np.pi * a / (per * per)
            if circ < 0.45:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            centers_fallback.append((cx, cy))
        centers = centers_fallback

    # For visualization: draw on a copy of the tile
    vis = tile_bgr.copy()
    for (cx, cy) in centers:
        cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 2, (0, 255, 0), -1)

    if dbg_prefix is not None:
        save_dbg(vis, dbg_prefix.with_name(dbg_prefix.stem + "_tile_overlay.png"), show, "tile_overlay")

    return centers, pip_bin


def find_tiles(bgr: np.ndarray, mask: np.ndarray) -> List[TileDetection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles: List[TileDetection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_TILE_AREA:
            continue
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), ang = rect
        if w <= 0 or h <= 0:
            continue
        ar = max(w, h) / max(1e-6, min(w, h))
        if not (ASPECT_MIN <= ar <= ASPECT_MAX):
            continue
        warp_rgb, M, Minv = warp_tile(bgr, rect)
        tiles.append(TileDetection(rect=rect, contour=c, M=M, Minv=Minv))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--show", action="store_true", help="Show debug windows (press any key to exit).")
    ap.add_argument("--save", type=Path, default=None, help="Save final overlay to this path.")
    ap.add_argument("--debug_dir", type=Path, default=None, help="Directory to dump debug images.")
    args = ap.parse_args()

    img0 = cv2.imread(str(args.image))
    if img0 is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    img = resize_max(img0, MAX_SIDE)

    debug_dir = args.debug_dir if args.debug_dir else (args.image.with_suffix("").parent / (args.image.stem + "_debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1) White-tile mask
    mask = white_tile_mask(img)
    save_dbg(mask, debug_dir / "01_tile_mask.png", args.show, "01_tile_mask")

    # 2) Find tiles
    tiles = find_tiles(img, mask)
    print(f"[info] Detected tiles: {len(tiles)}")

    # Visualize tile contours
    vis_tiles = img.copy()
    cv2.drawContours(vis_tiles, [t.contour for t in tiles], -1, (0, 255, 0), 2)
    save_dbg(vis_tiles, debug_dir / "02_tiles_contours.png", args.show, "02_tiles")

    total_pips = 0
    overlay = img.copy()

    for idx, t in enumerate(tiles):
        warp_rgb, M, Minv = warp_tile(img, t.rect)

        # 3) Detect pips per tile (with midline suppression)
        centers, pip_bin = detect_pips(
            warp_rgb,
            dbg_prefix=(debug_dir / f"tile_{idx:02d}"),
            show=args.show
        )

        total_pips += len(centers)

        # 4) Project centers back to original image and draw
        if centers:
            pts = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
            pts_src = cv2.perspectiveTransform(pts, t.Minv).reshape(-1, 2)
            for (x, y) in pts_src:
                cv2.circle(overlay, (int(x), int(y)), 12, (0, 255, 0), 2)
                cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)

    # 5) Summary overlay
    text = f"Tiles: {len(tiles)} | Total pips: {total_pips}"
    cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 255, 40), 2, cv2.LINE_AA)

    # Save + optional show
    out_path = args.save if args.save else (debug_dir / "final_overlay.png")
    cv2.imwrite(str(out_path), overlay)
    print(f"[info] Saved: {out_path}")
    print(f"[info] Grand total pips: {total_pips}")

    if args.show:
        cv2.imshow("final_overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
