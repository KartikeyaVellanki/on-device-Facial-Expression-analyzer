#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_landmarks_csv(path: Path) -> Dict[int, np.ndarray]:
    frames: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            frame = int(row[0])
            vals = [float(x) for x in row[1:] if x != ""]
            pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
            frames[frame] = pts
    return frames


def load_meta_csv(path: Optional[Path]) -> Dict[int, str]:
    status: Dict[int, str] = {}
    if path is None or not path.exists():
        return status
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            frame = int(row[0])
            status[frame] = row[1]
    return status


def load_raw_csv(path: Optional[Path]) -> Dict[int, np.ndarray]:
    d: Dict[int, np.ndarray] = {}
    if path is None or not path.exists():
        return d
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            frame = int(row[0])
            vals = [float(x) for x in row[1:]]
            d[frame] = np.array(vals, dtype=np.float32)
    return d


# Landmark index helpers (iBUG 68)
LE = [36, 37, 38, 39, 40, 41]
RE = [42, 43, 44, 45, 46, 47]
OUTER_LIP = list(range(48, 60))
INNER_LIP = list(range(60, 68))


def eye_aspect_ratio(pts: np.ndarray, idx: List[int]) -> float:
    p = pts[idx, :]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3]) + 1e-6
    return (v1 + v2) / (2.0 * h)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    p = pts
    A = np.linalg.norm(p[61] - p[67])
    B = np.linalg.norm(p[62] - p[66])
    C = np.linalg.norm(p[63] - p[65])
    D = np.linalg.norm(p[60] - p[64]) + 1e-6
    return (A + B + C) / (3.0 * D)


def brow_raise_metric(pts: np.ndarray) -> float:
    # average distance from brow (19,24) to corresponding eye centers normalized by face height
    left_eye_center = pts[LE, :].mean(axis=0)
    right_eye_center = pts[RE, :].mean(axis=0)
    left_brow = pts[19]
    right_brow = pts[24]
    dist = (np.linalg.norm(left_brow - left_eye_center) + np.linalg.norm(right_brow - right_eye_center)) / 2.0
    face_h = pts[:, 1].max() - pts[:, 1].min() + 1e-6
    return float(dist / face_h)


def smile_lift_metric(pts: np.ndarray) -> float:
    # corner lift relative to mouth center (negative when corners are below center)
    lc = pts[48]
    rc = pts[54]
    center = (pts[51] + pts[57]) / 2.0
    lift = ((center[1] - lc[1]) + (center[1] - rc[1])) / 2.0
    face_h = pts[:, 1].max() - pts[:, 1].min() + 1e-6
    return float(lift / face_h)


def head_ypr_from_raw(raw: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if raw is None or raw.size < 264:
        return None, None, None
    pitch = raw[258] * 90.0
    yaw = raw[259] * 90.0
    roll = raw[260] * 90.0
    return float(yaw), float(pitch), float(roll)


def classify_expression(pts: np.ndarray, raw: Optional[np.ndarray]) -> Tuple[str, Dict[str, float]]:
    ear_l = eye_aspect_ratio(pts, LE)
    ear_r = eye_aspect_ratio(pts, RE)
    ear = (ear_l + ear_r) / 2.0
    mar = mouth_aspect_ratio(pts)
    brow = brow_raise_metric(pts)
    smile_lift = smile_lift_metric(pts)
    yaw, pitch, roll = head_ypr_from_raw(raw)

    feats = {
        "ear": ear,
        "mar": mar,
        "brow": brow,
        "smile_lift": smile_lift,
        "yaw": yaw if yaw is not None else 0.0,
        "pitch": pitch if pitch is not None else 0.0,
        "roll": roll if roll is not None else 0.0,
    }

    # Thresholds (heuristic)
    blink_thr = 0.18
    eyes_low = ear < blink_thr
    mouth_open_thr = 0.55
    smile_thr = 0.015
    frown_thr = -0.007
    brow_high_thr = 0.10
    gaze_away = abs(feats["yaw"]) > 30.0

    label = "neutral"
    if eyes_low:
        label = "blink/eyes-closed"
    if mar > mouth_open_thr and brow > brow_high_thr:
        label = "surprised"
    elif smile_lift > smile_thr and mar < 0.7:
        label = "smile"
    elif smile_lift < frown_thr:
        label = "frown"
    if gaze_away:
        label = label + "+gaze-away" if label != "neutral" else "gaze-away"
    return label, feats


def frame_score(label: str) -> float:
    # Map expression to per-frame score in [0,1]
    base = 0.6  # neutral baseline
    if label.startswith("smile"):
        return 0.85
    if label.startswith("surprised"):
        return 0.4
    if "blink" in label:
        return 0.5
    if "frown" in label:
        return 0.35
    if "gaze-away" in label:
        return 0.45
    return base


def analyze(landmarks_csv: Path, meta_csv: Optional[Path], raw_csv: Optional[Path], out_dir: Path):
    lm = load_landmarks_csv(landmarks_csv)
    meta = load_meta_csv(meta_csv)
    raw = load_raw_csv(raw_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    expr_rows: List[List[str]] = []
    per_frame_scores: List[float] = []
    classes_count: Dict[str, int] = {}

    frames_sorted = sorted(lm.keys())
    for f in frames_sorted:
        status = meta.get(f, "ok")
        if status.startswith("no_face") or status == "unstable":
            continue
        pts = lm[f]
        raw_vec = raw.get(f)
        label, feats = classify_expression(pts, raw_vec)
        score = frame_score(label)
        classes_count[label] = classes_count.get(label, 0) + 1
        per_frame_scores.append(score)
        # Convert feats to plain float for JSON
        feats_jsonable = {k: float(v) for k, v in feats.items()}
        expr_rows.append([str(f), label, f"{score:.3f}", json.dumps(feats_jsonable)])

    overall = float(np.mean(per_frame_scores) * 100.0) if per_frame_scores else 0.0

    # Write outputs
    with open(out_dir / "expressions.csv", "w") as ef:
        ef.write("frame,label,score,features\n")
        ef.write("\n".join([",".join(r) for r in expr_rows]))
    with open(out_dir / "score.txt", "w") as sf:
        sf.write(f"{overall:.1f}\n")
    report = {
        "overall_score": overall,
        "frames_scored": len(per_frame_scores),
        "class_distribution": classes_count,
    }
    with open(out_dir / "report.json", "w") as jf:
        json.dump(report, jf, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Analyze facial expressions from landmarks CSV; output score out of 100.")
    ap.add_argument("--landmarks", required=True, help="Path to *_landmarks.csv")
    ap.add_argument("--meta", default=None, help="Optional path to *_meta.csv for status")
    ap.add_argument("--raw", default=None, help="Optional path to *_raw.csv for pose params")
    ap.add_argument("--out", default="out/analysis", help="Output directory for analysis results")
    args = ap.parse_args()

    analyze(Path(args.landmarks), Path(args.meta) if args.meta else None, Path(args.raw) if args.raw else None, Path(args.out))


if __name__ == "__main__":
    main()
