#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
try:
    import cv2  # for video IO and face detection
except Exception:
    cv2 = None
import onnxruntime as ort

# Optional Qualcomm FaceMap 3DMM utilities
try:
    from qai_hub_models.models.facemap_3dmm.utils import (
        project_landmark as qcom_project_landmark,
        transform_landmark_coordinates as qcom_transform_landmark,
    )
    HAS_QCOM_3DMM = True
except Exception:
    HAS_QCOM_3DMM = False


def debug(msg: str):
    print(f"[infer] {msg}")


def load_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 2  # warnings+
    providers = ["CPUExecutionProvider"]
    debug(f"Loading model: {model_path}")
    debug(f"Available providers: {ort.get_available_providers()}")
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    debug(f"Using provider(s): {sess.get_providers()}")
    return sess


def io_info(sess: ort.InferenceSession):
    def _shape_str(dims):
        return "x".join(str(d) if isinstance(d, int) else str(d) for d in dims)

    print("Inputs:")
    for i, t in enumerate(sess.get_inputs()):
        print(f"  [{i}] name={t.name} shape={_shape_str(t.shape)} dtype={t.type}")
    print("Outputs:")
    for i, t in enumerate(sess.get_outputs()):
        print(f"  [{i}] name={t.name} shape={_shape_str(t.shape)} dtype={t.type}")


def _dim_as_int(d):
    try:
        return int(d)
    except Exception:
        return -1


def determine_layout(input_shape: List[int]) -> Tuple[str, int, int, int]:
    """
    Infer layout (NCHW or NHWC) and return (layout, H, W, C).
    Unknown dims are -1.
    """
    dims = [_dim_as_int(x) for x in input_shape]
    # Remove batch dim if present
    if len(dims) == 4:
        n, a, b, c = dims
        # Try NCHW
        if b > 0 and c > 0 and a in (1, 3):
            return "NCHW", b, c, a
        # Try NHWC
        if a > 0 and b > 0 and c in (1, 3):
            return "NHWC", a, b, c
        # Fallback heuristics: assume channel dim is where 1 or 3 appears
        if dims[1] in (1, 3):
            return "NCHW", dims[2], dims[3], dims[1]
        if dims[-1] in (1, 3):
            return "NHWC", dims[1], dims[2], dims[3]
    elif len(dims) == 3:
        a, b, c = dims
        # CHW
        if a in (1, 3) and b > 0 and c > 0:
            return "CHW", b, c, a
        # HWC
        if c in (1, 3) and a > 0 and b > 0:
            return "HWC", a, b, c
    raise ValueError(f"Unsupported input shape: {input_shape}")


def preprocess_image(
    image_path: str,
    H: int,
    W: int,
    C: int,
    layout: str,
    normalize: str = "0_1",
    bgr: bool = False,
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((W, H), Image.BILINEAR)
    x = np.array(img_resized, dtype=np.float32)  # HxWx3
    if bgr:
        x = x[..., ::-1]
    if normalize == "none":
        pass
    elif normalize == "0_1":
        x = x / 255.0
    elif normalize == "-1_1":
        x = (x / 127.5) - 1.0
    elif normalize == "imagenet":
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    # Reorder layout
    if layout in ("NCHW", "CHW"):
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = x[np.newaxis, ...]  # add batch dim
    elif layout in ("NHWC", "HWC"):
        x = x[np.newaxis, ...]  # add batch dim
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return x.astype(np.float32)


def preprocess_from_array(
    frame: np.ndarray,
    H: int,
    W: int,
    C: int,
    layout: str,
    normalize: str = "0_1",
    bgr_input: bool = True,
    bgr: bool = False,
) -> np.ndarray:
    # frame: HxWxC (BGR if bgr_input True else RGB)
    if bgr_input:
        frame_rgb = frame[:, :, ::-1]
    else:
        frame_rgb = frame
    img_resized = Image.fromarray(frame_rgb).resize((W, H), Image.BILINEAR)
    x = np.array(img_resized, dtype=np.float32)  # HxWx3 RGB
    if bgr:
        x = x[..., ::-1]
    if normalize == "none":
        pass
    elif normalize == "0_1":
        x = x / 255.0
    elif normalize == "-1_1":
        x = (x / 127.5) - 1.0
    elif normalize == "imagenet":
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")
    if layout in ("NCHW", "CHW"):
        x = np.transpose(x, (2, 0, 1))
        x = x[np.newaxis, ...]
    elif layout in ("NHWC", "HWC"):
        x = x[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return x.astype(np.float32)


def run_inference(
    sess: ort.InferenceSession,
    input_name: str,
    x: np.ndarray,
    output_names: Optional[List[str]] = None,
):
    feeds = {input_name: x}
    if output_names:
        outputs = sess.run(output_names, feeds)
    else:
        outputs = sess.run(None, feeds)
    return outputs


def summarize_outputs(outputs: List[np.ndarray]) -> str:
    parts = []
    for i, o in enumerate(outputs):
        arr = np.asarray(o)
        parts.append(f"[{i}] shape={list(arr.shape)} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
    return "\n".join(parts)


def decode_3dmm_to_points(
    params: np.ndarray,
    resized_h: int,
    resized_w: int,
    out_h: int,
    out_w: int,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Decode 265-D FaceMap 3DMM params to 68x2 landmark points in pixel coords.

    Uses Qualcomm's reference utils if available. If unavailable or fails, returns None.
    We assume the face ROI is the full frame (no crop), so we map from resized (model input)
    to the output resolution directly.
    """
    if not HAS_QCOM_3DMM:
        return None
    try:
        flat = np.asarray(params).reshape(-1)
        if flat.size < 265:
            return None
        # Project landmarks in resized image space (center-origin coords)
        _torch = getattr(qcom_project_landmark, '__globals__', {}).get('torch', None)
        if _torch is None:
            return None
        lm = qcom_project_landmark(_torch.tensor(flat[:265], dtype=_torch.float32))  # torch.Tensor [68,2]
        # Transform to output pixel coordinates using ROI bbox (or full frame)
        if bbox is None:
            bbox = (0, 0, out_w - 1, out_h - 1)
        qcom_transform_landmark(lm, bbox=bbox, resized_height=resized_h, resized_width=resized_w)
        pts = lm.detach().cpu().numpy()  # [68,2] in pixel space of out_w x out_h
        return pts
    except Exception as e:
        debug(f"3DMM decode failed: {e}")
        return None


def detect_face_bbox(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    try:
        cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return (int(x), int(y), int(x + w), int(y + h))
    except Exception:
        return None


def crop_square_with_margin(bbox: Tuple[int, int, int, int], img_w: int, img_h: int, margin: float = 0.2) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    size = max(x1 - x0, y1 - y0)
    size = int(size * (1.0 + margin))
    half = size // 2
    nx0 = max(0, int(cx - half))
    ny0 = max(0, int(cy - half))
    nx1 = min(img_w - 1, int(cx + half))
    ny1 = min(img_h - 1, int(cy + half))
    # adjust to square
    w = nx1 - nx0
    h = ny1 - ny0
    if w != h:
        if w > h:
            ny1 = min(img_h - 1, ny1 + (w - h))
        else:
            nx1 = min(img_w - 1, nx1 + (h - w))
    return (nx0, ny0, nx1, ny1)


def pick_points_array(outputs: List[np.ndarray], assume_points: Optional[int] = None) -> Optional[np.ndarray]:
    # Prefer float arrays with a trailing dim of 2 or 3
    candidate = None
    for o in outputs:
        arr = np.asarray(o)
        if arr.dtype.kind not in ("f", "i"):  # floats or ints
            continue
        if arr.ndim >= 2 and arr.shape[-1] in (2, 3):
            candidate = arr.reshape(-1, arr.shape[-1])
            break
    if candidate is None:
        # Try 2D with last dim 2/3
        for o in outputs:
            arr = np.asarray(o)
            if arr.dtype.kind not in ("f", "i"):
                continue
            flat = arr.reshape(-1)
            if assume_points:
                for d in (2, 3):
                    need = assume_points * d
                    if flat.size >= need:
                        return flat[:need].reshape(assume_points, d)
            # Guess dims
            if flat.size % 2 == 0:
                return flat.reshape(-1, 2)
            if flat.size % 3 == 0:
                return flat.reshape(-1, 3)
        return None
    if assume_points:
        if candidate.shape[0] >= assume_points:
            return candidate[:assume_points]
    return candidate


def is_normalized(points: np.ndarray) -> bool:
    if points.size == 0:
        return True
    mn = float(points.min())
    mx = float(points.max())
    # Heuristic: if values are mostly in [-0.5, 1.5], treat as normalized
    return (mn >= -0.5) and (mx <= 1.5)


def draw_points(
    image_path: str,
    points: np.ndarray,
    out_path: str,
    size: Tuple[int, int],
    normalized: bool,
):
    W, H = size[1], size[0]
    base = Image.open(image_path).convert("RGB").resize((W, H), Image.BILINEAR)
    draw = ImageDraw.Draw(base)
    # Use x,y from first two columns
    pts = points[:, :2]
    if normalized:
        pts_px = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1)
    else:
        pts_px = pts
    for (x, y) in pts_px:
        r = 1.5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)
    debug(f"Saved overlay: {out_path}")


def draw_points_on_image(
    image: Image.Image,
    points: np.ndarray,
    normalized: bool,
    point_size: int = 3,
    color=(0, 255, 0),
):
    W, H = image.size
    draw = ImageDraw.Draw(image)
    pts = points[:, :2]
    if normalized:
        pts_px = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1)
    else:
        pts_px = pts
    for (x, y) in pts_px:
        r = float(point_size)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return image


def _connections_68() -> list[tuple[int, int]]:
    # Standard iBUG 68 connections (jaw, brows, nose, eyes, outer/inner lips)
    conns = []
    # Jaw 0-16
    conns += [(i, i + 1) for i in range(0, 16)]
    # Left brow 17-21
    conns += [(i, i + 1) for i in range(17, 21)]
    # Right brow 22-26
    conns += [(i, i + 1) for i in range(22, 26)]
    # Nose bridge 27-30
    conns += [(i, i + 1) for i in range(27, 30)]
    # Nose bottom 31-35
    conns += [(i, i + 1) for i in range(31, 35)]
    # Left eye 36-41 + loop
    conns += [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
    # Right eye 42-47 + loop
    conns += [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
    # Outer lip 48-59 + loop
    conns += [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
    # Inner lip 60-67 + loop
    conns += [(i, i + 1) for i in range(60, 67)] + [(67, 60)]
    return conns


def draw_connections_on_image(
    image: Image.Image,
    points: np.ndarray,
    normalized: bool,
    connections: list[tuple[int, int]],
    line_width: int = 2,
    color=(0, 255, 0),
):
    W, H = image.size
    draw = ImageDraw.Draw(image)
    pts = points[:, :2]
    if normalized:
        pts_px = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1)
    else:
        pts_px = pts
    for a, b in connections:
        if a < len(pts_px) and b < len(pts_px):
            x1, y1 = pts_px[a]
            x2, y2 = pts_px[b]
            draw.line((x1, y1, x2, y2), fill=color, width=line_width)
    return image


def run_video(
    sess: ort.InferenceSession,
    inp_name: str,
    layout: str,
    H: int,
    W: int,
    C: int,
    video_path: str,
    out_dir: Path,
    normalize: str = "0_1",
    bgr_model: bool = False,
    every: int = 1,
    max_frames: Optional[int] = None,
    save_video: bool = True,
    keep_res: bool = True,
    dump_npy: bool = False,
    dump_csv: bool = False,
    decode_3dmm: bool = False,
    face_detect: bool = True,
    face_margin: float = 0.2,
    assume_points: Optional[int] = None,
    assume_norm: bool = False,
    point_size: int = 3,
    line_width: int = 2,
    draw_contours: bool = True,
    draw_mesh: bool = False,
    dump_landmarks_csv: bool = False,
    skip_if_no_face: bool = False,
    jump_threshold: float = 25.0,
):
    if cv2 is None:
        raise SystemExit("OpenCV not installed. Please install opencv-python.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    # Output video: optionally keep original resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = out_dir / (Path(video_path).stem + "_overlay.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (orig_W, orig_H) if keep_res else (W, H)
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, out_size) if save_video else None

    log_lines = ["frame,min,max,shape0"]
    raw_csv_path = out_dir / "logs" / (Path(video_path).stem + "_raw.csv")
    header_written = False
    lm_csv_path = out_dir / "logs" / (Path(video_path).stem + "_landmarks.csv")
    lm_header_written = False
    meta_path = out_dir / "logs" / (Path(video_path).stem + "_meta.csv")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as mf:
        mf.write('frame,status,reason,x0,y0,x1,y1\n')

    frame_idx = 0
    kept = 0
    prev_vec = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if every > 1 and (frame_idx % every) != 0:
                continue
            kept += 1
            if max_frames and kept > max_frames:
                break

            # Prepare input: either whole frame or face crop
            used_bbox = None
            src_for_model = frame
            if face_detect:
                fb = detect_face_bbox(frame)
                if fb is not None:
                    x0, y0, x1, y1 = crop_square_with_margin(fb, frame.shape[1], frame.shape[0], margin=face_margin)
                    used_bbox = (x0, y0, x1, y1)
                    src_for_model = frame[y0:y1, x0:x1]
                elif skip_if_no_face:
                    with open(meta_path, 'a') as mf:
                        mf.write(f"{frame_idx},no_face,detector_failed,-1,-1,-1,-1\n")
                    vis_img = Image.fromarray(frame[:, :, ::-1])
                    draw = ImageDraw.Draw(vis_img)
                    draw.rectangle((0, 0, 220, 24), fill=(0, 0, 0))
                    draw.text((5, 5), "No face detected", fill=(255, 0, 0))
                    if writer is not None:
                        writer.write(np.array(vis_img)[:, :, ::-1])
                    continue
            x = preprocess_from_array(src_for_model, H, W, C, layout, normalize=normalize, bgr_input=True, bgr=bgr_model)
            outputs = run_inference(sess, inp_name, x, None)
            arr0 = np.asarray(outputs[0])
            log_lines.append(f"{frame_idx},{float(arr0.min()):.6f},{float(arr0.max()):.6f},{int(arr0.shape[0]) if arr0.ndim>0 else 1}")
            if dump_npy:
                np.save(out_dir / f"frame_{frame_idx:06d}.npy", arr0)
            if dump_csv:
                flat = arr0.reshape(-1)
                if not header_written:
                    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
                    with open(raw_csv_path, 'w') as f:
                        f.write('frame,' + ','.join(f'p{i}' for i in range(flat.size)) + '\n')
                    header_written = True
                with open(raw_csv_path, 'a') as f:
                    f.write(str(frame_idx) + ',' + ','.join(f'{v:.6f}' for v in flat.tolist()) + '\n')

            # Stability check (vector jump)
            unstable = False
            flat_vec = arr0.reshape(-1)
            if prev_vec is not None and flat_vec.size == prev_vec.size:
                delta = float(np.linalg.norm(flat_vec - prev_vec))
                if delta > jump_threshold:
                    unstable = True
            prev_vec = flat_vec.copy()

            # Visualization attempts
            pts = None
            if decode_3dmm and int(arr0.reshape(-1).size) >= 265:
                pts = decode_3dmm_to_points(
                    arr0,
                    H,
                    W,
                    orig_H if keep_res else H,
                    orig_W if keep_res else W,
                    used_bbox if keep_res else (0, 0, W - 1, H - 1) if used_bbox is None else (0, 0, W - 1, H - 1),
                )
            if pts is None:
                pts = pick_points_array(outputs, assume_points=assume_points)
            status = 'ok'
            reason = ''
            if face_detect and used_bbox is None:
                status, reason = 'warn', 'no_bbox'
            if unstable:
                status, reason = 'unstable', 'jump_threshold'
                pts = None
            bx0 = used_bbox[0] if used_bbox else -1
            by0 = used_bbox[1] if used_bbox else -1
            bx1 = used_bbox[2] if used_bbox else -1
            by1 = used_bbox[3] if used_bbox else -1
            with open(meta_path, 'a') as mf:
                mf.write(f"{frame_idx},{status},{reason},{bx0},{by0},{bx1},{by1}\n")
            # Optionally dump 68 landmark pixels per frame
            if dump_landmarks_csv and pts is not None and pts.size >= 4:
                pts_arr = np.asarray(pts)
                # if not decode path, convert to pixel coords for full frame
                if not decode_3dmm:
                    if assume_norm or is_normalized(pts_arr):
                        pts_arr = np.stack([pts_arr[:, 0] * orig_W, pts_arr[:, 1] * orig_H], axis=1)
                    else:
                        sx, sy = orig_W / float(W), orig_H / float(H)
                        pts_arr = np.stack([pts_arr[:, 0] * sx, pts_arr[:, 1] * sy], axis=1)
                # ensure header
                if not lm_header_written:
                    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
                    n = min(68, pts_arr.shape[0])
                    cols = []
                    for i in range(n):
                        cols += [f"x{i}", f"y{i}"]
                    with open(lm_csv_path, 'w') as f:
                        f.write('frame,' + ','.join(cols) + '\n')
                    lm_header_written = True
                # write row (truncate/pad to 68)
                n = min(68, pts_arr.shape[0])
                row_vals = []
                for i in range(n):
                    row_vals += [f"{float(pts_arr[i,0]):.2f}", f"{float(pts_arr[i,1]):.2f}"]
                with open(lm_csv_path, 'a') as f:
                    f.write(str(frame_idx) + ',' + ','.join(row_vals) + '\n')
            # Base image for overlay: original or resized
            if keep_res:
                vis_img = Image.fromarray(frame[:, :, ::-1])  # original size, RGB
                draw = ImageDraw.Draw(vis_img)
                if pts is not None and pts.size >= 4 and status == 'ok':
                    # If 3DMM decoded, pts are already pixel coords in output resolution.
                    # Otherwise, convert normalized/model-space to output pixel coords.
                    pts2 = np.asarray(pts)
                    if decode_3dmm:
                        pts_px = pts2
                    else:
                        if assume_norm or is_normalized(pts2):
                            pts_px = np.stack([pts2[:, 0] * orig_W, pts2[:, 1] * orig_H], axis=1)
                        else:
                            sx, sy = orig_W / float(W), orig_H / float(H)
                            pts_px = np.stack([pts2[:, 0] * sx, pts2[:, 1] * sy], axis=1)
                    # Points
                    r = float(point_size)
                    for (x1, y1) in pts_px:
                        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(0, 255, 0))
                    # Contours
                    if draw_contours and pts_px.shape[0] >= 68:
                        for a, b in _connections_68():
                            if a < len(pts_px) and b < len(pts_px):
                                x1, y1 = pts_px[a]; x2, y2 = pts_px[b]
                                draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=line_width)
                    # Mesh via Delaunay
                    if draw_mesh:
                        try:
                            from scipy.spatial import Delaunay
                            tri = Delaunay(pts_px)
                            for (i, j, k) in tri.simplices:
                                for (u, v) in [(i, j), (j, k), (k, i)]:
                                    x1, y1 = pts_px[u]; x2, y2 = pts_px[v]
                                    draw.line((x1, y1, x2, y2), fill=(0, 200, 255), width=max(1, line_width-1))
                        except Exception:
                            pass
                else:
                    msg = (
                        "No face" if status.startswith('no_face') else (
                            "Unstable" if status == 'unstable' else (
                                "3DMM decoded: none" if decode_3dmm else (
                                    "3DMM params (1x265); see logs" if int(np.asarray(outputs[0]).reshape(-1).size) == 265 else "No landmarks inferred"
                                )
                            )
                        )
                    )
                    draw.rectangle((0, 0, 260, 24), fill=(0, 0, 0))
                    draw.text((5, 5), msg, fill=(0, 255, 0))
                vis_bgr = np.array(vis_img)[:, :, ::-1]
            else:
                rgb_resized = Image.fromarray(frame[:, :, ::-1]).resize((W, H), Image.BILINEAR)
                if pts is not None and pts.size >= 4 and status == 'ok':
                    if decode_3dmm:
                        # Already pixel coords for resized dims
                        vis = rgb_resized.copy()
                        draw = ImageDraw.Draw(vis)
                        r = float(point_size)
                        for (x1, y1) in pts:
                            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(0, 255, 0))
                        if draw_contours and len(pts) >= 68:
                            for a, b in _connections_68():
                                if a < len(pts) and b < len(pts):
                                    x1, y1 = pts[a]; x2, y2 = pts[b]
                                    draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=line_width)
                        if draw_mesh:
                            try:
                                from scipy.spatial import Delaunay
                                tri = Delaunay(pts)
                                for (i, j, k) in tri.simplices:
                                    for (u, v) in [(i, j), (j, k), (k, i)]:
                                        x1, y1 = pts[u]; x2, y2 = pts[v]
                                        draw.line((x1, y1, x2, y2), fill=(0, 200, 255), width=max(1, line_width-1))
                            except Exception:
                                pass
                    else:
                        norm = assume_norm or is_normalized(pts)
                        vis = draw_points_on_image(rgb_resized, pts, normalized=norm, point_size=point_size)
                        if draw_contours and len(pts) >= 68:
                            vis = draw_connections_on_image(vis, pts, normalized=norm, connections=_connections_68(), line_width=line_width)
                        if draw_mesh:
                            try:
                                from scipy.spatial import Delaunay
                                Wt, Ht = vis.size
                                pts_draw = pts
                                if norm:
                                    pts_draw = np.stack([pts[:, 0] * Wt, pts[:, 1] * Ht], axis=1)
                                tri = Delaunay(pts_draw)
                                draw2 = ImageDraw.Draw(vis)
                                for (i, j, k) in tri.simplices:
                                    for (u, v) in [(i, j), (j, k), (k, i)]:
                                        x1, y1 = pts_draw[u]; x2, y2 = pts_draw[v]
                                        draw2.line((x1, y1, x2, y2), fill=(0, 200, 255), width=max(1, line_width-1))
                            except Exception:
                                pass
                    vis_bgr = np.array(vis)[:, :, ::-1]
                else:
                    # No landmarks; overlay simple text indicating 3DMM params or generic message
                    vis = rgb_resized.copy()
                    draw = ImageDraw.Draw(vis)
                    msg = (
                        "No face" if status.startswith('no_face') else (
                            "Unstable" if status == 'unstable' else (
                                "3DMM decoded: none" if decode_3dmm else (
                                    "3DMM params (1x265); see logs" if int(np.asarray(outputs[0]).reshape(-1).size) == 265 else "No landmarks inferred"
                                )
                            )
                        )
                    )
                    draw.rectangle((0, 0, W, 24), fill=(0, 0, 0))
                    draw.text((5, 5), msg, fill=(0, 255, 0))
                    vis_bgr = np.array(vis)[:, :, ::-1]

            if writer is not None:
                writer.write(vis_bgr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Write log
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / (Path(video_path).stem + "_summary.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    debug(f"Saved log: {log_path}")
    if save_video:
        debug(f"Saved video: {out_video_path}")


def main():
    ap = argparse.ArgumentParser(description="Run ONNX facial landmark model on an image.")
    ap.add_argument("--model", default="model.onnx", help="Path to model.onnx")
    ap.add_argument("--image", help="Path to input image")
    ap.add_argument("--video", help="Path to input video (.mov, .mp4)")
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame (video)")
    ap.add_argument("--max-frames", type=int, default=None, help="Max frames to process (video)")
    ap.add_argument("--save-video", action="store_true", help="Save annotated video to output dir")
    ap.add_argument("--output", default="out", help="Output directory for overlays/dumps")
    ap.add_argument("--clean", action="store_true", help="Delete output directory before running")
    ap.add_argument("--normalize", default="0_1", choices=["none", "0_1", "-1_1", "imagenet"], help="Input normalization")
    ap.add_argument("--bgr", action="store_true", help="Use BGR channel order for input")
    ap.add_argument("--input-name", default=None, help="Override input tensor name")
    ap.add_argument("--output-name", default=None, help="Comma-separated output tensor names to fetch")
    ap.add_argument("--assume-points", type=int, default=None, help="Assume output contains N landmark points")
    ap.add_argument("--assume-norm", action="store_true", help="Assume landmark points are normalized [0..1]")
    ap.add_argument("--dump-npy", action="store_true", help="Dump raw outputs as .npy files")
    ap.add_argument("--dump-csv", action="store_true", help="Append raw output vector(s) to CSV under out/logs")
    ap.add_argument(
        "--decode-3dmm",
        action="store_true",
        help="Decode 1x265 FaceMap 3DMM params to 68 landmarks (requires qai-hub-models)",
    )
    ap.add_argument("--face-detect", action="store_true", help="Detect face and crop ROI before inference (improves 3DMM decode)")
    ap.add_argument("--face-margin", type=float, default=0.2, help="Extra margin around detected face (fraction)")
    ap.add_argument("--point-size", type=int, default=3, help="Landmark point radius in px")
    ap.add_argument("--line-width", type=int, default=2, help="Connection/mesh line width")
    ap.add_argument("--no-contours", action="store_true", help="Disable 68-pt contour lines")
    ap.add_argument("--mesh", action="store_true", help="Draw Delaunay mesh over landmarks")
    ap.add_argument("--dump-landmarks-csv", action="store_true", help="Write per-frame 68 landmark pixel coords to CSV (with --decode-3dmm)")
    ap.add_argument("--skip-if-no-face", action="store_true", help="Skip inference and overlay when no face is detected")
    ap.add_argument("--jump-threshold", type=float, default=25.0, help="L2 threshold on raw param vector delta to mark frame unstable")
    ap.add_argument("--inspect", action="store_true", help="Print model I/O and exit")
    args = ap.parse_args()

    if args.inspect:
        sess = load_session(args.model)
        io_info(sess)
        return

    if not args.image and not args.video:
        raise SystemExit("--image or --video is required unless --inspect is used")

    sess = load_session(args.model)
    inputs = sess.get_inputs()
    if not inputs:
        raise SystemExit("Model has no inputs")
    inp = None
    if args.input_name:
        for t in inputs:
            if t.name == args.input_name:
                inp = t
                break
        if inp is None:
            raise SystemExit(f"Input name {args.input_name} not found. Use --inspect to list.")
    else:
        inp = inputs[0]

    layout, H, W, C = determine_layout(inp.shape)
    debug(f"Inferred layout={layout} H={H} W={W} C={C}")

    out_dir = Path(args.output)
    if args.clean and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        debug(f"Running on video: {args.video}")
        run_video(
            sess,
            inp.name,
            layout,
            H,
            W,
            C,
            args.video,
            out_dir,
            normalize=args.normalize,
            bgr_model=args.bgr,
            every=args.every,
            max_frames=args.max_frames,
            save_video=args.save_video,
            keep_res=True,
            dump_npy=args.dump_npy,
            dump_csv=args.dump_csv,
            decode_3dmm=args.decode_3dmm,
            face_detect=args.face_detect,
            face_margin=args.face_margin,
            assume_points=args.assume_points,
            assume_norm=args.assume_norm,
            point_size=args.point_size,
            line_width=args.line_width,
            draw_contours=not args.no_contours,
            draw_mesh=args.mesh,
            dump_landmarks_csv=args.dump_landmarks_csv,
            skip_if_no_face=getattr(args, 'skip_if_no_face', False),
            jump_threshold=args.jump_threshold,
        )
        print("Video processing complete. See output dir for results.")
        return
    else:
        x = preprocess_image(args.image, H, W, C, layout, normalize=args.normalize, bgr=args.bgr)
        out_names = None
        if args.output_name:
            out_names = [n.strip() for n in args.output_name.split(",") if n.strip()]
        outputs = run_inference(sess, inp.name, x, out_names)

        print("=== Outputs ===")
        print(summarize_outputs(outputs))

        if args.dump_npy:
            for i, o in enumerate(outputs):
                np.save(out_dir / f"output_{i}.npy", np.asarray(o))

        # Try to visualize
        pts = None
        if args.decode_3dmm and int(np.asarray(outputs[0]).reshape(-1).size) >= 265:
            pts = decode_3dmm_to_points(np.asarray(outputs[0]), H, W, H, W)
        if pts is None:
            pts = pick_points_array(outputs, assume_points=args.assume_points)
        if pts is not None and pts.size >= 4:
            overlay_path = out_dir / (Path(args.image).stem + "_overlay.jpg")
            if args.decode_3dmm:
                # pts already in pixel coords for resized dims
                base = Image.open(args.image).convert("RGB").resize((W, H), Image.BILINEAR)
                draw = ImageDraw.Draw(base)
                for (x1, y1) in pts:
                    r = 1.5
                    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(0, 255, 0))
                base.save(overlay_path)
                debug(f"Saved overlay: {overlay_path}")
            else:
                norm = args.assume_norm or is_normalized(pts)
                draw_points(args.image, pts, str(overlay_path), (H, W), normalized=norm)
            print(f"Overlay saved: {overlay_path}")
        else:
            print("Could not infer landmark points from outputs.")
            flat_sizes = sorted({int(np.asarray(o).reshape(-1).size) for o in outputs})
            if 265 in flat_sizes:
                print("Hint: Output appears to be 3DMM parameters (e.g., 1x265).")
                if HAS_QCOM_3DMM:
                    print("Try adding --decode-3dmm to enable landmark decoding.")
                else:
                    print("To visualize landmarks/mesh, install Qualcomm's post-processing:")
                    print('  pip install "qai-hub-models[facemap-3dmm]"')
                    print("Then try their demo:")
                    print("  python -m qai_hub_models.models.facemap_3dmm.demo")
            else:
                print("Use --assume-points or --output-name to guide landmark extraction.")


if __name__ == "__main__":
    main()
def detect_face_bbox(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    try:
        cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # pick largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return (int(x), int(y), int(x + w), int(y + h))
    except Exception:
        return None


def crop_square_with_margin(bbox: Tuple[int, int, int, int], img_w: int, img_h: int, margin: float = 0.2) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    size = max(x1 - x0, y1 - y0)
    size = int(size * (1.0 + margin))
    half = size // 2
    nx0 = max(0, int(cx - half))
    ny0 = max(0, int(cy - half))
    nx1 = min(img_w - 1, int(cx + half))
    ny1 = min(img_h - 1, int(cy + half))
    # adjust to square exactly
    w = nx1 - nx0
    h = ny1 - ny0
    if w != h:
        if w > h:
            d = w - h
            ny1 = min(img_h - 1, ny1 + d)
        else:
            d = h - w
            nx1 = min(img_w - 1, nx1 + d)
    return (nx0, ny0, nx1, ny1)
