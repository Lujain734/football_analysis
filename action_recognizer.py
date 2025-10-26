# action_recognizer.py
import os, re, cv2, numpy as np, torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics  # print version

# --- Limit Torch threads to reduce RAM/CPU thrash ---
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ===== Defaults =====
DEFAULT_CLASS_NAMES = ["dribble", "pass", "shoot"]

# ===== 34D -> num_classes MLP =====
class ActionMLP(nn.Module):
    def __init__(self, in_dim=34, hid=128, num_classes=3, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ===== Helpers =====
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _extract_index_from_name(fn: str):
    m = re.findall(r'\d+', fn)
    return int(m[0]) if m else None

def _smooth_probs(arr: np.ndarray, win: int):
    if win <= 1 or len(arr) == 0:
        return arr
    out = np.zeros_like(arr)
    n = len(arr)
    half = win // 2
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        out[i] = np.mean(arr[s:e], axis=0)
    return out

def _feat_from_det(kpts_xyn: np.ndarray, box_xyxy: np.ndarray, H: int, W: int) -> np.ndarray:
    """17 keypoints xyn -> 34D normalized inside player's box."""
    k = np.asarray(kpts_xyn, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] != 17:
        return np.zeros((34,), np.float32)

    if k.shape[1] == 2:
        xy, v = k, np.ones((17, 1), np.float32)
    else:
        xy, v = k[:, :2], k[:, 2:3]

    xy[:, 0] *= W
    xy[:, 1] *= H

    x1, y1, x2, y2 = map(float, box_xyxy)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx, cy = x1 + w / 2.0, y1 + h / 2.0

    nx = (xy[:, 0] - cx) / w
    ny = (xy[:, 1] - cy) / h

    feat = np.stack([nx, ny], axis=1) * v
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.reshape(-1).astype(np.float32)

# ===== Safe YOLO call wrapper =====
def yolo_infer(model: YOLO, img, conf: float, imgsz: int, max_det: int):
    """
    Unified YOLO inference that works with ultralytics 8.0.200+
    Handles both direct call and predict() methods
    """
    try:
        # Method 1: Direct call (most common)
        results = model(img, conf=conf, imgsz=imgsz, verbose=False, max_det=max_det)
        return results
    except AttributeError as e1:
        try:
            # Method 2: Using predict() 
            results = model.predict(img, conf=conf, imgsz=imgsz, save=False, verbose=False, max_det=max_det)
            return results
        except Exception as e2:
            try:
                # Method 3: Minimal parameters (fallback for older versions)
                print(f"⚠️ YOLO inference using fallback method")
                results = model(img, conf=conf, verbose=False)
                return results
            except Exception as e3:
                # Method 4: Absolute minimal (last resort)
                print(f"❌ All YOLO methods failed: {e1}, {e2}, {e3}")
                print(f"⚠️ Using absolute minimal inference")
                results = model(img)
                return results

# ===== Main API: return counts only =====
def infer_action_counts(
    crops_dir: str,
    pose_weights: str,
    mlp_weights: str,
    class_names=None,
    conf_threshold: float = 0.75,
    yolo_conf: float = 0.25,
    img_size: int = 736,
    max_det: int = 5,
    smooth_window: int = 7,
    min_seg_sec: float = 0.30,
    fps: float = 25.0
):
    print(f"[infer_action_counts] ultralytics version: {ultralytics.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = class_names if class_names else DEFAULT_CLASS_NAMES

    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"Missing crops_dir: {crops_dir}")
    if not os.path.isfile(pose_weights):
        raise FileNotFoundError(f"Missing pose weights: {pose_weights}")
    if not os.path.isfile(mlp_weights):
        raise FileNotFoundError(f"Missing MLP weights: {mlp_weights}")

    # Load YOLO Pose model
    print(f"📥 Loading YOLO Pose model from: {pose_weights}")
    yolo_pose = YOLO(pose_weights)
    
    # Load MLP model
    print(f"📥 Loading MLP model from: {mlp_weights}")
    mlp = ActionMLP(num_classes=len(classes)).to(device)
    sd = torch.load(mlp_weights, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mlp.load_state_dict(sd, strict=False)
    mlp.eval()

    records = []
    image_paths = sorted(
        [os.path.join(crops_dir, f) for f in os.listdir(crops_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))],
        key=_natural_key
    )

    print(f"🔍 Found {len(image_paths)} crop images")
    
    for idx, img_path in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"⏳ Processing crop {idx+1}/{len(image_paths)}...")
            
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Run YOLO Pose inference
        results = yolo_infer(yolo_pose, img, conf=yolo_conf, imgsz=img_size, max_det=max_det)
        r = results[0]

        # Check if we have valid detections
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        kpts  = r.keypoints.xyn.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        det_idx = int(np.argmax(confs))
        feat = _feat_from_det(kpts[det_idx], boxes[det_idx], img.shape[0], img.shape[1])

        with torch.no_grad():
            x = torch.from_numpy(feat[None, :]).to(device)
            logits = mlp(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        max_conf = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        pred_name = classes[pred_idx] if max_conf >= conf_threshold else "no_action"

        frame_idx = _extract_index_from_name(os.path.basename(img_path))
        frame_idx = frame_idx if frame_idx is not None else 0

        records.append((frame_idx, probs, pred_name))

    if not records:
        print("⚠️ No valid records found")
        return {c: 0 for c in classes}

    records.sort(key=lambda t: t[0])
    valid = [(fi, p) for (fi, p, name) in records if name != "no_action"]
    
    if not valid:
        print("⚠️ No actions detected (all no_action)")
        return {c: 0 for c in classes}

    print(f"✅ Found {len(valid)} valid action frames")

    frame_idx_arr = np.array([fi for fi, _ in valid], dtype=np.int32)
    probs_arr     = np.stack([p for _, p in valid], axis=0)

    probs_smooth = _smooth_probs(probs_arr, win=smooth_window)
    smooth_idx   = np.argmax(probs_smooth, axis=1)
    smooth_labels = [classes[i] for i in smooth_idx]

    segments = []
    cur_label = smooth_labels[0]
    cur_start = int(frame_idx_arr[0])

    for i in range(1, len(smooth_labels)):
        lab = smooth_labels[i]
        idx = int(frame_idx_arr[i])
        if lab != cur_label:
            cur_end = int(frame_idx_arr[i - 1])
            dur = (cur_end - cur_start + 1) / float(fps)
            if dur >= min_seg_sec:
                segments.append((cur_label, cur_start, cur_end, round(dur, 2)))
            cur_label = lab
            cur_start = idx

    cur_end = int(frame_idx_arr[-1])
    dur = (cur_end - cur_start + 1) / float(fps)
    if dur >= min_seg_sec:
        segments.append((cur_label, cur_start, cur_end, round(dur, 2)))

    counts = {c: 0 for c in classes}
    for (lab, _, _, _) in segments:
        if lab in counts:
            counts[lab] += 1

    print(f"📊 Action counts: {counts}")
    return counts
