# main.py
import os
import cv2
import numpy as np
import argparse
import traceback

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

# Action counting
from action_recognizer import infer_action_counts, DEFAULT_CLASS_NAMES

# ------------------------ CLI ------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Analysis with Target Player Tracking + Cropping + Action Counting"
    )
    parser.add_argument("--video-path", type=str, default=r"input_videos/header.mp4",
                        help="Path to input video")
    parser.add_argument("--target-xy", type=float, nargs=2,
                        help="Target player pixel coords (x y) in early frames")
    parser.add_argument("--crop-dir", type=str, default="target_crops",
                        help="Directory to save cropped target images")
    parser.add_argument("--zoom", type=float, default=1.3,
                        help="Zoom factor for target crop")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Output crop size (square)")
    parser.add_argument("--iou-thr", type=float, default=0.03,
                        help="IoU threshold for ID fallback continuity")

    # Pose/MLP weights + counting params
    parser.add_argument("--pose-weights", type=str, default=r"models/best1.pt",
                        help="Path to YOLOv8 Pose weights")
    parser.add_argument("--mlp-weights", type=str, default=r"models/action_mlp.pt",
                        help="Path to 34D MLP weights")
    parser.add_argument("--conf-thr", type=float, default=0.75,
                        help="Classification confidence threshold for accepting an action")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO pose confidence threshold")
    parser.add_argument("--img-size", type=int, default=736,
                        help="YOLO pose inference image size")
    parser.add_argument("--max-det", type=int, default=5,
                        help="Max detections per image for pose")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Smoothing window length over probabilities")
    parser.add_argument("--min-seg-sec", type=float, default=0.30,
                        help="Minimum duration (sec) for an action segment to be counted")
    return parser.parse_args()

# --------------------- Utilities ---------------------
def seed_target_from_first_k_frames(tracks_players, target_xy, K=10):
    if target_xy is None:
        return None, None, None
    best = (None, None, None)
    best_dist = float("inf")
    max_f = min(K, len(tracks_players))
    for f_idx in range(max_f):
        for pid, tr in tracks_players[f_idx].items():
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = np.hypot(cx - target_xy[0], cy - target_xy[1])
            if d < best_dist:
                best_dist = d
                best = (f_idx, pid, tr["bbox"])
    return best

def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def enlarge_bbox(bbox, zoom=2.7, W=None, H=None, pad_px=40):
    if W is None or H is None:
        raise ValueError("W and H must be provided")
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = w * zoom + 2 * pad_px
    nh = h * zoom + 2 * pad_px
    nx1 = int(clamp(cx - nw / 2, 0, W - 1))
    ny1 = int(clamp(cy - nh / 2, 0, H - 1))
    nx2 = int(clamp(cx + nw / 2, 0, W - 1))
    ny2 = int(clamp(cy + nh / 2, 0, H - 1))
    if nx2 <= nx1:
        nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H - 1, ny1 + 1)
    return [nx1, ny1, nx2, ny2]

# ------------------------ Main ------------------------
def main():
    args = parse_args()

    # Read video
    video_frames = read_video(args.video_path)
    if len(video_frames) == 0:
        print("❌ No frames read from video. Check --video-path.")
        return

    # Downscale frames to reduce RAM usage
    first_h, first_w = video_frames[0].shape[:2]
    max_w = 960  # reduce if memory is tight (e.g., 720)
    if first_w > max_w:
        scaled = []
        scale = max_w / float(first_w)
        for fr in video_frames:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            scaled.append(fr)
        video_frames = scaled
    H, W = video_frames[0].shape[:2]

    # FPS (fallback to 30 if unknown)
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
    except Exception:
        fps = 30.0
    cap.release()

    # Tracking players/ball
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)

    # Seed target from early frames (pixel coords)
    target_id = None
    last_bbox = None
    target_xy = tuple(args.target_xy) if args.target_xy else None
    if target_xy:
        f0, pid0, bbox0 = seed_target_from_first_k_frames(tracks["players"], target_xy, K=10)
        if pid0 is not None:
            target_id = pid0
            last_bbox = bbox0
            print(f"🎯 Target player ID: {target_id} (seeded from frame {f0})")
        else:
            print("⚠️ No player near the provided --target-xy in first frames.")

    # Camera motion + view transform + team colors + ball assignment
    cam = CameraMovementEstimator(video_frames[0])
    cam_movement = cam.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    cam.add_adjust_positions_to_tracks(tracks, cam_movement)

    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    ta = TeamAssigner()
    ta.assign_team_color(video_frames[0], tracks["players"][0])

    for f_idx, player_track in enumerate(tracks["players"]):
        for pid, tr in player_track.items():
            team = ta.get_player_team(video_frames[f_idx], tr["bbox"], pid)
            tracks["players"][f_idx][pid]["team"] = team
            tracks["players"][f_idx][pid]["team_color"] = ta.team_colors[team]

    pba = PlayerBallAssigner()
    team_ball_control = []
    for f_idx, player_track in enumerate(tracks["players"]):
        ball_dict = tracks["ball"][f_idx] if f_idx < len(tracks["ball"]) else {}
        ball_bbox = ball_dict.get(1, {}).get("bbox", None)
        assigned = -1
        if ball_bbox is not None:
            assigned = pba.assign_ball_to_player(player_track, ball_bbox)

        if assigned != -1:
            tracks["players"][f_idx][assigned]["has_ball"] = True
            team_ball_control.append(tracks["players"][f_idx][assigned]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) else 0)

    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control, target_id=target_id
    )

    # ------- Crop target player per frame + save to correct dir -------
    os.makedirs(args.crop_dir, exist_ok=True)
    print("🗂️ Crops will be saved to:", os.path.abspath(args.crop_dir))

    IOU_THR = float(args.iou_thr)
    for f_idx, frame in enumerate(video_frames):
        target_bbox = None
        if target_id is not None and target_id in tracks["players"][f_idx]:
            target_bbox = tracks["players"][f_idx][target_id]["bbox"]
            last_bbox = target_bbox
        elif last_bbox is not None:
            best_iou, best_bbox = 0.0, None
            for _, pdata in tracks["players"][f_idx].items():
                b = pdata["bbox"]
                i = calculate_iou(last_bbox, b)
                if i > best_iou:
                    best_iou, best_bbox = i, b
            if best_iou > IOU_THR:
                target_bbox = best_bbox
                last_bbox = target_bbox

        if target_bbox:
            ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
                crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg")
                ok = cv2.imwrite(crop_path, crop_resized)
                if not ok:
                    print("⚠️ imwrite failed at:", crop_path)

            # highlight target (visual only)
            x1, y1, x2, y2 = map(int, target_bbox)
            cv2.rectangle(output_video_frames[f_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_video_frames[f_idx], "TARGET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ------- Save annotated video -------
    os.makedirs("output_videos", exist_ok=True)
    save_video(output_video_frames, "output_videos/output_video.avi")
    print("✅ Saved video to output_videos/output_video.avi")

    # ------- Action counting from crops folder -------
    try:
        counts = infer_action_counts(
            crops_dir=args.crop_dir,
            pose_weights=args.pose_weights,
            mlp_weights=args.mlp_weights,
            class_names=DEFAULT_CLASS_NAMES,
            conf_threshold=args.conf_thr,
            yolo_conf=args.yolo_conf,
            img_size=args.img_size,
            max_det=args.max_det,
            smooth_window=args.smooth_window,
            min_seg_sec=args.min_seg_sec,
            fps=fps  # same as input video FPS
        )
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Action counting failed: {e}")
        return

    # Print counts in the required format (parsed by server.py)
    for cls in DEFAULT_CLASS_NAMES:
        print(f"{cls} = {counts.get(cls, 0)}")

if __name__ == "__main__":
    main()
