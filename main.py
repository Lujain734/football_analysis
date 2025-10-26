#!/usr/bin/env python3
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
    parser.add_argument("--iou-thr", type=float, default=0.3,  # ✅ Changed from 0.03 to 0.3
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
def seed_target_from_first_k_frames(tracks_players, target_xy, K=10, max_distance=150):
    """
    Find the player closest to target coordinates.
    
    Args:
        tracks_players: Player tracking data
        target_xy: Target coordinates (x, y)
        K: Number of frames to search
        max_distance: Maximum distance threshold (pixels) - ignore players further than this
    """
    if target_xy is None:
        return None, None, None
    
    print(f"🔍 DEBUG: Searching for player near coordinates: ({target_xy[0]:.2f}, {target_xy[1]:.2f})")
    print(f"🔍 DEBUG: Maximum search distance: {max_distance} pixels")
    
    best = (None, None, None)
    best_dist = float("inf")
    max_f = min(K, len(tracks_players))
    
    print(f"🔍 DEBUG: Checking first {max_f} frames")
    
    candidates = []  # Store all candidates with their distances
    
    for f_idx in range(max_f):
        print(f"\n📍 Frame {f_idx}: Found {len(tracks_players[f_idx])} players")
        
        for pid, tr in tracks_players[f_idx].items():
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = np.hypot(cx - target_xy[0], cy - target_xy[1])
            
            # Log each player
            print(f"  Player ID {pid}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], center=({cx:.1f}, {cy:.1f}), distance={d:.1f}")
            
            # Store candidate if within max distance
            if d <= max_distance:
                candidates.append((f_idx, pid, tr["bbox"], d))
                
                if d < best_dist:
                    best_dist = d
                    best = (f_idx, pid, tr["bbox"])
                    print(f"  ✅ NEW BEST: ID {pid}, distance={d:.1f}")
            else:
                print(f"  ❌ TOO FAR: Distance {d:.1f} > {max_distance}")
    
    if best[1] is not None:
        print(f"\n🎯 FINAL SELECTION: Frame {best[0]}, Player ID {best[1]}, Distance={best_dist:.1f}, BBox={best[2]}")
        print(f"📊 Total candidates found: {len(candidates)}")
    else:
        print(f"\n❌ NO PLAYER FOUND within {max_distance} pixels")
        if candidates:
            print(f"⚠️ Closest player was at distance {min(c[3] for c in candidates):.1f} pixels")
        else:
            print(f"⚠️ No players detected in the first {max_f} frames")
    
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
    max_w = 960
    
    if first_w > max_w:
        scaled = []
        scale = max_w / float(first_w)
        print(f"ℹ️ Downscaling frames from {first_w}px width to {max_w}px (scale: {scale:.2f})")
        
        for fr in video_frames:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            scaled.append(fr)
        
        video_frames = scaled
    
    H, W = video_frames[0].shape[:2]
    print(f"ℹ️ Video dimensions: {W}x{H}, Frames: {len(video_frames)}")
    
    # FPS
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    try:
        fps = float(fps)
        if fps <= 0 or np.isnan(fps):
            print("⚠️ Could not get FPS, defaulting to 30.")
            fps = 30.0
    except Exception as e:
        print(f"⚠️ Error getting FPS ({e}), defaulting to 30.")
        fps = 30.0
    
    cap.release()
    print(f"ℹ️ Video FPS: {fps:.2f}")
    
    # Tracking
    print("⏳ Running object tracking...")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)
    print("✅ Tracking complete.")
    
    # Seed target
    target_id = None
    last_bbox = None
    target_xy = tuple(args.target_xy) if args.target_xy else None
    
    print(f"ℹ️ Trying to seed target using coords: {target_xy}")
    
    # 🔍 DEBUG: Draw target point on first frame
    if target_xy and len(video_frames) > 0:
        debug_frame = video_frames[0].copy()
        tx, ty = int(target_xy[0]), int(target_xy[1])
        
        # Draw large red circle at target point
        cv2.circle(debug_frame, (tx, ty), 20, (0, 0, 255), -1)
        cv2.circle(debug_frame, (tx, ty), 22, (255, 255, 255), 2)
        
        # Draw all player bounding boxes in first frame
        if len(tracks["players"]) > 0:
            for pid, pdata in tracks["players"][0].items():
                x1, y1, x2, y2 = map(int, pdata["bbox"])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw player ID
                cv2.putText(debug_frame, f"ID:{pid}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(debug_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # Save debug frame
        debug_path = os.path.join("debug_output", "coordinate_debug_frame0.jpg")
        os.makedirs("debug_output", exist_ok=True)
        cv2.imwrite(debug_path, debug_frame)
        print(f"🖼️ DEBUG: Saved coordinate visualization to: {debug_path}")
        print(f"   Red circle = Target point ({tx}, {ty})")
        print(f"   Green boxes = Detected players")
        print(f"   Blue dots = Player centers")
        print("")
    
    if target_xy:
        f0, pid0, bbox0 = seed_target_from_first_k_frames(tracks["players"], target_xy, K=10)
        
        if pid0 is not None:
            target_id = pid0
            last_bbox = bbox0
            print(f"🎯 Target player ID: {target_id} (seeded from frame {f0} using bbox {bbox0})")
            print(f"✅✅✅ INITIAL TARGET ID SEEDED AS: {target_id} ✅✅✅")
        else:
            print("❌ No player found near the provided --target-xy in the first 10 frames.")
            print(f"❌❌❌ FAILED TO SEED INITIAL TARGET ID FROM COORDS {target_xy} ❌❌❌")
    else:
        print("⚠️ No --target-xy provided, cannot track a specific player.")
    
    # Camera motion
    print("⏳ Estimating camera movement...")
    cam = CameraMovementEstimator(video_frames[0])
    cam_movement = cam.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    cam.add_adjust_positions_to_tracks(tracks, cam_movement)
    print("✅ Camera movement estimated.")
    
    # View transformation
    print("⏳ Applying view transformation...")
    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)
    print("✅ View transformation applied.")
    
    # Ball interpolation
    print("⏳ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("✅ Ball positions interpolated.")
    
    # Team assignment
    print("⏳ Assigning team colors...")
    ta = TeamAssigner()
    
    first_player_frame_idx = -1
    for idx, frame_players in enumerate(tracks["players"]):
        if frame_players:
            first_player_frame_idx = idx
            break
    
    if first_player_frame_idx != -1:
        ta.assign_team_color(video_frames[first_player_frame_idx], tracks["players"][first_player_frame_idx])
        print(f"✅ Assigned team colors: Team 1 ~{ta.team_colors.get(1)}, Team 2 ~{ta.team_colors.get(2)}")
        print(f"✅ Initial team colors assigned using frame {first_player_frame_idx}")
        
        print("⏳ Assigning teams to players per frame...")
        for f_idx, player_track in enumerate(tracks["players"]):
            for pid, tr in player_track.items():
                team = ta.get_player_team(video_frames[f_idx], tr["bbox"], pid)
                tracks["players"][f_idx][pid]["team"] = team
                tracks["players"][f_idx][pid]["team_color"] = ta.team_colors.get(team, (255, 255, 255))
        print("✅ Teams assigned per frame.")
    else:
        print("⚠️ No players detected in any frame, cannot assign team colors.")
    
    # Ball possession
    print("⏳ Assigning ball possession...")
    pba = PlayerBallAssigner()
    team_ball_control = []
    
    for f_idx, player_track in enumerate(tracks["players"]):
        ball_dict = tracks["ball"][f_idx] if f_idx < len(tracks["ball"]) else {}
        ball_bbox = ball_dict.get(1, {}).get("bbox", None)
        assigned_player_id = -1
        
        if ball_bbox is not None:
            assigned_player_id = pba.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player_id != -1 and assigned_player_id in tracks["players"][f_idx]:
            tracks["players"][f_idx][assigned_player_id]["has_ball"] = True
            team_ball_control.append(tracks["players"][f_idx][assigned_player_id]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    
    print("✅ Ball possession assigned.")
    
    # Crop target player
    if target_id is not None:
        os.makedirs(args.crop_dir, exist_ok=True)
        print(f"🗂️ Cropping target player (ID: {target_id}) to: {os.path.abspath(args.crop_dir)}")
        
        IOU_THR = float(args.iou_thr)
        print(f"ℹ️ Using IoU threshold for fallback: {IOU_THR}")
        
        frames_cropped = 0
        
        for f_idx, frame in enumerate(video_frames):
            target_bbox = None
            
            # Direct detection
            if f_idx < len(tracks["players"]) and target_id in tracks["players"][f_idx]:
                target_bbox = tracks["players"][f_idx][target_id]["bbox"]
                last_bbox = target_bbox
            # IoU fallback
            elif last_bbox is not None and f_idx < len(tracks["players"]):
                best_iou = 0.0
                best_bbox_candidate = None
                
                for current_pid, pdata in tracks["players"][f_idx].items():
                    b = pdata["bbox"]
                    i = calculate_iou(last_bbox, b)
                    
                    if i > best_iou:
                        best_iou = i
                        best_bbox_candidate = b
                
                if best_iou > IOU_THR:
                    target_bbox = best_bbox_candidate
                    last_bbox = target_bbox
            
            # Crop if bbox found
            if target_bbox:
                try:
                    ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
                    
                    if ex1 < ex2 and ey1 < ey2:
                        crop = frame[ey1:ey2, ex1:ex2]
                        
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
                            crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg")
                            ok = cv2.imwrite(crop_path, crop_resized)
                            
                            if ok:
                                frames_cropped += 1
                except Exception as e_crop:
                    print(f"⚠️ Error cropping frame {f_idx}: {e_crop}")
        
        print(f"✅ Cropped target player in {frames_cropped} frames.")
    else:
        print("ℹ️ No target player ID was set, skipping cropping.")
    
    # Action counting
    if target_id is not None and os.path.exists(args.crop_dir):
        print("⏳ Performing action recognition on crops...")
        
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
                fps=fps
            )
            
            print("✅ Action recognition complete!")
            print("")
            print("--- FINAL ACTION COUNTS ---")
            for cls in DEFAULT_CLASS_NAMES:
                count_val = counts.get(cls, 0)
                print(f"{cls} = {count_val}")
        
        except Exception as e_action:
            traceback.print_exc()
            print(f"❌ Action counting failed: {e_action}")
            print("")
            print("--- ACTION COUNTING ERROR ---")
            for cls in DEFAULT_CLASS_NAMES:
                print(f"{cls} = 0")
    
    elif target_id is None:
        print("ℹ️ No target ID set, skipping action recognition.")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")
    else:
        print(f"⚠️ Crop directory '{args.crop_dir}' not found or empty, skipping action recognition.")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")

if __name__ == "__main__":
    try:
        main()
    except Exception as e_main:
        print(f"❌ CRITICAL ERROR in main function: {e_main}")
        traceback.print_exc()
        print("")
        print("--- MAIN FUNCTION ERROR ---")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")
