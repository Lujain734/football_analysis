#!/usr/bin/env python3
# main.py
import os
import cv2
import numpy as np
import argparse
import traceback
from utils import read_video, save_video
from trackers import Tracker # Ensure trackers.py contains the Tracker class
from team_assigner import TeamAssigner # Ensure team_assigner.py is correct
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
    parser.add_argument("--iou-thr", type=float, default=0.3, # Default value if not passed by server.py
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
# In main.py

def seed_target_from_first_k_frames(tracks_players, target_xy, K=10): # K is now ignored
    if target_xy is None:
        return None, None, None
    best = (None, None, None)
    best_dist = float("inf")
    # max_f = min(K, len(tracks_players)) # <<< OLD LINE
    max_f = 1 # <<< NEW LINE: Force check only the first frame (index 0)

    # Ensure tracks_players has at least one frame
    if len(tracks_players) == 0:
         print("⚠️ Warning: tracks_players is empty in seed_target function.")
         return None, None, None

    # Loop will only run for f_idx = 0
    for f_idx in range(max_f):
        # Ensure tracks_players[f_idx] is a dictionary
        if not isinstance(tracks_players[f_idx], dict):
             print(f"⚠️ Warning: tracks_players[{f_idx}] is not a dict: {tracks_players[f_idx]}")
             continue
        for pid, tr in tracks_players[f_idx].items():
            # Ensure 'bbox' key exists and is valid
            if 'bbox' not in tr or not isinstance(tr['bbox'], (list, tuple)) or len(tr['bbox']) != 4:
                # print(f"⚠️ Warning: Invalid bbox for pid {pid} in frame {f_idx}: {tr.get('bbox')}") # Optional Debug
                continue
            x1, y1, x2, y2 = tr["bbox"]
            # Ensure coordinates are numbers
            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                 # print(f"⚠️ Warning: Non-numeric bbox coords for pid {pid} in frame {f_idx}: {[x1, y1, x2, y2]}") # Optional Debug
                 continue
            # Ensure bbox has positive width/height
            if x1 >= x2 or y1 >= y2:
                # print(f"⚠️ Warning: Zero/negative area bbox for pid {pid} in frame {f_idx}: {[x1, y1, x2, y2]}") # Optional Debug
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = np.hypot(cx - target_xy[0], cy - target_xy[1])
            if d < best_dist:
                best_dist = d
                best = (f_idx, pid, tr["bbox"]) # f_idx will always be 0 here now

    # Add a check here: if no player was found in frame 0, return None
    if best == (None, None, None):
        print(f"⚠️ No suitable player found in frame 0 near coords {target_xy}")
        return None, None, None

    return best

def calculate_iou(a, b):
    # Ensure inputs are valid lists/tuples of 4 numbers
    if not isinstance(a, (list, tuple)) or len(a) != 4 or not all(isinstance(x, (int, float)) for x in a): return 0.0
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(isinstance(x, (int, float)) for x in b): return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Ensure coordinates are ordered correctly (handle potential tracker errors)
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter

    # Avoid division by zero
    return inter / denom if denom > 0 else 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def enlarge_bbox(bbox, zoom=1.3, W=None, H=None, pad_px=10): # Reduced default zoom/pad slightly
    if W is None or H is None:
        raise ValueError("W and H must be provided")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
         raise ValueError(f"Invalid bbox format for enlarge_bbox: {bbox}")

    x1, y1, x2, y2 = bbox
    # Ensure valid numbers before calculations
    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
         raise ValueError(f"Non-numeric coords in enlarge_bbox: {[x1, y1, x2, y2]}")

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0: # Handle zero-area bbox before calculating center
        print(f"⚠️ Warning: enlarge_bbox received zero-area bbox: {[x1, y1, x2, y2]}")
        # Return a minimal valid box at top-left or handle as error
        return [0, 0, 1, 1]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Increase size based on zoom and padding
    nw = w * zoom + 2 * pad_px
    nh = h * zoom + 2 * pad_px

    # Calculate new coords and clamp to image boundaries
    nx1 = int(clamp(cx - nw / 2, 0, W - 1))
    ny1 = int(clamp(cy - nh / 2, 0, H - 1))
    nx2 = int(clamp(cx + nw / 2, 0, W - 1))
    ny2 = int(clamp(cy + nh / 2, 0, H - 1))

    # Ensure box has at least 1 pixel width/height after clamping
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

    # Downscale frames (adjust max_w if needed)
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
        if fps <= 0 or np.isnan(fps): fps = 30.0
    except: fps = 30.0
    cap.release()
    print(f"ℹ️ Video FPS: {fps:.2f}")

    # Tracking
    print("⏳ Running object tracking...")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)
    print("✅ Tracking complete.")

    # Seed target
    target_id = None
    last_bbox = None
    target_xy = tuple(args.target_xy) if args.target_xy else None
    print(f"ℹ️ Trying to seed target using coords: {target_xy}")
    if target_xy:
        f0, pid0, bbox0 = seed_target_from_first_k_frames(tracks.get("players", []), target_xy, K=10) # Added .get()
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

    # --- Run other processing steps (Camera, ViewTransform, Teams, Ball) ---
    if "players" in tracks and tracks["players"]: # Check if tracks exist before proceeding
        print("⏳ Estimating camera movement...")
        cam = CameraMovementEstimator(video_frames[0])
        cam_movement = cam.get_camera_movement(video_frames, read_from_stub=False, stub_path="stubs/camera_movement_stub.pkl")
        cam.add_adjust_positions_to_tracks(tracks, cam_movement)
        print("✅ Camera movement estimated.")

        print("⏳ Applying view transformation...")
        vt = ViewTransformer()
        vt.add_transformed_position_to_tracks(tracks)
        print("✅ View transformation applied.")

        print("⏳ Interpolating ball positions...")
        if "ball" in tracks:
             tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
             print("✅ Ball positions interpolated.")
        else:
             print("⚠️ No ball tracks found to interpolate.")


        print("⏳ Assigning team colors...")
        ta = TeamAssigner()
        first_player_frame_idx = -1
        for idx, frame_players in enumerate(tracks["players"]):
            if frame_players:
                first_player_frame_idx = idx
                break

        if first_player_frame_idx != -1:
            ta.assign_team_color(video_frames[first_player_frame_idx], tracks["players"][first_player_frame_idx])
            # print(f"DEBUG: Assigned team colors: Team 1 ~{ta.team_colors.get(1)}, Team 2 ~{ta.team_colors.get(2)}")
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


        print("⏳ Assigning ball possession...")
        pba = PlayerBallAssigner()
        team_ball_control = []
        for f_idx, player_track in enumerate(tracks["players"]):
            ball_dict = tracks.get("ball", [])[f_idx] if f_idx < len(tracks.get("ball", [])) else {}
            ball_bbox = ball_dict.get(1, {}).get("bbox", None)
            assigned_player_id = -1
            if ball_bbox is not None:
                assigned_player_id = pba.assign_ball_to_player(player_track, ball_bbox)

            current_team = 0 # Default if no one has ball
            if assigned_player_id != -1 and assigned_player_id in tracks["players"][f_idx]:
                 tracks["players"][f_idx][assigned_player_id]["has_ball"] = True
                 current_team = tracks["players"][f_idx][assigned_player_id].get("team", 0)

            team_ball_control.append(current_team if current_team != 0 else (team_ball_control[-1] if team_ball_control else 0))
        print("✅ Ball possession assigned.")

    else:
        print("⚠️ No player tracks found, skipping dependent processing steps.")
        # Ensure default counts are printed later if processing stops here
        tracks = {"players": [], "referees": [], "ball": []} # Prevent crashes later

    # --- Cropping Loop with Enhanced Debugging ---
    if target_id is not None:
        os.makedirs(args.crop_dir, exist_ok=True)
        print(f"🗂️ Cropping target player (Original ID: {target_id}) to: {os.path.abspath(args.crop_dir)}")
        IOU_THR = float(args.iou_thr)
        print(f"ℹ️ Using IoU threshold for fallback: {IOU_THR}")
        frames_cropped = 0
        original_target_id = target_id # Store the initial ID

        for f_idx, frame in enumerate(video_frames):
            target_bbox = None
            log_prefix = f" F[{f_idx:04d}] Target({original_target_id}):" # Consistent prefix for logs
            current_frame_players = tracks.get("players", [])[f_idx] if f_idx < len(tracks.get("players", [])) else {}

            # STEP 1: Look for the original target ID
            if original_target_id in current_frame_players:
                target_bbox = current_frame_players[original_target_id].get("bbox")
                if target_bbox:
                    print(f"{log_prefix} Found original ID {original_target_id}.")
                    last_bbox = target_bbox # Update last known GOOD position
                else:
                    print(f"{log_prefix} Original ID {original_target_id} present but missing bbox!")
                    target_bbox = None # Treat as lost if bbox is invalid

            # STEP 2: Fallback if original ID lost and we have a previous location
            elif last_bbox is not None:
                print(f"{log_prefix} Original ID lost. Last known bbox: {last_bbox}. Searching fallback...")
                best_iou = 0.0
                best_pid_fallback = None
                best_bbox_candidate = None

                for current_pid, pdata in current_frame_players.items():
                    if current_pid == original_target_id: continue # Skip original if already checked
                    b = pdata.get("bbox")
                    if not b: continue # Skip if candidate has no bbox

                    try: # Add try-except around calculate_iou
                         i = calculate_iou(last_bbox, b)
                    except Exception as e_iou:
                         print(f"{log_prefix} ⚠️ Error calculating IoU between {last_bbox} and {b}: {e_iou}")
                         continue

                    # print(f"{log_prefix}   -> Candidate ID {current_pid}, IoU with last: {i:.3f}") # Very verbose
                    if i > best_iou:
                        best_iou = i
                        best_pid_fallback = current_pid
                        best_bbox_candidate = b

                if best_iou > IOU_THR:
                    print(f"{log_prefix}   FALLBACK USED! Found Candidate ID {best_pid_fallback} (IoU: {best_iou:.3f} > {IOU_THR}). Using its bbox for THIS frame.")
                    target_bbox = best_bbox_candidate
                    # --- DO NOT update last_bbox here ---
                else:
                    print(f"{log_prefix}   Fallback FAILED. Max IoU {best_iou:.3f} <= {IOU_THR}. Target lost.")
                    # Keep last_bbox as is, maybe the original ID reappears

            # STEP 3: Crop if a bbox was found (original or fallback)
            if target_bbox:
                print(f"{log_prefix} Cropping using bbox: {target_bbox}")
                try:
                    # Ensure bbox is valid before enlarging
                    if not isinstance(target_bbox, (list, tuple)) or len(target_bbox) != 4 or not all(isinstance(c, (int, float)) for c in target_bbox) or target_bbox[0] >= target_bbox[2] or target_bbox[1] >= target_bbox[3]:
                         print(f"{log_prefix} ⚠️ Invalid target_bbox before enlarge: {target_bbox}. Skipping crop.")
                         continue

                    ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
                    print(f"{log_prefix}   Enlarged to: [{ex1},{ey1},{ex2},{ey2}]") # Debug enlarged box

                    if ex1 < ex2 and ey1 < ey2:
                        crop = frame[ey1:ey2, ex1:ex2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
                            crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg")
                            ok = cv2.imwrite(crop_path, crop_resized)
                            if ok: frames_cropped += 1
                            # else: print(f"{log_prefix} ⚠️ imwrite failed for crop: {crop_path}")
                        # else: print(f"{log_prefix} ⚠️ Empty crop generated after slicing.")
                    # else: print(f"{log_prefix} ⚠️ Invalid enlarged bbox after clamping.")
                except Exception as e_crop:
                    print(f"{log_prefix} ❌ ERROR during cropping/saving: {e_crop}")
                    print(f"   Input BBox: {target_bbox}")

            # else: # No bbox found for this frame
            #     print(f"{log_prefix} No bbox found (direct or fallback). Skipping crop.")

        print(f"✅ Cropped target player in {frames_cropped} frames.")
    else:
        print("ℹ️ No target player ID was set, skipping cropping.")

    # --- Action Counting ---
    counts_printed = False # Flag to ensure counts are printed once
    if target_id is not None and os.path.exists(args.crop_dir):
        print("⏳ Performing action recognition on crops...")
        try:
            counts = infer_action_counts(
                crops_dir=args.crop_dir, pose_weights=args.pose_weights, mlp_weights=args.mlp_weights,
                class_names=DEFAULT_CLASS_NAMES, conf_threshold=args.conf_thr, yolo_conf=args.yolo_conf,
                img_size=args.img_size, max_det=args.max_det, smooth_window=args.smooth_window,
                min_seg_sec=args.min_seg_sec, fps=fps
            )
            print("✅ Action recognition complete!")
            print("\n--- FINAL ACTION COUNTS ---") # Added newline for clarity
            for cls in DEFAULT_CLASS_NAMES:
                count_val = counts.get(cls, 0)
                print(f"{cls} = {count_val}")
            counts_printed = True
        except Exception as e_action:
            traceback.print_exc()
            print(f"❌ Action counting failed: {e_action}")

    # --- Print default counts if not already printed (due to errors or no target) ---
    if not counts_printed:
        if target_id is None: print("ℹ️ No target ID set, skipping action recognition.")
        elif not os.path.exists(args.crop_dir): print(f"⚠️ Crop directory '{args.crop_dir}' not found/empty, skipping action recognition.")
        else: print("--- ACTION COUNTING ERROR ---") # If action counting failed exception

        print("\n--- FINAL ACTION COUNTS (Default) ---")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")

if __name__ == "__main__":
    try:
        main()
        print("\n✅ Script finished successfully.") # Confirmation message
    except Exception as e_main:
        print(f"❌ CRITICAL ERROR in main function: {e_main}")
        traceback.print_exc()
        # Ensure default counts are printed if main fails badly
        print("\n--- MAIN FUNCTION ERROR ---")
        for cls in DEFAULT_CLASS_NAMES:
             print(f"{cls} = 0")

