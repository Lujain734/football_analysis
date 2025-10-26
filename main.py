#!/usr/bin/env python3
# main.py
import os
import cv2
import numpy as np
import argparse
import traceback
from utils import read_video, save_video
from trackers import Tracker  # Ensure trackers.py contains the Tracker class
from team_assigner import TeamAssigner  # Ensure team_assigner.py is correct
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
                        help="Target player normalized coords (x y, 0-1) in early frames") # Changed help text
    parser.add_argument("--crop-dir", type=str, default="target_crops",
                        help="Directory to save cropped target images")
    parser.add_argument("--zoom", type=float, default=1.3,
                        help="Zoom factor for target crop")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Output crop size (square)")
    parser.add_argument("--iou-thr", type=float, default=0.5,  # Increased default based on server.py
                        help="IoU threshold for ID fallback continuity")
    parser.add_argument("--original-width", type=float, default=None,
                        help="Original video width from client")
    parser.add_argument("--original-height", type=float, default=None,
                        help="Original video height from client")
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
def seed_target_from_first_k_frames(tracks_players, target_xy_pixels, K=1): # Changed default K to 1
    """Finds player closest to target_xy_pixels ONLY in the specified frames."""
    if target_xy_pixels is None:
        return None, None, None
    best = (None, None, None)
    best_dist = float("inf")
    # Ensure K is at least 1 and within bounds
    max_f = min(max(1, K), len(tracks_players)) # Check at least frame 0 up to K

    if len(tracks_players) == 0:
         print("⚠️ Warning: tracks_players is empty in seed_target function.")
         return None, None, None

    # Check only frame 0 if K=1
    for f_idx in range(max_f):
        if not isinstance(tracks_players[f_idx], dict):
             print(f"⚠️ Warning: tracks_players[{f_idx}] is not a dict: {tracks_players[f_idx]}")
             continue
        for pid, tr in tracks_players[f_idx].items():
            if 'bbox' not in tr or not isinstance(tr['bbox'], (list, tuple)) or len(tr['bbox']) != 4: continue
            x1, y1, x2, y2 = tr["bbox"]
            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]): continue
            if x1 >= x2 or y1 >= y2: continue # Skip zero area boxes silently now

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            d = np.hypot(cx - target_xy_pixels[0], cy - target_xy_pixels[1])
            # Verbose logging of distance checks:
            print(f"   [Seed Check] Frame {f_idx}, Player {pid}: center=({cx:.1f}, {cy:.1f}), dist_to_target= {d:.2f}")
            if d < best_dist:
                best_dist = d
                best = (f_idx, pid, tr["bbox"])

    if best == (None, None, None):
        print(f"⚠️ No suitable player found in first {max_f} frame(s) near coords {target_xy_pixels}")
    else:
        # Unpack for clarity in logging
        f0, pid0, bbox0 = best
        print(f"✅ Selected player {pid0} in frame {f0} (bbox = {bbox0}) as initial target, distance = {best_dist:.2f}")
    return best

def calculate_iou(a, b):
    # Basic validation
    if not isinstance(a, (list, tuple)) or len(a) != 4 or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in a): return 0.0
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in b): return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Ensure coordinates order (minmax)
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    # Calculate intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0

    # Calculate union
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union == 0: return 0.0 # Avoid division by zero if areas are zero

    return inter / union

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def enlarge_bbox(bbox, zoom=1.3, W=None, H=None, pad_px=10):
    if W is None or H is None: raise ValueError("W and H must be provided")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4: raise ValueError(f"Invalid bbox: {bbox}")
    if not all(isinstance(c, (int, float)) and np.isfinite(c) for c in bbox): raise ValueError(f"Non-numeric bbox: {bbox}")

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return [0, 0, 1, 1] # Handle zero-area input

    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    nw, nh = w * zoom + 2 * pad_px, h * zoom + 2 * pad_px

    nx1 = int(clamp(cx - nw / 2, 0, W - 1))
    ny1 = int(clamp(cy - nh / 2, 0, H - 1))
    nx2 = int(clamp(cx + nw / 2, 0, W - 1))
    ny2 = int(clamp(cy + nh / 2, 0, H - 1))

    # Ensure minimum size
    if nx2 <= nx1: nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H - 1, ny1 + 1)
    return [nx1, ny1, nx2, ny2]

# ------------------------ Main ------------------------
def main():
    args = parse_args()

    # Read video
    video_frames = read_video(args.video_path)
    if not video_frames:
        print("❌ No frames read from video. Check --video-path.")
        # Print default counts for server compatibility
        print("\n--- MAIN FUNCTION ERROR (NO FRAMES) ---")
        for cls in DEFAULT_CLASS_NAMES: print(f"{cls} = 0")
        return

    # Processed dimensions (after potential downscaling)
    H, W = video_frames[0].shape[:2]
    print(f"ℹ️ Video dimensions (potentially downscaled): {W}x{H}, Frames: {len(video_frames)}")

    # Calculate scaling factor if original dimensions provided
    scale_x, scale_y = 1.0, 1.0
    if args.original_width and args.original_height and args.original_width > 0 and args.original_height > 0:
        scale_x = W / args.original_width
        scale_y = H / args.original_height
        print(f"ℹ️ Scaling factors based on original {args.original_width}x{args.original_height}: sx={scale_x:.3f}, sy={scale_y:.3f}")

    # Scale target coordinates from NORMALIZED input to PIXEL coordinates of the processed video
    target_xy_pixels = None
    if args.target_xy:
        try:
            # Input xy is assumed normalized (0-1) based on Swift code
            norm_x, norm_y = args.target_xy
            if not (0 <= norm_x <= 1 and 0 <= norm_y <= 1):
                 print(f"⚠️ Warning: Received target_xy {args.target_xy} seems outside normalized range [0, 1].")
                 # Clamp to be safe, though this might indicate an issue in Swift
                 norm_x = clamp(norm_x, 0, 1)
                 norm_y = clamp(norm_y, 0, 1)

            # Scale normalized coords to the *processed* video dimensions (W, H)
            target_xy_pixels = (norm_x * W, norm_y * H)
            print(f"ℹ️ Converted normalized target {args.target_xy} to pixel coords: {target_xy_pixels} for {W}x{H} video")
        except Exception as e_scale:
            print(f"❌ Error scaling target coordinates: {e_scale}. Cannot seed target.")
            target_xy_pixels = None # Ensure it's None if scaling fails

    # FPS
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    try: fps = float(fps); assert fps > 0 and np.isfinite(fps)
    except: fps = 30.0; print("⚠️ FPS read failed, defaulting to 30.")
    print(f"ℹ️ Video FPS: {fps:.2f}")

    # Tracking
    print("⏳ Running object tracking...")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl")
    tracker.add_position_to_tracks(tracks)
    print("✅ Tracking complete.")

    # Debug: Print players in first frame
    if tracks.get("players", []) and isinstance(tracks["players"][0], dict):
        print(f"ℹ️ Players detected in first frame: {len(tracks['players'][0])}")
        # for pid, tr in tracks["players"][0].items(): print(f"  Player {pid}: bbox={tr.get('bbox')}") # Optional verbose
    else: print("⚠️ No players detected in first frame.")

    # Seed target - NOW using target_xy_pixels and checking ONLY FRAME 0
    target_id = None
    last_bbox = None
    f0 = None # Store the frame index where target was seeded (should be 0 now)
    print(f"ℹ️ Trying to seed target using PIXEL coords: {target_xy_pixels}")
    if target_xy_pixels:
        # --- MODIFIED: Force K=1 to check only frame 0 ---
        f0_cand, pid0, bbox0 = seed_target_from_first_k_frames(tracks.get("players", []), target_xy_pixels, K=1)
        if pid0 is not None:
            target_id = pid0
            last_bbox = bbox0
            f0 = f0_cand # Should be 0
            print(f"🎯 Target player ID: {target_id} (seeded from frame {f0} using bbox {bbox0})")
            print(f"✅✅✅ INITIAL TARGET ID SEEDED AS: {target_id} ✅✅✅")
        else:
            print("❌ No player found near the provided coordinates in frame 0.")
            print(f"❌❌❌ FAILED TO SEED INITIAL TARGET ID FROM COORDS {target_xy_pixels} ❌❌❌")
    else:
        print("⚠️ No valid target coordinates provided, cannot track a specific player.")

    # --- Run other processing steps ---
    if "players" in tracks and tracks["players"]:
        print("⏳ Estimating camera movement...")
        cam = CameraMovementEstimator(video_frames[0])
        cam_movement = cam.get_camera_movement(video_frames, read_from_stub=False, stub_path="stubs/camera_movement_stub.pkl")
        cam.add_adjust_positions_to_tracks(tracks, cam_movement)
        print("✅ Camera movement estimated.")

        print("⏳ Applying view transformation...")
        vt = ViewTransformer(); vt.add_transformed_position_to_tracks(tracks)
        print("✅ View transformation applied.")

        print("⏳ Interpolating ball positions...")
        if "ball" in tracks and tracks["ball"]: tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"]); print("✅ Ball positions interpolated.")
        else: print("⚠️ No ball tracks found.")

        print("⏳ Assigning team colors...")
        ta = TeamAssigner()
        first_player_frame_idx = next((idx for idx, fr in enumerate(tracks["players"]) if fr), -1)
        if first_player_frame_idx != -1:
            ta.assign_team_color(video_frames[first_player_frame_idx], tracks["players"][first_player_frame_idx])
            print(f"✅ Initial team colors assigned using frame {first_player_frame_idx}.")
            print("⏳ Assigning teams per frame...")
            for f_idx, player_track in enumerate(tracks["players"]):
                for pid, tr in player_track.items():
                    # Ensure bbox is valid before getting team
                    if 'bbox' in tr and isinstance(tr['bbox'], (list, tuple)) and len(tr['bbox']) == 4:
                         team = ta.get_player_team(video_frames[f_idx], tr["bbox"], pid)
                         tracks["players"][f_idx][pid]["team"] = team
                         tracks["players"][f_idx][pid]["team_color"] = ta.team_colors.get(team, (255, 255, 255))
                    else:
                         # Assign default if bbox invalid
                         tracks["players"][f_idx][pid]["team"] = 0
                         tracks["players"][f_idx][pid]["team_color"] = (255, 255, 255)
            print("✅ Teams assigned.")
        else: print("⚠️ No players detected, cannot assign teams.")

        print("⏳ Assigning ball possession...")
        pba = PlayerBallAssigner()
        team_ball_control = []
        last_team_with_ball = 0
        for f_idx, player_track in enumerate(tracks["players"]):
            ball_dict = tracks.get("ball", [])[f_idx] if f_idx < len(tracks.get("ball", [])) else {}
            ball_bbox = ball_dict.get(1, {}).get("bbox", None)
            assigned_player_id = pba.assign_ball_to_player(player_track, ball_bbox) if ball_bbox else -1

            current_team = 0
            if assigned_player_id != -1 and assigned_player_id in tracks["players"][f_idx]:
                 tracks["players"][f_idx][assigned_player_id]["has_ball"] = True
                 current_team = tracks["players"][f_idx][assigned_player_id].get("team", 0)
                 last_team_with_ball = current_team # Update last known team

            team_ball_control.append(current_team if current_team != 0 else last_team_with_ball)
        print("✅ Ball possession assigned.")
    else:
        print("⚠️ No player tracks found, skipping dependent processing steps.")
        tracks = {"players": [], "referees": [], "ball": []} # Prevent crashes

    # --- Cropping Loop (Corrected Logic) ---
    if target_id is not None:
        os.makedirs(args.crop_dir, exist_ok=True)
        print(f"🗂️ Cropping target player (Original ID: {target_id}) to: {os.path.abspath(args.crop_dir)}")
        IOU_THR = float(args.iou_thr)
        print(f"ℹ️ Using IoU threshold for fallback: {IOU_THR}")
        frames_cropped = 0
        original_target_id = target_id
        # Use last_bbox which was set during seeding (should be from frame 0)
        last_bbox_of_original = last_bbox
        # Get the original target's team from the seeding frame (f0) if possible
        original_team = None
        if f0 is not None and tracks.get("players") and f0 < len(tracks["players"]) and original_target_id in tracks["players"][f0]:
             original_team = tracks["players"][f0][original_target_id].get("team")
        print(f"ℹ️ Original target team (from frame {f0}): {original_team}")


        for f_idx, frame in enumerate(video_frames):
            target_bbox = None # Bbox to use for cropping this frame
            pid_for_this_crop = None # ID whose bbox is used
            log_prefix = f" F[{f_idx:04d}] Target({original_target_id}):"
            current_frame_players = tracks.get("players", [])[f_idx] if f_idx < len(tracks.get("players", [])) else {}
            found_by_direct_id = False
            found_by_fallback = False

            # STEP 1: Look for the original target ID
            if original_target_id in current_frame_players:
                bbox_candidate = current_frame_players[original_target_id].get("bbox")
                # Check validity
                if bbox_candidate and isinstance(bbox_candidate, (list, tuple)) and len(bbox_candidate) == 4 and all(isinstance(c, (int, float)) and np.isfinite(c) for c in bbox_candidate):
                     target_bbox = bbox_candidate
                     last_bbox_of_original = target_bbox # Update last known GOOD position
                     pid_for_this_crop = original_target_id
                     found_by_direct_id = True
                     print(f"{log_prefix} Found original ID {original_target_id}.")
                else: print(f"{log_prefix} Original ID {original_target_id} present but bbox invalid: {bbox_candidate}!")

            # STEP 2: Fallback if original ID was NOT found AND we have a previous location
            if not found_by_direct_id and last_bbox_of_original is not None:
                print(f"{log_prefix} Original ID lost. Last known original bbox: {last_bbox_of_original}. Searching fallback...")
                best_iou = 0.0
                best_pid_fallback = None
                best_bbox_candidate = None

                for current_pid, pdata in current_frame_players.items():
                    if current_pid == original_target_id: continue
                    b = pdata.get("bbox")
                    if not b: continue

                    try: i = calculate_iou(last_bbox_of_original, b)
                    except Exception as e_iou: print(f"{log_prefix} ⚠️ Error calculating IoU: {e_iou}"); continue

                    candidate_team = pdata.get("team")
                    # Check IoU AND Team match (allow if original_team unknown)
                    if i > best_iou and (original_team is None or candidate_team is None or candidate_team == original_team):
                        best_iou = i
                        best_pid_fallback = current_pid
                        best_bbox_candidate = b
                    # elif i > best_iou: print(f"{log_prefix}   -> Candidate ID {current_pid} IoU {i:.3f} REJECTED (Team mismatch: {candidate_team} vs {original_team})")

                if best_iou > IOU_THR:
                    print(f"{log_prefix}   FALLBACK USED! Found Candidate ID {best_pid_fallback} (IoU: {best_iou:.3f} > {IOU_THR}). Using its bbox for THIS frame.")
                    target_bbox = best_bbox_candidate
                    pid_for_this_crop = best_pid_fallback # Record the ID actually used
                    found_by_fallback = True
                    # DO NOT update last_bbox_of_original
                else:
                    print(f"{log_prefix}   Fallback FAILED. Max IoU {best_iou:.3f} <= {IOU_THR}. Target lost this frame.")

            # STEP 3: Crop if a bbox was found
            if target_bbox:
                # print(f"{log_prefix} Cropping using bbox: {target_bbox} (from ID: {pid_for_this_crop})") # Debug log
                try:
                    if not (isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4 and all(isinstance(c, (int, float)) and np.isfinite(c) for c in target_bbox) and target_bbox[0] < target_bbox[2] and target_bbox[1] < target_bbox[3]):
                         print(f"{log_prefix} ⚠️ Invalid target_bbox before enlarge: {target_bbox}. Skipping crop.")
                         continue

                    ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
                    # print(f"{log_prefix}   Enlarged to: [{ex1},{ey1},{ex2},{ey2}]") # Debug

                    if ex1 < ex2 and ey1 < ey2:
                        crop = frame[ey1:ey2, ex1:ex2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
                            crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg") # Original filename
                            ok = cv2.imwrite(crop_path, crop_resized)
                            if ok:
                                frames_cropped += 1
                                # Log the ID whose bbox was actually used
                                print(f"{log_prefix}   Crop saved: {os.path.basename(crop_path)}, using bbox from ID: {pid_for_this_crop}") # Corrected Variable Name
                        # else: print(f"{log_prefix} ⚠️ Empty crop generated after slicing.")
                    # else: print(f"{log_prefix} ⚠️ Invalid enlarged bbox after clamping.")
                except Exception as e_crop:
                    print(f"{log_prefix} ❌ ERROR during cropping/saving: {e_crop}")
                    print(f"   Input BBox: {target_bbox}")

            # else: # No bbox found
            #    print(f"{log_prefix} No bbox found. Skipping crop.")


        print(f"✅ Cropped target player in {frames_cropped} frames.")
    else:
        print("ℹ️ No target player ID was set, skipping cropping.")

    # --- Action Counting ---
    counts_printed = False
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
            print("\n--- FINAL ACTION COUNTS ---")
            for cls in DEFAULT_CLASS_NAMES:
                count_val = counts.get(cls, 0)
                print(f"{cls} = {count_val}")
            counts_printed = True
        except Exception as e_action:
            traceback.print_exc()
            print(f"❌ Action counting failed: {e_action}")

    if not counts_printed:
        # Determine why counts weren't printed and log appropriately
        reason = ""
        if target_id is None: reason = "No target ID was set."
        elif not os.path.exists(args.crop_dir): reason = f"Crop directory '{args.crop_dir}' not found/empty."
        else: reason = "Action counting function raised an error."
        print(f"⚠️ Skipping action counting: {reason}")
        print("\n--- FINAL ACTION COUNTS (Default) ---")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")

if __name__ == "__main__":
    try:
        main()
        print("\n✅ Script finished successfully.")
    except Exception as e_main:
        print(f"❌ CRITICAL ERROR in main function: {e_main}")
        traceback.print_exc()
        # Ensure default counts are printed
        print("\n--- MAIN FUNCTION ERROR ---")
        for cls in DEFAULT_CLASS_NAMES:
             print(f"{cls} = 0")
