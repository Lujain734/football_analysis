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
    # Receive normalized coords (0-1 range)
    parser.add_argument("--target-xy", type=float, nargs=2,
                        help="Target player NORMALIZED coords (x y) relative to original frame")
    parser.add_argument("--crop-dir", type=str, default="target_crops",
                        help="Directory to save cropped target images")
    parser.add_argument("--zoom", type=float, default=1.3,
                        help="Zoom factor for target crop")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Output crop size (square)")
    parser.add_argument("--iou-thr", type=float, default=0.5, # Default value if not passed by server.py
                        help="IoU threshold for ID fallback continuity")
    # Receive original video dimensions
    parser.add_argument("--original-width", type=float, default=None,
                        help="Original video width before any downscaling")
    parser.add_argument("--original-height", type=float, default=None,
                        help="Original video height before any downscaling")

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
# --- NEW Seeding Function (Checks Frame 0 Only) ---
def seed_target_from_first_frame(tracks_players_frame0, target_xy_pixels):
    """
    Seeds the target player using ONLY frame 0.
    Prioritizes players whose bbox contains the target_xy_pixels.
    Falls back to closest center if no bbox contains the point.
    """
    if target_xy_pixels is None:
        return None, None, None # No coordinates provided

    target_x, target_y = target_xy_pixels
    best_direct_hit = (None, None, None) # PID, Bbox for direct contains
    best_fallback = (None, None, None)   # PID, Bbox for closest center
    min_dist_fallback = float("inf")

    if not isinstance(tracks_players_frame0, dict):
         print(f"⚠️ [Seed Check] Warning: tracks_players_frame0 is not a dict: {tracks_players_frame0}")
         return None, None, None

    print(f"ℹ️ [Seed Check] Searching Frame 0 for target near PIXEL coords ({target_x:.1f}, {target_y:.1f})...")

    for pid, tr in tracks_players_frame0.items():
        # Basic Bbox Validation
        bbox = tr.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4: continue
        x1, y1, x2, y2 = bbox
        if not all(isinstance(c, (int, float)) for c in [x1, y1, x2, y2]): continue
        if x1 >= x2 or y1 >= y2: continue # Skip zero-area boxes

        # METHOD 1: Check if target_xy is INSIDE the bbox
        if x1 <= target_x <= x2 and y1 <= target_y <= y2:
            print(f"   [Seed Check] Player {pid}: DIRECT HIT! Coords are inside bbox {bbox}.")
            # Prioritize direct hit
            best_direct_hit = (pid, bbox)
            # Return immediately on first direct hit in frame 0
            print(f"✅ Selected player {best_direct_hit[0]} (bbox = {best_direct_hit[1]}) via DIRECT HIT in Frame 0.")
            return 0, best_direct_hit[0], best_direct_hit[1] # Return frame 0, pid, bbox

        # METHOD 2: Calculate distance to center (for fallback)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = np.hypot(cx - target_x, cy - target_y)
        print(f"   [Seed Check] Player {pid}: center=({cx:.1f}, {cy:.1f}), dist={dist:.2f} (Not inside bbox)") # Log distance

        if dist < min_dist_fallback:
            min_dist_fallback = dist
            best_fallback = (pid, bbox)

    # --- Decide Winner ---
    # If loop finishes without returning a direct hit:
    if best_fallback[0] is not None:
        print(f"⚠️ No direct bbox hit found in Frame 0. Using fallback: Closest center.")
        print(f"✅ Selected player {best_fallback[0]} (bbox = {best_fallback[1]}) via closest center in Frame 0, distance = {min_dist_fallback:.2f}")
        return 0, best_fallback[0], best_fallback[1] # Return frame 0, pid, bbox
    else:
        print(f"⚠️ No suitable player found (direct hit or fallback) in Frame 0 near coords {target_xy_pixels}")
        return None, None, None


def calculate_iou(a, b):
    # (Keep the robust calculate_iou function from previous version)
    if not isinstance(a, (list, tuple)) or len(a) != 4 or not all(isinstance(x, (int, float)) for x in a): return 0.0
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(isinstance(x, (int, float)) for x in b): return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
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
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def enlarge_bbox(bbox, zoom=1.3, W=None, H=None, pad_px=10):
    # (Keep the robust enlarge_bbox function from previous version)
    if W is None or H is None: raise ValueError("W and H must be provided")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4: raise ValueError(f"Invalid bbox format for enlarge_bbox: {bbox}")
    x1, y1, x2, y2 = bbox
    if not all(isinstance(coord, (int, float)) and np.isfinite(coord) for coord in [x1, y1, x2, y2]): raise ValueError(f"Non-numeric coords in enlarge_bbox: {[x1, y1, x2, y2]}")
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0: return [0, 0, 1, 1] # Handle zero-area bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = w * zoom + 2 * pad_px
    nh = h * zoom + 2 * pad_px
    nx1 = int(clamp(cx - nw / 2, 0, W - 1))
    ny1 = int(clamp(cy - nh / 2, 0, H - 1))
    nx2 = int(clamp(cx + nw / 2, 0, W - 1))
    ny2 = int(clamp(cy + nh / 2, 0, H - 1))
    if nx2 <= nx1: nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H - 1, ny1 + 1)
    return [nx1, ny1, nx2, ny2]

# ------------------------ Main ------------------------
def main():
    args = parse_args()

    # Read video
    video_frames = read_video(args.video_path)
    if len(video_frames) == 0:
        print("❌ No frames read from video. Check --video-path.")
        return

    # Store original dimensions before potential downscaling
    original_h_video, original_w_video = video_frames[0].shape[:2]

    # Downscale frames (adjust max_w if needed)
    max_w = 960
    scale_x = 1.0
    scale_y = 1.0 # Keep track of separate scales if aspect ratio changes (unlikely with INTER_AREA)
    H, W = original_h_video, original_w_video

    if original_w_video > max_w:
        scale_x = max_w / float(original_w_video)
        # Assuming INTER_AREA preserves aspect ratio, scale_y should be the same
        scale_y = scale_x
        new_w = int(original_w_video * scale_x)
        new_h = int(original_h_video * scale_y)
        print(f"ℹ️ Downscaling frames from {original_w_video}x{original_h_video} to {new_w}x{new_h} (scale: {scale_x:.3f})")
        scaled_frames = []
        for fr in video_frames:
            scaled_frames.append(cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_AREA))
        video_frames = scaled_frames
        H, W = new_h, new_w # Update dimensions to the processed size
    else:
        print(f"ℹ️ No downscaling needed. Using original dimensions: {W}x{H}")

    print(f"ℹ️ Processed video dimensions: {W}x{H}, Frames: {len(video_frames)}")

    # Calculate TARGET PIXEL coordinates based on NORMALIZED input and PROCESSED dimensions
    target_xy_pixels = None
    if args.target_xy:
        # Args.target_xy is normalized (0-1) from Swift
        norm_x, norm_y = args.target_xy
        # Convert normalized to pixel coordinates of the *processed* frame size (W, H)
        target_xy_pixels = (norm_x * W, norm_y * H)
        print(f"ℹ️ Converted normalized target [{norm_x:.3f}, {norm_y:.3f}] to pixel coords: ({target_xy_pixels[0]:.1f}, {target_xy_pixels[1]:.1f}) for {W}x{H} video")
    else:
         print("⚠️ No --target-xy provided.")


    # FPS
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps);
        if fps <= 0 or np.isnan(fps): fps = 30.0
    except: fps = 30.0
    cap.release()
    print(f"ℹ️ Video FPS: {fps:.2f}")

    # --- Tracking (Reverted to original call structure) ---
    print("⏳ Running object tracking...")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl",
    )
    # Check if tracking actually returned player data
    if not tracks or not tracks.get("players"):
        print("❌ Tracking failed or returned no player data.")
        # Print default counts and exit cleanly
        print("\n--- TRACKING FAILED ---")
        for cls in DEFAULT_CLASS_NAMES: print(f"{cls} = 0")
        return
    tracker.add_position_to_tracks(tracks)
    print("✅ Tracking complete.")

    # --- Debug prints for first frame tracks ---
    if tracks.get("players"):
         print(f"ℹ️ Players detected in first frame: {len(tracks['players'][0])}")
         # for pid, tr in tracks["players"][0].items():
         #      print(f"   Player {pid}: bbox = {tr.get('bbox')}") # Can be very verbose
    else:
         print("⚠️ No players list found in tracks after tracking.")


    # --- Seed target using ONLY FRAME 0 and PIXEL Coords ---
    target_id = None
    last_bbox_of_original = None
    f0 = None # Frame index where target was seeded (should be 0)

    if target_xy_pixels and tracks.get("players"): # Check if we have coords and player tracks
        tracks_frame0 = tracks["players"][0] if len(tracks["players"]) > 0 else {}
        # Call the new seeding function which only checks frame 0
        f0, pid0, bbox0 = seed_target_from_first_frame(tracks_frame0, target_xy_pixels)
        if pid0 is not None:
            target_id = pid0
            last_bbox_of_original = bbox0 # Initialize last known good position
            # Log confirmation
            print(f"🎯 Target player ID: {target_id} (seeded from frame {f0} using bbox {bbox0})")
            print(f"✅✅✅ INITIAL TARGET ID SEEDED AS: {target_id} ✅✅✅")
        else:
            # Seeding failed even in frame 0
            print("❌ No player found near the provided coordinates in frame 0.")
            print(f"❌❌❌ FAILED TO SEED INITIAL TARGET ID FROM COORDS {target_xy_pixels} IN FRAME 0 ❌❌❌")
            target_id = None # Ensure target_id is None if seeding fails
    elif not target_xy_pixels:
        print("⚠️ No target coordinates provided, cannot track a specific player.")
    else: # No player tracks exist
         print("⚠️ No player tracks available to seed from.")


    # --- Run other processing steps ONLY if tracking succeeded ---
    if tracks.get("players"):
        # (Camera, ViewTransform, Teams, Ball assignment code remains the same as previous correct version)
        # Make sure this code correctly handles cases where 'ball' might be missing etc.
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
        if "ball" in tracks and tracks["ball"]:
            try:
                 tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
                 print("✅ Ball positions interpolated.")
            except Exception as e_interp:
                 print(f"⚠️ Error interpolating ball positions: {e_interp}")
                 tracks["ball"] = [] # Clear ball tracks if interpolation fails
        else:
            print("⚠️ No ball tracks found to interpolate.")

        print("⏳ Assigning team colors...")
        ta = TeamAssigner()
        first_player_frame_idx = -1
        for idx, frame_players in enumerate(tracks["players"]):
            if frame_players: first_player_frame_idx = idx; break

        if first_player_frame_idx != -1:
            try:
                ta.assign_team_color(video_frames[first_player_frame_idx], tracks["players"][first_player_frame_idx])
                print(f"✅ Initial team colors assigned using frame {first_player_frame_idx}.")
                print("⏳ Assigning teams to players per frame...")
                for f_idx, player_track in enumerate(tracks["players"]):
                    for pid, tr in player_track.items():
                        # Make sure bbox exists before getting team
                        if "bbox" in tr:
                             team = ta.get_player_team(video_frames[f_idx], tr["bbox"], pid)
                             tracks["players"][f_idx][pid]["team"] = team
                             tracks["players"][f_idx][pid]["team_color"] = ta.team_colors.get(team, (255, 255, 255))
                        else:
                             tracks["players"][f_idx][pid]["team"] = 0 # Assign default team if no bbox
                             tracks["players"][f_idx][pid]["team_color"] = (255, 255, 255)
                print("✅ Teams assigned.")
            except Exception as e_team:
                 print(f"⚠️ Error during team assignment: {e_team}")
                 # Assign default team colors if assignment fails
                 for f_idx, player_track in enumerate(tracks["players"]):
                      for pid in player_track:
                           tracks["players"][f_idx][pid]["team"] = 0
                           tracks["players"][f_idx][pid]["team_color"] = (255, 255, 255)

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

            current_team = 0
            if assigned_player_id != -1 and assigned_player_id in tracks["players"][f_idx]:
                tracks["players"][f_idx][assigned_player_id]["has_ball"] = True
                current_team = tracks["players"][f_idx][assigned_player_id].get("team", 0)

            team_ball_control.append(current_team if current_team != 0 else (team_ball_control[-1] if team_ball_control else 0))
        print("✅ Ball possession assigned.")

    else: # Handle case where initial tracking failed
        print("⚠️ No player tracks available, skipping dependent processing steps.")
        tracks = {"players": [], "referees": [], "ball": []} # Ensure structure exists

    # --- Cropping Loop (Using Refined Logic from Previous Response) ---
    if target_id is not None:
        os.makedirs(args.crop_dir, exist_ok=True)
        print(f"🗂️ Cropping target player (Original ID: {target_id}) to: {os.path.abspath(args.crop_dir)}")
        IOU_THR = float(args.iou_thr)
        print(f"ℹ️ Using IoU threshold for fallback: {IOU_THR}")
        frames_cropped = 0
        original_target_id = target_id
        # last_bbox_of_original is already initialized after seeding

        for f_idx, frame in enumerate(video_frames):
            target_bbox = None
            pid_for_this_crop = None
            log_prefix = f" F[{f_idx:04d}] Target({original_target_id}):"
            current_frame_players = tracks.get("players", [])[f_idx] if f_idx < len(tracks.get("players", [])) else {}
            found_by_direct_id = False
            found_by_fallback = False

            # STEP 1: PRIORITIZE finding the ORIGINAL target_id
            if original_target_id in current_frame_players:
                bbox_candidate = current_frame_players[original_target_id].get("bbox")
                if bbox_candidate:
                    if isinstance(bbox_candidate, (list, tuple)) and len(bbox_candidate) == 4 and all(isinstance(c, (int, float)) for c in bbox_candidate):
                        target_bbox = bbox_candidate
                        last_bbox_of_original = target_bbox
                        pid_for_this_crop = original_target_id
                        found_by_direct_id = True
                        print(f"{log_prefix} Found original ID {original_target_id}.")
                    else: print(f"{log_prefix} Original ID {original_target_id} present but bbox invalid: {bbox_candidate}!")
                # else: print(f"{log_prefix} Original ID {original_target_id} present but missing bbox!") # Can be noisy

            # STEP 2: Fallback ONLY if original ID NOT found AND we have last_bbox_of_original
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

                    # REMOVED team check for simplicity now
                    if i > best_iou:
                        best_iou = i
                        best_pid_fallback = current_pid
                        best_bbox_candidate = b

                if best_iou > IOU_THR:
                    print(f"{log_prefix}   FALLBACK USED! Found Candidate ID {best_pid_fallback} (IoU: {best_iou:.3f} > {IOU_THR}). Using its bbox for THIS frame.")
                    target_bbox = best_bbox_candidate
                    pid_for_this_crop = best_pid_fallback
                    found_by_fallback = True
                    # --- DO NOT update last_bbox_of_original here ---
                else:
                    print(f"{log_prefix}   Fallback FAILED. Max IoU {best_iou:.3f} <= {IOU_THR}. Target lost this frame.")

            # STEP 3: Crop if a bbox was found
            if target_bbox:
                try:
                    if not isinstance(target_bbox, (list, tuple)) or len(target_bbox) != 4 or not all(isinstance(c, (int, float)) and np.isfinite(c) for c in target_bbox) or target_bbox[0] >= target_bbox[2] or target_bbox[1] >= target_bbox[3]:
                        print(f"{log_prefix} ⚠️ Invalid target_bbox before enlarge: {target_bbox}. Skipping crop.")
                        continue

                    ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
                    # print(f"{log_prefix} Cropping bbox: {target_bbox}, Enlarged: [{ex1},{ey1},{ex2},{ey2}]") # Verbose

                    if ex1 < ex2 and ey1 < ey2:
                        crop = frame[ey1:ey2, ex1:ex2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
                            crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg")
                            ok = cv2.imwrite(crop_path, crop_resized)
                            if ok:
                                frames_cropped += 1
                                print(f"{log_prefix}   Crop saved: {os.path.basename(crop_path)}, using bbox from ID: {pid_for_this_crop}") # Log the ID used
                        # else: print(f"{log_prefix} ⚠️ Empty crop generated after slicing.") # Verbose
                    # else: print(f"{log_prefix} ⚠️ Invalid enlarged bbox after clamping.") # Verbose
                except Exception as e_crop:
                    print(f"{log_prefix} ❌ ERROR during cropping/saving: {e_crop}")
                    print(f"   Input BBox: {target_bbox}")
            # else: print(f"{log_prefix} No bbox found. Skipping crop.") # Verbose

        print(f"✅ Cropped target player in {frames_cropped} frames.")
    else:
        print("ℹ️ No target player ID was set or seeding failed, skipping cropping.")

    # --- Action Counting ---
    counts_printed = False
    if target_id is not None and os.path.exists(args.crop_dir) and frames_cropped > 0: # Added check for frames_cropped
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

    # --- Print default counts if not already printed ---
    if not counts_printed:
        if target_id is None: print("ℹ️ No target ID set, skipping action recognition.")
        elif not os.path.exists(args.crop_dir): print(f"⚠️ Crop directory '{args.crop_dir}' not found/empty, skipping action recognition.")
        elif frames_cropped == 0: print("⚠️ No frames were cropped for the target player, skipping action recognition.")
        else: print("--- ACTION COUNTING ERROR ---")

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
        print("\n--- MAIN FUNCTION ERROR ---")
        for cls in DEFAULT_CLASS_NAMES:
             print(f"{cls} = 0")
