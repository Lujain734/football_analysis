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
    parser.add_argument("--iou-thr", type=float, default=0.03, # Default value if not passed by server.py
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

    # Downscale frames to reduce RAM usage (Optional, adjust max_w if needed)
    first_h, first_w = video_frames[0].shape[:2]
    max_w = 960  # reduce if memory is tight (e.g., 720)
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

    # FPS (fallback to 30 if unknown)
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

    # Tracking players/ball
    print("⏳ Running object tracking...")
    tracker = Tracker("models/best.pt") # Assuming DETECTION_WEIGHTS path is models/best.pt
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False, # Set to True to load cached tracks for faster debugging
        stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)
    print("✅ Tracking complete.")

    # Seed target from early frames (pixel coords)
    target_id = None
    last_bbox = None
    target_xy = tuple(args.target_xy) if args.target_xy else None
    print(f"ℹ️ Trying to seed target using coords: {target_xy}")
    if target_xy:
        f0, pid0, bbox0 = seed_target_from_first_k_frames(tracks["players"], target_xy, K=10) # Checks first 10 frames
        if pid0 is not None:
            target_id = pid0
            last_bbox = bbox0
            print(f"🎯 Target player ID: {target_id} (seeded from frame {f0} using bbox {bbox0})")
            # --- ADDED LOGGING ---
            print(f"✅✅✅ INITIAL TARGET ID SEEDED AS: {target_id} ✅✅✅")
            # --- END ADDED LOGGING ---
        else:
            print("⚠️ No player found near the provided --target-xy in the first 10 frames.")
            # --- ADDED LOGGING ---
            print(f"❌❌❌ FAILED TO SEED INITIAL TARGET ID FROM COORDS {target_xy} ❌❌❌")
            # --- END ADDED LOGGING ---
    else:
        print("⚠️ No --target-xy provided, cannot track a specific player.")

    # Camera motion + view transform + team colors + ball assignment
    print("⏳ Estimating camera movement...")
    cam = CameraMovementEstimator(video_frames[0])
    cam_movement = cam.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    cam.add_adjust_positions_to_tracks(tracks, cam_movement)
    print("✅ Camera movement estimated.")

    print("⏳ Applying view transformation...")
    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)
    print("✅ View transformation applied.")

    print("⏳ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("✅ Ball positions interpolated.")

    print("⏳ Assigning team colors...")
    ta = TeamAssigner()
    # Use the first frame with players to assign initial team colors
    first_player_frame_idx = -1
    for idx, frame_players in enumerate(tracks["players"]):
        if frame_players:
             first_player_frame_idx = idx
             break

    if first_player_frame_idx != -1:
        ta.assign_team_color(video_frames[first_player_frame_idx], tracks["players"][first_player_frame_idx])
        print("✅ Initial team colors assigned using frame", first_player_frame_idx)

        print("⏳ Assigning teams to players per frame...")
        for f_idx, player_track in enumerate(tracks["players"]):
            for pid, tr in player_track.items():
                team = ta.get_player_team(video_frames[f_idx], tr["bbox"], pid)
                tracks["players"][f_idx][pid]["team"] = team
                tracks["players"][f_idx][pid]["team_color"] = ta.team_colors.get(team, (255, 255, 255)) # Default white if team missing
        print("✅ Teams assigned per frame.")
    else:
        print("⚠️ No players detected in any frame, cannot assign team colors.")


    print("⏳ Assigning ball possession...")
    pba = PlayerBallAssigner()
    team_ball_control = []
    for f_idx, player_track in enumerate(tracks["players"]):
        ball_dict = tracks["ball"][f_idx] if f_idx < len(tracks["ball"]) else {}
        # Assuming ball has ID 1, check if this is correct for your tracker
        ball_bbox = ball_dict.get(1, {}).get("bbox", None)
        assigned_player_id = -1
        if ball_bbox is not None:
            assigned_player_id = pba.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player_id != -1 and assigned_player_id in tracks["players"][f_idx]:
            tracks["players"][f_idx][assigned_player_id]["has_ball"] = True
            team_ball_control.append(tracks["players"][f_idx][assigned_player_id]["team"])
        else:
            # If no player assigned or player ID invalid, carry over last known control
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0) # Default team 0 if no history
    print("✅ Ball possession assigned.")

    # --- Draw annotations (optional, for output video) ---
    # print("⏳ Drawing annotations...")
    # output_video_frames = tracker.draw_annotations(
    #     video_frames, tracks, team_ball_control, target_id=target_id
    # )
    # print("✅ Annotations drawn.")
    # Use original frames if annotations are not needed for crops
    output_video_frames = video_frames

    # ------- Crop target player per frame + save to correct dir -------
    if target_id is not None:
        os.makedirs(args.crop_dir, exist_ok=True)
        print(f"🗂️ Cropping target player (ID: {target_id}) to: {os.path.abspath(args.crop_dir)}")

        IOU_THR = float(args.iou_thr) # Get threshold from args
        print(f"ℹ️ Using IoU threshold for fallback: {IOU_THR}")
        frames_cropped = 0

        for f_idx, frame in enumerate(video_frames):
            target_bbox = None
            player_found_in_frame = False

            # Check if target ID exists directly in this frame's tracks
            if f_idx < len(tracks["players"]) and target_id in tracks["players"][f_idx]:
                target_bbox = tracks["players"][f_idx][target_id]["bbox"]
                last_bbox = target_bbox # Update last known good position
                player_found_in_frame = True
            # Fallback: If target ID lost, try finding player by IoU overlap with last known bbox
            elif last_bbox is not None and f_idx < len(tracks["players"]):
                best_iou = 0.0
                best_pid = None
                best_bbox_candidate = None
                # print(f"DEBUG Frame {f_idx}: Target ID {target_id} lost. Last bbox: {last_bbox}. Searching candidates...") # Verbose debug

                for current_pid, pdata in tracks["players"][f_idx].items():
                    b = pdata["bbox"]
                    i = calculate_iou(last_bbox, b)
                    # print(f"  Candidate ID {current_pid}, bbox {b}, IoU with last: {i:.3f}") # Verbose debug
                    if i > best_iou:
                        best_iou = i
                        best_pid = current_pid
                        best_bbox_candidate = b

                # If a player with sufficient overlap is found, assume it's the target
                if best_iou > IOU_THR:
                    # print(f"  FALLBACK SUCCESS: Found ID {best_pid} with IoU {best_iou:.3f} > {IOU_THR}. Assuming target.") # Verbose debug
                    target_bbox = best_bbox_candidate
                    last_bbox = target_bbox # Update last known position with the fallback
                    # --- IMPORTANT: SHOULD WE UPDATE target_id HERE? ---
                    # If we update target_id = best_pid, we permanently switch.
                    # If we don't, we risk losing them again if the original ID reappears.
                    # For now, let's NOT update target_id, just use the bbox for this frame.
                    player_found_in_frame = True
                # else:
                    # print(f"  FALLBACK FAILED: Max IoU {best_iou:.3f} <= {IOU_THR}. Target lost this frame.") # Verbose debug


            # If we have a bounding box for the target (either direct or fallback)
            if target_bbox:
                try:
                    ex1, ey1, ex2, ey2 = map(int, enlarge_bbox(target_bbox, args.zoom, W, H))
                    # Ensure coordinates are valid after enlargement
                    if ex1 < ex2 and ey1 < ey2:
                        crop = frame[ey1:ey2, ex1:ex2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR for resizing
                            crop_path = os.path.join(args.crop_dir, f"frame_{f_idx:05d}.jpg")
                            ok = cv2.imwrite(crop_path, crop_resized)
                            if ok:
                                frames_cropped += 1
                            else:
                                print(f"⚠️ imwrite failed for crop: {crop_path}")
                        else:
                            print(f"⚠️ Empty crop generated for frame {f_idx} at bbox {target_bbox} enlarged to [{ex1},{ey1},{ex2},{ey2}]")
                    else:
                         print(f"⚠️ Invalid enlarged bbox [{ex1},{ey1},{ex2},{ey2}] for frame {f_idx}")

                    # --- Draw target box on output video (optional) ---
                    # if f_idx < len(output_video_frames):
                    #     x1_orig, y1_orig, x2_orig, y2_orig = map(int, target_bbox)
                    #     cv2.rectangle(output_video_frames[f_idx], (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
                    #     cv2.putText(output_video_frames[f_idx], f"TARGET (ID:{target_id})", (x1_orig, y1_orig - 10),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e_crop:
                    print(f"❌ Error cropping frame {f_idx}: {e_crop}")
                    print(f"  Target BBox: {target_bbox}")

            # If player wasn't found even with fallback
            # elif target_id is not None:
                # print(f"ℹ️ Target player ID {target_id} not found in frame {f_idx}, skipping crop.")
                # We already reset last_bbox if the direct ID wasn't found and fallback failed

        print(f"✅ Cropped target player in {frames_cropped} frames.")
    else:
        print("ℹ️ No target player ID was set, skipping cropping.")


    # ------- Save annotated video (optional) -------
    # os.makedirs("output_videos", exist_ok=True)
    # save_video(output_video_frames, "output_videos/output_video.avi", fps) # Pass FPS
    # print("✅ Saved annotated video to output_videos/output_video.avi")

    # ------- Action counting from crops folder -------
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
                fps=fps  # Pass the video's FPS
            )
            print("✅ Action recognition complete.")

            # Print counts in the required format (parsed by server.py)
            print("📊 Final Action Counts:")
            final_counts = {}
            for cls in DEFAULT_CLASS_NAMES:
                count_val = counts.get(cls, 0)
                print(f"{cls} = {count_val}")
                final_counts[cls] = count_val

            # Optional: Print total counts for verification
            # total_actions = sum(final_counts.values())
            # print(f"Total actions counted: {total_actions}")

        except Exception as e_action:
            traceback.print_exc()
            print(f"❌ Action counting failed: {e_action}")
            # Ensure *some* output is printed for the server to parse, even on error
            print("--- ACTION COUNTING ERROR ---")
            for cls in DEFAULT_CLASS_NAMES:
                print(f"{cls} = 0")
            return # Exit if action counting failed critically

    elif target_id is None:
         print("ℹ️ No target ID set, skipping action recognition.")
         for cls in DEFAULT_CLASS_NAMES:
             print(f"{cls} = 0")
    else: # Crop dir doesn't exist (likely no crops were saved)
        print(f"⚠️ Crop directory '{args.crop_dir}' not found or empty, skipping action recognition.")
        for cls in DEFAULT_CLASS_NAMES:
            print(f"{cls} = 0")


if __name__ == "__main__":
    try:
        main()
    except Exception as e_main:
        print(f"❌ CRITICAL ERROR in main function: {e_main}")
        traceback.print_exc()
        # Ensure default counts are printed if main fails badly
        # This helps server.py parse *something*
        print("--- MAIN FUNCTION ERROR ---")
        for cls in DEFAULT_CLASS_NAMES:
             print(f"{cls} = 0")
