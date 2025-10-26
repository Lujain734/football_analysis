#!/usr/bin/env python3
"""
Haddaf Backend Server - Football Action Recognition API
"""

import os
import sys
import shutil
import subprocess
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import hf_hub_download

# ================== App / Paths ==================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ================== Models ==================
HF_REPO_ID = "lujain-721/haddaf-models"

DETECTION_WEIGHTS = os.path.join(MODELS_DIR, "best.pt")
POSE_WEIGHTS      = os.path.join(MODELS_DIR, "best1.pt")
MLP_WEIGHTS       = os.path.join(MODELS_DIR, "action_mlp.pt")

def ensure_models_downloaded():
    """Download models from HF if missing."""
    models_needed = {
        "best.pt": DETECTION_WEIGHTS,
        "best1.pt": POSE_WEIGHTS,
        "action_mlp.pt": MLP_WEIGHTS
    }
    for fname, local_path in models_needed.items():
        if not os.path.exists(local_path):
            print(f"📥 Downloading {fname} from Hugging Face...")
            try:
                downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=fname, cache_dir=None)
                shutil.copy(downloaded, local_path)
                print(f"✅ Downloaded {fname} -> {local_path}")
            except Exception as e:
                print(f"❌ Error downloading {fname}: {e}")
                print(f"   Check: https://huggingface.co/{HF_REPO_ID}")
                raise
        else:
            print(f"✅ {fname} already exists")

print("🔍 Checking models...")
ensure_models_downloaded()

# ================== Routes ==================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Haddaf Action Recognition API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/health": "Server status",
            "/analyze": "POST a video + (x,y) to analyze",
            "/view-crops/current": "View last crops in browser"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "detection_model": os.path.exists(DETECTION_WEIGHTS),
            "pose_model": os.path.exists(POSE_WEIGHTS),
            "mlp_model": os.path.exists(MLP_WEIGHTS),
        },
        "python_version": sys.version,
    })

@app.route("/crops/current/<path:filename>")
def serve_crop(filename):
    crops_path = os.path.join(DEBUG_DIR, "current", "crops")
    return send_from_directory(crops_path, filename)

@app.route("/view-crops/current")
def view_crops():
    crops_path = os.path.join(DEBUG_DIR, "current", "crops")
    if not os.path.exists(crops_path):
        return f"<h1>❌ No crops folder found</h1><p>{crops_path}</p>", 404

    images = sorted([f for f in os.listdir(crops_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
    if not images:
        return f"<h1>⚠️ No images found</h1><p>{crops_path}</p>", 404

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Target Player Crops</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                padding: 20px;
                background: #0f1222;
                color: white;
                min-height: 100vh;
            }}
            h1 {{ color: #4CAF50; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }}
            .item {{ background: #1a1f36; padding: 12px; border-radius: 12px; border: 1px solid #26304a; }}
            img {{ width: 100%; height: 250px; object-fit: cover; border-radius: 8px; }}
            .name {{ margin-top: 8px; color: #9ae6b4; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>🎯 Target Player Crops ({len(images)})</h1>
        <div class="grid">
    """
    for img in images:
        html += f"""
            <div class="item">
                <img src="/crops/current/{img}" alt="{img}" loading="lazy" />
                <div class="name">{img}</div>
            </div>
        """
    html += "</div></body></html>"
    return html

@app.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Analyze a video for action counts.
    Form-data:
      - video: file
      - x, y: floats (target pixel coordinates in ORIGINAL video dimensions)
      - video_width, video_height: ints (original video dimensions from iOS)
    """
    try:
        if "video" not in request.files:
            return jsonify({"success": False, "error": "No video file provided"}), 400
        video_file = request.files["video"]
        if not video_file.filename:
            return jsonify({"success": False, "error": "Empty filename"}), 400

        # Coordinates
        try:
            x = float(request.form.get("x", 0))
            y = float(request.form.get("y", 0))
            video_width = int(request.form.get("video_width", 0))
            video_height = int(request.form.get("video_height", 0))
        except ValueError:
            return jsonify({"success": False, "error": "Invalid coordinates or dimensions"}), 400

        print(f"📹 Received: {video_file.filename}")
        print(f"📍 Target pixel coords (original): x={x}, y={y}")
        print(f"📐 Original video dimensions: {video_width}x{video_height}")

        # Working dir
        work_dir = os.path.join(DEBUG_DIR, "current")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        # Save video
        video_path = os.path.join(work_dir, "input_video.mp4")
        video_file.save(video_path)
        print(f"✅ Video saved: {video_path}")

        # Get actual video dimensions after saving
        import cv2
        cap = cv2.VideoCapture(video_path)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"📐 Saved video dimensions: {actual_width}x{actual_height}")

        # Calculate scale if main.py will downscale
        max_width = 960
        scale_x = 1.0
        scale_y = 1.0
        
        if actual_width > max_width:
            # main.py will downscale, so we need to adjust coordinates
            scale_factor = max_width / float(actual_width)
            scaled_width = max_width
            scaled_height = int(actual_height * scale_factor)
            print(f"⚠️ Video will be downscaled in main.py to {scaled_width}x{scaled_height} (scale: {scale_factor:.3f})")
            
            # Adjust coordinates based on original vs actual dimensions
            if video_width > 0 and video_height > 0:
                # User sent coordinates based on their view, adjust to actual video first
                x_actual = x * (actual_width / float(video_width))
                y_actual = y * (actual_height / float(video_height))
                print(f"📍 Adjusted to actual video: ({x_actual:.2f}, {y_actual:.2f})")
                
                # Then adjust for downscaling
                x_scaled = x_actual * scale_factor
                y_scaled = y_actual * scale_factor
                print(f"📍 After downscaling adjustment: ({x_scaled:.2f}, {y_scaled:.2f})")
            else:
                # No original dimensions provided, assume coordinates are for actual video
                x_scaled = x * scale_factor
                y_scaled = y * scale_factor
                print(f"📍 Direct downscale adjustment: ({x_scaled:.2f}, {y_scaled:.2f})")
            
            final_x = x_scaled
            final_y = y_scaled
        else:
            # No downscaling will happen
            if video_width > 0 and video_height > 0:
                # Adjust from iOS view to actual video
                final_x = x * (actual_width / float(video_width))
                final_y = y * (actual_height / float(video_height))
                print(f"📍 Adjusted coordinates: ({final_x:.2f}, {final_y:.2f})")
            else:
                final_x = x
                final_y = y

        print(f"🎯 Final coordinates for main.py: ({final_x:.2f}, {final_y:.2f})")

        # Crops dir
        crops_dir = os.path.join(work_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        # Build command
        main_script = os.path.join(BASE_DIR, "main.py")
        cmd = [
            sys.executable, main_script,
            "--video-path", video_path,
            "--target-xy", str(final_x), str(final_y),
            "--crop-dir", crops_dir,
            "--pose-weights", POSE_WEIGHTS,
            "--mlp-weights", MLP_WEIGHTS,
            "--zoom", "1.3",
            "--crop-size", "224",
            "--yolo-conf", "0.25",
            "--img-size", "736",
            "--max-det", "5",
            "--smooth-window", "7",
            "--min-seg-sec", "0.30",
            "--iou-thr", "0.3"
        ]
        print(f"🚀 Running: {' '.join(cmd)}")

        # Limit threads
        env_limited = dict(os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            BLIS_NUM_THREADS="1",
            OPENCV_OPENCL_RUNTIME="disabled",
            MALLOC_ARENA_MAX="2",
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=BASE_DIR,
            env=env_limited,
        )
        print(f"📤 Return code: {result.returncode}")

        if result.returncode != 0:
            print(f"❌ STDERR:\n{result.stderr}")
            return jsonify({
                "success": False,
                "error": "Processing failed",
                "details": result.stderr[-2000:],
            }), 500

        # Parse output
        output_lines = result.stdout.strip().splitlines()
        action_counts = {}
        print("📊 Output lines:")
        for line in output_lines:
            print("  ", line)
            if "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    k = parts[0].strip()
                    try:
                        v = int(parts[1].strip())
                        action_counts[k] = v
                    except ValueError:
                        pass

        if not action_counts:
            action_counts = {"dribble": 0, "pass": 0, "shoot": 0}
            print("⚠️ No action counts parsed, defaulting to zeros.")

        # Count crops
        crop_count = 0
        if os.path.exists(crops_dir):
            crop_files = [f for f in os.listdir(crops_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            crop_count = len(crop_files)
        print(f"📸 Total crops: {crop_count}")
        
        base_url = request.host_url.rstrip("/")
        crops_url = f"{base_url}/view-crops/current"
        print(f"🌐 View crops at: {crops_url}")

        return jsonify({
            "success": True,
            "action_counts": action_counts,
            "target_coordinates": {
                "original_x": x, 
                "original_y": y,
                "adjusted_x": final_x,
                "adjusted_y": final_y
            },
            "video_dimensions": {
                "original": f"{video_width}x{video_height}",
                "actual": f"{actual_width}x{actual_height}",
                "processed": f"{max_width if actual_width > max_width else actual_width}x{int(actual_height * (max_width/actual_width)) if actual_width > max_width else actual_height}"
            },
            "crops_url": crops_url,
            "total_crops": crop_count,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ Exception: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(_):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Haddaf Backend Server Starting...")
    print("=" * 60)
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🤖 Models directory: {MODELS_DIR}")
    print(f"📸 Debug output directory: {DEBUG_DIR}")
    print(f"✅ Detection model: {os.path.exists(DETECTION_WEIGHTS)}")
    print(f"✅ Pose model: {os.path.exists(POSE_WEIGHTS)}")
    print(f"✅ MLP model: {os.path.exists(MLP_WEIGHTS)}")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
