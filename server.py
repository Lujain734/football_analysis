#!/usr/bin/env python3
"""
Haddaf Backend Server - Football Action Recognition
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import os
import sys
import tempfile
import shutil
import subprocess
import traceback
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===== Hugging Face Configuration =====
HF_REPO_ID = "lujain-721/haddaf-models"

# ===== إعدادات المسارات =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")

# المودلات المطلوبة
DETECTION_WEIGHTS = os.path.join(MODELS_DIR, "best.pt")
POSE_WEIGHTS = os.path.join(MODELS_DIR, "best1.pt")
MLP_WEIGHTS = os.path.join(MODELS_DIR, "action_mlp.pt")

# ===== تحميل المودلات من Hugging Face =====
def ensure_models_downloaded():
    """تحميل المودلات من Hugging Face لو مو موجودة"""
    
    models_needed = {
        'best.pt': DETECTION_WEIGHTS,
        'best1.pt': POSE_WEIGHTS,
        'action_mlp.pt': MLP_WEIGHTS
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_filename, local_path in models_needed.items():
        if not os.path.exists(local_path):
            print(f"📥 Downloading {model_filename} from Hugging Face...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=model_filename,
                    cache_dir=None
                )
                
                shutil.copy(downloaded_path, local_path)
                print(f"✅ Downloaded {model_filename} to {local_path}")
            except Exception as e:
                print(f"❌ Error downloading {model_filename}: {e}")
                print(f"   Make sure the model exists at: https://huggingface.co/{HF_REPO_ID}")
                raise
        else:
            print(f"✅ {model_filename} already exists")

# تحميل المودلات عند بداية التشغيل
print("🔍 Checking models...")
ensure_models_downloaded()

# إنشاء مجلد debug_output
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===== API Endpoints =====

@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        'service': 'Haddaf Action Recognition API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Check server status',
            '/analyze': 'Analyze football video (POST)',
            '/view-crops/current': 'View crop images in browser'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """للتحقق من أن السيرفر يعمل"""
    models_status = {
        'detection_model': os.path.exists(DETECTION_WEIGHTS),
        'pose_model': os.path.exists(POSE_WEIGHTS),
        'mlp_model': os.path.exists(MLP_WEIGHTS)
    }
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': models_status,
        'python_version': sys.version
    })

@app.route('/crops/current/<path:filename>')
def serve_crop(filename):
    """عرض صورة معينة"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    return send_from_directory(crops_path, filename)

@app.route('/view-crops/current')
def view_crops():
    """عرض كل الصور في صفحة HTML"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    
    if not os.path.exists(crops_path):
        return f"<h1>❌ No crops folder found</h1><p>{crops_path}</p>", 404
    
    images = sorted([f for f in os.listdir(crops_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
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
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                min-height: 100vh;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ 
                color: #4CAF50; 
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
            }}
            .stats {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
            }}
            .stat {{
                background: rgba(76, 175, 80, 0.1);
                padding: 15px 30px;
                border-radius: 10px;
                border: 2px solid rgba(76, 175, 80, 0.3);
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .stat-label {{
                color: #aaa;
                margin-top: 5px;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
                gap: 25px;
                margin-top: 30px;
            }}
            .item {{ 
                background: rgba(255,255,255,0.08);
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
                border-color: rgba(76, 175, 80, 0.5);
            }}
            img {{ 
                width: 100%;
                height: 280px;
                object-fit: cover;
                border-radius: 8px;
                border: 2px solid rgba(76, 175, 80, 0.2);
            }}
            .filename {{ 
                margin-top: 12px;
                font-size: 13px;
                color: #4CAF50;
                font-family: 'Courier New', monospace;
            }}
            .frame-num {{
                display: inline-block;
                background: rgba(76, 175, 80, 0.2);
                padding: 3px 8px;
                border-radius: 5px;
                margin-top: 5px;
                font-size: 11px;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Target Player Crops</h1>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(images)}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
    """
    
    for img in images:
        frame_num = ''.join(filter(str.isdigit, img.split('.')[0]))
        
        html += f"""
            <div class="item">
                <img src="/crops/current/{img}" alt="{img}" loading="lazy">
                <div class="filename">{img}</div>
                <div class="frame-num">Frame #{frame_num}</div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    تحليل الفيديو وإرجاع عدد الأكشنز
    """
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        try:
            x = float(request.form.get('x', 0))
            y = float(request.form.get('y', 0))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid x or y coordinates'
            }), 400
        
        print(f"📹 Received video: {video_file.filename}")
        print(f"📍 Normalized coordinates: x={x:.3f}, y={y:.3f}")
        
        work_dir = os.path.join(DEBUG_DIR, 'current')
        
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        os.makedirs(work_dir, exist_ok=True)
        print(f"📁 Working directory: {work_dir}")
        
        try:
            video_path = os.path.join(work_dir, 'input_video.mp4')
            video_file.save(video_path)
            print(f"✅ Video saved: {video_path}")
            
            crops_dir = os.path.join(work_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            print(f"📸 Crops will be saved to: {crops_dir}")
            
            main_script = os.path.join(BASE_DIR, 'main.py')
            
            cmd = [
                sys.executable,
                main_script,
                '--video-path', video_path,
                '--target-xy', str(x), str(y),
                '--crop-dir', crops_dir,
                '--pose-weights', POSE_WEIGHTS,
                '--mlp-weights', MLP_WEIGHTS,
                '--zoom', '1.3',
                '--crop-size', '224'
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=BASE_DIR
            )
            
            print(f"📤 Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"❌ STDERR:\n{result.stderr}")
                return jsonify({
                    'success': False,
                    'error': 'Processing failed',
                    'details': result.stderr[-1000:]
                }), 500
            
            output_lines = result.stdout.strip().split('\n')
            action_counts = {}
            
            print("📊 Output lines:")
            for line in output_lines:
                print(f"  {line}")
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        action = parts[0].strip()
                        try:
                            count = int(parts[1].strip())
                            action_counts[action] = count
                        except ValueError:
                            pass
            
            if not action_counts:
                print("⚠️  No action counts found in output")
                action_counts = {
                    'dribble': 0,
                    'pass': 0,
                    'shoot': 0
                }
            
            print(f"✅ Final counts: {action_counts}")
            
            crop_count = 0
            if os.path.exists(crops_dir):
                crop_files = [f for f in os.listdir(crops_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                crop_count = len(crop_files)
            
            print(f"📸 Total crops saved: {crop_count}")
            print(f"🌐 View crops at: /view-crops/current")
            
            base_url = request.host_url.rstrip('/')
            crops_url = f"{base_url}/view-crops/current"
            
            return jsonify({
                'success': True,
                'action_counts': action_counts,
                'target_coordinates': {'x': x, 'y': y},
                'crops_url': crops_url,
                'total_crops': crop_count,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            print(traceback.format_exc())
            raise
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
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
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )    models_needed = {
        'best.pt': DETECTION_WEIGHTS,
        'best1.pt': POSE_WEIGHTS,
        'action_mlp.pt': MLP_WEIGHTS
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_filename, local_path in models_needed.items():
        if not os.path.exists(local_path):
            print(f"📥 Downloading {model_filename} from Hugging Face...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=model_filename,
                    cache_dir=None  # استخدام الـcache الافتراضي
                )
                
                # نسخ للمكان الصحيح
                shutil.copy(downloaded_path, local_path)
                print(f"✅ Downloaded {model_filename} to {local_path}")
            except Exception as e:
                print(f"❌ Error downloading {model_filename}: {e}")
                print(f"   Make sure the model exists at: https://huggingface.co/{HF_REPO_ID}")
                raise
        else:
            print(f"✅ {model_filename} already exists")

# تحميل المودلات عند بداية التشغيل
print("🔍 Checking models...")
ensure_models_downloaded()

# إنشاء مجلد debug_output 🔥 إضافة جديدة
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===== API Endpoints =====

@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        'service': 'Haddaf Action Recognition API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Check server status',
            '/analyze': 'Analyze football video (POST)',
            '/view-crops/current': 'View crop images in browser'  # 🔥 إضافة جديدة
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """للتحقق من أن السيرفر يعمل"""
    models_status = {
        'detection_model': os.path.exists(DETECTION_WEIGHTS),
        'pose_model': os.path.exists(POSE_WEIGHTS),
        'mlp_model': os.path.exists(MLP_WEIGHTS)
    }
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': models_status,
        'python_version': sys.version
    })

# 🔥 Endpoint جديد: عرض صورة واحدة
@app.route('/crops/current/<path:filename>')
def serve_crop(filename):
    """عرض صورة معينة"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    return send_from_directory(crops_path, filename)

# 🔥 Endpoint جديد: عرض كل الصور في HTML
@app.route('/view-crops/current')
def view_crops():
    """عرض كل الصور في صفحة HTML"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    
    if not os.path.exists(crops_path):
        return f"<h1>❌ No crops folder found</h1><p>{crops_path}</p>", 404
    
    # قائمة الصور
    images = sorted([f for f in os.listdir(crops_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not images:
        return f"<h1>⚠️ No images found</h1><p>{crops_path}</p>", 404
    
    # HTML بسيط وجميل
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
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                min-height: 100vh;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ 
                color: #4CAF50; 
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
            }}
            .stats {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
            }}
            .stat {{
                background: rgba(76, 175, 80, 0.1);
                padding: 15px 30px;
                border-radius: 10px;
                border: 2px solid rgba(76, 175, 80, 0.3);
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .stat-label {{
                color: #aaa;
                margin-top: 5px;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
                gap: 25px;
                margin-top: 30px;
            }}
            .item {{ 
                background: rgba(255,255,255,0.08);
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
                border-color: rgba(76, 175, 80, 0.5);
            }}
            img {{ 
                width: 100%;
                height: 280px;
                object-fit: cover;
                border-radius: 8px;
                border: 2px solid rgba(76, 175, 80, 0.2);
            }}
            .filename {{ 
                margin-top: 12px;
                font-size: 13px;
                color: #4CAF50;
                font-family: 'Courier New', monospace;
            }}
            .frame-num {{
                display: inline-block;
                background: rgba(76, 175, 80, 0.2);
                padding: 3px 8px;
                border-radius: 5px;
                margin-top: 5px;
                font-size: 11px;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Target Player Crops</h1>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(images)}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
    """
    
    for img in images:
        # استخراج رقم الفريم
        frame_num = ''.join(filter(str.isdigit, img.split('.')[0]))
        
        html += f"""
            <div class="item">
                <img src="/crops/current/{img}" alt="{img}" loading="lazy">
                <div class="filename">{img}</div>
                <div class="frame-num">Frame #{frame_num}</div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    تحليل الفيديو وإرجاع عدد الأكشنز
    
    Parameters:
    - video (file): ملف الفيديو
    - x (float): إحداثية x للاعب (normalized 0-1)
    - y (float): إحداثية y للاعب (normalized 0-1)
    
    Returns:
    - action_counts: {dribble: X, pass: Y, shoot: Z}
    """
    try:
        # 1. التحقق من البيانات
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # قراءة الإحداثيات (normalized 0-1)
        try:
            x = float(request.form.get('x', 0))
            y = float(request.form.get('y', 0))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid x or y coordinates'
            }), 400
        
        print(f"📹 Received video: {video_file.filename}")
        print(f"📍 Normalized coordinates: x={x:.3f}, y={y:.3f}")
        
        # 2. مجلد ثابت واحد 🔥 التعديل الرئيسي
        work_dir = os.path.join(DEBUG_DIR, 'current')
        
        # امسح المجلد القديم واعمل واحد جديد
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        os.makedirs(work_dir, exist_ok=True)
        print(f"📁 Working directory: {work_dir}")
        
        try:
            # حفظ الفيديو
            video_path = os.path.join(work_dir, 'input_video.mp4')
            video_file.save(video_path)
            print(f"✅ Video saved: {video_path}")
            
            # مجلد القصّات
            crops_dir = os.path.join(work_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            print(f"📸 Crops will be saved to: {crops_dir}")
            
            # 3. تشغيل main.py
            main_script = os.path.join(BASE_DIR, 'main.py')
            
            cmd = [
                sys.executable,
                main_script,
                '--video-path', video_path,
                '--target-xy', str(x), str(y),
                '--crop-dir', crops_dir,
                '--pose-weights', POSE_WEIGHTS,
                '--mlp-weights', MLP_WEIGHTS,
                '--zoom', '1.3',
                '--crop-size', '224'
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 دقيقة
                cwd=BASE_DIR
            )
            
            print(f"📤 Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"❌ STDERR:\n{result.stderr}")
                return jsonify({
                    'success': False,
                    'error': 'Processing failed',
                    'details': result.stderr[-1000:]
                }), 500
            
            # 4. استخراج النتائج من stdout
            output_lines = result.stdout.strip().split('\n')
            action_counts = {}
            
            print("📊 Output lines:")
            for line in output_lines:
                print(f"  {line}")
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        action = parts[0].strip()
                        try:
                            count = int(parts[1].strip())
                            action_counts[action] = count
                        except ValueError:
                            pass
            
            if not action_counts:
                print("⚠️  No action counts found in output")
                action_counts = {
                    'dribble': 0,
                    'pass': 0,
                    'shoot': 0
                }
            
            print(f"✅ Final counts: {action_counts}")
            
            # 🔥 عد الصور
            crop_count = 0
            if os.path.exists(crops_dir):
                crop_files = [f for f in os.listdir(crops_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                crop_count = len(crop_files)
            
            print(f"📸 Total crops saved: {crop_count}")
            print(f"🌐 View crops at: /view-crops/current")
            
            # 5. إرجاع النتائج 🔥 مع رابط الصور
            base_url = request.host_url.rstrip('/')
            crops_url = f"{base_url}/view-crops/current"
            
            return jsonify({
                'success': True,
                'action_counts': action_counts,
                'target_coordinates': {'x': x, 'y': y},
                'crops_url': crops_url,  # 🔥 إضافة جديدة
                'total_crops': crop_count,  # 🔥 إضافة جديدة
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            print(traceback.format_exc())
            raise
            
        # 🔥 حذفنا الـ finally block اللي كان يمسح الملفات
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ===== Error Handlers =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ===== Main =====

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 Haddaf Backend Server Starting...")
    print("=" * 60)
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🤖 Models directory: {MODELS_DIR}")
    print(f"📸 Debug output directory: {DEBUG_DIR}")  # 🔥 إضافة جديدة
    print(f"✅ Detection model: {os.path.exists(DETECTION_WEIGHTS)}")
    print(f"✅ Pose model: {os.path.exists(POSE_WEIGHTS)}")
    print(f"✅ MLP model: {os.path.exists(MLP_WEIGHTS)}")
    print("=" * 60)
    
    # الحصول على PORT من البيئة (مهم للـCloud)
    port = int(os.environ.get('PORT', 5000))
    
    # تشغيل السيرفر
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )    models_needed = {
        'best.pt': DETECTION_WEIGHTS,
        'best1.pt': POSE_WEIGHTS,
        'action_mlp.pt': MLP_WEIGHTS
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_filename, local_path in models_needed.items():
        if not os.path.exists(local_path):
            print(f"📥 Downloading {model_filename} from Hugging Face...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=model_filename,
                    cache_dir=None  # استخدام الـcache الافتراضي
                )
                
                # نسخ للمكان الصحيح
                shutil.copy(downloaded_path, local_path)
                print(f"✅ Downloaded {model_filename} to {local_path}")
            except Exception as e:
                print(f"❌ Error downloading {model_filename}: {e}")
                print(f"   Make sure the model exists at: https://huggingface.co/{HF_REPO_ID}")
                raise
        else:
            print(f"✅ {model_filename} already exists")

# تحميل المودلات عند بداية التشغيل
print("🔍 Checking models...")
ensure_models_downloaded()

# إنشاء مجلد debug_output 🔥 إضافة جديدة
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===== API Endpoints =====

@app.route('/', methods=['GET'])
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        'service': 'Haddaf Action Recognition API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Check server status',
            '/analyze': 'Analyze football video (POST)',
            '/view-crops/current': 'View crop images in browser'  # 🔥 إضافة جديدة
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """للتحقق من أن السيرفر يعمل"""
    models_status = {
        'detection_model': os.path.exists(DETECTION_WEIGHTS),
        'pose_model': os.path.exists(POSE_WEIGHTS),
        'mlp_model': os.path.exists(MLP_WEIGHTS)
    }
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': models_status,
        'python_version': sys.version
    })

# 🔥 Endpoint جديد: عرض صورة واحدة
@app.route('/crops/current/<path:filename>')
def serve_crop(filename):
    """عرض صورة معينة"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    return send_from_directory(crops_path, filename)

# 🔥 Endpoint جديد: عرض كل الصور في HTML
@app.route('/view-crops/current')
def view_crops():
    """عرض كل الصور في صفحة HTML"""
    crops_path = os.path.join(DEBUG_DIR, 'current', 'crops')
    
    if not os.path.exists(crops_path):
        return f"<h1>❌ No crops folder found</h1><p>{crops_path}</p>", 404
    
    # قائمة الصور
    images = sorted([f for f in os.listdir(crops_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not images:
        return f"<h1>⚠️ No images found</h1><p>{crops_path}</p>", 404
    
    # HTML بسيط وجميل
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
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                min-height: 100vh;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ 
                color: #4CAF50; 
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
            }}
            .stats {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
            }}
            .stat {{
                background: rgba(76, 175, 80, 0.1);
                padding: 15px 30px;
                border-radius: 10px;
                border: 2px solid rgba(76, 175, 80, 0.3);
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .stat-label {{
                color: #aaa;
                margin-top: 5px;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
                gap: 25px;
                margin-top: 30px;
            }}
            .item {{ 
                background: rgba(255,255,255,0.08);
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
                border-color: rgba(76, 175, 80, 0.5);
            }}
            img {{ 
                width: 100%;
                height: 280px;
                object-fit: cover;
                border-radius: 8px;
                border: 2px solid rgba(76, 175, 80, 0.2);
            }}
            .filename {{ 
                margin-top: 12px;
                font-size: 13px;
                color: #4CAF50;
                font-family: 'Courier New', monospace;
            }}
            .frame-num {{
                display: inline-block;
                background: rgba(76, 175, 80, 0.2);
                padding: 3px 8px;
                border-radius: 5px;
                margin-top: 5px;
                font-size: 11px;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Target Player Crops</h1>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(images)}</div>
                    <div class="stat-label">Total Frames</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
    """
    
    for img in images:
        # استخراج رقم الفريم
        frame_num = ''.join(filter(str.isdigit, img.split('.')[0]))
        
        html += f"""
            <div class="item">
                <img src="/crops/current/{img}" alt="{img}" loading="lazy">
                <div class="filename">{img}</div>
                <div class="frame-num">Frame #{frame_num}</div>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    تحليل الفيديو وإرجاع عدد الأكشنز
    
    Parameters:
    - video (file): ملف الفيديو
    - x (float): إحداثية x للاعب
    - y (float): إحداثية y للاعب
    
    Returns:
    - action_counts: {dribble: X, pass: Y, shoot: Z}
    """
    try:
        # 1. التحقق من البيانات
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # قراءة الإحداثيات
        try:
            x = float(request.form.get('x', 0))
            y = float(request.form.get('y', 0))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid x or y coordinates'
            }), 400
        
        print(f"📹 Received video: {video_file.filename}")
        print(f"📍 Target coordinates: x={x}, y={y}")
        
        # 2. مجلد ثابت واحد 🔥 التعديل الرئيسي
        work_dir = os.path.join(DEBUG_DIR, 'current')
        
        # امسح المجلد القديم واعمل واحد جديد
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        os.makedirs(work_dir, exist_ok=True)
        print(f"📁 Working directory: {work_dir}")
        
        try:
            # حفظ الفيديو
            video_path = os.path.join(work_dir, 'input_video.mp4')
            video_file.save(video_path)
            print(f"✅ Video saved: {video_path}")
            
            # مجلد القصّات
            crops_dir = os.path.join(work_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            print(f"📸 Crops will be saved to: {crops_dir}")
            
            # 3. تشغيل main.py
            main_script = os.path.join(BASE_DIR, 'main.py')
            
            cmd = [
                sys.executable,
                main_script,
                '--video-path', video_path,
                '--target-xy', str(x), str(y),
                '--crop-dir', crops_dir,
                '--pose-weights', POSE_WEIGHTS,
                '--mlp-weights', MLP_WEIGHTS,
                '--zoom', '1.3',
                '--crop-size', '224'
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 دقيقة
                cwd=BASE_DIR
            )
            
            print(f"📤 Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"❌ STDERR:\n{result.stderr}")
                return jsonify({
                    'success': False,
                    'error': 'Processing failed',
                    'details': result.stderr[-1000:]
                }), 500
            
            # 4. استخراج النتائج من stdout
            output_lines = result.stdout.strip().split('\n')
            action_counts = {}
            
            print("📊 Output lines:")
            for line in output_lines:
                print(f"  {line}")
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        action = parts[0].strip()
                        try:
                            count = int(parts[1].strip())
                            action_counts[action] = count
                        except ValueError:
                            pass
            
            if not action_counts:
                print("⚠️  No action counts found in output")
                action_counts = {
                    'dribble': 0,
                    'pass': 0,
                    'shoot': 0
                }
            
            print(f"✅ Final counts: {action_counts}")
            
            # 🔥 عد الصور
            crop_count = 0
            if os.path.exists(crops_dir):
                crop_files = [f for f in os.listdir(crops_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                crop_count = len(crop_files)
            
            print(f"📸 Total crops saved: {crop_count}")
            print(f"🌐 View crops at: /view-crops/current")
            
            # 5. إرجاع النتائج 🔥 مع رابط الصور
            base_url = request.host_url.rstrip('/')
            crops_url = f"{base_url}/view-crops/current"
            
            return jsonify({
                'success': True,
                'action_counts': action_counts,
                'target_coordinates': {'x': x, 'y': y},
                'crops_url': crops_url,  # 🔥 إضافة جديدة
                'total_crops': crop_count,  # 🔥 إضافة جديدة
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            print(traceback.format_exc())
            raise
            
        # 🔥 حذفنا الـ finally block اللي كان يمسح الملفات
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ===== Error Handlers =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ===== Main =====

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 Haddaf Backend Server Starting...")
    print("=" * 60)
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🤖 Models directory: {MODELS_DIR}")
    print(f"📸 Debug output directory: {DEBUG_DIR}")  # 🔥 إضافة جديدة
    print(f"✅ Detection model: {os.path.exists(DETECTION_WEIGHTS)}")
    print(f"✅ Pose model: {os.path.exists(POSE_WEIGHTS)}")
    print(f"✅ MLP model: {os.path.exists(MLP_WEIGHTS)}")
    print("=" * 60)
    
    # الحصول على PORT من البيئة (مهم للـCloud)
    port = int(os.environ.get('PORT', 5000))
    
    # تشغيل السيرفر
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
