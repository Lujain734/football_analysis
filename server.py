#!/usr/bin/env python3
"""
Haddaf Backend Server - Football Action Recognition
"""

from flask import Flask, request, jsonify
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
HF_REPO_ID = "lujain-721/haddaf-models"  # ✅ حسابك في Hugging Face

# ===== إعدادات المسارات =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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
            '/analyze': 'Analyze football video (POST)'
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
        
        # 2. إنشاء مجلد مؤقت
        temp_dir = tempfile.mkdtemp(prefix='haddaf_')
        print(f"📁 Working directory: {temp_dir}")
        
        try:
            # حفظ الفيديو
            video_path = os.path.join(temp_dir, 'input_video.mp4')
            video_file.save(video_path)
            print(f"✅ Video saved: {video_path}")
            
            # مجلد القصّات
            crops_dir = os.path.join(temp_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            
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
            
            # 5. إرجاع النتائج
            return jsonify({
                'success': True,
                'action_counts': action_counts,
                'target_coordinates': {'x': x, 'y': y},
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            # 6. تنظيف الملفات المؤقتة
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"🧹 Cleaned up: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
    
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