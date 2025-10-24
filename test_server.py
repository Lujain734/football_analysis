import requests
import os

print("=" * 60)
print("🧪 Testing Haddaf Backend Server")
print("=" * 60)

# 1. اختبار Health
print("\n1️⃣ Testing /health endpoint...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   ✅ Status Code: {response.status_code}")
    print(f"   📄 Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. اختبار Analyze
print("\n2️⃣ Testing /analyze endpoint...")

# غيري المسار للفيديو عندك
video_path = "input_videos/A1606b0e6_0 (10).mp4"  # تأكدي من الامتداد

if os.path.exists(video_path):
    print(f"   📹 Found video: {video_path}")
    print("   ⏳ Uploading and analyzing... (this may take 1-2 minutes)")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'x': 939, 'y': 505}
            
            response = requests.post(
                "http://localhost:8000/analyze",
                files=files,
                data=data,
                timeout=1800
            )
        
        print(f"   ✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   📊 Action Counts:")
            for action, count in result.get('action_counts', {}).items():
                print(f"      - {action}: {count}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print(f"   ⚠️  Video not found: {video_path}")
    print(f"   💡 Current directory: {os.getcwd()}")
    print(f"   💡 Try full path like: /Users/alhussan/Desktop/football_analysis/input_videos/video.mp4")

print("\n" + "=" * 60)
print("✅ Testing complete!")
print("=" * 60)