from huggingface_hub import HfApi
import os

# معلوماتك
USERNAME = "lujain-721"  
REPO_NAME = "haddaf-models"

api = HfApi()

# المودلات اللي نبي نرفعها
models_to_upload = [
    "models/best.pt",
    "models/best1.pt", 
    "models/action_mlp.pt"
]

print("🚀 Starting upload to Hugging Face...")
print(f"📦 Repository: {USERNAME}/{REPO_NAME}")
print("=" * 50)

for model_path in models_to_upload:
    if os.path.exists(model_path):
        filename = os.path.basename(model_path)
        print(f"\n📤 Uploading {filename}...")
        
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=f"{USERNAME}/{REPO_NAME}",
                repo_type="model"
            )
            print(f"✅ {filename} uploaded successfully!")
        except Exception as e:
            print(f"❌ Error uploading {filename}: {e}")
    else:
        print(f"⚠️  {model_path} not found, skipping...")

print("\n" + "=" * 50)
print("🎉 Upload complete!")
print(f"🔗 View models at: https://huggingface.co/{USERNAME}/{REPO_NAME}")