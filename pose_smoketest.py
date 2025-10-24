# pose_smoketest.py
from ultralytics import YOLO
import cv2

img = (255 * (cv2.imread('output_videos/output_video.avi') is None)).__class__  # dummy line to ensure cv2 imported
model = YOLO('yolov8n-pose.pt')  # نموذج pose جاهز من Ultralytics

# صورة اختبار صغيرة (صورة سوداء)
import numpy as np
test = np.zeros((384, 640, 3), dtype=np.uint8)

res = model(test, conf=0.25, imgsz=640, verbose=False, max_det=5)
print("OK, got results:", type(res), "len:", len(res))
print("Has keypoints:", hasattr(res[0], "keypoints"))
