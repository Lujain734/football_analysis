# استخدام Ubuntu مع Python
FROM python:3.11-slim

# تثبيت المكتبات المطلوبة للنظام
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# مجلد العمل
WORKDIR /app

# نسخ الملفات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# المنفذ
EXPOSE 8080

# تشغيل السيرفر
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "1800"]
