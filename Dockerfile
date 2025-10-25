FROM python:3.11-slim

# تحديث وتثبيت المكتبات (أسماء جديدة)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "1800"]
