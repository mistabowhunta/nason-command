FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System libs for OpenCV video decoding + graphics libraries for ultralytics export
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics onnx

# Generate yolov8s.onnx natively via Python on disk during build
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); model.export(format='onnx', imgsz=640)" && \
    mkdir -p /app/models && \
    mv yolov8s.onnx /app/models/yolov8s.onnx

COPY ./app /app/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]