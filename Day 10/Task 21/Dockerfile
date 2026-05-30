FROM python:3.11-slim

WORKDIR /Task 21

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -r requirements.txt 

COPY train/weights/best.pt .
COPY predict.py .

CMD ["python", "predict.py"]