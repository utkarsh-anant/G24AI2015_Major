FROM python:3.10-slim

WORKDIR /app

# install system deps for pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# copy requirements first to leverage docker cache
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# copy app and model
COPY app.py .
# NOTE: savedmodel.pth must be present in the build context so it is copied into image
COPY savedmodel.pth .

EXPOSE 5000

CMD ["python", "app.py"]
