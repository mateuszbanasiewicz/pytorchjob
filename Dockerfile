FROM registry.redhat.io/rhoai/odh-training-cuda128-torch28-py312-rhel9:v3.0

WORKDIR /workspace

# Zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skrypt treningowy
COPY train.py .

# OpenShift: kontenery działają jako losowy UID — upewnij się że katalogi są dostępne
RUN mkdir -p /data /output \
    && chmod -R 777 /data /output /workspace

# Domyślne uruchomienie
CMD ["python", "train.py"]
