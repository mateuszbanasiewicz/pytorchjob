FROM registry.redhat.io/rhoai/odh-training-cuda128-torch28-py312-rhel9:v3.0

WORKDIR /workspace

# Zależności systemowe
RUN yum install -y wget curl git \
    && yum clean all \
    && rm -rf /var/cache/yum

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
