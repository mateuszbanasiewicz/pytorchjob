FROM registry.redhat.io/rhoai/odh-training-cuda128-torch28-py312-rhel9:v3.0

WORKDIR /workspace

# Zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skrypt treningowy
COPY train.py .

# Domyślne uruchomienie
CMD ["python", "train.py"]
