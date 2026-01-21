FROM python:3.11-slim

# Installer FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Créer le dossier de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY main.py .

# Créer le dossier pour les fichiers temporaires
RUN mkdir -p /tmp/audio

# Exposer le port
EXPOSE 8000

# Démarrer le serveur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
