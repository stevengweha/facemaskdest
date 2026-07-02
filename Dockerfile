FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installer les dépendances
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port utilisé par uvicorn (Hugging Face Spaces préfère 7860/8080)
EXPOSE 7860

# Commande de lancement (modifiez si votre app est différente)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
