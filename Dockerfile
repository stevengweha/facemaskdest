# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8080

# Commande pour exécuter l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]