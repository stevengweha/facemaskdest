# FaceMask Web Page

Ce dépôt contient une petite application Python (FastAPI) pour la détection de masque facial.
Les fichiers importants présents : `app.py`, `best.onnx`, `best.pt`, le dossier `static/` et `template/`.

## Fichiers ajoutés
- Dockerfile : image pour déployer l'application (utile pour Hugging Face Spaces avec runtime Docker).
- requirements.txt : dépendances Python nécessaires.
- README.md : ce fichier d'instructions.

## Construire et exécuter localement (Docker)

```bash
# depuis la racine du projet
docker build -t facemask-app .

docker run --rm -p 7860:7860 \
  --name facemask-app \
  facemask-app
```

## Lancer sans Docker (local)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Publier sur Hugging Face Spaces
1. Créez un nouveau Space sur https://huggingface.co/spaces en choisissant "Docker" comme runtime.
2. Poussez votre dépôt (avec `Dockerfile` inclus) vers le repo du Space. Le Space construira l'image automatiquement.

Exemple minimal :

```bash
git init
git add .
git commit -m "Add Dockerfile and requirements for HF Space"
git remote add origin https://huggingface.co/spaces/<votre-compte>/<nom-du-space>.git
git push origin main
```

Remarques:
- Si votre application n'utilise pas `uvicorn` comme point d'entrée, modifiez la commande dans le `Dockerfile`.
- Si vous avez besoin d'une image avec CUDA (GPU), adaptez la base du `Dockerfile` en conséquence et vérifiez la disponibilité GPU sur le Space.

Si vous voulez, je peux aussi :
- Ajouter des versions précises aux dépendances.
- Générer un `.dockerignore` pour réduire l'image.
- Préparer le dépôt pour le push (branch, commit, instructions personnalisées).
