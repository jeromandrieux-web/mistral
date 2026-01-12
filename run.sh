#!/usr/bin/env bash
set -e

# Se placer dans le dossier du script (important si tu le lances par double-clic)
cd "$(dirname "$0")"

# --------- Chargement des variables d’environnement ------------
if [ -f ".env" ]; then
  echo "[INFO] Chargement du fichier .env"
  export $(grep -v '^#' .env | xargs)
else
  echo "[WARN] Aucun fichier .env trouvé, lancement avec variables système."
fi

# --------- Création & activation du venv -----------------------
if [ ! -d ".venv" ]; then
  echo "[INFO] Création de l'environnement virtuel .venv"
  python3 -m venv .venv
fi

# Activation du venv
# shellcheck disable=SC1091
source .venv/bin/activate

# --------- Installation des requirements -----------------------
if [ ! -f ".venv/installed.flag" ]; then
  echo "[INFO] Installation des dépendances"
  pip install --upgrade pip
  pip install -r requirements.txt
  touch .venv/installed.flag
else
  echo "[INFO] Dépendances déjà installées."
fi

echo "[INFO] Lancement de Uvicorn sur http://127.0.0.1:8000"

# --------- Vérifie si le port est déjà utilisé -----------------------
if lsof -i :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
  PID=$(lsof -i :8000 -sTCP:LISTEN -t | head -n 1)
  echo "[ERROR] Le port 8000 est déjà utilisé (PID=$PID)."
  echo "        Ferme l'autre serveur ou lance: kill -9 $PID"
  exit 1
fi

# On lance uvicorn en arrière-plan
uvicorn main:app --host 0.0.0.0 --port 8000 &


UVICORN_PID=$!

# Petite pause pour le démarrage
sleep 1

# Ouverture automatique du navigateur sur la bonne URL
open "http://127.0.0.1:8000/"

# On attend que uvicorn se termine (Ctrl+C)
wait "$UVICORN_PID"
