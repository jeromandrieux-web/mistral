#!/usr/bin/env bash
set -e

echo "=== Axion PDF · Install ==="

# 1) Vérif Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERREUR] python3 introuvable. Installe Python 3.x d'abord."
  exit 1
fi

# 2) Création du venv
if [ ! -d ".venv" ]; then
  echo "[INFO] Création de l'environnement virtuel .venv"
  python3 -m venv .venv
else
  echo "[INFO] Environnement .venv déjà présent"
fi

# 3) Activation du venv
echo "[INFO] Activation de .venv"
# shellcheck disable=SC1091
source .venv/bin/activate

# 4) Installation des dépendances
if [ ! -f "requirements.txt" ]; then
  echo "[ERREUR] requirements.txt introuvable dans le dossier courant."
  exit 1
fi

echo "[INFO] Installation / mise à jour des dépendances"
pip install --upgrade pip
pip install -r requirements.txt

# 5) .env de base (si tu ne l'as pas déjà)
if [ ! -f ".env" ]; then
  echo "[INFO] Création d'un .env par défaut"
  cat > .env << 'EOF'
############################################
# Endpoints / LLM
############################################
OLLAMA_LOCAL_URL=http://127.0.0.1:11434
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_HOST=http://127.0.0.1:11434

# Si Ollama nécessite un bearer, ce sera lu par le backend
OLLAMA_API_KEY=dummy

# Liste des modèles servis par Ollama
OLLAMA_MODELS=mistral-large-3:675b-cloud
OLLAMA_DEFAULT_MODEL=mistral-large-3:675b-cloud
CHAT_ALLOWED_MODELS=mistral-large-3:675b-cloud

############################################
# Paramètres globaux de génération
############################################
OLLAMA_NUM_CTX=65536
OLLAMA_NUM_PREDICT=768
OLLAMA_TEMPERATURE=0.2
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40

# Nombre de tokens de sortie par défaut côté ChatService / profils
DEFAULT_MAX_TOKENS=3072

MODEL_NAME=mistral-large-3:675b-cloud
LLM_API_BASE_URL=http://localhost:11434/v1/chat/completions
EOF
else
  echo "[INFO] .env déjà présent, je ne le touche pas."
fi

# 6) Rendre run.sh exécutable
if [ -f "run.sh" ]; then
  chmod +x run.sh
  echo "[INFO] run.sh est maintenant exécutable."
else
  echo "[WARN] run.sh introuvable, je ne peux pas le rendre exécutable."
fi

# 7) Rendre install.sh exécutable lui-même (au cas où)
chmod +x install.sh

echo
echo "=== Installation terminée ==="
echo "Pour lancer le serveur :"
echo "  ./run.sh"
