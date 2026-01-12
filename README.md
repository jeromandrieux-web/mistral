# Analyste PDF — Installation OCR (macOS)

Prérequis système (macOS Homebrew) :
- brew install poppler tesseract

(Optionnel) installer le modèle de langue français pour Tesseract :
- curl -L -o fra.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata
- sudo mv fra.traineddata /usr/local/share/tessdata/

Installation Python :
1. Créer un environnement virtuel et l'activer
   - python3 -m venv .venv
   - source .venv/bin/activate

2. Installer les dépendances
   - pip install -r requirements.txt

Lancer l'API en développement :
- uvicorn main:app --reload --host 0.0.0.0 --port 8000

Variables d'environnement utiles :
- OLLAMA_BASE_URL (ex: http://127.0.0.1:11434)
- OLLAMA_API_KEY (si nécessaire)

Notes OCR :
- pdf2image nécessite `poppler` (brew install poppler).
- pytesseract nécessite `tesseract` (brew install tesseract) et les fichiers de langue (tessdata) pour de meilleurs résultats.
- Ajuster la valeur `dpi` dans `ocr_pdf_bytes()` si la qualité d'OCR est insuffisante (300 recommandé pour les scans lisibles).

Si tu veux, j'ajoute ces fichiers directement dans le dépôt.  