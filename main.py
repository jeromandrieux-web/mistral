from __future__ import annotations

import os
import uuid
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO
from dataclasses import dataclass, field

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# DOCX
try:
    from docx import Document
except Exception:
    Document = None

# Ajout pour OCR
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    convert_from_bytes = None
    pytesseract = None

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import re


# ========= ANONYMISATION =========

@dataclass
class AnonContext:
    real2token: Dict[str, str] = field(default_factory=dict)
    token2real: Dict[str, str] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=lambda: {
        "PERSON": 0, "NAME": 0, "PLATE": 0, "EMAIL": 0, "PHONE": 0, "IBAN": 0
    })

    def token_for(self, kind: str, real: str) -> str:
        if real in self.real2token:
            return self.real2token[real]
        self.counters[kind] = self.counters.get(kind, 0) + 1
        token = f"__{kind}_{self.counters[kind]}__"
        self.real2token[real] = token
        self.token2real[token] = real
        return token


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+33|0)\s*(?:\d[\s.-]?){8,}\d\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b", re.IGNORECASE)

TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])\s*(?:[:hH]\s*)[0-5]\d\b")
PLATE_FR_RE = re.compile(r"\b[A-Z]{2}[\s.-]?\d{3}[\s.-]?[A-Z]{2}\b", re.IGNORECASE)
PLATE_OLD_FR_RE = re.compile(r"\b\d{1,4}[\s.-]?[A-Z]{2,3}\b", re.IGNORECASE)
PLATE_CONTEXT_RE = re.compile(r"\b(immatriculation|immat|plaque|vehicule|véhicule|voiture|auto|moto|carte\s*grise)\b", re.IGNORECASE)
PLATE_NEG_CONTEXT_RE = re.compile(r"\b(localisation|géolocalisation|geolocalisation|horaires?|heure|timestamp|antenne|cellule|imei|imsi|msisdn)\b", re.IGNORECASE)

PARTICLES = r"(?:DE|DU|DES|D'|D’|LA|LE|LES|DEL|DA|DI|VAN|VON)"
NAME_WORD = r"[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’\-]{1,}"
FIRST_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-þ'’\-]{1,}|[A-ZÀ-ÖØ-Þ]{2,}"
IDENTITY_LABELS_RE = re.compile(r"\b(nom|prénom|prenom|identit[eé]|etat\s*civil|état\s*civil|suspect|mis\s+en\s+cause|victime|t[eé]moin|interpell[eé])\b", re.IGNORECASE)
TITLE_RE = re.compile(r"\b(M\.|MME|MADAME|MONSIEUR|MLLE|Mademoiselle)\b", re.IGNORECASE)
FIELD_NOM_RE = re.compile(r"(?im)\bNom\s*:\s*(?P<value>.+)$")
FIELD_PRENOM_RE = re.compile(r"(?im)\bPr[ée]nom\s*:\s*(?P<value>.+)$")
PERSON_INLINE_RE = re.compile(rf"\b(?P<first>{FIRST_WORD})(?:\s+)(?P<last>(?:{PARTICLES}\s+)?{NAME_WORD}(?:\s+(?:{PARTICLES}\s+)?{NAME_WORD})?)\b")
PERSON_INVERTED_RE = re.compile(rf"\b(?P<last>(?:{PARTICLES}\s+)?{NAME_WORD}(?:\s+(?:{PARTICLES}\s+)?{NAME_WORD})?)\s+(?P<first>{FIRST_WORD})\b")

DENY_ALLCAPS = {
    "VOL", "VIOLENCES", "STUPEFIANTS", "STUPÉFIANTS", "ARME", "MENACES",
    "AUDITION", "PROCEDURE", "PROCÉDURE", "ENQUETE", "ENQUÊTE", "GARDE", "VUE",
    "GAV", "OPJ", "PJ", "BTA", "IGGN", "ITT", "RAPPORT", "NOTE",
    "FRANCE", "PARIS", "LYON", "MARSEILLE",
}


def _looks_like_non_person(token: str) -> bool:
    t = token.strip().strip(",;:.()[]{}").upper()
    if not t:
        return True
    if t in DENY_ALLCAPS:
        return True
    if len(t) <= 2:
        return True
    return False


def anonymize_plates(text: str, ctx: AnonContext, window_lines: int = 1) -> str:
    lines = text.splitlines()
    out_lines: List[str] = []
    for i, line in enumerate(lines):
        if not (PLATE_FR_RE.search(line) or PLATE_OLD_FR_RE.search(line)):
            out_lines.append(line)
            continue

        lo = max(0, i - window_lines)
        hi = min(len(lines), i + window_lines + 1)
        context_blob = "\n".join(lines[lo:hi])
        has_pos = bool(PLATE_CONTEXT_RE.search(context_blob))
        has_neg = bool(PLATE_NEG_CONTEXT_RE.search(context_blob))

        def repl(m: re.Match) -> str:
            raw = m.group(0)
            if TIME_RE.search(raw):
                return raw
            if not (re.search(r"[A-Z]", raw, re.IGNORECASE) and re.search(r"\d", raw)):
                return raw
            if not has_pos or has_neg:
                return raw
            return ctx.token_for("PLATE", raw)

        line2 = PLATE_FR_RE.sub(repl, line)
        if has_pos and not has_neg:
            line2 = PLATE_OLD_FR_RE.sub(repl, line2)
        out_lines.append(line2)

    return "\n".join(out_lines)


def anonymize_persons(text: str, ctx: AnonContext, window_lines: int = 1) -> str:
    lines = text.splitlines()
    out_lines: List[str] = []
    speaker_re = re.compile(r"(?m)^(?P<speaker>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]{1,})\s*:\s+")

    for i, line in enumerate(lines):
        lo = max(0, i - window_lines)
        hi = min(len(lines), i + window_lines + 1)
        context_blob = "\n".join(lines[lo:hi])
        has_identity_context = bool(IDENTITY_LABELS_RE.search(context_blob) or TITLE_RE.search(context_blob))

        def repl_nom_field(m: re.Match) -> str:
            value = m.group("value").strip()
            head = re.split(r"[;,/|]", value)[0].strip()
            if _looks_like_non_person(head):
                return m.group(0)
            tok = ctx.token_for("NAME", head)
            return m.group(0).replace(head, tok, 1)

        def repl_prenom_field(m: re.Match) -> str:
            value = m.group("value").strip()
            head = re.split(r"[;,/|]", value)[0].strip()
            if _looks_like_non_person(head):
                return m.group(0)
            tok = ctx.token_for("NAME", head)
            return m.group(0).replace(head, tok, 1)

        def repl_speaker(m: re.Match) -> str:
            sp = m.group("speaker")
            if _looks_like_non_person(sp):
                return m.group(0)
            tok = ctx.token_for("NAME", sp)
            return m.group(0).replace(sp, tok, 1)

        line2 = FIELD_NOM_RE.sub(repl_nom_field, line)
        line2 = FIELD_PRENOM_RE.sub(repl_prenom_field, line2)
        line2 = speaker_re.sub(repl_speaker, line2)

        if has_identity_context:
            def repl_person(m: re.Match) -> str:
                first = m.group("first").strip()
                last = m.group("last").strip()
                if _looks_like_non_person(first) or _looks_like_non_person(last):
                    return m.group(0)
                return ctx.token_for("PERSON", f"{first} {last}")

            line2 = PERSON_INLINE_RE.sub(repl_person, line2)
            line2 = PERSON_INVERTED_RE.sub(repl_person, line2)

        out_lines.append(line2)

    return "\n".join(out_lines)


def anonymize_text(text: str, ctx: Optional[AnonContext] = None) -> Tuple[str, AnonContext]:
    if ctx is None:
        ctx = AnonContext()
    text = EMAIL_RE.sub(lambda m: ctx.token_for("EMAIL", m.group(0)), text)
    text = PHONE_RE.sub(lambda m: ctx.token_for("PHONE", m.group(0)), text)
    text = IBAN_RE.sub(lambda m: ctx.token_for("IBAN", m.group(0)), text)
    text = anonymize_persons(text, ctx, window_lines=1)
    text = anonymize_plates(text, ctx, window_lines=1)
    return text, ctx


def deanonymize_text(text: str, ctx: AnonContext) -> str:
    for token in sorted(ctx.token2real.keys(), key=len, reverse=True):
        text = text.replace(token, ctx.token2real[token])
    return text


# ========= CONFIG =========

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_URL = os.getenv("LLM_API_BASE_URL", f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions")

DEFAULT_MODEL_NAME = (
    os.getenv("OLLAMA_DEFAULT_MODEL")
    or os.getenv("MODEL_NAME")
    or "mistral-large-3:675b-cloud"
)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


NUM_CTX = _env_int("OLLAMA_NUM_CTX", 262144)
NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", -1)
TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)
TOP_P = _env_float("OLLAMA_TOP_P", 0.95)
TOP_K = _env_int("OLLAMA_TOP_K", 0)

TOP_K_CHUNKS = 16

MAX_UPLOAD_MB = 500
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# ========= STOPWORDS =========
# (inchangé)
FRENCH_STOPWORDS = {
    "a", "à", "â", "abord", "afin", "ah", "ai", "aie", "aient", "ainsi", "allaient",
    "allo", "allô", "allons", "après", "assez", "attendu", "au", "aucun", "aucune",
    "aujourd", "aujourd'hui", "auquel", "aura", "aurai", "auraient", "aurais",
    "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi",
    "autre", "autres", "aux", "auxquelles", "auxquels", "avaient", "avais",
    "avait", "avant", "avec", "avoir", "avons", "ayant", "ayez", "ayons",
    "bah", "beaucoup", "bien", "boum", "bravo", "brrr",
    "c", "ç", "ça", "car", "ce", "ceci", "cela", "celle", "celles", "celui",
    "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet",
    "cette", "ceux", "chacun", "chaque", "cher", "chère", "chères", "chers",
    "chez", "chose", "ci", "combien", "comme", "comment", "concernant",
    "contre", "courant", "d", "da", "dans", "de", "dedans", "dehors", "delà",
    "depuis", "des", "désormais", "desquelles", "desquels", "dessous", "dessus",
    "deux", "devant", "devers", "devra", "devrait", "différent", "différente",
    "différentes", "différents", "dire", "divers", "diverse", "diverses", "doit",
    "doivent", "donc", "dont", "du", "duquel",
    "durant", "e", "effet", "eh", "elle", "elles", "en", "encore", "enfin",
    "entre", "envers", "es", "est", "et", "étaient", "étais", "était", "étant",
    "etc", "été", "être", "eu", "eue", "eues", "eus", "fait", "faites", "fois",
    "font", "furent", "fut", "fûmes", "gare", "grâce", "ha", "hé", "hein", "hem",
    "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue",
    "hui", "huit", "il", "ils", "importe", "j", "je", "jusqu", "jusque",
    "l", "la", "là", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels",
    "leur", "leurs", "lorsque", "lui", "lui-même", "m", "ma", "maint", "mais",
    "malgré", "me", "même", "mes", "mien", "mienne", "miennes", "miens",
    "moins", "mon", "mot", "n", "na", "ne", "néanmoins", "ni", "non", "nos",
    "notre", "nous", "nous-mêmes", "nul", "nulle", "oh", "on", "ont", "or",
    "ou", "où", "ouais", "ouf", "ouias", "par", "parmi", "pas",
    "passé", "pendant", "personne", "peu", "peut", "peuvent",
    "plus", "plusieurs", "plutôt", "pour", "pourquoi", "premier", "première",
    "premières", "premiers", "près", "puis", "puisque", "qu", "quand", "quant",
    "quanta", "quant-à-soi", "quarante", "quatorze", "quatre", "que", "quel",
    "quelconque", "quelle", "quelles", "quelque", "quelques", "quels",
    "qui", "quiconque", "quinze", "quoi", "quoique", "revoici", "revoilà",
    "rien", "s", "sa", "sans", "sauf", "se", "seize", "selon", "sept",
    "sera", "serai", "seraient", "serais", "serait", "seras", "serez",
    "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seules",
    "seuls", "si", "sien", "sienne", "siennes", "siens", "sinon", "six",
    "soi", "soi-même", "soit", "sommes", "son", "sont", "sous", "souvent",
    "soyez", "soyons", "suis", "suivant", "sur", "t", "ta", "tandis", "te",
    "tel", "telle", "telles", "tels", "tes", "toi", "toi-même", "ton", "toujours",
    "tous", "tout", "toute", "toutes", "treize", "trente", "très", "trois",
    "tu", "un", "une", "unes", "uns", "valeur", "vers", "via", "vingt",
    "voici", "voilà", "vont", "vos", "votre", "vous", "vous-mêmes", "vu",
    "y", "zéro"
}


# ========= STRUCTURES EN MEMOIRE =========

class Chunk(BaseModel):
    text: str
    page: int
    pdf_id: str


class PdfIndex:
    def __init__(self, pdf_id: str, file_name: str, chunks: List[Chunk]):
        self.pdf_id = pdf_id
        self.file_name = file_name
        self.chunks = chunks

        self.texts: List[str] = []
        for c in self.chunks:
            t = (c.text or "").strip()
            self.texts.append(t if len(t) >= 1 else "")

        fit_texts: List[str] = []
        fit_idx: List[int] = []
        for i, t in enumerate(self.texts):
            t2 = " ".join(t.split())
            if len(t2) >= 30:
                fit_texts.append(t2)
                fit_idx.append(i)

        if not fit_texts:
            raise ValueError("Aucun contenu textuel exploitable après nettoyage (PDF scanné/extraction vide).")

        def _fit(vectorizer: TfidfVectorizer):
            mat = vectorizer.fit_transform(fit_texts)
            return mat, fit_idx

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(FRENCH_STOPWORDS),
            max_features=40000,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1
        )
        try:
            self.matrix, self.fit_idx = _fit(self.vectorizer)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words=None,
                    max_features=40000,
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=1
                )
                self.matrix, self.fit_idx = _fit(self.vectorizer)
            else:
                raise


PDF_STORE: Dict[str, PdfIndex] = {}


# ========= MODELES API =========

class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_NAME
    message: str
    pdf_ids: List[str] = []
    return_sources: bool = True
    history: List[ChatHistoryItem] = []
    stream: bool = False


class SourceItem(BaseModel):
    id: str
    file_name: Optional[str] = None
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceItem]] = None


class SummaryRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_NAME
    pdf_ids: List[str] = []
    mode: str = "executif"
    max_chunks: int = 20


# ========= APP FASTAPI =========

app = FastAPI(title="Analyste PDF · Mistral Large 3")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    rel_static = os.path.join(os.getcwd(), "static")
    if os.path.isdir(rel_static):
        app.mount("/static", StaticFiles(directory=rel_static), name="static")

app.mount("/assets", StaticFiles(directory=BASE_DIR), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= UTILS PDF / RAG =========

def read_pdf_bytes(data: bytes) -> List[str]:
    pages_text: List[str] = []
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = text.replace("\x00", " ").strip()
            pages_text.append(text)
    except PdfReadError:
        pages_text = []
    except Exception:
        pages_text = []

    def letter_count(s: str) -> int:
        return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s or ""))

    need_full_ocr = False
    if not pages_text:
        need_full_ocr = True
    else:
        low_pages = [i for i, t in enumerate(pages_text) if letter_count(t) < 8]
        if len(low_pages) > max(1, len(pages_text) // 3):
            need_full_ocr = True

    if need_full_ocr:
        if convert_from_bytes is None or pytesseract is None:
            raise HTTPException(
                status_code=500,
                detail="OCR non disponible : installez pdf2image+pytesseract (voir README)."
            )
        try:
            ocr_pages = ocr_pdf_bytes(data)
            if pages_text and len(ocr_pages) == len(pages_text):
                pages_text = [
                    ocr_pages[i] if letter_count(pages_text[i]) < letter_count(ocr_pages[i]) else pages_text[i]
                    for i in range(len(pages_text))
                ]
            else:
                pages_text = ocr_pages
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur OCR PDF: {e}")
    else:
        weak_idx = [i for i, t in enumerate(pages_text) if letter_count(t) < 8]
        if weak_idx and convert_from_bytes and pytesseract:
            try:
                ocr_pages = ocr_pdf_bytes(data)
                if len(ocr_pages) == len(pages_text):
                    for i in weak_idx:
                        if letter_count(ocr_pages[i]) > letter_count(pages_text[i]):
                            pages_text[i] = ocr_pages[i]
            except Exception:
                pass

    return [(p or "").replace("\x00", " ").strip() for p in pages_text]


def ocr_pdf_bytes(data: bytes, lang: str = "fra", dpi: int = 300) -> List[str]:
    if convert_from_bytes is None or pytesseract is None:
        raise RuntimeError(
            "pdf2image/pytesseract non installés. "
            "Installez: pip install pdf2image pytesseract pillow ; brew install poppler tesseract"
        )
    images = convert_from_bytes(data, dpi=dpi, fmt="jpeg")
    texts: List[str] = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 3")
        except Exception:
            txt = pytesseract.image_to_string(img)
        texts.append((txt or "").replace("\x00", " ").strip())
    return texts


def read_docx_bytes(data: bytes) -> List[str]:
    if Document is None:
        raise HTTPException(status_code=500, detail="python-docx non installé. Installez: pip install python-docx")
    try:
        doc = Document(BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le document Word: {e}")
    text = "\n".join(p.text for p in doc.paragraphs)
    return [(text or "").replace("\x00", " ").strip()]


def read_text_bytes(data: bytes, encoding: str = "utf-8") -> List[str]:
    try:
        text = data.decode(encoding, errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier texte: {e}")
    return [(text or "").replace("\x00", " ").strip()]


def chunk_page_text(text: str, pdf_id: str, page: int) -> List[Chunk]:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    buf = ""
    max_len = 1800

    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_len:
            buf += "\n\n" + p
        else:
            chunks.append(Chunk(text=buf, page=page, pdf_id=pdf_id))
            buf = p

    if buf:
        chunks.append(Chunk(text=buf, page=page, pdf_id=pdf_id))
    return chunks


def build_pdf_index(pdf_id: str, file_name: str, pages_text: List[str]) -> PdfIndex:
    all_chunks: List[Chunk] = []
    for page_num, page_text in enumerate(pages_text, start=1):
        all_chunks.extend(chunk_page_text(page_text, pdf_id=pdf_id, page=page_num))
    if not all_chunks:
        all_chunks.append(Chunk(text="", page=1, pdf_id=pdf_id))
    return PdfIndex(pdf_id=pdf_id, file_name=file_name, chunks=all_chunks)


def retrieve_relevant_chunks(question: str, pdf_ids: List[str]) -> List[Chunk]:
    if not pdf_ids:
        return []

    scored_chunks = []
    for pdf_id in pdf_ids:
        index = PDF_STORE.get(pdf_id)
        if not index:
            continue

        q_vec = index.vectorizer.transform([question])
        scores = (index.matrix @ q_vec.T).toarray().ravel()
        if len(scores) == 0:
            continue

        top_k = min(TOP_K_CHUNKS, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_k]

        for idx in top_idx:
            score = scores[idx]
            real_i = index.fit_idx[idx]
            chunk = index.chunks[real_i]
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_scored = scored_chunks[:TOP_K_CHUNKS]
    return [c for _, c in top_scored]


def build_context_and_sources(chunks: List[Chunk]) -> Tuple[str, List[SourceItem]]:
    if not chunks:
        return "", []

    lines: List[str] = []
    sources: List[SourceItem] = []
    for c in chunks:
        pdf_index = PDF_STORE.get(c.pdf_id)
        file_name = pdf_index.file_name if pdf_index else None
        header = f"[PDF {c.pdf_id} · page {c.page}]"
        lines.append(f"{header}\n{c.text}")
        sources.append(SourceItem(id=c.pdf_id, file_name=file_name, page=c.page))

    context_text = "\n\n---\n\n".join(lines)
    return context_text, sources


# ========= CALL LLM =========

def _build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers


def call_mistral_chat(model: str, messages: List[Dict[str, str]]) -> str:
    model_name = model or DEFAULT_MODEL_NAME

    req_payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            **({} if NUM_PREDICT == -1 else {"num_predict": NUM_PREDICT}),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            **({} if TOP_K == 0 else {"top_k": TOP_K}),
        },
    }

    resp = requests.post(CHAT_URL, json=req_payload, headers=_build_headers(), timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Erreur LLM {resp.status_code}: {resp.text}")

    data = resp.json()
    if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if content:
            return content

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


def stream_mistral_chat(model: str, messages: List[Dict[str, str]], ctx: AnonContext):
    """
    Stream SSE vers le navigateur.
    - ne strip pas (préserve espaces)
    - ne réutilise pas `payload` pour éviter UnboundLocalError
    """
    model_name = model or DEFAULT_MODEL_NAME

    req_payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            **({} if NUM_PREDICT == -1 else {"num_predict": NUM_PREDICT}),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            **({} if TOP_K == 0 else {"top_k": TOP_K}),
        },
    }

    headers = {
        **_build_headers(),
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }

    def generate():
        try:
            resp = requests.post(
                CHAT_URL,
                json=req_payload,
                headers=headers,
                stream=True,
                timeout=120,
            )
        except Exception as e:
            yield f"event: error\ndata: [ERROR] upstream connection failed: {e}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        if resp.status_code >= 400:
            txt = getattr(resp, "text", "")
            yield f"event: error\ndata: [ERROR] upstream {resp.status_code}: {txt}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            try:
                resp.close()
            except Exception:
                pass
            return

        try:
            yield ": keep-alive\n\n"

            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

                if line.startswith("data:"):
                    sse_data = line[len("data:"):]
                    if sse_data.startswith(" "):
                        sse_data = sse_data[1:]
                    line = sse_data

                if line == "[DONE]":
                    break

                # Parse JSON or forward text
                piece = ""
                try:
                    obj = json.loads(line)
                except Exception:
                    piece = line
                else:
                    if isinstance(obj, dict):
                        # OpenAI streaming: choices[0].delta.content
                        choices = obj.get("choices") or []
                        if choices:
                            ch0 = choices[0] or {}
                            if isinstance(ch0, dict):
                                delta = ch0.get("delta") or {}
                                if isinstance(delta, dict):
                                    piece = delta.get("content") or ""
                                if not piece:
                                    msg = ch0.get("message") or {}
                                    if isinstance(msg, dict):
                                        piece = msg.get("content") or ""
                        # Ollama style: {"message":{"content":...}}
                        if not piece:
                            msg2 = obj.get("message")
                            if isinstance(msg2, dict):
                                piece = msg2.get("content") or ""
                        if not piece and "content" in obj:
                            piece = obj.get("content") or ""
                    else:
                        piece = str(obj)

                if piece:
                    try:
                        out = deanonymize_text(piece, ctx)
                    except Exception:
                        out = piece
                    yield f"data: {out}\n\n"

            yield "event: done\ndata: [DONE]\n\n"

        finally:
            try:
                resp.close()
            except Exception:
                pass

    return generate


# ========= API =========

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document"
    name_lower = filename.lower()

    if not any(name_lower.endswith(ext) for ext in (".pdf", ".txt", ".md", ".docx")):
        raise HTTPException(status_code=400, detail="Formats acceptés : PDF, DOCX, TXT, MD.")

    raw = await file.read()
    size_bytes = len(raw)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    if size_bytes > MAX_UPLOAD_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux ({size_mb:.1f} Mo). Limite: {MAX_UPLOAD_MB} Mo."
        )

    doc_id = "doc_" + uuid.uuid4().hex[:10]

    try:
        if name_lower.endswith(".pdf"):
            pages_text = read_pdf_bytes(raw)
        elif name_lower.endswith((".txt", ".md")):
            pages_text = read_text_bytes(raw)
        elif name_lower.endswith(".docx"):
            pages_text = read_docx_bytes(raw)
        else:
            raise HTTPException(status_code=400, detail="Type de fichier non supporté.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Erreur lecture document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lecture document: {e}")

    full_text = " ".join(pages_text).strip() if pages_text else ""
    letter_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", full_text))
    if letter_count < 20:
        raise HTTPException(status_code=400, detail="Document sans texte exploitable (extraction vide ou PDF scanné).")

    if len(full_text) < 100:
        raise HTTPException(status_code=400, detail="Document trop court pour être indexé (texte extrait insuffisant).")

    try:
        index = build_pdf_index(doc_id, filename, pages_text)
        PDF_STORE[doc_id] = index
        page_count = len(pages_text)
    except Exception as e:
        print(f"[ERROR] Erreur indexation document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur indexation document: {e}")

    return JSONResponse({
        "id": doc_id,
        "file_name": filename,
        "pages": page_count,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    })


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model_name = req.model or DEFAULT_MODEL_NAME
    print(f"[CHAT] stream={req.stream} pdf_ids={len(req.pdf_ids)}")

    chunks = retrieve_relevant_chunks(req.message, req.pdf_ids)
    context_text, sources = build_context_and_sources(chunks)

    try:
        ctx = AnonContext()
        anon_question, ctx = anonymize_text(req.message, ctx)
        anon_context, ctx = anonymize_text(context_text, ctx)

        if req.return_sources:
            full_prompt = f"""
Vous êtes un assistant intelligent. Répondez à la question en vous basant sur les documents fournis.

Documents:
{anon_context}

Question: {anon_question}
Réponse:"""
            messages = [{"role": "user", "content": full_prompt}]
        else:
            messages = [{"role": "user", "content": anon_question}]

        if req.stream:
            print(f"[CHAT] starting stream model={model_name} pdf_count={len(req.pdf_ids)}")
            gen = stream_mistral_chat(model_name, messages, ctx)
            return StreamingResponse(gen(), media_type="text/event-stream")

        print(f"[CHAT] calling sync model={model_name}")
        answer = call_mistral_chat(model_name, messages)
        answer = deanonymize_text(answer or "", ctx)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au modèle: {e}")

    answer = re.sub(r"\n{2,}", "\n", (answer or "")).strip()
    return ChatResponse(answer=answer, sources=sources)


@app.post("/api/summary")
async def summary(req: SummaryRequest):
    model_name = req.model or DEFAULT_MODEL_NAME

    all_chunks: List[Chunk] = []
    for pdf_id in req.pdf_ids:
        index = PDF_STORE.get(pdf_id)
        if not index:
            continue
        all_chunks.extend(index.chunks)

    if req.max_chunks > 0 and len(all_chunks) > req.max_chunks:
        all_chunks = all_chunks[:req.max_chunks]

    if not all_chunks:
        raise HTTPException(status_code=400, detail="Aucun chunk disponible pour le résumé.")

    context_text = "\n\n---\n\n".join(c.text for c in all_chunks)

    max_chars = 120_000
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    ctx = AnonContext()
    anon_context, ctx = anonymize_text(context_text, ctx)

    prompt = (
        f"Tu es un assistant. Fais un résumé {req.mode} du texte.\n"
        f"Règles:\n"
        f"- Réponds en français.\n"
        f"- Ne réponds pas vide.\n"
        f"- Ne commente pas les tokens __NAME_1__, etc.\n\n"
        f"TEXTE:\n{anon_context}\n\nRÉSUMÉ:"
    )

    messages = [{"role": "user", "content": prompt}]
    summary_text = call_mistral_chat(model_name, messages) or ""
    summary_text = deanonymize_text(summary_text, ctx)

    print(f"[SUMMARY] in={len(context_text)} chars | out={len(summary_text)} chars")
    return {"summary": summary_text.strip(), "answer": summary_text.strip()}


@app.get("/api/ping")
async def ping():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "API up"}


@app.post("/api/chat_stream")
async def chat_stream(req: ChatRequest):
    model_name = req.model or DEFAULT_MODEL_NAME

    chunks = retrieve_relevant_chunks(req.message, req.pdf_ids)
    context_text, _sources = build_context_and_sources(chunks)

    ctx = AnonContext()
    anon_question, ctx = anonymize_text(req.message, ctx)
    anon_context, ctx = anonymize_text(context_text, ctx)

    system = (
        "Tu es un assistant. "
        "Ne commente jamais les tokens d’anonymisation (ex: __NAME_1__). "
        "Ne mentionne pas l’anonymisation. "
        "Réponds normalement."
    )

    full_prompt = f"""Documents:
{anon_context}

Question: {anon_question}
Réponse:"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": full_prompt},
    ]

    gen = stream_mistral_chat(model_name, messages, ctx)
    return StreamingResponse(gen(), media_type="text/event-stream")
