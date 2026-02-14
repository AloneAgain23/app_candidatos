import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

from google import genai

# =========================
# CONFIG (EDITA ESTO)
# =========================
DATA_DIR = r"C:\ruta\a\DATA_DIR"              # carpeta raíz
CANDIDATE_FOLDER = "Candidato_X"             # nombre exacto de la carpeta del candidato
QUESTIONS_XLSX = r"C:\ruta\preguntas.xlsx"   # Excel con columnas: Tema | Pregunta (Tema opcional)
SHEET_NAME = 0                               # hoja (0 = primera)

# Embeddings / retrieval
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 6                      # candidatos a chunks (antes de filtrar)
KEEP_CONTEXT_CHUNKS = 2        # chunks finales que van a Gemini
MIN_SCORE = 0.30               # umbral: si no supera -> sin_datos_suficientes (no llama a Gemini)

# Chunking
CHUNK_MAX_CHARS = 2600
CHUNK_OVERLAP_CHARS = 250

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.0

# Output
OUT_XLSX = "posturas_1_candidato.xlsx"
CACHE_JSON = "gemini_cache_1_candidato.json"  # para no re-gastar llamadas si re-ejecutas

STANCE_ENUM = [
    "totalmente_a_favor",
    "a_favor",
    "neutro",
    "en_contra",
    "totalmente_en_contra",
    "sin_datos_suficientes",
]

STANCE_LABEL = {
    "totalmente_a_favor": "Totalmente a favor",
    "a_favor": "A favor",
    "neutro": "Neutro",
    "en_contra": "En contra",
    "totalmente_en_contra": "Totalmente en contra",
    "sin_datos_suficientes": "Sin datos suficientes",
}

# =========================
# UTILIDADES: lectura TXT/PDF
# =========================
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts)

def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Formato no soportado: {ext}")

def normalize(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    text = normalize(text)
    if len(text) <= max_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out

def infer_source(path: str) -> str:
    folder = os.path.basename(os.path.dirname(path)).lower()
    if "entrev" in folder:
        return "interview"
    if "plan" in folder:
        return "plan"
    return folder

def short_snippet(t: str, n: int = 450) -> str:
    t = re.sub(r"\s+", " ", (t or "")).strip()
    return t[:n] + ("..." if len(t) > n else "")

# =========================
# ESTRUCTURA DE CHUNKS
# =========================
@dataclass
class Chunk:
    chunk_id: int
    source: str
    file: str
    text: str

def collect_chunks_one_candidate(data_dir: str, candidate_folder: str) -> List[Chunk]:
    base = os.path.join(data_dir, candidate_folder)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No existe la carpeta del candidato: {base}")

    paths = []
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith((".txt", ".pdf")):
                paths.append(os.path.join(root, fn))

    if not paths:
        raise FileNotFoundError(f"No encontré TXT/PDF dentro de: {base}")

    chunks: List[Chunk] = []
    cid = 0
    for path in tqdm(paths, desc="Leyendo documentos"):
        try:
            txt = load_text(path)
            if len(normalize(txt)) < 200:
                # típico PDF escaneado o vacío
                continue
        except Exception as e:
            print("SKIP:", path, "->", e)
            continue

        src = infer_source(path)
        for ch in chunk_text(txt):
            if len(ch.strip()) < 120:
                continue
            chunks.append(Chunk(cid, src, os.path.basename(path), ch))
            cid += 1

    if not chunks:
        raise RuntimeError("No se generaron chunks (¿PDFs sin texto extraíble?).")

    return chunks

# =========================
# EMBEDDINGS + FAISS
# =========================
def build_index(chunks: List[Chunk], model_name: str):
    model = SentenceTransformer(model_name)
    embs = model.encode(
        [c.text for c in chunks],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cos sim si normalizado
    index.add(embs.astype(np.float32))
    return model, index

def retrieve(model, index, chunks: List[Chunk], query: str, top_k: int):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, top_k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        c = chunks[int(idx)]
        hits.append({
            "score": float(score),
            "source": c.source,
            "file": c.file,
            "text": c.text
        })
    return hits

# =========================
# GEMINI (JSON estructurado) + CACHE
# =========================
client = genai.Client()  # usa GEMINI_API_KEY o GOOGLE_API_KEY en variables de entorno

SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": STANCE_ENUM},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "certainty": {"type": "string", "enum": ["alta", "media", "baja"]},
    },
    "required": ["stance", "summary", "evidence", "certainty"],
}

def load_cache(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, cache: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def make_cache_key(theme: str, question: str, context: str) -> str:
    h = hashlib.sha256()
    h.update((theme + "\n" + question + "\n" + context).encode("utf-8", errors="ignore"))
    return h.hexdigest()

def gemini_extract(theme: str, question: str, context: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = make_cache_key(theme, question, context)
    if key in cache:
        return cache[key]

    prompt = f"""
Eres un analista político. Debes evaluar si el DISCURSO DEL CANDIDATO coincide o no
con el siguiente ENUNCIADO DE POSTURA.

IMPORTANTE:
- El enunciado es un criterio de postura (no es una pregunta al candidato).
- En el contexto puede aparecer entrevistador y candidato: prioriza frases atribuibles al candidato cuando sea posible.
- Usa SOLO el contexto. No inferir. No inventar.

Clasifica en EXACTAMENTE UNA:
totalmente_a_favor, a_favor, neutro, en_contra, totalmente_en_contra, sin_datos_suficientes

Reglas:
- "sin_datos_suficientes" si el contexto no permite determinar postura.
- "neutro" si el candidato reconoce el tema pero no toma posición clara o muestra ambivalencia.
- Evidencia: máximo 2 citas textuales cortas y literales del contexto.

Tema: {theme}

ENUNCIADO DE POSTURA:
\"\"\"{question}\"\"\"

CONTEXTO:
\"\"\"{context}\"\"\"
""".strip()

    last_err = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": SCHEMA,
                    "temperature": GEMINI_TEMPERATURE,
                },
            )
            data = json.loads(resp.text)
            cache[key] = data
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini falló tras reintentos. Último error: {last_err}")

# =========================
# MAIN
# =========================
def main():
    # 1) Leer preguntas (Tema opcional)
    qdf = pd.read_excel(QUESTIONS_XLSX, sheet_name=SHEET_NAME).fillna("")
    if "Pregunta" not in qdf.columns:
        raise ValueError(f"Falta columna 'Pregunta'. Encontré: {list(qdf.columns)}")
    if "Tema" not in qdf.columns:
        qdf["Tema"] = "NA"

    # 2) Leer docs y chunking
    chunks = collect_chunks_one_candidate(DATA_DIR, CANDIDATE_FOLDER)
    print("Chunks generados:", len(chunks))

    # 3) Indexar
    model, index = build_index(chunks, EMB_MODEL)

    # 4) Cache Gemini
    cache = load_cache(CACHE_JSON)

    results = []
    for _, r in tqdm(qdf.iterrows(), total=len(qdf), desc="Procesando"):
        theme = str(r["Tema"]).strip() if "Tema" in qdf.columns else "NA"
        question = str(r["Pregunta"]).strip()

        # Query de recuperación (incluye el enunciado tal cual)
        query = f"{theme}. {question}. postura, propuesta, enfoque, medidas, declarar, apoyar, rechazar"
        hits = retrieve(model, index, chunks, query, TOP_K)

        # Filtro por score (si nada relevante, no gastar Gemini)
        hits = [h for h in hits if h["score"] >= MIN_SCORE]
        if not hits:
            results.append({
                "Candidato": CANDIDATE_FOLDER,
                "Tema": theme,
                "Pregunta": question,
                "Postura": "sin_datos_suficientes",
                "Postura_label": STANCE_LABEL["sin_datos_suficientes"],
                "Certeza": "alta",
                "Resumen": "No se encontraron fragmentos suficientemente relevantes en los documentos disponibles.",
                "Evidencia_1": "",
                "Evidencia_2": "",
                "Fuentes_top": "",
                "Snippets_top": "",
                "Scores_top": "",
            })
            continue

        # Contexto compacto para Gemini
        top = sorted(hits, key=lambda x: x["score"], reverse=True)[:KEEP_CONTEXT_CHUNKS]
        context = "\n\n---\n\n".join(
            [f"[Fuente={h['source']} | Archivo={h['file']} | Score={h['score']:.3f}]\n{h['text']}"
             for h in top]
        )
        snips = " || ".join([short_snippet(h["text"]) for h in top])
        sources = "; ".join(sorted({f"{h['source']}:{h['file']}" for h in top}))
        scores = "; ".join([f"{h['score']:.3f}" for h in top])

        out = gemini_extract(theme, question, context, cache)

        stance = out.get("stance", "sin_datos_suficientes")
        if stance not in STANCE_ENUM:
            stance = "sin_datos_suficientes"

        ev = out.get("evidence", []) or []
        results.append({
            "Candidato": CANDIDATE_FOLDER,
            "Tema": theme,
            "Pregunta": question,
            "Postura": stance,
            "Postura_label": STANCE_LABEL.get(stance, stance),
            "Certeza": out.get("certainty", "media"),
            "Resumen": out.get("summary", ""),
            "Evidencia_1": ev[0] if len(ev) > 0 else "",
            "Evidencia_2": ev[1] if len(ev) > 1 else "",
            "Fuentes_top": sources,
            "Snippets_top": snips,
            "Scores_top": scores,
        })

        # guarda cache en cada iteración (por si se corta el proceso)
        save_cache(CACHE_JSON, cache)

    df = pd.DataFrame(results)

    # Export Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="resultados")
        df["Postura_label"].value_counts().rename_axis("Postura").to_frame("n").to_excel(w, sheet_name="conteo_posturas")

    print("OK ->", OUT_XLSX)
    print("Cache ->", CACHE_JSON)

if __name__ == "__main__":
    main()
