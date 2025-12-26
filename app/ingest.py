# app/ingest.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
OUT_DIR = PROJECT_ROOT / "data" / "faiss_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = OUT_DIR / "meta.jsonl"
EMB_PATH = OUT_DIR / "embeddings.npy"

# Taille cible du chunk (en caractères) une fois reconstitué
CHUNK_SIZE = 900

# Overlap en nombre de phrases (bien mieux que overlap caractères)
OVERLAP_SENTENCES = 2


def clean_pdf_text(txt: str) -> str:
    """Nettoyage léger mais efficace pour PDF -> texte RAG."""
    txt = (txt or "").replace("\x00", " ")
    txt = txt.replace("\u00ad", "")  # soft hyphen

    # corrige mots coupés en fin de ligne : "assuran-\nce" -> "assurance"
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)

    # supprime lignes "page number" seules
    txt = re.sub(r"(?m)^\s*\d+\s*$", "", txt)

    # supprime lignes de pointillés / tirets répétitifs
    txt = re.sub(r"[.\u2026]{6,}", " ", txt)
    txt = re.sub(r"[-_=]{6,}", " ", txt)

    # Normalise retours : conserve les doubles sauts (paragraphes)
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    # Convertit les sauts simples en espaces (pour éviter texte cassé)
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)

    # Élimine quelques lignes très “bruit” typiques (tél, ICE, RC, etc.)
    # (optionnel mais utile)
    noise_patterns = [
        r"Tél\s*[:：]", r"Télécopie", r"\bRC\s*[:：]", r"\bICE\s*[:：]",
        r"\bCNSS\s*[:：]", r"\bIF\s*[:：]"
    ]
    for p in noise_patterns:
        txt = re.sub(p + r".{0,120}", " ", txt)

    # Compacte les espaces
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def split_sentences(text: str) -> List[str]:
    """Découpe simple en phrases (sans dépendances externes)."""
    text = (text or "").strip()
    if not text:
        return []

    # force un espace après ponctuation si collé
    text = re.sub(r"([.!?])([A-Za-zÀ-ÿ])", r"\1 \2", text)

    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def chunk_by_sentences(text: str, chunk_size: int = CHUNK_SIZE, overlap_sentences: int = OVERLAP_SENTENCES) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunk = " ".join(cur).strip()
            if len(chunk) >= 120:  # évite mini-chunks inutiles
                chunks.append(chunk)
        # overlap: garde les dernières phrases
        if overlap_sentences > 0 and cur:
            cur = cur[-overlap_sentences:]
            cur_len = sum(len(x) + 1 for x in cur)
        else:
            cur = []
            cur_len = 0

    for s in sents:
        s_len = len(s) + 1
        if cur_len + s_len > chunk_size and cur:
            flush()
        cur.append(s)
        cur_len += s_len

    flush()
    return chunks


def extract_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    out: List[Dict[str, Any]] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        cleaned = clean_pdf_text(raw)
        for chunk in chunk_by_sentences(cleaned):
            out.append({
                "source": pdf_path.name,
                "page": i,
                "chunk_text": chunk,
            })
    return out


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"📄 PDFs trouvés: {len(pdfs)}")
    for p in pdfs:
        print(" -", p.name)

    all_rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        all_rows.extend(extract_pdf(pdf))

    if not all_rows:
        print("⚠️ Aucun texte extrait. Ajoute des PDFs dans data/docs")
        return

    # Option A (rapide, ok)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Option B (meilleur en FR, un peu + lent) :
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print(f"🧠 Embeddings: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [r["chunk_text"] for r in all_rows]
    emb = model.encode(texts, convert_to_numpy=True).astype("float32")

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    with META_PATH.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    np.save(str(EMB_PATH), emb)
    print(f"💾 Sauvegarde: {META_PATH} + {EMB_PATH}")
    print(f"✅ OK: {len(all_rows)} chunks indexés.")


if __name__ == "__main__":
    main()
