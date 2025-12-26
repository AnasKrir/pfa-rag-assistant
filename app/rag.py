# app/rag.py
from __future__ import annotations

import json
import logging
import os
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from requests.exceptions import RequestException, Timeout
from sentence_transformers import SentenceTransformer

# --- Logs ---
log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"
META_PATH = INDEX_DIR / "meta.jsonl"
EMB_PATH = INDEX_DIR / "embeddings.npy"

# ---------------- Llama-server ----------------
LLAMA_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
LLAMA_COMPLETION_ENDPOINT = f"{LLAMA_URL}/completion"

# ---------------- Embeddings model ----------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Stability on mac (avoid over-threading)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------- Tuning ----------------
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "420"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

DEFAULT_N_PREDICT = int(os.getenv("LLM_N_PREDICT", "120"))
DEFAULT_READ_TIMEOUT = int(os.getenv("LLM_READ_TIMEOUT", "60"))
DISABLE_READ_TIMEOUT = os.getenv("LLM_NO_TIMEOUT", "0") == "1"

HYBRID_LEX_WEIGHT = float(os.getenv("RAG_LEX_WEIGHT", "0.18"))  # 0..1
MIN_EMBED_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.10"))     # plus permissif

# ---------------- Globals ----------------
_embedder: Optional[SentenceTransformer] = None
_meta_rows: Optional[List[Dict[str, Any]]] = None
_emb_norm: Optional[np.ndarray] = None

_STORE_LOCK = threading.Lock()
_EMBED_LOCK = threading.Lock()
_LLM_SEM = threading.Semaphore(1)

_http = requests.Session()
_http.headers.update({"Content-Type": "application/json"})

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+", re.UNICODE)

_NOISE_SENT_RE = re.compile(
    r"(CONVENTION\s+DE\s+STAGE|EMPLOI\s+DU\s+TEMPS|Adresse|Rue|Tél|Télécopie|RC|ICE|CNSS|IF|"
    r"EMSI[-\s]*RABAT|Patrice\s+Lumumba|Anis\s+BOULAL|E\.?M\.?S\.?I)",
    re.IGNORECASE,
)


def _norm_source(s: str) -> str:
    return (s or "").strip().lower()


def _infer_source_hint(question: str) -> Optional[str]:
    """
    Route “fort” vers un document, sinon None.
    On renvoie un motif (lower) à chercher dans row["source"].lower().
    """
    q = (question or "").lower()

    # PFA
    if re.search(r"\bpfa\b", q):
        return "pfa.pdf"

    # Convention de stage
    if ("convention" in q and "stage" in q) or re.search(r"\bconvention\b.*\bstage\b", q):
        return "convention de stage"

    # Emploi du temps / horaires
    if ("emploi du temps" in q) or ("horaire" in q) or ("planning" in q):
        return "emploi du temps"

    return None


def _tokenize_words(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _jaccard_overlap(q: str, t: str) -> float:
    qw = set(_tokenize_words(q))
    if not qw:
        return 0.0
    tw = set(_tokenize_words(t))
    if not tw:
        return 0.0
    inter = len(qw & tw)
    denom = (len(qw) + len(tw) - inter) or 1
    return inter / denom


def _has_long_ngram_overlap(answer: str, contexts: List[Dict[str, Any]], n: int = 7) -> bool:
    ans = " ".join(_tokenize_words(answer))
    if not ans:
        return False

    ctx_text = " ".join((c.get("chunk_text") or "") for c in contexts)
    ctx = " ".join(_tokenize_words(ctx_text))
    if not ctx:
        return False

    ans_words = ans.split()
    ctx_words = ctx.split()
    if len(ans_words) < n or len(ctx_words) < n:
        return False

    ctx_ngrams = set(" ".join(ctx_words[i:i + n]) for i in range(len(ctx_words) - n + 1))
    for i in range(len(ans_words) - n + 1):
        if " ".join(ans_words[i:i + n]) in ctx_ngrams:
            return True
    return False


def _clean_answer_one_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = re.sub(r"^\s*(A\s*:\s*)+", "", t).strip()
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    m = re.search(r"[.!?]", t)
    if m:
        t = t[: m.end()].strip()

    if t and t[-1] not in ".!?":
        t += "."
    return t


def _strip_headers_and_noise(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\bCONVENTION\s+DE\s+STAGE\b[^.]{0,220}", " ", s, flags=re.I)
    s = re.sub(r"\bRue\b.{0,140}", " ", s, flags=re.I)
    s = re.sub(r"\bTél\b.{0,140}", " ", s, flags=re.I)
    s = re.sub(r"\bTélécopie\b.{0,140}", " ", s, flags=re.I)
    s = re.sub(r"\bRC\b.{0,100}", " ", s, flags=re.I)
    s = re.sub(r"\bICE\b.{0,100}", " ", s, flags=re.I)
    s = re.sub(r"\bCNSS\b.{0,100}", " ", s, flags=re.I)
    s = re.sub(r"\bIF\b.{0,100}", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_sentences(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return []
    return re.split(r"(?<=[.!?])\s+", s)


def _ctx_for_llm(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").replace("\r", " ").replace("\n", " ")).strip()
    if not s:
        return ""

    s = re.sub(r"\b\d+\.\s*(En ce qui concerne)\b", r"\1", s, flags=re.IGNORECASE)

    sents = _split_sentences(s)
    kept = []
    for sent in sents:
        st = sent.strip()
        if not st:
            continue
        if _NOISE_SENT_RE.search(st):
            continue
        if len(st) < 25:
            continue
        kept.append(st)

    if not kept:
        s2 = _strip_headers_and_noise(s)
        kept = _split_sentences(s2)[:3]

    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------- Embedding store ----------------
def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder

    with _EMBED_LOCK:
        if _embedder is not None:
            return _embedder
        try:
            import torch  # noqa
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        return _embedder


def _load_store() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    global _meta_rows, _emb_norm

    if _meta_rows is not None and _emb_norm is not None:
        return _meta_rows, _emb_norm

    with _STORE_LOCK:
        if _meta_rows is not None and _emb_norm is not None:
            return _meta_rows, _emb_norm

        if not META_PATH.exists():
            raise FileNotFoundError(f"meta introuvable: {META_PATH}. Lance: python -m app.ingest")
        if not EMB_PATH.exists():
            raise FileNotFoundError(f"embeddings introuvables: {EMB_PATH}. Lance: python -m app.ingest")

        rows: List[Dict[str, Any]] = []
        with META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        emb = np.load(str(EMB_PATH)).astype("float32")
        if len(rows) != emb.shape[0]:
            raise ValueError(f"Incohérence meta/embeddings: meta={len(rows)} vs embeddings={emb.shape[0]}")

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb_norm = emb / norms

        _meta_rows = rows
        _emb_norm = emb_norm
        return rows, emb_norm


def retrieve(query: str, top_k: int = 16) -> List[Dict[str, Any]]:
    if not (query or "").strip():
        return []

    rows, emb_norm = _load_store()
    embedder = _get_embedder()

    with _EMBED_LOCK:
        q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    scores = emb_norm @ q_emb

    top_k = max(1, int(top_k))
    top_k = min(top_k, scores.shape[0])

    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]

    out: List[Dict[str, Any]] = []
    for i in idx:
        r = dict(rows[int(i)])
        r["score"] = float(scores[int(i)])
        out.append(r)
    return out


def _keyword_fallback(question: str, top_k: int, source_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    q = (question or "").strip().lower()
    if not q:
        return []

    rows, _ = _load_store()
    q_tokens = set(_tokenize_words(q))
    if not q_tokens:
        return []

    hint = (source_hint or "").lower().strip()

    hits: List[Dict[str, Any]] = []
    for r in rows:
        src = _norm_source(r.get("source", ""))
        if hint and hint not in src:
            continue

        txt = (r.get("chunk_text") or "")
        low = txt.lower()

        hit = sum(1 for tok in q_tokens if tok and tok in low)
        # IMPORTANT: si question contient "pfa", ce mot n'est pas forcément dans le texte,
        # donc on accepte hit==0 mais avec un bonus si on est déjà sur la bonne source.
        if hit <= 0 and not hint:
            continue

        rr = dict(r)
        rr["lex_score"] = float(_jaccard_overlap(question, txt))
        rr["hybrid_score"] = float(hit) + rr["lex_score"] + (0.50 if hint else 0.0)
        hits.append(rr)

    hits.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hits[: max(1, int(top_k))]


def _select_contexts_hybrid(question: str, top_k: int) -> List[Dict[str, Any]]:
    hint = _infer_source_hint(question)  # ex: "pfa.pdf" / "convention de stage" / "emploi du temps"
    raw = retrieve(question, top_k=max(20, top_k * 8))

    # Filtrage par source si hint
    if hint:
        raw2 = [r for r in raw if hint in _norm_source(r.get("source", ""))]
        raw = raw2 if raw2 else raw  # si vide, on garde raw pour ne pas tout casser

    if not raw:
        return _keyword_fallback(question, top_k, source_hint=hint)

    scored: List[Dict[str, Any]] = []
    for r in raw:
        if float(r.get("score", 0.0)) < MIN_EMBED_SCORE and not hint:
            continue
        lex = _jaccard_overlap(question, r.get("chunk_text", ""))
        hybrid = (1.0 - HYBRID_LEX_WEIGHT) * float(r.get("score", 0.0)) + HYBRID_LEX_WEIGHT * lex
        if hint and hint in _norm_source(r.get("source", "")):
            hybrid += 0.35  # bonus fort si c'est la source attendue

        rr = dict(r)
        rr["lex_score"] = float(lex)
        rr["hybrid_score"] = float(hybrid)
        scored.append(rr)

    if not scored:
        return _keyword_fallback(question, top_k, source_hint=hint)

    # Choix meilleure source (somme des meilleurs passages)
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in scored:
        by_src.setdefault(r.get("source", "unknown"), []).append(r)

    best_src = None
    best_score = -1e9
    for src, items in by_src.items():
        items_sorted = sorted(items, key=lambda x: x["hybrid_score"], reverse=True)
        s = sum(x["hybrid_score"] for x in items_sorted[: min(3, len(items_sorted))])
        if s > best_score:
            best_score = s
            best_src = src

    items = sorted(by_src.get(best_src, []), key=lambda x: x["hybrid_score"], reverse=True)
    return items[:top_k]


# ---------------- LLM ----------------
def call_llama(
    prompt: str,
    *,
    n_predict: int = DEFAULT_N_PREDICT,
    temperature: float = 0.1,
    read_timeout: int = DEFAULT_READ_TIMEOUT,
) -> str:
    p = (prompt or "").strip()
    if not p:
        return ""

    # Force format instruct
    if p.startswith("[INST]"):
        p = "<s>" + p

    payload = {
        "prompt": p,
        "n_predict": int(n_predict),
        "temperature": float(temperature),
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.15,
        "stream": False,
        "stop": ["</s>", "[INST]"],
    }

    _LLM_SEM.acquire()
    try:
        timeout = None if DISABLE_READ_TIMEOUT else (10, int(read_timeout))
        r = _http.post(LLAMA_COMPLETION_ENDPOINT, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("content") or "").strip()
    except Timeout as e:
        raise TimeoutError("LLM timeout.") from e
    except RequestException as e:
        raise RuntimeError(f"LLM HTTP error: {e}") from e
    finally:
        _LLM_SEM.release()


# ---------------- Deterministic synthesis ----------------
def _answer_pfa_subject(contexts: List[Dict[str, Any]]) -> str:
    text = " ".join(c.get("chunk_text") or "" for c in contexts)
    text = re.sub(r"\s+", " ", text).strip()

    m = re.search(r"Sujet\s+proposé\s*:\s*(.+?)(?:Objectifs\b|Livrables\b|Axes\b|Plan\b|$)", text, flags=re.I)
    if m:
        subj = m.group(1).strip()
    else:
        # fallback court
        subj = text[:240].strip()

    subj = re.sub(r"\s+", " ", subj).strip(" -–:;,.")
    if not subj:
        return "Non précisé dans les extraits."
    return f"Le sujet du PFA est : {subj}."


def _answer_convention_infos(contexts: List[Dict[str, Any]]) -> str:
    full = " ".join(c.get("chunk_text") or "" for c in contexts)
    full = _strip_headers_and_noise(full)

    periode = ""
    m = re.search(r"s'effectuera\s+entre\s+(.+?)\s+d['’]une\s+durée\s+de\s+(.+?)(?:\.|$)", full, flags=re.I)
    if m:
        periode = f"entre {m.group(1).strip()}, pour {m.group(2).strip()}"

    parts = [
        "les parties (école et organisme d’accueil)",
        "l’identité du stagiaire",
        "le contenu/libellé du stage",
        f"la période et la durée ({periode})" if periode else "la période et la durée du stage",
        "le statut de l’étudiant et ses obligations (horaires, discrétion/secret)",
        "l’assurance sociale",
        "les modalités d’évaluation",
    ]
    return "Une convention de stage précise " + ", ".join(parts) + "."


def _generic_answer_from_contexts(contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "Non précisé dans les extraits."
    txt = _ctx_for_llm(contexts[0].get("chunk_text") or "")
    if not txt:
        return "Non précisé dans les extraits."
    sents = _split_sentences(txt)
    if not sents:
        return "Non précisé dans les extraits."
    out = sents[0].strip()
    if len(out) < 60 and len(sents) > 1:
        out = (out + " " + sents[1].strip()).strip()
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _build_llm_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        chunk = _ctx_for_llm(c.get("chunk_text") or c.get("text") or "")
        chunk = chunk[:MAX_CHUNK_CHARS]
        ctx_lines.append(f"[{i}] {chunk}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "Aucun extrait trouvé."

    return (
        "[INST] Tu réponds uniquement à partir des extraits.\n"
        "Écris UNE seule phrase claire et cohérente, et termine par un point.\n"
        "N'écris pas de titres, noms propres, adresses ou numéros inutiles.\n"
        "Reformule: ne recopie pas mot à mot des extraits.\n"
        "Si l'info n'est pas dans les extraits: répond exactement 'Non précisé dans les extraits.'.\n"
        f"Q: {question}\n"
        f"Extraits:\n{context_block}\n"
        "R: [/INST]"
    )


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[Dict[str, Any]]]:
    question = (question or "").strip()
    if not question:
        return "Non précisé dans les extraits.\nSources: (aucune)", []

    hint = _infer_source_hint(question)
    contexts = _select_contexts_hybrid(question, top_k=top_k)

    # ✅ HARD FIX: si on a un hint, mais la meilleure source n'est pas celle attendue,
    # on force un fallback keyword RESTRICT sur la source hint (ça corrige ton screen PFA->Convention).
    if hint and contexts:
        src0 = _norm_source(contexts[0].get("source", ""))
        if hint not in src0:
            forced = _keyword_fallback(question, top_k, source_hint=hint)
            if forced:
                contexts = forced

    q_low = question.lower()

    # Baseline robuste (si le modèle déraille)
    if hint == "pfa.pdf":
        baseline = _answer_pfa_subject(contexts)
    elif hint == "convention de stage":
        baseline = _answer_convention_infos(contexts)
    else:
        baseline = _generic_answer_from_contexts(contexts)

    # Tentative LLM + garde-fous
    answer = ""
    if contexts:
        try:
            prompt = _build_llm_prompt(question, contexts)
            llm_out = call_llama(
                prompt,
                n_predict=DEFAULT_N_PREDICT,
                temperature=0.2,
                read_timeout=DEFAULT_READ_TIMEOUT,
            )
            llm_out = _clean_answer_one_sentence(llm_out)

            bad = False
            if not llm_out:
                bad = True
            if llm_out == "Non précisé dans les extraits.":
                bad = True
            if _NOISE_SENT_RE.search(llm_out):
                bad = True
            if hint == "convention de stage" and not llm_out.lower().startswith("une convention de stage"):
                bad = True
            if hint == "pfa.pdf" and not llm_out.lower().startswith("le sujet"):
                bad = True
            if _has_long_ngram_overlap(llm_out, contexts, n=7):
                bad = True

            answer = baseline if bad else llm_out
        except Exception as e:
            log.exception("LLM error: %s", e)
            answer = baseline
    else:
        answer = "Non précisé dans les extraits."

    # Sources
    if contexts:
        answer += "\nSources: " + ", ".join([f"[{i}]" for i in range(1, len(contexts) + 1)])
    else:
        answer += "\nSources: (aucune)"

    return answer, contexts


if __name__ == "__main__":
    tests = [
        "C'est quoi le sujet de PFA ?",
        "Quelles sont les informations importantes d'une convention de stage ?",
    ]
    for q in tests:
        ans, ctx = answer_question(q, top_k=3)
        print("\nQ:", q)
        print("A:", ans)
        if ctx:
            print("Top source:", ctx[0].get("source"))
