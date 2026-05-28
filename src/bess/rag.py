"""RAG store over NEM market documents (Chroma + Hugging Face embeddings).

Ingests local docs (``.md`` / ``.txt`` / ``.pdf``) describing AEMO market
mechanics, tariffs, and BESS economics, chunks them, embeds with a sentence-
transformers model (Hugging Face), and persists to a Chroma collection. Retrieval
returns chunks *with source + section metadata* so the agent can cite where each
claim came from — citation is the whole point of grounding the advisor in docs
rather than letting the LLM freelance market rules.

Embedding model: ``sentence-transformers/all-MiniLM-L6-v2`` — 384-dim, fast on
CPU, strong retrieval/size trade-off. Swap via ``RagStore(embed_model=...)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "docs" / "knowledge"
DEFAULT_STORE = ROOT / "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Citation:
    text: str
    source: str
    section: str
    score: float


def _read_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_markdown(text: str, source: str, *, max_chars: int = 1200) -> list[dict]:
    """Split on markdown headings, then pack paragraphs up to ``max_chars``.

    Keeps the nearest heading as the chunk's ``section`` so citations point at a
    real part of the document, not just a filename.
    """
    chunks: list[dict] = []
    section = "(intro)"
    buf: list[str] = []

    def flush():
        if buf:
            body = "\n".join(buf).strip()
            if body:
                chunks.append({"text": body, "source": source, "section": section})
            buf.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", block)
        if m:
            flush()
            section = m.group(1).strip()
            continue
        if sum(len(b) for b in buf) + len(block) > max_chars:
            flush()
        buf.append(block)
    flush()
    return chunks


class RagStore:
    def __init__(self, store_dir: Path | str = DEFAULT_STORE, embed_model: str = EMBED_MODEL):
        self.store_dir = Path(store_dir)
        self.embed_model = embed_model
        self._client = None
        self._collection = None
        self._embedder = None

    # -- lazy heavy deps ----------------------------------------------------
    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.store_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                "nem_docs", metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    # -- ingest / query -----------------------------------------------------
    def ingest_dir(self, docs_dir: Path | str = DEFAULT_DOCS) -> int:
        docs_dir = Path(docs_dir)
        paths = [p for p in docs_dir.rglob("*") if p.suffix.lower() in (".md", ".txt", ".pdf")]
        all_chunks: list[dict] = []
        for p in paths:
            all_chunks.extend(chunk_markdown(_read_doc(p), source=p.name))
        if not all_chunks:
            return 0
        ids = [f"{c['source']}::{i}" for i, c in enumerate(all_chunks)]
        self.collection.upsert(
            ids=ids,
            documents=[c["text"] for c in all_chunks],
            embeddings=self._embed([c["text"] for c in all_chunks]),
            metadatas=[{"source": c["source"], "section": c["section"]} for c in all_chunks],
        )
        return len(all_chunks)

    def query(self, question: str, k: int = 4) -> list[Citation]:
        res = self.collection.query(
            query_embeddings=self._embed([question]), n_results=k
        )
        out: list[Citation] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            out.append(
                Citation(
                    text=text,
                    source=meta.get("source", "?"),
                    section=meta.get("section", "?"),
                    score=float(1.0 - dist),  # cosine distance -> similarity
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()
