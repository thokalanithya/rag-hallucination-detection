"""
app.py — FastAPI server with SSE streaming for the RAG hallucination detection UI.

Run with:
    cd src
    uvicorn app:app --reload --port 8000
"""

import sys
import json
import queue
import asyncio
import threading
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── File logger ───────────────────────────────────────────────────────────────
_log_dir = Path(__file__).parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = RotatingFileHandler(
    _log_dir / "rag_debug.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)

logger = logging.getLogger("rag_app")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.propagate = False   # don't double-log to uvicorn's root logger

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import run_rag_pipeline

app = FastAPI(title="RAG Hallucination Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


class QueryRequest(BaseModel):
    question: str


def _serialize_result(result: dict) -> dict:
    """Convert non-JSON-serializable values (numpy floats, int keys) in the result."""
    from collections import defaultdict

    out = dict(result)

    chunks = []
    for chunk in out.get("retrieved_chunks", []):
        c = dict(chunk)
        for field in ("similarity", "rerank_score"):
            if field in c:
                c[field] = float(c[field])
        chunks.append(c)
    out["retrieved_chunks"] = chunks

    # Build paper_id → merged chunk text for context enrichment
    paper_chunks: dict = defaultdict(list)
    for chunk in chunks:
        pid = chunk.get("paper_id", "")
        if pid:
            paper_chunks[pid].append(chunk.get("text", ""))

    # Enrich RAG citations with the retrieved chunk text
    enriched_citations = {}
    for k, v in out.get("citations", {}).items():
        enriched = dict(v)
        pid = enriched.get("paper_id", "")
        if pid in paper_chunks:
            enriched["context"] = "\n\n".join(paper_chunks[pid])
        enriched_citations[str(k)] = enriched
    out["citations"] = enriched_citations

    # Web citations already carry a snippet; normalise keys to str
    out["web_citations"] = {
        str(k): v for k, v in out.get("web_citations", {}).items()
    }

    return out


@app.get("/")
async def root():
    return HTMLResponse((_static_dir / "index.html").read_text())


@app.post("/api/query")
async def query(request: QueryRequest):
    sync_queue: queue.Queue = queue.Queue()

    def status_callback(event_type: str, message: str, data=None):
        sync_queue.put_nowait({"type": event_type, "message": message, "data": data})

    def run_pipeline():
        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"QUERY    : {request.question!r}")
        try:
            result = run_rag_pipeline(
                question=request.question,
                collection_name="qasper_chunks", # qasper_chunks
                status_callback=status_callback,
            )
            elapsed = time.time() - t0

            # ── Retrieval ─────────────────────────────────────────
            chunks = result.get("retrieved_chunks", [])
            logger.debug(f"CHUNKS   : {len(chunks)} retrieved after re-ranking")
            for i, c in enumerate(chunks, 1):
                logger.debug(
                    f"  [{i}] paper={c.get('paper_id','?'):12s} "
                    f"rerank={float(c.get('rerank_score', 0)):+.4f}  "
                    f"sim={float(c.get('similarity', 0)):.4f}  "
                    f"| {c.get('text', '')[:100].strip()!r}"
                )

            # ── Source & answer ───────────────────────────────────
            logger.info(f"SOURCE   : {result['source']}")
            logger.info(f"ANSWER   :\n{result['answer']}")

            # ── Citations ─────────────────────────────────────────
            for k, v in result.get("citations", {}).items():
                logger.info(
                    f"  CITE  [{k}] arXiv:{v.get('paper_id','')} "
                    f"— {v.get('title','')[:70]}"
                )
            for k, v in result.get("web_citations", {}).items():
                logger.info(
                    f"  WCITE [{k}] {v.get('title','')[:60]} "
                    f"| {v.get('url','')}"
                )

            # ── Hallucination verdict ─────────────────────────────
            verdict = "HALLUCINATED" if result["is_hallucinated"] else "GROUNDED"
            logger.info(
                f"VERDICT  : {verdict}  score={result['score']:.4f}  "
                f"type={result['hallucination_type']}"
            )
            if result["is_hallucinated"]:
                logger.warning(f"EXPLAIN  : {result['explanation']}")
                for s in result.get("problematic_sentences", []):
                    logger.warning(f"  FLAGGED: {s!r}")
            else:
                logger.info(f"EXPLAIN  : {result['explanation']}")

            logger.info(f"ELAPSED  : {elapsed:.1f}s")
            logger.info("=" * 60)

            sync_queue.put_nowait({"type": "result", "data": _serialize_result(result)})
        except Exception as exc:
            logger.error(f"ERROR    : {exc}", exc_info=True)
            logger.info("=" * 60)
            sync_queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            sync_queue.put_nowait(None)

    threading.Thread(target=run_pipeline, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(
                None, lambda: sync_queue.get(timeout=300)
            )
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
