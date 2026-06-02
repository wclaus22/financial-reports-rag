"""FastAPI surface: POST query, return generated answer."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.generation import Generator
from app.logging import configure_logging
from app.models import QueryRequest, QueryResponse, Source
from app.pipeline import run_query
from app.retrieval import Retriever
from app.safety import GenerationSafety, NoSafetyMechanism

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Swiss Financial Reports RAG", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


_retriever: Retriever | None = None
_generator: Generator | None = None


def get_retriever() -> Retriever:
    # Not thread-safe: safe under a single asyncio event loop (Uvicorn default),
    # but concurrent first-requests across threads could construct two Retrievers.
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_generator() -> Generator:
    # Not thread-safe: safe under a single asyncio event loop (Uvicorn default),
    # but concurrent first-requests across threads could construct two Generators.
    global _generator
    if _generator is None:
        safety = GenerationSafety(NoSafetyMechanism())
        _generator = Generator(safety)
    return _generator


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # TODO: not async thread safe, but fine under Uvicorn's single-threaded event loop model
    try:
        retriever = get_retriever()
    except Exception as e:
        logger.exception("Retriever construction failed")
        raise HTTPException(
            500,
            f"Retriever unavailable. Has the ingestion job been run? Error: {e}",
        )
    try:
        generator = get_generator()
    except Exception as e:
        logger.exception("Generator construction failed")
        raise HTTPException(
            500,
            f"Generator unavailable. Error: {e}",
        )

    result = run_query(request.question, request.top_k, retriever, generator)
    sources = [Source(**hit.__dict__) for hit in result.grounding]
    return QueryResponse(answer=result.answer, sources=sources)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
