# financial-reports-rag

Retrieval-augmented generation over annual reports of five Swiss-listed companies: Roche, Novartis, UBS, Nestlé, and Zurich Insurance (2021–2025).

PDFs are parsed page-by-page, chunked with overlap, embedded with Voyage AI, and indexed in a local Chroma vector store. Retrieval results are passed to Claude to answer questions with citations back to the source company, year, and page.

A FastAPI service exposes a `/query` endpoint and a minimal static HTML frontend (single-shot textbox — interactive chat is upcoming). A pluggable safety layer sits between retrieval and generation.

## Stack

- **Embeddings:** Voyage AI (`voyage-4`)
- **LLM:** Anthropic Claude (`claude-haiku-4-5`)
- **Vector store:** Chroma (local, persisted)
- **Parsing:** `pypdf`
- **Chunking:** `langchain-text-splitters` (recursive, 1000 chars / 100 overlap)
- **Serving:** FastAPI + Uvicorn
- **Frontend:** static HTML/JS served by FastAPI (no build step)
- **Tests:** `pytest`
- **Packaging:** `uv` + `hatchling`
- **Container:** Dockerfile + `docker-compose`

## Project layout

```
app/             FastAPI app, retrieval, generation, safety, config
ingest/          PDF parsing, chunking, embedding, indexing
frontend/        Static single-page UI (index.html)
tests/           Pytest suite (retrieval, generation, ingestion)
data/
  raw/           Annual report PDFs, one folder per company (gitignored)
  company_metadata.json
Dockerfile
docker-compose.yml
```

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
make install
```

Create a `.env` from the template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `VOYAGE_API_KEY` | Voyage AI embeddings |
| `ANTHROPIC_API_KEY` | Claude API |
| `CHROMA_PERSIST_DIRECTORY` | Local path for the Chroma DB (e.g. `./chroma_db`) |
| `COLLECTION_NAME` | Chroma collection name |
| `EMBEDDING_MODEL` | Voyage model (default `voyage-4`) |
| `LLM_MODEL` | Claude model (default `claude-haiku-4-5`) |
| `TOP_K` | Number of chunks to retrieve per query |

## Data

PDFs are not committed (the corpus is ~110 MB). Drop annual report PDFs into `data/raw/<company>/<company>_<year>.pdf`, for example:

```
data/raw/roche/roche_2024.pdf
data/raw/ubs/ubs_2023.pdf
```

The companies are defined in [data/company_metadata.json](data/company_metadata.json). Add a new key there to extend the corpus.

## Ingest

Parses every PDF under `data/raw/`, chunks pages, embeds with Voyage, and writes to Chroma:

```bash
make ingest
```

The collection is recreated from scratch on each run.

## Run

```bash
make run
```

Serves FastAPI on `http://localhost:8080`:

- `GET /` — static frontend (single-shot question box, renders the answer plus expandable source chunks)
- `POST /query` — `{ "question": str, "top_k": int }` → `{ "answer": str, "sources": [...] }`
- `GET /health` — liveness probe

The frontend is intentionally static for now — no streaming, no multi-turn chat. Conversational chat is on the roadmap.

### Docker

```bash
make docker-up      # build + run via docker-compose
make docker-down
```

The compose file mounts `./chroma_db` and `./frontend` so the index and UI can be iterated on without rebuilding.

## Safety

Retrieved hits pass through a `GenerationSafety` strategy before being formatted into the LLM prompt. See [app/safety.py](app/safety.py). The protocol exposes two hooks:

- `add_hit(hit, filtered_hits)` — per-hit filter, decides whether a chunk is kept
- `check_hits(filtered_hits)` — collection-level check across the kept hits

The default `NoSafetyMechanism` is a pass-through. Real mechanisms (PII redaction, content-policy filters, etc.) plug in by implementing the same protocol.

## Tests

```bash
uv run pytest
```

Covers the retriever, the generator (with the safety pipeline), and the ingestion path.

## License

[MIT](LICENSE)
