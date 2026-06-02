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
  pipeline.py    Shared retrieve→generate orchestration (route + eval call this)
ingest/          PDF parsing, chunking, embedding, indexing
frontend/        Static single-page UI (index.html)
evals/           Evaluation harness (question set, runner, metrics, LLM judge)
  eval_set.json  Gold question set with expected answers and source pages
  results/       Timestamped run outputs (gitignored)
tests/           Pytest suite (retrieval, generation, ingestion, pipeline, judge)
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
| `LLM_MODEL` | Claude model for answer generation (default `claude-haiku-4-5`) |
| `JUDGE_MODEL` | Claude model for the eval judge (default `claude-sonnet-4-6`) |
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

## Evaluation

The [evals/](evals/) harness scores the system against a gold question set in [evals/eval_set.json](evals/eval_set.json). The runner drives the **same** `run_query` pipeline the `/query` endpoint uses ([app/pipeline.py](app/pipeline.py)) — in-process, no server required — so the eval can never drift from what's served.

```bash
make eval                  # full run: retrieval scoring + LLM-judged behaviour
make eval-retrieval        # retrieval only (skips the judge; still generates answers)

# extra flags pass straight through via ARGS:
make eval ARGS="--limit 3 --category single_doc_factual"
make eval ARGS="--id A001 --id B002"
```

A live run needs the Voyage + Anthropic keys and an ingested Chroma index, and spends API calls (one generation + one judge call per answerable question). Results are written to `evals/results/eval_<utc-timestamp>.json` and a per-category summary is printed.

### Two scoring axes

Each question is scored on two independent axes:

- **Retrieval** (deterministic) — did the retrieved chunks cover the question's gold source pages? Only applies to questions that have gold sources.
- **Behaviour** (LLM judge, [evals/judge.py](evals/judge.py)) — did the system do the right thing: answer correctly, refuse appropriately, and *not fabricate*? The judge is **not** told the expected behaviour, so it grades what actually happened; fabrication is judged only against the retrieved excerpts.

The summary table columns:

| Column | Meaning |
| --- | --- |
| `n` | questions in the category |
| `ret_n` | questions with gold sources (retrieval-applicable; refusal questions are excluded) |
| `ret_pass` | % where retrieval hit **all** required gold sources (or **any**, per `gold_match`) |
| `recall` | mean *fraction* of gold sources hit (partial-credit view of `ret_pass`) |
| `mrr` | mean reciprocal rank of the first gold-page hit (1.0 = top result, 0 = not in top-k) |
| `behav` | % passing the behaviour judge, over all `n` questions |

### Question schema

Each entry in `eval_set.json` carries an `expected_behavior` and a `gold_match`:

- **`expected_behavior`**: `answer` (must answer correctly), `refuse` (must decline — out-of-corpus, false-premise, injection, etc.), or `answer_or_refuse` (sparse topics where a grounded answer *or* an honest refusal both pass, but fabrication fails).
- **`gold_match`**: `all` (every gold source must be retrieved — cross-doc/multi-year) or `any` (one is sufficient — e.g. the same fact stated in two reports). Within a single source, hitting any of its listed pages counts.

## Tests

```bash
make test          # or: uv run pytest -q
```

Covers the retriever, the generator (with the safety pipeline), the shared query pipeline, the eval judge, and the ingestion path.

## License

[MIT](LICENSE)
