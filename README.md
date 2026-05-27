# financial-reports-rag

Retrieval-augmented generation over annual reports of five Swiss-listed companies: Roche, Novartis, UBS, Nestlé, and Zurich Insurance (2021–2025).

PDFs are parsed page-by-page, chunked with overlap, embedded with Voyage AI, and indexed in a local Chroma vector store. Retrieval results are passed to Claude to answer questions with citations back to the source company, year, and page.

## Stack

- **Embeddings:** Voyage AI (`voyage-4`)
- **LLM:** Anthropic Claude (`claude-haiku-4-5`)
- **Vector store:** Chroma (local, persisted)
- **Parsing:** `pypdf`
- **Chunking:** `langchain-text-splitters` (recursive, 1000 chars / 100 overlap)
- **Serving:** FastAPI + Uvicorn
- **Packaging:** `uv` + `hatchling`

## Project layout

```
app/             FastAPI app + config
ingest/          PDF parsing, chunking, embedding, indexing
data/
  raw/           Annual report PDFs, one folder per company (gitignored)
  company_metadata.json
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

Serves FastAPI on `http://localhost:8080`.

## License

[MIT](LICENSE)
