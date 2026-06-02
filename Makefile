.PHONY: install ingest run test eval eval-retrieval docker-up docker-down clean

install:
	uv sync

ingest:
	uv run python -m ingest.embed_and_index

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

test:
	uv run pytest -q

# Run the evaluation set in-process (retrieval + LLM-judged behaviour).
# Pass extra flags via ARGS, e.g. `make eval ARGS="--limit 3 --category single_doc_factual"`.
eval:
	uv run python -m evals.run $(ARGS)

# Retrieval-only: skip the LLM judge (still generates answers).
eval-retrieval:
	uv run python -m evals.run --no-judge $(ARGS)

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	rm -rf chroma_db .venv __pycache__ */__pycache__
