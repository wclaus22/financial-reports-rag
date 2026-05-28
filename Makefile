.PHONY: install ingest run docker-up docker-down clean

install:
	uv sync

ingest:
	uv run python -m ingest.embed_and_index

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	rm -rf chroma_db .venv __pycache__ */__pycache__
