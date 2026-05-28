FROM python:3.11-slim

# uv installer
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project metadata and install deps into a venv
COPY pyproject.toml ./
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -r pyproject.toml

# Copy code
COPY app/ ./app/
COPY frontend/ ./frontend/

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
