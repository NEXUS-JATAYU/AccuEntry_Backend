# AccuEntry Backend — Cloud Run (PORT=8080) / local: docker compose up → :8000
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python - <<'PY'
from pathlib import Path

source = Path('requirements.txt')
data = source.read_bytes()

for encoding in ('utf-8-sig', 'utf-16', 'utf-16le', 'utf-16be'):
    try:
        text = data.decode(encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise SystemExit('Could not decode requirements.txt')

Path('/tmp/requirements.txt').write_text(text, encoding='utf-8')
PY
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'))"

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
