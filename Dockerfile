FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir upstash-redis
RUN python -c "from pathlib import Path; data=Path('requirements.txt').read_bytes(); text=(data.decode('utf-16') if data[:2] in (b'\\xff\\xfe', b'\\xfe\\xff') else data.decode('utf-8-sig')); Path('/tmp/requirements.txt').write_text(text, encoding='utf-8')"
RUN pip install --no-cache-dir --ignore-installed --prefix=/install -r /tmp/requirements.txt

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8082

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8082'))"

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8082}"]
