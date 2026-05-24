import logging
import os
import redis.asyncio as redis
from dotenv import load_dotenv

try:
    from upstash_redis.asyncio import Redis as UpstashRedis
except Exception:  # pragma: no cover - optional dependency.
    UpstashRedis = None

load_dotenv()
logger = logging.getLogger(__name__)


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _build_redis_client():
    rest_url = _strip_wrapping_quotes(os.getenv("UPSTASH_REDIS_REST_URL", ""))
    rest_token = _strip_wrapping_quotes(os.getenv("UPSTASH_REDIS_REST_TOKEN", ""))
    if rest_url and rest_token and UpstashRedis is not None:
        try:
            client = UpstashRedis(url=rest_url, token=rest_token, allow_telemetry=False)
            logger.info("Upstash Redis REST client configured")
            return client
        except Exception as exc:
            logger.warning("Upstash REST config invalid (%s)", exc)

    redis_url = _strip_wrapping_quotes(os.getenv("REDIS_URL", ""))
    if redis_url:
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            logger.info("Redis client configured from REDIS_URL (URL redacted)")
            return client
        except Exception as exc:
            logger.warning("REDIS_URL invalid (%s)", exc)

    logger.warning("Redis is not configured; Redis session/OTP cache disabled")
    return None


redis_client = _build_redis_client()
