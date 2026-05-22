import logging
import os
import redis.asyncio as redis
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
redis_url = os.getenv("REDIS_URL")
if redis_url:
    redis_client = redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis client configured (URL redacted)")
else:
    logger.warning("REDIS_URL not set; Redis session/OTP cache disabled")
    redis_client = None
