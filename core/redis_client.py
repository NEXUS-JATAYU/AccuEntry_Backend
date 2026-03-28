import redis.asyncio as redis 
from dotenv import load_dotenv
import os 

load_dotenv()
redis_url = os.getenv("REDIS_URL")
print(f"Connecting to Redis at: {redis_url}")
if redis_url:
	redis_client = redis.from_url(redis_url, decode_responses=True)
else:
	print("[DEBUG][redis] REDIS_URL not set; Redis session cache disabled")
	redis_client = None
