from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Ensure scripts that import this module directly also get .env values.
load_dotenv()

MONGO_URL = os.getenv("MONGO_DB_URL")

if not MONGO_URL:
	raise RuntimeError("MONGO_DB_URL is not set. Configure it in AccuEntry_Backend/.env")

client = MongoClient(MONGO_URL)
aml_db = client["aml_db"]
kyc_db = client["kyc_db"]
