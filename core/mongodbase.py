import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Ensure scripts that import this module directly also get .env values.
load_dotenv()

MONGO_URL = os.getenv("MONGO_DB_URL")

if not MONGO_URL:
	print("WARNING: MONGO_DB_URL is not set; using local MongoDB fallback for startup.")
	MONGO_URL = "mongodb://127.0.0.1:27017"

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
aml_db = client["aml_db"]
kyc_db = client["kyc_db"]
