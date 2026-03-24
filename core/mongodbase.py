from pymongo import MongoClient
import os
from dotenv import load_dotenv

MONGO_URL = os.getenv("MONGO_DB_URL")

client = MongoClient(MONGO_URL)
aml_db = client["aml_db"]
kyc_db = client["kyc_db"]
