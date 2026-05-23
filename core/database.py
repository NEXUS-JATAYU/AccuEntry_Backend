from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD") or "")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL
    logger.info("PostgreSQL configured via DATABASE_URL")
elif all([DB_USER, DB_HOST, DB_PORT, DB_NAME]):
    logger.info("PostgreSQL configured host=%s port=%s db=%s user=%s", DB_HOST, DB_PORT, DB_NAME, DB_USER)
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    logger.warning("PostgreSQL env incomplete — set DATABASE_URL or DB_USER, DB_HOST, DB_PORT, DB_NAME")
    SQLALCHEMY_DATABASE_URL = "postgresql://localhost/accuentry_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
