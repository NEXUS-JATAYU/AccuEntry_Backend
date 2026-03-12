import os
from dotenv import load_dotenv
load_dotenv()

from core.database import engine, Base
import models.customer_info

def test_db():
    print("Testing DB connection...")
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully (or already exist).")
        
        # Test connection by selecting
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Count customers
        count = db.query(models.customer_info.CustomerDetails).count()
        print(f"Current customer count: {count}")
        db.close()
        print("DB connection successful.")
    except Exception as e:
        print(f"Error testing DB: {e}")

if __name__ == "__main__":
    test_db()
