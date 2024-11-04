from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "postgresql://practiceDB_owner:RDCok52HPzaE@ep-falling-cloud-a59e5jpy.us-east-2.aws.neon.tech/practiceDB?sslmode=require"
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost:5432/RAG_FastAPI"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
