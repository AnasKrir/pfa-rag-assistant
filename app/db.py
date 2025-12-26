# app/db.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
import json
from sqlalchemy import Boolean

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

# ---- DB PATH (stored inside /app for simplicity) ----
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False)  # student | teacher | admin

    chats = relationship("ChatLog", back_populates="user", cascade="all, delete-orphan")


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="chats")



class DocRequest(Base):
    __tablename__ = "doc_requests"

    id = Column(Integer, primary_key=True, index=True)

    doc_type = Column(String(50), nullable=False, index=True)  # internship_attestation
    status = Column(String(30), nullable=False, index=True)    # pending|approved|refused|generated

    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    approved_by_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    processed_by_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    admin_comment = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=False)                # JSON string des champs
    generated_path = Column(String(255), nullable=True)

    student = relationship("User", foreign_keys=[student_id])
    approved_by = relationship("User", foreign_keys=[approved_by_id])
    processed_by = relationship("User", foreign_keys=[processed_by_id])

    def payload(self) -> dict:
        try:
            return json.loads(self.payload_json or "{}")
        except Exception:
            return {}



def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create tables if they don't exist.
    """
    Base.metadata.create_all(bind=engine)