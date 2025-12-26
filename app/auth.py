# app/auth.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Literal

from fastapi import Depends, Request, HTTPException
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .db import User, get_db

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

Role = Literal["student", "teacher", "admin"]

# Cookie name (simple session token = username)
SESSION_COOKIE = "pfa_session_user"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    username = request.cookies.get(SESSION_COOKIE)
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()


def require_user(user: Optional[User]) -> User:
    if user is None:
        # keep it simple: caller will redirect in routes
        raise PermissionError("Not authenticated")
    return user


def require_role(user: User, allowed: set[str]) -> None:
    if user.role not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden")


def seed_default_users(db: Session) -> None:
    """
    Create 3 default users if they don't exist.
    Username / Password:
      - student / student123
      - teacher / teacher123
      - admin   / admin123
    """
    defaults = [
        ("student", "student123", "student"),
        ("teacher", "teacher123", "teacher"),
        ("admin", "admin123", "admin"),
    ]

    for username, plain_password, role in defaults:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            # If role is wrong, fix it (helps during dev)
            if existing.role != role:
                existing.role = role
                db.add(existing)
            continue

        u = User(
            username=username,
            password_hash=hash_password(plain_password),
            role=role,
        )
        db.add(u)

    db.commit()