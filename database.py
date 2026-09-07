from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Boolean, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ─── Models ───────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─── Init & Migration ────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_add_user_id()


def _migrate_add_user_id():
    """Add user_id column to existing tables if they don't have it (SQLite migration)."""
    with engine.connect() as conn:
        # Check conversations table
        result = conn.execute(text("PRAGMA table_info(conversations)"))
        columns = [row[1] for row in result]
        if "user_id" not in columns:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN user_id INTEGER"))
            conn.commit()
            print("[OK] Migrated: added user_id to conversations")

        # Check chat_messages table
        result = conn.execute(text("PRAGMA table_info(chat_messages)"))
        columns = [row[1] for row in result]
        if "user_id" not in columns:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN user_id INTEGER"))
            conn.commit()
            print("[OK] Migrated: added user_id to chat_messages")


# ─── Conversation CRUD ───────────────────────────────────────

def create_or_update_conversation(
    thread_id: str,
    first_message: str | None = None,
    user_id: int | None = None
):
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            if first_message:
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                user_id=user_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.add(conversation)

        else:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def list_conversations(user_id: int | None = None):
    """List conversations for a specific user."""
    db = SessionLocal()

    try:
        query = db.query(Conversation)

        if user_id is not None:
            query = query.filter(Conversation.user_id == user_id)

        return (
            query
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def list_all_conversations():
    """List ALL conversations (for admin)."""
    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def list_conversations_by_user(user_id: int):
    """List conversations for a specific user (admin use)."""
    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def get_conversation_owner(thread_id: str) -> int | None:
    """Get the user_id who owns a conversation."""
    db = SessionLocal()
    try:
        conv = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )
        return conv.user_id if conv else None
    finally:
        db.close()


# ─── Chat Messages ────────────────────────────────────────────

def save_chat_message(
    thread_id: str,
    role: str,
    content: str,
    user_id: int | None = None
):
    db = SessionLocal()

    try:
        msg = ChatMessage(
            thread_id=thread_id,
            user_id=user_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )

        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def get_chat_history(thread_id: str):
    db = SessionLocal()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()


# ─── Long-Term Memory ────────────────────────────────────────

def save_memory(thread_id: str, memory: str):
    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=datetime.utcnow()
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()


def search_memory(thread_id: str, query: str):
    db = SessionLocal()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()


# ─── Admin Stats ──────────────────────────────────────────────

def get_admin_stats():
    """Get overall stats for admin dashboard."""
    db = SessionLocal()
    try:
        from auth import User
        total_users = db.query(User).count()
        total_conversations = db.query(Conversation).count()
        total_messages = db.query(ChatMessage).count()
        return {
            "total_users": total_users,
            "total_conversations": total_conversations,
            "total_messages": total_messages
        }
    finally:
        db.close()


def count_user_conversations(user_id: int) -> int:
    """Count conversations for a specific user."""
    db = SessionLocal()
    try:
        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .count()
        )
    finally:
        db.close()


def delete_conversation(thread_id: str, user_id: int = None, is_admin: bool = False):
    """Delete a conversation and all its messages.
    Normal user can only delete their own. Admin can delete any."""
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        if not conv:
            return False

        # Check ownership for non-admin
        if not is_admin and conv.user_id != user_id:
            return False

        # Delete messages first
        db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).delete()
        # Delete conversation
        db.query(Conversation).filter(Conversation.thread_id == thread_id).delete()
        # Delete memory
        db.query(LongTermMemory).filter(LongTermMemory.thread_id == thread_id).delete()

        db.commit()
        return True
    finally:
        db.close()