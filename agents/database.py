from sqlalchemy import create_engine, update, Column, Integer, String, Date, DateTime, Text, TIMESTAMP, BigInteger, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert, JSON
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import os
from typing import Optional, Dict, List, Any
from datetime import date

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not defined in .env")

# Create SQLAlchemy engine with resilient settings
# - pool_pre_ping: validates connections before use to avoid "SSL SYSCALL EOF" on stale conns
# - pool_recycle: proactively refresh connections after N seconds (handles server idle timeouts)
# - sslmode=require: ensure SSL for Neon if not already in the URL
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,  # recycle connections every 30 minutes
    pool_size=5,
    max_overflow=10,
    connect_args={
        "sslmode": "require"
    } if DATABASE_URL.startswith("postgresql") and "sslmode=" not in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Import json for database operations
import json

class User(Base):
    """User model for database operations"""
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    dob = Column(Date, nullable=False)
    agent_type = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    education_field = Column(String)
    education_level = Column(String)
    # New registration fields
    genai_usage_frequency = Column(String(64))
    field_of_education_other = Column(String(100))
    current_level_of_education_other = Column(String(100))
    disability_knowledge = Column(String, nullable=False)
    genai_course_exp = Column(String, nullable=False)
    token = Column(String, unique=True, index=True)
    registration_time = Column(DateTime, default=func.now())

class Conversation(Base):
    """Conversation model for storing chat messages"""
    __tablename__ = "conversation"

    conversation_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_type = Column(String, nullable=False)  # "user_input" or "agent_response" (existing)
    content = Column(Text, nullable=False)  # The actual message content
    timestamp = Column(DateTime, default=func.now())
    character_count = Column(Integer)
    sequence_number = Column(Integer)
    # New columns (will be added by SQL script)
    role = Column(String)  # "user" or "assistant" (new)
    mode = Column(String)  # "chat" or "eval" (new)
    agent_type = Column(Integer)  # Which agent was used (new)

class MemoryVector(Base):
    """Memory vector model for RAG functionality"""
    __tablename__ = "memory_vectors"

    memory_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    memory_type = Column(String(50), nullable=False)  # Type of memory with constraint
    source_conversations = Column(Text)  # Source conversation IDs
    memory_content = Column(Text, nullable=False)  # The text content
    embedding = Column(Vector(1536), nullable=False)  # VECTOR(1536) type for pgvector extension
    created_at = Column(DateTime, default=func.now())
    _metadata = Column(JSON)  # JSONB metadata

class DatabaseManager:
    """Database manager for user operations"""

    def __init__(self):
        self.db = SessionLocal()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def get_user_by_token(self, token: str) -> Optional[Dict]:
        """Get user by token using ORM"""
        try:
            user = self.db.query(User).filter(User.token == token).first()
            if user:
                return {
                    "user_id": user.user_id,
                    "email": user.email,
                    "dob": user.dob,
                    "gender": user.gender,
                    "education_field": user.education_field,
                    "education_level": user.education_level,
                    "genai_usage_frequency": user.genai_usage_frequency,
                    "field_of_education_other": user.field_of_education_other,
                    "current_level_of_education_other": user.current_level_of_education_other,
                    "agent_type": user.agent_type,
                    "disability_knowledge": user.disability_knowledge,
                    "genai_course_exp": user.genai_course_exp,
                    "token": user.token,
                    "registration_time": user.registration_time
                }
            return None
        except Exception as e:
            print(f"Error getting user by token: {e}")
            return None

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get complete user profile using ORM"""
        try:
            user = self.db.query(User).filter(User.user_id == user_id).first()
            if user:
                return {
                    "user_id": user.user_id,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "dob": user.dob,
                    "agent_type": user.agent_type,
                    "gender": user.gender,
                    "education_field": user.education_field,
                    "education_level": user.education_level,
                    "genai_usage_frequency": user.genai_usage_frequency,
                    "field_of_education_other": user.field_of_education_other,
                    "current_level_of_education_other": user.current_level_of_education_other,
                    "disability_knowledge": user.disability_knowledge,
                    "genai_course_exp": user.genai_course_exp,
                    "token": user.token,
                    "registration_time": user.registration_time
                }
            return None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None

    def save_conversation_message(self, user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
        """Save a conversation message to database"""
        try:
            # Get the next sequence number for this user
            last_conversation = self.db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.sequence_number.desc()).first()

            next_sequence = 1 if not last_conversation else last_conversation.sequence_number + 1

            # Map role to message_type format
            message_type = "user_input" if role == "user" else "agent_response"

            conversation = Conversation(
                user_id=user_id,
                message_type=message_type,  # "USER" or "ASSISTANT"
                content=message,
                character_count=len(message),
                sequence_number=next_sequence,
                role=role,  # "user" or "assistant"
                mode=mode,  # "chat" or "eval"
                agent_type=agent_type
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation.conversation_id
        except Exception as e:
            self.db.rollback()
            print(f"Error saving conversation message: {e}")
            return None

    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's conversation history from database"""
        try:
            conversations = self.db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.timestamp.asc()).limit(limit).all()

            return [
                {
                    "conversation_id": conv.conversation_id,
                    "content": conv.content,
                    "message_type": conv.message_type,
                    "timestamp": conv.timestamp,
                }
                for conv in conversations
            ]
        except Exception as e:
            print(f"Error getting user conversations: {e}")
            return []

# Convenience functions
def get_user_by_token(token: str) -> Optional[Dict]:
    """Get user by token"""
    with DatabaseManager() as db:
        return db.get_user_by_token(token)

def get_user_profile(user_id: int) -> Optional[Dict]:
    """Get complete user profile"""
    with DatabaseManager() as db:
        return db.get_user_profile(user_id)

def save_conversation_message(user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
    """Save a conversation message"""
    with DatabaseManager() as db:
        return db.save_conversation_message(user_id, message, role, mode, agent_type)

def get_user_conversations(user_id: int, limit: int = 50) -> List[Dict]:
    """Get user's conversation history"""
    with DatabaseManager() as db:
        return db.get_user_conversations(user_id, limit)

def get_user_message_count(user_id: int, mode: str) -> int:
    """Get total message count for a user in chat mode"""
    with DatabaseManager() as db:
        try:
            # Only support chat mode
            result = db.db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.mode == mode
            ).count()
            return result
        except Exception as e:
            print(f"Error getting message count for user {user_id}, mode {mode}: {e}")
            return 0

# Async database functions using thread pool for synchronous operations
# Note: These are not truly async I/O but provide async interface for compatibility

async def save_conversation_message_async(user_id: int, message: str, role: str, mode: str, agent_type: int) -> Optional[Column[int]]:
    """Save conversation message using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, save_conversation_message, user_id, message, role, mode, agent_type)

async def get_user_conversations_async(user_id: int, limit: int = 50) -> List[Dict]:
    """Get user conversations using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_conversations, user_id, limit)

async def get_user_profile_async(user_id: int) -> Optional[Dict]:
    """Get user profile using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_profile, user_id)

async def get_user_by_token_async(token: str) -> Optional[Dict]:
    """Get user by token using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_by_token, token)

async def get_user_message_count_async(user_id: int, mode: str) -> int:
    """Get total message count for a user in a specific mode using thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_user_message_count, user_id, mode)
