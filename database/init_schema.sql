-- Database Initialization Script for Chatbot + RAG Testing
-- Database: hackathon_test
-- Purpose: Local testing of Chatbot Agent with Memory System

-- ============================================
-- Step 1: Create Database (run this manually first)
-- ============================================
-- CREATE DATABASE hackathon_test;
-- \c hackathon_test

-- ============================================
-- Step 2: Install pgvector Extension
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- Step 3: Create Tables
-- ============================================

-- User Table
CREATE TABLE IF NOT EXISTS "user" (
    user_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255) UNIQUE NOT NULL,
    dob DATE NOT NULL,
    agent_type INTEGER NOT NULL,
    gender VARCHAR(50) NOT NULL,
    education_field VARCHAR(100),
    education_level VARCHAR(100),
    genai_usage_frequency VARCHAR(64),
    field_of_education_other VARCHAR(100),
    current_level_of_education_other VARCHAR(100),
    disability_knowledge VARCHAR(100) NOT NULL,
    genai_course_exp VARCHAR(100) NOT NULL,
    token VARCHAR(255) UNIQUE,
    registration_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation Table
CREATE TABLE IF NOT EXISTS conversation (
    conversation_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    message_type VARCHAR(50) NOT NULL,  -- "user_input" or "agent_response"
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    character_count INTEGER,
    sequence_number INTEGER,
    role VARCHAR(20),  -- "user" or "assistant"
    mode VARCHAR(20),  -- "chat" or "eval"
    agent_type INTEGER
);

-- Memory Vectors Table (for RAG)
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    memory_type VARCHAR(50) NOT NULL,  -- "conversation_chunk", "round_summary", "eval_summary"
    source_conversations TEXT,  -- JSON array of conversation IDs
    memory_content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,  -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    _metadata JSONB
);

-- ============================================
-- Step 4: Create Indexes for Performance
-- ============================================

-- User indexes
CREATE INDEX IF NOT EXISTS idx_user_email ON "user"(email);
CREATE INDEX IF NOT EXISTS idx_user_token ON "user"(token);

-- Conversation indexes
CREATE INDEX IF NOT EXISTS idx_conversation_user_id ON conversation(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation(timestamp);
CREATE INDEX IF NOT EXISTS idx_conversation_mode ON conversation(mode);

-- Memory vectors indexes
CREATE INDEX IF NOT EXISTS idx_memory_user_id ON memory_vectors(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_vectors(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_vectors(created_at);

-- Vector similarity index (IVFFlat for fast approximate search)
-- Note: Only create this after inserting some data (>100 rows recommended)
-- CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_vectors
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================
-- Step 5: Verify Installation
-- ============================================

-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check tables
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- ============================================
-- Notes
-- ============================================
-- 1. Run this script using: psql -U your_username -d hackathon_test -f init_schema.sql
-- 2. Make sure to create the database first manually
-- 3. pgvector must be installed on your PostgreSQL instance
-- 4. For production, consider adding foreign key constraints
