"""
config.py - Database configuration and constants.

This module contains database path configuration, schema definitions,
and migration management.
"""

import os

# Database path - use /data for persistence in Docker
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR is None:
    raise ValueError("DATA_DIR environment variable is not set")
DB_PATH = os.path.join(DATA_DIR, "rag.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# SQL Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
    file_hash TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT,
    source_type TEXT NOT NULL DEFAULT 'pdf',
    file_status TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS thread_files (
    thread_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, file_hash),
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    FOREIGN KEY (file_hash) REFERENCES files(file_hash)
);

CREATE TABLE IF NOT EXISTS thread_file_annotations (
    thread_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    annotations_json TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, file_hash),
    FOREIGN KEY (thread_id, file_hash)
        REFERENCES thread_files(thread_id, file_hash)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    context_compact TEXT,
    reasoning TEXT,
    reasoning_available INTEGER NOT NULL DEFAULT 0,
    reasoning_format TEXT NOT NULL DEFAULT 'none',
    web_sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_files_thread_id ON thread_files(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_files_file_hash ON thread_files(file_hash);

CREATE TABLE IF NOT EXISTS thread_stats (
    thread_id       TEXT PRIMARY KEY,
    total_qa_pairs  INTEGER NOT NULL DEFAULT 0,
    total_qa_chars  INTEGER NOT NULL DEFAULT 0,
    avg_qa_chars    REAL    NOT NULL DEFAULT 0.0,
    last_qa_at      TIMESTAMP,
    documents_meta  TEXT    NOT NULL DEFAULT '{}',
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
"""


# Database migrations
MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN reasoning TEXT",
    "ALTER TABLE messages ADD COLUMN reasoning_available INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE messages ADD COLUMN reasoning_format TEXT NOT NULL DEFAULT 'none'",
    "ALTER TABLE messages ADD COLUMN context_compact TEXT",
    "ALTER TABLE threads ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE messages ADD COLUMN web_sources TEXT",
    "ALTER TABLE files ADD COLUMN source_type TEXT NOT NULL DEFAULT 'pdf'",
    "ALTER TABLE files ADD COLUMN parsed_sentences_json TEXT",
    "ALTER TABLE files ADD COLUMN file_status TEXT NOT NULL DEFAULT '{}'",
]
