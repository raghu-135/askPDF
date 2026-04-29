"""
test_schema_validation_pytest.py - DB-agnostic schema validation tests.

This module validates the database schema definition to ensure:
- All required tables are defined
- All required columns exist with correct types
- Foreign keys and indexes are properly defined
- Check constraints are present

These tests parse the SCHEMA string from config.py and validate it
without requiring a database connection, making them fast and DB-agnostic.
"""

import os
import sys
import re
from typing import Dict, List, Set

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db.config import SCHEMA


class TestSchemaValidation:
    """Test suite for validating database schema definition."""

    @pytest.fixture
    def parsed_schema(self) -> Dict[str, Dict]:
        """
        Parse the SCHEMA string into a structured format.
        
        Returns:
            Dict mapping table names to their column definitions
        """
        tables = {}
        
        # Extract CREATE TABLE statements
        table_pattern = r'CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\);'
        for match in re.finditer(table_pattern, SCHEMA, re.DOTALL):
            table_name = match.group(1)
            table_body = match.group(2)
            
            # Parse columns and constraints
            columns = {}
            for line in table_body.split(','):
                line = line.strip()
                if not line:
                    continue
                
                # Skip constraints that span multiple lines
                if line.startswith('FOREIGN KEY') or line.startswith('PRIMARY KEY'):
                    continue
                
                # Parse column definition
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0]
                    col_type = parts[1]
                    columns[col_name] = col_type
            
            tables[table_name] = columns
        
        return tables

    @pytest.fixture
    def parsed_indexes(self) -> List[Dict]:
        """
        Parse index definitions from SCHEMA.
        
        Returns:
            List of index definitions with table and column info
        """
        indexes = []
        
        # Extract CREATE INDEX statements
        index_pattern = r'CREATE INDEX IF NOT EXISTS (\w+) ON (\w+)\((\w+)\)'
        for match in re.finditer(index_pattern, SCHEMA):
            indexes.append({
                'name': match.group(1),
                'table': match.group(2),
                'column': match.group(3)
            })
        
        return indexes

    @pytest.fixture
    def parsed_foreign_keys(self) -> List[Dict]:
        """
        Parse foreign key constraints from SCHEMA.
        
        Returns:
            List of foreign key definitions
        """
        foreign_keys = []
        
        # Extract FOREIGN KEY constraints
        fk_pattern = r'FOREIGN KEY \((\w+)\) REFERENCES (\w+)\((\w+)\)(?: ON DELETE (\w+))?'
        for match in re.finditer(fk_pattern, SCHEMA):
            foreign_keys.append({
                'column': match.group(1),
                'ref_table': match.group(2),
                'ref_column': match.group(3),
                'on_delete': match.group(4) if match.group(4) else None
            })
        
        return foreign_keys

    @pytest.fixture
    def parsed_check_constraints(self) -> List[Dict]:
        """
        Parse CHECK constraints from SCHEMA.
        
        Returns:
            List of check constraint definitions
        """
        check_constraints = []
        
        # Extract CHECK constraints
        check_pattern = r'CHECK \((.*?)\)'
        for match in re.finditer(check_pattern, SCHEMA):
            check_constraints.append({
                'constraint': match.group(1)
            })
        
        return check_constraints

    # Table existence tests
    def test_threads_table_exists(self, parsed_schema):
        """Test that threads table is defined."""
        assert 'threads' in parsed_schema, "threads table not found in schema"

    def test_files_table_exists(self, parsed_schema):
        """Test that files table is defined."""
        assert 'files' in parsed_schema, "files table not found in schema"

    def test_thread_files_table_exists(self, parsed_schema):
        """Test that thread_files table is defined."""
        assert 'thread_files' in parsed_schema, "thread_files table not found in schema"

    def test_thread_file_annotations_table_exists(self, parsed_schema):
        """Test that thread_file_annotations table is defined."""
        assert 'thread_file_annotations' in parsed_schema, "thread_file_annotations table not found in schema"

    def test_messages_table_exists(self, parsed_schema):
        """Test that messages table is defined."""
        assert 'messages' in parsed_schema, "messages table not found in schema"

    def test_thread_stats_table_exists(self, parsed_schema):
        """Test that thread_stats table is defined."""
        assert 'thread_stats' in parsed_schema, "thread_stats table not found in schema"

    # Thread table column tests
    def test_threads_table_has_id_column(self, parsed_schema):
        """Test that threads table has id column."""
        assert 'id' in parsed_schema['threads'], "threads.id column not found"
        assert parsed_schema['threads']['id'] == 'TEXT', "threads.id should be TEXT"

    def test_threads_table_has_name_column(self, parsed_schema):
        """Test that threads table has name column."""
        assert 'name' in parsed_schema['threads'], "threads.name column not found"
        assert parsed_schema['threads']['name'] == 'TEXT', "threads.name should be TEXT"

    def test_threads_table_has_embed_model_column(self, parsed_schema):
        """Test that threads table has embed_model column."""
        assert 'embed_model' in parsed_schema['threads'], "threads.embed_model column not found"
        assert parsed_schema['threads']['embed_model'] == 'TEXT', "threads.embed_model should be TEXT"

    def test_threads_table_has_settings_column(self, parsed_schema):
        """Test that threads table has settings column."""
        assert 'settings' in parsed_schema['threads'], "threads.settings column not found"
        assert parsed_schema['threads']['settings'] == 'TEXT', "threads.settings should be TEXT"

    def test_threads_table_has_created_at_column(self, parsed_schema):
        """Test that threads table has created_at column."""
        assert 'created_at' in parsed_schema['threads'], "threads.created_at column not found"
        assert parsed_schema['threads']['created_at'] == 'TIMESTAMP', "threads.created_at should be TIMESTAMP"

    # File table column tests
    def test_files_table_has_file_hash_column(self, parsed_schema):
        """Test that files table has file_hash column."""
        assert 'file_hash' in parsed_schema['files'], "files.file_hash column not found"
        assert parsed_schema['files']['file_hash'] == 'TEXT', "files.file_hash should be TEXT"

    def test_files_table_has_file_name_column(self, parsed_schema):
        """Test that files table has file_name column."""
        assert 'file_name' in parsed_schema['files'], "files.file_name column not found"
        assert parsed_schema['files']['file_name'] == 'TEXT', "files.file_name should be TEXT"

    def test_files_table_has_file_path_column(self, parsed_schema):
        """Test that files table has file_path column."""
        assert 'file_path' in parsed_schema['files'], "files.file_path column not found"
        assert parsed_schema['files']['file_path'] == 'TEXT', "files.file_path should be TEXT"

    def test_files_table_has_source_type_column(self, parsed_schema):
        """Test that files table has source_type column."""
        assert 'source_type' in parsed_schema['files'], "files.source_type column not found"
        assert parsed_schema['files']['source_type'] == 'TEXT', "files.source_type should be TEXT"

    def test_files_table_has_file_status_column(self, parsed_schema):
        """Test that files table has file_status column."""
        assert 'file_status' in parsed_schema['files'], "files.file_status column not found"
        assert parsed_schema['files']['file_status'] == 'TEXT', "files.file_status should be TEXT"

    # Thread files table column tests
    def test_thread_files_table_has_thread_id_column(self, parsed_schema):
        """Test that thread_files table has thread_id column."""
        assert 'thread_id' in parsed_schema['thread_files'], "thread_files.thread_id column not found"
        assert parsed_schema['thread_files']['thread_id'] == 'TEXT', "thread_files.thread_id should be TEXT"

    def test_thread_files_table_has_file_hash_column(self, parsed_schema):
        """Test that thread_files table has file_hash column."""
        assert 'file_hash' in parsed_schema['thread_files'], "thread_files.file_hash column not found"
        assert parsed_schema['thread_files']['file_hash'] == 'TEXT', "thread_files.file_hash should be TEXT"

    def test_thread_files_table_has_added_at_column(self, parsed_schema):
        """Test that thread_files table has added_at column."""
        assert 'added_at' in parsed_schema['thread_files'], "thread_files.added_at column not found"
        assert parsed_schema['thread_files']['added_at'] == 'TIMESTAMP', "thread_files.added_at should be TIMESTAMP"

    # Thread file annotations table column tests
    def test_thread_file_annotations_table_has_thread_id_column(self, parsed_schema):
        """Test that thread_file_annotations table has thread_id column."""
        assert 'thread_id' in parsed_schema['thread_file_annotations'], "thread_file_annotations.thread_id column not found"
        assert parsed_schema['thread_file_annotations']['thread_id'] == 'TEXT', "thread_file_annotations.thread_id should be TEXT"

    def test_thread_file_annotations_table_has_file_hash_column(self, parsed_schema):
        """Test that thread_file_annotations table has file_hash column."""
        assert 'file_hash' in parsed_schema['thread_file_annotations'], "thread_file_annotations.file_hash column not found"
        assert parsed_schema['thread_file_annotations']['file_hash'] == 'TEXT', "thread_file_annotations.file_hash should be TEXT"

    def test_thread_file_annotations_table_has_annotations_json_column(self, parsed_schema):
        """Test that thread_file_annotations table has annotations_json column."""
        assert 'annotations_json' in parsed_schema['thread_file_annotations'], "thread_file_annotations.annotations_json column not found"
        assert parsed_schema['thread_file_annotations']['annotations_json'] == 'TEXT', "thread_file_annotations.annotations_json should be TEXT"

    def test_thread_file_annotations_table_has_created_at_column(self, parsed_schema):
        """Test that thread_file_annotations table has created_at column."""
        assert 'created_at' in parsed_schema['thread_file_annotations'], "thread_file_annotations.created_at column not found"
        assert parsed_schema['thread_file_annotations']['created_at'] == 'TIMESTAMP', "thread_file_annotations.created_at should be TIMESTAMP"

    def test_thread_file_annotations_table_has_updated_at_column(self, parsed_schema):
        """Test that thread_file_annotations table has updated_at column."""
        assert 'updated_at' in parsed_schema['thread_file_annotations'], "thread_file_annotations.updated_at column not found"
        assert parsed_schema['thread_file_annotations']['updated_at'] == 'TIMESTAMP', "thread_file_annotations.updated_at should be TIMESTAMP"

    # Messages table column tests
    def test_messages_table_has_id_column(self, parsed_schema):
        """Test that messages table has id column."""
        assert 'id' in parsed_schema['messages'], "messages.id column not found"
        assert parsed_schema['messages']['id'] == 'TEXT', "messages.id should be TEXT"

    def test_messages_table_has_thread_id_column(self, parsed_schema):
        """Test that messages table has thread_id column."""
        assert 'thread_id' in parsed_schema['messages'], "messages.thread_id column not found"
        assert parsed_schema['messages']['thread_id'] == 'TEXT', "messages.thread_id should be TEXT"

    def test_messages_table_has_role_column(self, parsed_schema):
        """Test that messages table has role column."""
        assert 'role' in parsed_schema['messages'], "messages.role column not found"
        assert parsed_schema['messages']['role'] == 'TEXT', "messages.role should be TEXT"

    def test_messages_table_has_content_column(self, parsed_schema):
        """Test that messages table has content column."""
        assert 'content' in parsed_schema['messages'], "messages.content column not found"
        assert parsed_schema['messages']['content'] == 'TEXT', "messages.content should be TEXT"

    def test_messages_table_has_context_compact_column(self, parsed_schema):
        """Test that messages table has context_compact column."""
        assert 'context_compact' in parsed_schema['messages'], "messages.context_compact column not found"
        assert parsed_schema['messages']['context_compact'] == 'TEXT', "messages.context_compact should be TEXT"

    def test_messages_table_has_reasoning_column(self, parsed_schema):
        """Test that messages table has reasoning column."""
        assert 'reasoning' in parsed_schema['messages'], "messages.reasoning column not found"
        assert parsed_schema['messages']['reasoning'] == 'TEXT', "messages.reasoning should be TEXT"

    def test_messages_table_has_reasoning_available_column(self, parsed_schema):
        """Test that messages table has reasoning_available column."""
        assert 'reasoning_available' in parsed_schema['messages'], "messages.reasoning_available column not found"
        assert parsed_schema['messages']['reasoning_available'] == 'INTEGER', "messages.reasoning_available should be INTEGER"

    def test_messages_table_has_reasoning_format_column(self, parsed_schema):
        """Test that messages table has reasoning_format column."""
        assert 'reasoning_format' in parsed_schema['messages'], "messages.reasoning_format column not found"
        assert parsed_schema['messages']['reasoning_format'] == 'TEXT', "messages.reasoning_format should be TEXT"

    def test_messages_table_has_web_sources_column(self, parsed_schema):
        """Test that messages table has web_sources column."""
        assert 'web_sources' in parsed_schema['messages'], "messages.web_sources column not found"
        assert parsed_schema['messages']['web_sources'] == 'TEXT', "messages.web_sources should be TEXT"

    def test_messages_table_has_created_at_column(self, parsed_schema):
        """Test that messages table has created_at column."""
        assert 'created_at' in parsed_schema['messages'], "messages.created_at column not found"
        assert parsed_schema['messages']['created_at'] == 'TIMESTAMP', "messages.created_at should be TIMESTAMP"

    # Thread stats table column tests
    def test_thread_stats_table_has_thread_id_column(self, parsed_schema):
        """Test that thread_stats table has thread_id column."""
        assert 'thread_id' in parsed_schema['thread_stats'], "thread_stats.thread_id column not found"
        assert parsed_schema['thread_stats']['thread_id'] == 'TEXT', "thread_stats.thread_id should be TEXT"

    def test_thread_stats_table_has_total_qa_pairs_column(self, parsed_schema):
        """Test that thread_stats table has total_qa_pairs column."""
        assert 'total_qa_pairs' in parsed_schema['thread_stats'], "thread_stats.total_qa_pairs column not found"
        assert parsed_schema['thread_stats']['total_qa_pairs'] == 'INTEGER', "thread_stats.total_qa_pairs should be INTEGER"

    def test_thread_stats_table_has_total_qa_chars_column(self, parsed_schema):
        """Test that thread_stats table has total_qa_chars column."""
        assert 'total_qa_chars' in parsed_schema['thread_stats'], "thread_stats.total_qa_chars column not found"
        assert parsed_schema['thread_stats']['total_qa_chars'] == 'INTEGER', "thread_stats.total_qa_chars should be INTEGER"

    def test_thread_stats_table_has_avg_qa_chars_column(self, parsed_schema):
        """Test that thread_stats table has avg_qa_chars column."""
        assert 'avg_qa_chars' in parsed_schema['thread_stats'], "thread_stats.avg_qa_chars column not found"
        assert parsed_schema['thread_stats']['avg_qa_chars'] == 'REAL', "thread_stats.avg_qa_chars should be REAL"

    def test_thread_stats_table_has_last_qa_at_column(self, parsed_schema):
        """Test that thread_stats table has last_qa_at column."""
        assert 'last_qa_at' in parsed_schema['thread_stats'], "thread_stats.last_qa_at column not found"
        assert parsed_schema['thread_stats']['last_qa_at'] == 'TIMESTAMP', "thread_stats.last_qa_at should be TIMESTAMP"

    def test_thread_stats_table_has_documents_meta_column(self, parsed_schema):
        """Test that thread_stats table has documents_meta column."""
        assert 'documents_meta' in parsed_schema['thread_stats'], "thread_stats.documents_meta column not found"
        assert parsed_schema['thread_stats']['documents_meta'] == 'TEXT', "thread_stats.documents_meta should be TEXT"

    def test_thread_stats_table_has_last_updated_at_column(self, parsed_schema):
        """Test that thread_stats table has last_updated_at column."""
        assert 'last_updated_at' in parsed_schema['thread_stats'], "thread_stats.last_updated_at column not found"
        assert parsed_schema['thread_stats']['last_updated_at'] == 'TIMESTAMP', "thread_stats.last_updated_at should be TIMESTAMP"

    # Index tests
    def test_messages_thread_id_index_exists(self, parsed_indexes):
        """Test that idx_messages_thread_id index exists."""
        index_names = [idx['name'] for idx in parsed_indexes]
        assert 'idx_messages_thread_id' in index_names, "idx_messages_thread_id index not found"

    def test_thread_files_thread_id_index_exists(self, parsed_indexes):
        """Test that idx_thread_files_thread_id index exists."""
        index_names = [idx['name'] for idx in parsed_indexes]
        assert 'idx_thread_files_thread_id' in index_names, "idx_thread_files_thread_id index not found"

    def test_thread_files_file_hash_index_exists(self, parsed_indexes):
        """Test that idx_thread_files_file_hash index exists."""
        index_names = [idx['name'] for idx in parsed_indexes]
        assert 'idx_thread_files_file_hash' in index_names, "idx_thread_files_file_hash index not found"

    # Foreign key tests
    def test_thread_files_has_foreign_key_to_threads(self, parsed_foreign_keys):
        """Test that thread_files has foreign key to threads."""
        thread_file_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['ref_table'] == 'threads']
        assert len(thread_file_fks) > 0, "thread_files should have foreign key to threads"

    def test_thread_files_has_foreign_key_to_files(self, parsed_foreign_keys):
        """Test that thread_files has foreign key to files."""
        thread_file_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'file_hash' and fk['ref_table'] == 'files']
        assert len(thread_file_fks) > 0, "thread_files should have foreign key to files"

    def test_messages_has_foreign_key_to_threads(self, parsed_foreign_keys):
        """Test that messages has foreign key to threads."""
        message_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['ref_table'] == 'threads']
        assert len(message_fks) > 0, "messages should have foreign key to threads"

    def test_thread_stats_has_foreign_key_to_threads(self, parsed_foreign_keys):
        """Test that thread_stats has foreign key to threads."""
        stats_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['ref_table'] == 'threads']
        assert len(stats_fks) > 0, "thread_stats should have foreign key to threads"

    # Check constraint tests
    def test_messages_role_check_constraint(self, parsed_check_constraints):
        """Test that messages.role has check constraint for user/assistant."""
        role_checks = [c for c in parsed_check_constraints if 'role' in c['constraint'] and 'user' in c['constraint'] and 'assistant' in c['constraint']]
        assert len(role_checks) > 0, "messages.role should have check constraint for user/assistant"

    # Cascade delete tests
    def test_thread_files_cascade_delete(self, parsed_foreign_keys):
        """Test that thread_files has CASCADE delete on thread_id."""
        cascade_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['on_delete'] == 'CASCADE']
        assert len(cascade_fks) > 0, "thread_files should have CASCADE delete on thread_id"

    def test_messages_cascade_delete(self, parsed_foreign_keys):
        """Test that messages has CASCADE delete on thread_id."""
        cascade_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['on_delete'] == 'CASCADE']
        assert len(cascade_fks) > 0, "messages should have CASCADE delete on thread_id"

    def test_thread_stats_cascade_delete(self, parsed_foreign_keys):
        """Test that thread_stats has CASCADE delete on thread_id."""
        cascade_fks = [fk for fk in parsed_foreign_keys if fk['column'] == 'thread_id' and fk['on_delete'] == 'CASCADE']
        assert len(cascade_fks) > 0, "thread_stats should have CASCADE delete on thread_id"
