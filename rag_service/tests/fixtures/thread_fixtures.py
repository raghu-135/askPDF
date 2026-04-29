"""
thread_fixtures.py - Sample thread data for testing.

This module provides sample thread data structures for use in tests.
"""

from datetime import datetime


def sample_thread_data():
    """Generate sample thread data."""
    return {
        "name": "Test Thread",
        "embed_model": "BAAI/bge-m3",
        "settings": {
            "max_iterations": 10,
            "token_budget": 8192,
            "temperature": 0.7
        }
    }


def sample_thread_with_complex_settings():
    """Generate thread with complex nested settings."""
    return {
        "name": "Complex Settings Thread",
        "embed_model": "openai/text-embedding-3-small",
        "settings": {
            "max_iterations": 20,
            "token_budget": 16384,
            "nested": {
                "level1": {
                    "level2": {
                        "value": 42,
                        "array": [1, 2, 3]
                    }
                }
            },
            "features": {
                "web_search": True,
                "code_execution": False
            }
        }
    }


def sample_threads_list(count=3):
    """Generate a list of sample threads."""
    models = ["BAAI/bge-m3", "openai/text-embedding-3-small", "cohere/embed-english-v3.0"]
    threads = []
    for i in range(count):
        threads.append({
            "name": f"Thread {i}",
            "embed_model": models[i % len(models)],
            "settings": {
                "max_iterations": 10 + i,
                "token_budget": 8192 * (i + 1)
            }
        })
    return threads


def sample_thread_settings_variations():
    """Generate threads with different setting configurations."""
    return [
        {
            "name": "Minimal Thread",
            "embed_model": "BAAI/bge-m3",
            "settings": {}
        },
        {
            "name": "Conservative Thread",
            "embed_model": "BAAI/bge-m3",
            "settings": {
                "max_iterations": 5,
                "token_budget": 4096,
                "temperature": 0.3
            }
        },
        {
            "name": "Aggressive Thread",
            "embed_model": "BAAI/bge-m3",
            "settings": {
                "max_iterations": 30,
                "token_budget": 32768,
                "temperature": 1.0
            }
        }
    ]
