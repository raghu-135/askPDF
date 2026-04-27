"""
message_fixtures.py - Sample message data for testing.

This module provides sample message data structures for use in tests.
"""

from datetime import datetime


def sample_message_data(thread_id="thread-123"):
    """Generate sample message data."""
    return {
        "thread_id": thread_id,
        "role": "user",
        "content": "Hello, world!",
        "context_compact": "Compact context text",
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


def sample_assistant_message(thread_id="thread-123"):
    """Generate sample assistant message with reasoning."""
    return {
        "thread_id": thread_id,
        "role": "assistant",
        "content": "Here's the answer to your question.",
        "context_compact": "User asked about X",
        "reasoning": "Step 1: Analyze question\nStep 2: Search for information\nStep 3: Formulate answer",
        "reasoning_available": True,
        "reasoning_format": "markdown",
        "web_sources": [
            {"url": "http://example.com", "title": "Example", "snippet": "Example content"}
        ]
    }


def sample_conversation(thread_id="thread-123", count=6):
    """Generate a sample conversation with alternating roles."""
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user":
            messages.append({
                "thread_id": thread_id,
                "role": role,
                "content": f"User message {i // 2 + 1}",
                "context_compact": None,
                "reasoning": None,
                "reasoning_available": False,
                "reasoning_format": "none",
                "web_sources": []
            })
        else:
            messages.append({
                "thread_id": thread_id,
                "role": role,
                "content": f"Assistant response {i // 2 + 1}",
                "context_compact": f"User asked: User message {i // 2 + 1}",
                "reasoning": f"Reasoning for response {i // 2 + 1}",
                "reasoning_available": True,
                "reasoning_format": "markdown",
                "web_sources": []
            })
    return messages


def sample_message_with_web_sources(thread_id="thread-123"):
    """Generate message with multiple web sources."""
    return {
        "thread_id": thread_id,
        "role": "assistant",
        "content": "Based on the sources I found...",
        "web_sources": [
            {
                "url": "http://example1.com",
                "title": "Example 1",
                "snippet": "Content from example 1"
            },
            {
                "url": "http://example2.com",
                "title": "Example 2",
                "snippet": "Content from example 2"
            },
            {
                "url": "http://example3.com",
                "title": "Example 3",
                "snippet": "Content from example 3"
            }
        ]
    }


def sample_system_message(thread_id="thread-123"):
    """Generate a system message."""
    return {
        "thread_id": thread_id,
        "role": "system",
        "content": "You are a helpful assistant.",
        "context_compact": None,
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


def sample_long_message(thread_id="thread-123"):
    """Generate a message with very long content."""
    long_content = "This is a test. " * 1000  # ~15,000 characters
    return {
        "thread_id": thread_id,
        "role": "assistant",
        "content": long_content,
        "context_compact": "Long response to user query",
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


def sample_unicode_message(thread_id="thread-123"):
    """Generate message with unicode characters."""
    return {
        "thread_id": thread_id,
        "role": "user",
        "content": "Hello 世界 🌍 Привет مرحبا こんにちは",
        "context_compact": None,
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


def sample_messages_list(thread_id="thread-123", count=10):
    """Generate a list of sample messages."""
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "thread_id": thread_id,
            "role": role,
            "content": f"Message {i}",
            "context_compact": None,
            "reasoning": None,
            "reasoning_available": False,
            "reasoning_format": "none",
            "web_sources": []
        })
    return messages
