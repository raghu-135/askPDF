"""Shared string enums for RAG tool search and evidence payload values."""

from enum import Enum


class ThreadTimelineSource(str, Enum):
    ALL = "all"
    CONVERSATION = "conversation"
    DOCUMENTS = "documents"
    WEB_CACHE = "web_cache"


class ThreadTimelineOrder(str, Enum):
    RELEVANCE = "relevance"
    OLDEST = "oldest"
    NEWEST = "newest"


class TimelineSourceType(str, Enum):
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    WEB_CACHE = "web_cache"


class TimelineEventType(str, Enum):
    MESSAGE_CREATED = "message_created"
    DOCUMENT_ADDED_TO_THREAD = "document_added_to_thread"
    WEB_SEARCH_PERFORMED = "web_search_performed"


class ReembedSkipReason(str, Enum):
    EMBEDDING_MODEL_NOT_READY = "embedding_model_not_ready"
    REEMBED_IN_PROGRESS = "reembed_in_progress"
