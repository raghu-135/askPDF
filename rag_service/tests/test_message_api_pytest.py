"""
test_message_api_pytest.py - Message API endpoint contract tests.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.api import messages as messages_api


class TestMessageEndpoints:
    """Test suite for message endpoints."""

    @pytest.mark.asyncio
    async def test_delete_missing_message_is_idempotent(self):
        """Deleting a message that is already gone should not surface a 404."""
        with patch(
            "app.api.messages.get_message",
            new_callable=AsyncMock,
            return_value=None,
        ):
            data = await messages_api.delete_message_endpoint("missing-message-id")

        assert data == {"status": "not_found", "deleted_ids": []}
