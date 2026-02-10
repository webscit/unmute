"""Regression test for instructions type handling.

This test verifies that the unmute_handler correctly handles both string and
Instructions object types when updating sessions, preventing AttributeError
when instructions.make_system_prompt() is called on a string.

Error fixed:
    AttributeError: 'str' object has no attribute 'make_system_prompt'
    at unmute/llm/chatbot.py:98 in set_instructions
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unmute.llm.system_prompt import ConstantInstructions, Instructions
from unmute.unmute_handler import UnmuteHandler


@pytest.fixture
def mock_handler():
    """Create a minimal mock UnmuteHandler for testing."""
    with patch("unmute.unmute_handler.Chatbot") as mock_chatbot_class:
        mock_chatbot = MagicMock()
        mock_chatbot_class.return_value = mock_chatbot

        # Create handler with minimal mocking
        with patch.object(UnmuteHandler, "__init__", lambda self: None):
            handler = UnmuteHandler()
            handler.chatbot = mock_chatbot
            handler.tts_voice = "default"

        return handler


@pytest.mark.asyncio
async def test_update_session_with_string_instructions(mock_handler):
    """Test that string instructions are converted to ConstantInstructions."""
    # Arrange: Create a session with string instructions
    session = {
        "instructions": "You are a helpful assistant.",
        "voice": "test_voice",
    }

    # Act: Update session
    await mock_handler.update_session(session)

    # Assert: set_instructions was called with a ConstantInstructions object
    mock_handler.chatbot.set_instructions.assert_called_once()
    call_args = mock_handler.chatbot.set_instructions.call_args[0][0]
    assert isinstance(call_args, ConstantInstructions)
    assert call_args.text == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_update_session_with_instructions_object(mock_handler):
    """Test that Instructions objects are passed through unchanged."""
    # Arrange: Create a session with an Instructions object
    instructions = ConstantInstructions(text="Custom instructions", language="fr")
    session = {
        "instructions": instructions,
        "voice": "test_voice",
    }

    # Act: Update session
    await mock_handler.update_session(session)

    # Assert: set_instructions was called with the same Instructions object
    mock_handler.chatbot.set_instructions.assert_called_once_with(instructions)


@pytest.mark.asyncio
async def test_update_session_with_no_instructions(mock_handler):
    """Test that sessions without instructions don't call set_instructions."""
    # Arrange: Create a session without instructions
    session = {
        "voice": "test_voice",
    }

    # Act: Update session
    await mock_handler.update_session(session)

    # Assert: set_instructions was not called
    mock_handler.chatbot.set_instructions.assert_not_called()


@pytest.mark.asyncio
async def test_update_session_with_none_instructions(mock_handler):
    """Test that None instructions don't call set_instructions."""
    # Arrange: Create a session with None instructions
    session = {
        "instructions": None,
        "voice": "test_voice",
    }

    # Act: Update session
    await mock_handler.update_session(session)

    # Assert: set_instructions was not called
    mock_handler.chatbot.set_instructions.assert_not_called()


def test_chatbot_set_instructions_type_signature():
    """Verify that Chatbot.set_instructions expects Instructions type.

    This test ensures that the type signature hasn't changed and still
    requires an Instructions object (not a string).
    """
    from inspect import signature

    from unmute.llm.chatbot import Chatbot

    sig = signature(Chatbot.set_instructions)
    instructions_param = sig.parameters["instructions"]

    # The annotation should be Instructions (not str)
    assert instructions_param.annotation is not None
    assert "Instructions" in str(instructions_param.annotation)
    assert "str" not in str(instructions_param.annotation).replace(
        "Instructions", ""
    )  # Avoid false positive from "Instructions"


def test_constant_instructions_has_make_system_prompt():
    """Verify ConstantInstructions has make_system_prompt method.

    This test demonstrates that ConstantInstructions (which we convert strings to)
    has the required make_system_prompt() method, preventing the AttributeError:
        'str' object has no attribute 'make_system_prompt'
    """
    # Arrange: Create a ConstantInstructions object
    instructions = ConstantInstructions(text="You are a helpful assistant.")

    # Act & Assert: Should have make_system_prompt method
    assert hasattr(instructions, "make_system_prompt")
    assert callable(instructions.make_system_prompt)

    # Act: Call make_system_prompt (what the bug was failing on)
    with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
        system_prompt = instructions.make_system_prompt()

    # Assert: Returns a valid string prompt
    assert isinstance(system_prompt, str)
    assert "You are a helpful assistant." in system_prompt
