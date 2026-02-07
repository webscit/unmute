"""Regression test for unmute-87r: EventDispatcher AttributeError fix.

This test verifies that EventDispatcher correctly accesses audio_buffer via
handler.audio_buffer_handler.audio_buffer, not handler.audio_buffer directly.

The test uses AST parsing to avoid import issues with broken dependencies.
"""

import ast
from pathlib import Path


def test_event_dispatcher_uses_correct_audio_buffer_path():
    """Verify EventDispatcher accesses audio_buffer via audio_buffer_handler.

    This is a regression test for bug unmute-87r where:
        AttributeError: 'UnmuteHandler' object has no attribute 'audio_buffer'

    The fix changed:
        OLD: self.audio_buffer = handler.audio_buffer
        NEW: self.audio_buffer = handler.audio_buffer_handler.audio_buffer
    """
    # Read the event_dispatcher.py file
    dispatcher_file = (
        Path(__file__).parent.parent / "unmute" / "services" / "event_dispatcher.py"
    )
    source_code = dispatcher_file.read_text()

    # Parse the AST
    tree = ast.parse(source_code)

    # Find the EventDispatcher class
    event_dispatcher_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EventDispatcher":
            event_dispatcher_class = node
            break

    assert (
        event_dispatcher_class is not None
    ), "EventDispatcher class not found in event_dispatcher.py"

    # Find the __init__ method
    init_method = None
    for item in event_dispatcher_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_method = item
            break

    assert init_method is not None, "__init__ method not found in EventDispatcher"

    # Find the assignment to self.audio_buffer
    audio_buffer_assignment = None
    for node in ast.walk(init_method):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "audio_buffer"
                ):
                    audio_buffer_assignment = node
                    break

    assert (
        audio_buffer_assignment is not None
    ), "self.audio_buffer assignment not found in __init__"

    # Verify the RHS is handler.audio_buffer_handler.audio_buffer
    # Expected AST structure:
    #   Attribute(
    #       value=Attribute(
    #           value=Name(id='handler'),
    #           attr='audio_buffer_handler'
    #       ),
    #       attr='audio_buffer'
    #   )

    rhs = audio_buffer_assignment.value
    assert isinstance(
        rhs, ast.Attribute
    ), f"RHS should be an Attribute node, got {type(rhs)}"
    assert (
        rhs.attr == "audio_buffer"
    ), f"Expected final attribute to be 'audio_buffer', got '{rhs.attr}'"

    # Check that it's accessing through audio_buffer_handler
    assert isinstance(
        rhs.value, ast.Attribute
    ), "Expected intermediate attribute access (handler.audio_buffer_handler)"
    assert (
        rhs.value.attr == "audio_buffer_handler"
    ), f"Expected attribute 'audio_buffer_handler', got '{rhs.value.attr}'"

    # Check that the base is 'handler'
    assert isinstance(
        rhs.value.value, ast.Name
    ), "Expected base to be a Name node (handler)"
    assert (
        rhs.value.value.id == "handler"
    ), f"Expected base name to be 'handler', got '{rhs.value.value.id}'"

    # If we got here, the fix is correct:
    # self.audio_buffer = handler.audio_buffer_handler.audio_buffer


def test_unmute_handler_has_audio_buffer_handler():
    """Verify that UnmuteHandler has audio_buffer_handler attribute (not audio_buffer).

    This checks that the structure matches what EventDispatcher expects.
    """
    handler_file = Path(__file__).parent.parent / "unmute" / "unmute_handler.py"
    source_code = handler_file.read_text()

    # We expect to find: self.audio_buffer_handler = AudioBufferHandler(...)
    # But NOT: self.audio_buffer = ...
    assert (
        "self.audio_buffer_handler" in source_code
    ), "UnmuteHandler should have audio_buffer_handler attribute"

    # Make sure we're not setting self.audio_buffer directly in __init__
    # (it should only be accessible via audio_buffer_handler.audio_buffer)
    lines = source_code.split("\n")
    in_init = False
    for line in lines:
        if "def __init__" in line:
            in_init = True
        elif in_init and line.strip().startswith("def "):
            # Exited __init__
            in_init = False
        elif in_init and "self.audio_buffer = " in line:
            # Check this isn't setting audio_buffer_handler
            if "audio_buffer_handler" not in line:
                raise AssertionError(
                    f"Found direct self.audio_buffer assignment in UnmuteHandler.__init__: {line.strip()}"
                )


def test_audio_buffer_handler_has_audio_buffer():
    """Verify AudioBufferHandler has the audio_buffer attribute."""
    handler_file = (
        Path(__file__).parent.parent
        / "unmute"
        / "handlers"
        / "audio_buffer_handler.py"
    )
    source_code = handler_file.read_text()

    assert (
        "self.audio_buffer" in source_code
    ), "AudioBufferHandler should have audio_buffer attribute"
