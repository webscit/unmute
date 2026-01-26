"""Snapshot tests for tool schema descriptors.

These tests ensure that the generated tool schemas match what the LLM expects
and that changes to tool definitions are intentional and reviewed.
"""

import json

from unmute.tooling.tool_schemas import ACTUATOR_TOOLS, get_tool_schemas


class TestToolSchemaSnapshots:
    """Snapshot tests for tool schema outputs."""

    def test_tool_schemas_snapshot(self):
        """Test that tool schemas match expected structure.

        This snapshot test ensures that changes to tool schemas are intentional.
        If this test fails, review the changes to ensure they're correct, then
        update the expected snapshot below.
        """
        schemas = get_tool_schemas()

        # Serialize to JSON for consistent formatting
        schemas_json = json.dumps(schemas, indent=2, sort_keys=True)

        # Expected snapshot - update this if tool schemas intentionally change
        expected_snapshot = json.dumps(
            [
                {
                    "name": "move_arm",
                    "parameters": {
                        "properties": {
                            "speed": {
                                "default": 0.5,
                                "description": "Movement speed factor from 0.1 (slow) to 1.0 (fast)",
                                "maximum": 1.0,
                                "minimum": 0.1,
                                "type": "number",
                            },
                            "timeout_ms": {
                                "default": 5000,
                                "description": "Maximum time to wait for completion in milliseconds",
                                "maximum": 30000,
                                "minimum": 100,
                                "type": "integer",
                            },
                            "x": {
                                "description": "X position in meters, range -1.0 to 1.0",
                                "maximum": 1.0,
                                "minimum": -1.0,
                                "type": "number",
                            },
                            "y": {
                                "description": "Y position in meters, range -1.0 to 1.0",
                                "maximum": 1.0,
                                "minimum": -1.0,
                                "type": "number",
                            },
                            "z": {
                                "description": "Z position in meters, range 0.0 to 1.5",
                                "maximum": 1.5,
                                "minimum": 0.0,
                                "type": "number",
                            },
                        },
                        "required": ["x", "y", "z"],
                        "type": "object",
                    },
                    "type": "function",
                    "description": "Move the robot arm to a specified position in 3D space. Coordinates are relative to the robot base frame.",
                },
                {
                    "name": "rotate_base",
                    "parameters": {
                        "properties": {
                            "angle": {
                                "description": "Rotation angle in degrees, range -180 to 180",
                                "maximum": 180.0,
                                "minimum": -180.0,
                                "type": "number",
                            },
                            "speed": {
                                "default": 0.5,
                                "description": "Rotation speed factor from 0.1 (slow) to 1.0 (fast)",
                                "maximum": 1.0,
                                "minimum": 0.1,
                                "type": "number",
                            },
                            "timeout_ms": {
                                "default": 3000,
                                "description": "Maximum time to wait for completion in milliseconds",
                                "maximum": 15000,
                                "minimum": 100,
                                "type": "integer",
                            },
                        },
                        "required": ["angle"],
                        "type": "object",
                    },
                    "type": "function",
                    "description": "Rotate the robot base by a specified angle. Positive angles rotate counterclockwise.",
                },
                {
                    "name": "set_gaze",
                    "parameters": {
                        "properties": {
                            "mode": {
                                "description": "Control mode: 'target' to look at a point, 'angles' for direct pan/tilt control",
                                "enum": ["target", "angles"],
                                "type": "string",
                            },
                            "pan": {
                                "description": "Pan angle in degrees (required if mode='angles')",
                                "maximum": 90.0,
                                "minimum": -90.0,
                                "type": "number",
                            },
                            "target_x": {
                                "description": "Target X coordinate in meters (required if mode='target')",
                                "maximum": 5.0,
                                "minimum": -5.0,
                                "type": "number",
                            },
                            "target_y": {
                                "description": "Target Y coordinate in meters (required if mode='target')",
                                "maximum": 5.0,
                                "minimum": -5.0,
                                "type": "number",
                            },
                            "target_z": {
                                "description": "Target Z coordinate in meters (required if mode='target')",
                                "maximum": 3.0,
                                "minimum": -2.0,
                                "type": "number",
                            },
                            "tilt": {
                                "description": "Tilt angle in degrees (required if mode='angles')",
                                "maximum": 45.0,
                                "minimum": -45.0,
                                "type": "number",
                            },
                            "timeout_ms": {
                                "default": 2000,
                                "description": "Maximum time to wait for completion in milliseconds",
                                "maximum": 10000,
                                "minimum": 100,
                                "type": "integer",
                            },
                        },
                        "required": ["mode"],
                        "type": "object",
                    },
                    "type": "function",
                    "description": "Control where the robot is looking. Can specify either a target point in 3D space or explicit pan/tilt angles.",
                },
                {
                    "name": "play_sound",
                    "parameters": {
                        "properties": {
                            "blocking": {
                                "default": False,
                                "description": "If true, wait for playback to complete before returning",
                                "type": "boolean",
                            },
                            "sound_id": {
                                "description": "Identifier for the sound (effect name, file path, or text to speak)",
                                "maxLength": 200,
                                "minLength": 1,
                                "type": "string",
                            },
                            "sound_type": {
                                "description": "Type of audio: 'effect' for built-in sounds, 'file' for audio files, 'speech' for TTS",
                                "enum": ["effect", "file", "speech"],
                                "type": "string",
                            },
                            "timeout_ms": {
                                "default": 10000,
                                "description": "Maximum time to wait for completion in milliseconds",
                                "maximum": 60000,
                                "minimum": 100,
                                "type": "integer",
                            },
                            "volume": {
                                "default": 0.7,
                                "description": "Playback volume from 0.0 (mute) to 1.0 (max)",
                                "maximum": 1.0,
                                "minimum": 0.0,
                                "type": "number",
                            },
                        },
                        "required": ["sound_type", "sound_id"],
                        "type": "object",
                    },
                    "type": "function",
                    "description": "Play an audio sound effect, file, or synthesized speech through the robot's speakers.",
                },
                {
                    "name": "capture_frame",
                    "parameters": {
                        "properties": {
                            "camera": {
                                "default": "front",
                                "description": "Which camera to capture from",
                                "enum": ["front", "wrist", "overhead"],
                                "type": "string",
                            },
                            "include_metadata": {
                                "default": True,
                                "description": "Include capture timestamp and camera parameters in response",
                                "type": "boolean",
                            },
                            "resolution": {
                                "default": "medium",
                                "description": "Image resolution/quality",
                                "enum": ["low", "medium", "high"],
                                "type": "string",
                            },
                            "timeout_ms": {
                                "default": 2000,
                                "description": "Maximum time to wait for capture in milliseconds",
                                "maximum": 10000,
                                "minimum": 100,
                                "type": "integer",
                            },
                        },
                        "required": [],
                        "type": "object",
                    },
                    "type": "function",
                    "description": "Capture an image frame from one of the robot's cameras. Returns a reference ID that can be used to retrieve the image.",
                },
            ],
            indent=2,
            sort_keys=True,
        )

        # Compare snapshots
        assert schemas_json == expected_snapshot, (
            "Tool schema snapshot mismatch. If this is intentional, update the "
            "expected_snapshot in this test. Diff:\n"
            f"Generated:\n{schemas_json}\n\n"
            f"Expected:\n{expected_snapshot}"
        )

    def test_all_tools_have_required_fields(self):
        """Test that all tool schemas have required OpenAI format fields."""
        schemas = get_tool_schemas()

        for schema in schemas:
            # Required top-level fields
            assert "type" in schema
            assert schema["type"] == "function"
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

            # Required parameters structure
            params = schema["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

            # All tools should have timeout_ms parameter
            assert "timeout_ms" in params["properties"]
            timeout = params["properties"]["timeout_ms"]
            assert timeout["type"] == "integer"
            assert "minimum" in timeout
            assert "maximum" in timeout
            assert "default" in timeout

    def test_tool_count(self):
        """Test that we have the expected number of tools."""
        schemas = get_tool_schemas()
        assert len(schemas) == 5, "Expected 5 actuator tools"

        tool_names = {s["name"] for s in schemas}
        expected_names = {
            "move_arm",
            "rotate_base",
            "set_gaze",
            "play_sound",
            "capture_frame",
        }
        assert tool_names == expected_names
