"""Actuator tool schemas for Unmute robot control.

This module defines validated schemas for robot actuator tools including
movement, gaze control, sound playback, and sensor capture. All tools
include argument validation, units, and timeout budgets.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Tool Argument Models
# ============================================================================


class MoveArmArgs(BaseModel):
    """Arguments for moving the robot arm.

    Coordinates are relative to the robot's base frame.
    Units: meters for position, degrees for angles.
    """

    x: float = Field(
        ..., ge=-1.0, le=1.0, description="X position in meters (-1.0 to 1.0)"
    )
    y: float = Field(
        ..., ge=-1.0, le=1.0, description="Y position in meters (-1.0 to 1.0)"
    )
    z: float = Field(
        ..., ge=0.0, le=1.5, description="Z position in meters (0.0 to 1.5)"
    )
    speed: float = Field(
        0.5, ge=0.1, le=1.0, description="Movement speed factor (0.1 to 1.0)"
    )
    timeout_ms: int = Field(
        5000, ge=100, le=30000, description="Operation timeout in milliseconds"
    )


class RotateBaseArgs(BaseModel):
    """Arguments for rotating the robot base.

    Units: degrees for angle, seconds for duration.
    """

    angle: float = Field(
        ..., ge=-180.0, le=180.0, description="Rotation angle in degrees (-180 to 180)"
    )
    speed: float = Field(
        0.5, ge=0.1, le=1.0, description="Rotation speed factor (0.1 to 1.0)"
    )
    timeout_ms: int = Field(
        3000, ge=100, le=15000, description="Operation timeout in milliseconds"
    )


class SetGazeArgs(BaseModel):
    """Arguments for setting gaze direction.

    Gaze can be specified by target point or by pan/tilt angles.
    Units: meters for target, degrees for angles.
    """

    mode: Literal["target", "angles"] = Field(
        ..., description="Gaze control mode: 'target' or 'angles'"
    )

    # Target mode parameters
    target_x: float | None = Field(
        None, ge=-5.0, le=5.0, description="Target X in meters (for target mode)"
    )
    target_y: float | None = Field(
        None, ge=-5.0, le=5.0, description="Target Y in meters (for target mode)"
    )
    target_z: float | None = Field(
        None, ge=-2.0, le=3.0, description="Target Z in meters (for target mode)"
    )

    # Angles mode parameters
    pan: float | None = Field(
        None, ge=-90.0, le=90.0, description="Pan angle in degrees (for angles mode)"
    )
    tilt: float | None = Field(
        None, ge=-45.0, le=45.0, description="Tilt angle in degrees (for angles mode)"
    )

    timeout_ms: int = Field(
        2000, ge=100, le=10000, description="Operation timeout in milliseconds"
    )

    @model_validator(mode="after")
    def validate_mode_params(self) -> "SetGazeArgs":
        """Validate that required parameters are provided based on mode."""
        if self.mode == "target":
            if self.target_x is None or self.target_y is None or self.target_z is None:
                raise ValueError(
                    "target_x, target_y, and target_z are required when mode='target'"
                )
        elif self.mode == "angles":
            if self.pan is None or self.tilt is None:
                raise ValueError("pan and tilt are required when mode='angles'")
        return self


class PlaySoundArgs(BaseModel):
    """Arguments for playing a sound or audio file.

    Supports both named sound effects and custom audio files.
    """

    sound_type: Literal["effect", "file", "speech"] = Field(
        ..., description="Type of sound to play"
    )
    sound_id: str = Field(
        ..., min_length=1, max_length=200, description="Sound identifier or file path"
    )
    volume: float = Field(
        0.7, ge=0.0, le=1.0, description="Playback volume (0.0 to 1.0)"
    )
    blocking: bool = Field(
        False, description="Whether to wait for playback to complete"
    )
    timeout_ms: int = Field(
        10000, ge=100, le=60000, description="Operation timeout in milliseconds"
    )

    @field_validator("sound_id")
    @classmethod
    def validate_sound_id(cls, v: str) -> str:
        """Validate sound ID format."""
        if not v.strip():
            raise ValueError("sound_id cannot be empty or whitespace")
        # Additional validation could check allowed characters, extensions, etc.
        return v.strip()


class CaptureFrameArgs(BaseModel):
    """Arguments for capturing a camera frame.

    Captures image from specified camera and returns reference ID.
    """

    camera: Literal["front", "wrist", "overhead"] = Field(
        "front", description="Camera to capture from"
    )
    resolution: Literal["low", "medium", "high"] = Field(
        "medium", description="Capture resolution"
    )
    include_metadata: bool = Field(
        True, description="Include timestamp and camera parameters"
    )
    timeout_ms: int = Field(
        2000, ge=100, le=10000, description="Operation timeout in milliseconds"
    )


# ============================================================================
# Tool Call Data Structures
# ============================================================================


class ToolCallDelta(BaseModel):
    """Represents a streaming chunk of a tool call.

    Used when the LLM streams function call arguments incrementally.
    """

    call_id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Tool name")
    arguments_delta: str = Field(..., description="Incremental JSON chunk of arguments")


class ToolCallResult(BaseModel):
    """Represents the result of a tool execution.

    Returned after a tool has been invoked and completed (or failed).
    """

    call_id: str = Field(..., description="Tool call ID this result corresponds to")
    success: bool = Field(..., description="Whether the tool executed successfully")
    output: str = Field(..., description="Tool output or error message")
    execution_time_ms: int | None = Field(
        None, description="Actual execution time in milliseconds"
    )


# ============================================================================
# Tool Schema Definitions (OpenAI Format)
# ============================================================================


ACTUATOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "move_arm",
        "description": "Move the robot arm to a specified position in 3D space. Coordinates are relative to the robot base frame.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X position in meters, range -1.0 to 1.0",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": "Y position in meters, range -1.0 to 1.0",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "z": {
                    "type": "number",
                    "description": "Z position in meters, range 0.0 to 1.5",
                    "minimum": 0.0,
                    "maximum": 1.5,
                },
                "speed": {
                    "type": "number",
                    "description": "Movement speed factor from 0.1 (slow) to 1.0 (fast)",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for completion in milliseconds",
                    "minimum": 100,
                    "maximum": 30000,
                    "default": 5000,
                },
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "type": "function",
        "name": "rotate_base",
        "description": "Rotate the robot base by a specified angle. Positive angles rotate counterclockwise.",
        "parameters": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Rotation angle in degrees, range -180 to 180",
                    "minimum": -180.0,
                    "maximum": 180.0,
                },
                "speed": {
                    "type": "number",
                    "description": "Rotation speed factor from 0.1 (slow) to 1.0 (fast)",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for completion in milliseconds",
                    "minimum": 100,
                    "maximum": 15000,
                    "default": 3000,
                },
            },
            "required": ["angle"],
        },
    },
    {
        "type": "function",
        "name": "set_gaze",
        "description": "Control where the robot is looking. Can specify either a target point in 3D space or explicit pan/tilt angles.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["target", "angles"],
                    "description": "Control mode: 'target' to look at a point, 'angles' for direct pan/tilt control",
                },
                "target_x": {
                    "type": "number",
                    "description": "Target X coordinate in meters (required if mode='target')",
                    "minimum": -5.0,
                    "maximum": 5.0,
                },
                "target_y": {
                    "type": "number",
                    "description": "Target Y coordinate in meters (required if mode='target')",
                    "minimum": -5.0,
                    "maximum": 5.0,
                },
                "target_z": {
                    "type": "number",
                    "description": "Target Z coordinate in meters (required if mode='target')",
                    "minimum": -2.0,
                    "maximum": 3.0,
                },
                "pan": {
                    "type": "number",
                    "description": "Pan angle in degrees (required if mode='angles')",
                    "minimum": -90.0,
                    "maximum": 90.0,
                },
                "tilt": {
                    "type": "number",
                    "description": "Tilt angle in degrees (required if mode='angles')",
                    "minimum": -45.0,
                    "maximum": 45.0,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for completion in milliseconds",
                    "minimum": 100,
                    "maximum": 10000,
                    "default": 2000,
                },
            },
            "required": ["mode"],
        },
    },
    {
        "type": "function",
        "name": "play_sound",
        "description": "Play an audio sound effect, file, or synthesized speech through the robot's speakers.",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_type": {
                    "type": "string",
                    "enum": ["effect", "file", "speech"],
                    "description": "Type of audio: 'effect' for built-in sounds, 'file' for audio files, 'speech' for TTS",
                },
                "sound_id": {
                    "type": "string",
                    "description": "Identifier for the sound (effect name, file path, or text to speak)",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "volume": {
                    "type": "number",
                    "description": "Playback volume from 0.0 (mute) to 1.0 (max)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                },
                "blocking": {
                    "type": "boolean",
                    "description": "If true, wait for playback to complete before returning",
                    "default": False,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for completion in milliseconds",
                    "minimum": 100,
                    "maximum": 60000,
                    "default": 10000,
                },
            },
            "required": ["sound_type", "sound_id"],
        },
    },
    {
        "type": "function",
        "name": "capture_frame",
        "description": "Capture an image frame from one of the robot's cameras. Returns a reference ID that can be used to retrieve the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["front", "wrist", "overhead"],
                    "description": "Which camera to capture from",
                    "default": "front",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Image resolution/quality",
                    "default": "medium",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include capture timestamp and camera parameters in response",
                    "default": True,
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for capture in milliseconds",
                    "minimum": 100,
                    "maximum": 10000,
                    "default": 2000,
                },
            },
            "required": [],
        },
    },
]


# Map tool names to their argument models for validation
_TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "move_arm": MoveArmArgs,
    "rotate_base": RotateBaseArgs,
    "set_gaze": SetGazeArgs,
    "play_sound": PlaySoundArgs,
    "capture_frame": CaptureFrameArgs,
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get all actuator tool schemas in OpenAI function calling format.

    Returns:
        List of tool schema dictionaries ready for use in LLM requests.
    """
    return ACTUATOR_TOOLS.copy()


def register_tools_in_prompt(
    base_prompt: str, include_tools: list[str] | None = None
) -> str:
    """Add tool descriptions to a system prompt.

    Args:
        base_prompt: The base system prompt text.
        include_tools: Optional list of tool names to include. If None, includes all tools.

    Returns:
        Enhanced prompt with tool documentation appended.
    """
    if include_tools is None:
        tools_to_document = ACTUATOR_TOOLS
    else:
        tools_to_document = [t for t in ACTUATOR_TOOLS if t["name"] in include_tools]

    if not tools_to_document:
        return base_prompt

    tool_docs = "\n\n## Available Robot Control Tools\n\n"
    tool_docs += "You have access to the following robot actuator functions:\n\n"

    for tool in tools_to_document:
        tool_docs += f"### {tool['name']}\n"
        tool_docs += f"{tool['description']}\n\n"

        # Document parameters
        if "parameters" in tool and "properties" in tool["parameters"]:
            tool_docs += "**Parameters:**\n"
            props = tool["parameters"]["properties"]
            required = tool["parameters"].get("required", [])

            for param_name, param_spec in props.items():
                required_marker = (
                    " (required)" if param_name in required else " (optional)"
                )
                param_desc = param_spec.get("description", "")
                tool_docs += f"- `{param_name}`: {param_desc}{required_marker}\n"

            tool_docs += "\n"

    return base_prompt + tool_docs


def validate_and_parse_tool_call(
    tool_name: str,
    arguments_json: str,
) -> BaseModel:
    """Validate and parse tool call arguments into a typed dataclass.

    Args:
        tool_name: Name of the tool being called.
        arguments_json: JSON string containing the tool arguments.

    Returns:
        Validated Pydantic model instance for the tool's arguments.

    Raises:
        ValueError: If tool_name is unknown or arguments are invalid.
    """
    if tool_name not in _TOOL_ARG_MODELS:
        raise ValueError(
            f"Unknown tool: {tool_name}. "
            f"Known tools: {', '.join(_TOOL_ARG_MODELS.keys())}"
        )

    model_class = _TOOL_ARG_MODELS[tool_name]

    # Parse JSON and validate against the model
    import json

    try:
        args_dict = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool arguments: {e}") from e

    try:
        return model_class.model_validate(args_dict)
    except Exception as e:
        raise ValueError(f"Invalid arguments for {tool_name}: {e}") from e
