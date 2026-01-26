"""Unit tests for actuator tool schemas."""

import json

import pytest
from pydantic import ValidationError

from unmute.tooling.tool_schemas import (
    ACTUATOR_TOOLS,
    CaptureFrameArgs,
    MoveArmArgs,
    PlaySoundArgs,
    RotateBaseArgs,
    SetGazeArgs,
    ToolCallDelta,
    ToolCallResult,
    get_tool_schemas,
    register_tools_in_prompt,
    validate_and_parse_tool_call,
)


class TestMoveArmArgs:
    """Tests for MoveArmArgs validation."""

    def test_valid_args(self):
        """Test valid arm movement arguments."""
        args = MoveArmArgs(x=0.5, y=-0.3, z=0.8)
        assert args.x == 0.5
        assert args.y == -0.3
        assert args.z == 0.8
        assert args.speed == 0.5  # Default
        assert args.timeout_ms == 5000  # Default

    def test_valid_args_with_optional(self):
        """Test with all parameters specified."""
        args = MoveArmArgs(x=0.0, y=0.0, z=1.0, speed=0.8, timeout_ms=3000)
        assert args.speed == 0.8
        assert args.timeout_ms == 3000

    def test_x_out_of_range_low(self):
        """Test X coordinate below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=-1.5, y=0.0, z=0.5)
        assert "x" in str(exc_info.value)

    def test_x_out_of_range_high(self):
        """Test X coordinate above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=1.5, y=0.0, z=0.5)
        assert "x" in str(exc_info.value)

    def test_z_negative(self):
        """Test negative Z coordinate."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=-0.1)
        assert "z" in str(exc_info.value)

    def test_z_too_high(self):
        """Test Z coordinate above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=2.0)
        assert "z" in str(exc_info.value)

    def test_speed_too_low(self):
        """Test speed below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=0.5, speed=0.05)
        assert "speed" in str(exc_info.value)

    def test_speed_too_high(self):
        """Test speed above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=0.5, speed=1.5)
        assert "speed" in str(exc_info.value)

    def test_timeout_too_low(self):
        """Test timeout below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=0.5, timeout_ms=50)
        assert "timeout_ms" in str(exc_info.value)

    def test_timeout_too_high(self):
        """Test timeout above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            MoveArmArgs(x=0.0, y=0.0, z=0.5, timeout_ms=50000)
        assert "timeout_ms" in str(exc_info.value)


class TestRotateBaseArgs:
    """Tests for RotateBaseArgs validation."""

    def test_valid_args(self):
        """Test valid rotation arguments."""
        args = RotateBaseArgs(angle=45.0)
        assert args.angle == 45.0
        assert args.speed == 0.5  # Default
        assert args.timeout_ms == 3000  # Default

    def test_negative_angle(self):
        """Test negative rotation angle."""
        args = RotateBaseArgs(angle=-90.0)
        assert args.angle == -90.0

    def test_angle_too_low(self):
        """Test angle below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            RotateBaseArgs(angle=-200.0)
        assert "angle" in str(exc_info.value)

    def test_angle_too_high(self):
        """Test angle above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            RotateBaseArgs(angle=200.0)
        assert "angle" in str(exc_info.value)

    def test_custom_speed_and_timeout(self):
        """Test with custom speed and timeout."""
        args = RotateBaseArgs(angle=30.0, speed=0.3, timeout_ms=5000)
        assert args.speed == 0.3
        assert args.timeout_ms == 5000


class TestSetGazeArgs:
    """Tests for SetGazeArgs validation."""

    def test_target_mode_valid(self):
        """Test valid target mode gaze control."""
        args = SetGazeArgs(mode="target", target_x=1.0, target_y=0.5, target_z=1.5)
        assert args.mode == "target"
        assert args.target_x == 1.0
        assert args.target_y == 0.5
        assert args.target_z == 1.5

    def test_angles_mode_valid(self):
        """Test valid angles mode gaze control."""
        args = SetGazeArgs(mode="angles", pan=30.0, tilt=-15.0)
        assert args.mode == "angles"
        assert args.pan == 30.0
        assert args.tilt == -15.0

    def test_target_mode_missing_coords(self):
        """Test target mode without required coordinates."""
        with pytest.raises(ValidationError) as exc_info:
            SetGazeArgs(mode="target", target_x=1.0)
        assert "target_y" in str(exc_info.value) or "target_z" in str(exc_info.value)

    def test_angles_mode_missing_angles(self):
        """Test angles mode without required angles."""
        with pytest.raises(ValidationError) as exc_info:
            SetGazeArgs(mode="angles", pan=30.0)
        assert "tilt" in str(exc_info.value)

    def test_pan_out_of_range(self):
        """Test pan angle out of range."""
        with pytest.raises(ValidationError) as exc_info:
            SetGazeArgs(mode="angles", pan=100.0, tilt=0.0)
        assert "pan" in str(exc_info.value)

    def test_tilt_out_of_range(self):
        """Test tilt angle out of range."""
        with pytest.raises(ValidationError) as exc_info:
            SetGazeArgs(mode="angles", pan=0.0, tilt=-50.0)
        assert "tilt" in str(exc_info.value)


class TestPlaySoundArgs:
    """Tests for PlaySoundArgs validation."""

    def test_valid_effect(self):
        """Test valid sound effect playback."""
        args = PlaySoundArgs(sound_type="effect", sound_id="beep")
        assert args.sound_type == "effect"
        assert args.sound_id == "beep"
        assert args.volume == 0.7  # Default
        assert args.blocking is False  # Default

    def test_valid_file(self):
        """Test valid file playback."""
        args = PlaySoundArgs(sound_type="file", sound_id="/path/to/audio.wav", volume=0.9, blocking=True)
        assert args.sound_type == "file"
        assert args.sound_id == "/path/to/audio.wav"
        assert args.volume == 0.9
        assert args.blocking is True

    def test_valid_speech(self):
        """Test valid TTS playback."""
        args = PlaySoundArgs(sound_type="speech", sound_id="Hello, world!")
        assert args.sound_type == "speech"
        assert args.sound_id == "Hello, world!"

    def test_empty_sound_id(self):
        """Test empty sound ID."""
        with pytest.raises(ValidationError) as exc_info:
            PlaySoundArgs(sound_type="effect", sound_id="")
        assert "sound_id" in str(exc_info.value)

    def test_whitespace_sound_id(self):
        """Test whitespace-only sound ID."""
        with pytest.raises(ValidationError) as exc_info:
            PlaySoundArgs(sound_type="effect", sound_id="   ")
        assert "sound_id" in str(exc_info.value)

    def test_sound_id_too_long(self):
        """Test sound ID exceeding max length."""
        with pytest.raises(ValidationError) as exc_info:
            PlaySoundArgs(sound_type="effect", sound_id="x" * 201)
        assert "sound_id" in str(exc_info.value)

    def test_volume_out_of_range_low(self):
        """Test volume below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            PlaySoundArgs(sound_type="effect", sound_id="beep", volume=-0.1)
        assert "volume" in str(exc_info.value)

    def test_volume_out_of_range_high(self):
        """Test volume above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            PlaySoundArgs(sound_type="effect", sound_id="beep", volume=1.5)
        assert "volume" in str(exc_info.value)


class TestCaptureFrameArgs:
    """Tests for CaptureFrameArgs validation."""

    def test_defaults(self):
        """Test default capture arguments."""
        args = CaptureFrameArgs()
        assert args.camera == "front"
        assert args.resolution == "medium"
        assert args.include_metadata is True
        assert args.timeout_ms == 2000

    def test_custom_camera(self):
        """Test custom camera selection."""
        args = CaptureFrameArgs(camera="wrist", resolution="high")
        assert args.camera == "wrist"
        assert args.resolution == "high"

    def test_overhead_camera_low_res(self):
        """Test overhead camera with low resolution."""
        args = CaptureFrameArgs(camera="overhead", resolution="low", include_metadata=False)
        assert args.camera == "overhead"
        assert args.resolution == "low"
        assert args.include_metadata is False


class TestToolCallDataStructures:
    """Tests for ToolCallDelta and ToolCallResult."""

    def test_tool_call_delta(self):
        """Test ToolCallDelta model."""
        delta = ToolCallDelta(
            call_id="call_123",
            name="move_arm",
            arguments_delta='{"x": 0.5',
        )
        assert delta.call_id == "call_123"
        assert delta.name == "move_arm"
        assert delta.arguments_delta == '{"x": 0.5'

    def test_tool_call_result_success(self):
        """Test successful ToolCallResult."""
        result = ToolCallResult(
            call_id="call_123",
            success=True,
            output="Arm moved successfully",
            execution_time_ms=1234,
        )
        assert result.call_id == "call_123"
        assert result.success is True
        assert result.output == "Arm moved successfully"
        assert result.execution_time_ms == 1234

    def test_tool_call_result_failure(self):
        """Test failed ToolCallResult."""
        result = ToolCallResult(
            call_id="call_456",
            success=False,
            output="Error: Position out of reach",
        )
        assert result.call_id == "call_456"
        assert result.success is False
        assert result.execution_time_ms is None


class TestToolSchemas:
    """Tests for tool schema definitions."""

    def test_get_tool_schemas(self):
        """Test getting all tool schemas."""
        schemas = get_tool_schemas()
        assert len(schemas) == 5
        assert all(s["type"] == "function" for s in schemas)

        tool_names = [s["name"] for s in schemas]
        assert "move_arm" in tool_names
        assert "rotate_base" in tool_names
        assert "set_gaze" in tool_names
        assert "play_sound" in tool_names
        assert "capture_frame" in tool_names

    def test_move_arm_schema_structure(self):
        """Test move_arm schema structure."""
        schema = next(s for s in ACTUATOR_TOOLS if s["name"] == "move_arm")
        assert schema["type"] == "function"
        assert "description" in schema
        assert "parameters" in schema

        params = schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # Check required parameters
        assert set(params["required"]) == {"x", "y", "z"}

        # Check all properties exist
        props = params["properties"]
        assert "x" in props
        assert "y" in props
        assert "z" in props
        assert "speed" in props
        assert "timeout_ms" in props

        # Check constraints
        assert props["x"]["minimum"] == -1.0
        assert props["x"]["maximum"] == 1.0
        assert props["z"]["minimum"] == 0.0
        assert props["z"]["maximum"] == 1.5

    def test_rotate_base_schema_structure(self):
        """Test rotate_base schema structure."""
        schema = next(s for s in ACTUATOR_TOOLS if s["name"] == "rotate_base")
        assert schema["name"] == "rotate_base"
        assert set(schema["parameters"]["required"]) == {"angle"}

        props = schema["parameters"]["properties"]
        assert props["angle"]["minimum"] == -180.0
        assert props["angle"]["maximum"] == 180.0

    def test_set_gaze_schema_structure(self):
        """Test set_gaze schema structure."""
        schema = next(s for s in ACTUATOR_TOOLS if s["name"] == "set_gaze")
        assert schema["name"] == "set_gaze"
        assert set(schema["parameters"]["required"]) == {"mode"}

        props = schema["parameters"]["properties"]
        assert props["mode"]["enum"] == ["target", "angles"]
        assert "target_x" in props
        assert "pan" in props

    def test_play_sound_schema_structure(self):
        """Test play_sound schema structure."""
        schema = next(s for s in ACTUATOR_TOOLS if s["name"] == "play_sound")
        assert schema["name"] == "play_sound"
        assert set(schema["parameters"]["required"]) == {"sound_type", "sound_id"}

        props = schema["parameters"]["properties"]
        assert props["sound_type"]["enum"] == ["effect", "file", "speech"]
        assert props["volume"]["minimum"] == 0.0
        assert props["volume"]["maximum"] == 1.0

    def test_capture_frame_schema_structure(self):
        """Test capture_frame schema structure."""
        schema = next(s for s in ACTUATOR_TOOLS if s["name"] == "capture_frame")
        assert schema["name"] == "capture_frame"
        assert schema["parameters"]["required"] == []  # All optional

        props = schema["parameters"]["properties"]
        assert props["camera"]["enum"] == ["front", "wrist", "overhead"]
        assert props["resolution"]["enum"] == ["low", "medium", "high"]

    def test_timeout_defaults(self):
        """Test that all tools have reasonable timeout defaults."""
        for schema in ACTUATOR_TOOLS:
            props = schema["parameters"]["properties"]
            assert "timeout_ms" in props
            timeout_default = props["timeout_ms"].get("default")
            assert timeout_default is not None
            assert timeout_default >= 1000  # At least 1 second
            assert timeout_default <= 60000  # At most 60 seconds


class TestRegisterToolsInPrompt:
    """Tests for register_tools_in_prompt function."""

    def test_register_all_tools(self):
        """Test registering all tools in prompt."""
        base = "You are a helpful robot assistant."
        enhanced = register_tools_in_prompt(base)

        assert "You are a helpful robot assistant." in enhanced
        assert "Available Robot Control Tools" in enhanced
        assert "move_arm" in enhanced
        assert "rotate_base" in enhanced
        assert "set_gaze" in enhanced
        assert "play_sound" in enhanced
        assert "capture_frame" in enhanced

    def test_register_specific_tools(self):
        """Test registering specific tools only."""
        base = "System prompt."
        enhanced = register_tools_in_prompt(base, include_tools=["move_arm", "capture_frame"])

        assert "move_arm" in enhanced
        assert "capture_frame" in enhanced
        assert "rotate_base" not in enhanced
        assert "set_gaze" not in enhanced
        assert "play_sound" not in enhanced

    def test_register_no_tools(self):
        """Test with empty tool list."""
        base = "System prompt."
        enhanced = register_tools_in_prompt(base, include_tools=[])

        assert enhanced == base  # Should be unchanged

    def test_register_unknown_tool(self):
        """Test with unknown tool name."""
        base = "System prompt."
        enhanced = register_tools_in_prompt(base, include_tools=["unknown_tool"])

        # Should not add tool documentation section
        assert "Available Robot Control Tools" not in enhanced


class TestValidateAndParseToolCall:
    """Tests for validate_and_parse_tool_call function."""

    def test_valid_move_arm_call(self):
        """Test parsing valid move_arm call."""
        args_json = '{"x": 0.5, "y": -0.3, "z": 0.8, "speed": 0.7}'
        result = validate_and_parse_tool_call("move_arm", args_json)

        assert isinstance(result, MoveArmArgs)
        assert result.x == 0.5
        assert result.y == -0.3
        assert result.z == 0.8
        assert result.speed == 0.7

    def test_valid_rotate_base_call(self):
        """Test parsing valid rotate_base call."""
        args_json = '{"angle": 45.0}'
        result = validate_and_parse_tool_call("rotate_base", args_json)

        assert isinstance(result, RotateBaseArgs)
        assert result.angle == 45.0

    def test_valid_set_gaze_target_call(self):
        """Test parsing valid set_gaze call in target mode."""
        args_json = '{"mode": "target", "target_x": 1.0, "target_y": 0.0, "target_z": 1.5}'
        result = validate_and_parse_tool_call("set_gaze", args_json)

        assert isinstance(result, SetGazeArgs)
        assert result.mode == "target"
        assert result.target_x == 1.0

    def test_valid_play_sound_call(self):
        """Test parsing valid play_sound call."""
        args_json = '{"sound_type": "effect", "sound_id": "beep", "volume": 0.8}'
        result = validate_and_parse_tool_call("play_sound", args_json)

        assert isinstance(result, PlaySoundArgs)
        assert result.sound_type == "effect"
        assert result.sound_id == "beep"
        assert result.volume == 0.8

    def test_valid_capture_frame_call(self):
        """Test parsing valid capture_frame call."""
        args_json = '{"camera": "wrist", "resolution": "high"}'
        result = validate_and_parse_tool_call("capture_frame", args_json)

        assert isinstance(result, CaptureFrameArgs)
        assert result.camera == "wrist"
        assert result.resolution == "high"

    def test_unknown_tool(self):
        """Test parsing call to unknown tool."""
        with pytest.raises(ValueError) as exc_info:
            validate_and_parse_tool_call("unknown_tool", '{"arg": "value"}')
        assert "Unknown tool: unknown_tool" in str(exc_info.value)

    def test_invalid_json(self):
        """Test parsing call with invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            validate_and_parse_tool_call("move_arm", '{invalid json')
        assert "Invalid JSON" in str(exc_info.value)

    def test_invalid_arguments(self):
        """Test parsing call with invalid arguments."""
        # Missing required parameter
        with pytest.raises(ValueError) as exc_info:
            validate_and_parse_tool_call("move_arm", '{"x": 0.5, "y": 0.5}')
        assert "Invalid arguments for move_arm" in str(exc_info.value)

    def test_out_of_range_arguments(self):
        """Test parsing call with out-of-range values."""
        args_json = '{"x": 5.0, "y": 0.0, "z": 0.5}'  # x out of range
        with pytest.raises(ValueError) as exc_info:
            validate_and_parse_tool_call("move_arm", args_json)
        assert "Invalid arguments for move_arm" in str(exc_info.value)

    def test_defaults_applied(self):
        """Test that default values are applied correctly."""
        args_json = '{"x": 0.0, "y": 0.0, "z": 0.5}'  # Minimal required args
        result = validate_and_parse_tool_call("move_arm", args_json)

        assert result.speed == 0.5  # Default
        assert result.timeout_ms == 5000  # Default
