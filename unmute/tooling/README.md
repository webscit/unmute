# Actuator Tool Schemas

This module provides validated schemas for robot actuator tools including movement, gaze control, sound playback, and sensor capture.

## Features

- **Type-safe tool definitions** using Pydantic models
- **Argument validation** with units, ranges, and timeout budgets
- **OpenAI-compatible schemas** for LLM function calling
- **Helper functions** for registration and parsing

## Available Tools

### 1. move_arm
Move the robot arm to a specified 3D position.

**Parameters:**
- `x`, `y`, `z` (required): Position in meters
- `speed` (optional): Movement speed factor (0.1-1.0, default: 0.5)
- `timeout_ms` (optional): Operation timeout (100-30000ms, default: 5000ms)

### 2. rotate_base
Rotate the robot base by a specified angle.

**Parameters:**
- `angle` (required): Rotation angle in degrees (-180 to 180)
- `speed` (optional): Rotation speed factor (0.1-1.0, default: 0.5)
- `timeout_ms` (optional): Operation timeout (100-15000ms, default: 3000ms)

### 3. set_gaze
Control where the robot is looking.

**Parameters:**
- `mode` (required): "target" or "angles"
- For target mode: `target_x`, `target_y`, `target_z` (meters)
- For angles mode: `pan`, `tilt` (degrees)
- `timeout_ms` (optional): Operation timeout (100-10000ms, default: 2000ms)

### 4. play_sound
Play audio through the robot's speakers.

**Parameters:**
- `sound_type` (required): "effect", "file", or "speech"
- `sound_id` (required): Sound identifier or text
- `volume` (optional): Playback volume (0.0-1.0, default: 0.7)
- `blocking` (optional): Wait for completion (default: false)
- `timeout_ms` (optional): Operation timeout (100-60000ms, default: 10000ms)

### 5. capture_frame
Capture an image from a camera.

**Parameters:**
- `camera` (optional): "front", "wrist", or "overhead" (default: "front")
- `resolution` (optional): "low", "medium", or "high" (default: "medium")
- `include_metadata` (optional): Include timestamp/params (default: true)
- `timeout_ms` (optional): Operation timeout (100-10000ms, default: 2000ms)

## Usage Examples

### Getting Tool Schemas for LLM

```python
from unmute.tooling import get_tool_schemas

# Get all tool schemas in OpenAI format
tools = get_tool_schemas()

# Use with OpenAI API
response = await client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=tools,
)
```

### Registering Tools in System Prompt

```python
from unmute.tooling import register_tools_in_prompt

base_prompt = "You are a helpful robot assistant."

# Add all tools
enhanced_prompt = register_tools_in_prompt(base_prompt)

# Or select specific tools
enhanced_prompt = register_tools_in_prompt(
    base_prompt,
    include_tools=["move_arm", "set_gaze"]
)
```

### Validating Tool Call Arguments

```python
from unmute.tooling import validate_and_parse_tool_call

# When LLM generates a function call
tool_name = "move_arm"
arguments_json = '{"x": 0.5, "y": -0.3, "z": 0.8, "speed": 0.7}'

try:
    # Parse and validate
    args = validate_and_parse_tool_call(tool_name, arguments_json)

    # args is a MoveArmArgs instance with validated fields
    print(f"Move arm to ({args.x}, {args.y}, {args.z})")
    print(f"Speed: {args.speed}, Timeout: {args.timeout_ms}ms")

except ValueError as e:
    print(f"Invalid tool call: {e}")
```

### Using with UnmuteHandler

```python
from unmute.unmute_handler import UnmuteHandler

handler = UnmuteHandler()

# Configure tools in session
await handler.update_session({
    "tools": [
        {"name": "move_arm"},
        {"name": "set_gaze"},
        {"name": "play_sound"},
    ],
    "tool_choice": "auto",
})

# Validate tool calls when they arrive
try:
    validated_args = handler.validate_tool_call(
        tool_name="move_arm",
        arguments_json='{"x": 0.5, "y": 0.0, "z": 0.8}'
    )
    # Execute tool with validated args...
except ValueError as e:
    # Handle validation error
    pass
```

### Working with Tool Arguments Directly

```python
from unmute.tooling.tool_schemas import (
    MoveArmArgs,
    SetGazeArgs,
    PlaySoundArgs,
)

# Create and validate arguments
arm_args = MoveArmArgs(x=0.5, y=-0.3, z=0.8, speed=0.7)

# Conditional validation based on mode
gaze_target = SetGazeArgs(
    mode="target",
    target_x=1.0,
    target_y=0.5,
    target_z=1.5
)

gaze_angles = SetGazeArgs(
    mode="angles",
    pan=30.0,
    tilt=-15.0
)

# With defaults
sound_args = PlaySoundArgs(
    sound_type="effect",
    sound_id="beep"
)
# sound_args.volume == 0.7 (default)
# sound_args.blocking == False (default)
```

### Tool Call Results

```python
from unmute.tooling import ToolCallResult

# Success
result = ToolCallResult(
    call_id="call_123",
    success=True,
    output="Arm moved successfully",
    execution_time_ms=1234
)

# Failure
result = ToolCallResult(
    call_id="call_456",
    success=False,
    output="Error: Position out of reach"
)
```

## Integration with LLM Prompt Builder

The tool schemas integrate with the system prompt builder:

```python
from unmute.llm.system_prompt import ConstantInstructions

instructions = ConstantInstructions()

# Generate system prompt with tools
prompt = instructions.make_system_prompt(
    include_tools=["move_arm", "set_gaze", "capture_frame"]
)
```

This adds formatted tool documentation to the system prompt.

## Testing

Comprehensive tests are available in `tests/test_tool_schemas.py`:

```bash
pytest tests/test_tool_schemas.py -v
```

Snapshot tests ensure schema stability:

```bash
pytest tests/test_tool_schema_snapshots.py -v
```

## Future Work

This module provides the foundation for tool execution. Future tasks will:

1. Implement tool routing and execution handlers
2. Add actuator-specific execution logic
3. Handle tool call interrupts and cancellation
4. Implement tool result persistence and replay
