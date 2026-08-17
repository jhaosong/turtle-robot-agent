# Robot Agent Function Reference

This document inventories the current `src/robot_agent` callable surface by
file. It distinguishes model-visible LangChain tools from internal Python
functions and methods.

Notation:

- `Tool`: directly selectable by the lead agent.
- `Public`: callable application API.
- `Internal`: implementation helper; not directly visible to the LLM.
- Constructors and Pydantic/dataclass fields are summarized only when they are
  needed to understand a function input.
- Package-only `__init__.py` files are omitted because they only re-export
  symbols.

## Model-visible tools

All tools below are created in `src/robot_agent/tools/registry.py` by
`RobotToolRegistry.build()` and return a normalized `ToolResult` dictionary.

| Tool name | Input | Description |
|---|---|---|
| `get_known_locations` | None | Return the configured named locations and their map poses. |
| `get_robot_state` | None | Read the latest real pose when possible and return concise semantic robot state. |
| `navigate_to` | `location: str` | Navigate with Nav2 to one configured named location. |
| `navigate_to_pose` | `x: float`, `y: float`, `yaw: float = 0`, `frame_id: "map" = "map"` | Navigate to explicit map coordinates and verify the final observed pose. |
| `move_relative` | `distance_m: float` in `[-2, 2]` | Read the live map pose, compute a target along the current heading, then navigate to it. |
| `find_object` | `color: str | None`, `label: str | None` | Query existing detections in the semantic world model; does not capture a new image. |
| `inspect_for_color` | `color: "red" | "green" | "blue"` | Inspect the current camera frame once and update visible objects. |
| `search_for_object` | `route: list[str]`, `color: str | None`, `label: str | None` | Navigate through an ordered route while detecting; cancel navigation on a confident match and optionally center it. |
| `run_behavior_tree` | `goal: str` | Generate, validate, persist, and execute a bounded behavior tree. |
| `stop_robot` | None | Cancel this run's active Nav2 goal when supported and publish zero velocity. |
| `wait_seconds` | `seconds: float` | Wait for a bounded duration, or record the wait in dry-run mode. |
| `request_clarification` | `question: str`, `reason: str` | Return structured `needs_input` when no safe unambiguous action is available. |

## Top-level orchestration

### `src/robot_agent/cli.py`

| Name | Input | Description |
|---|---|---|
| `validate_location_file` | `settings: RobotAgentSettings` | Fail early when the configured location YAML does not exist. |
| `main` | None; reads CLI input and environment | Start the interactive robot-agent command loop. |

### `src/robot_agent/models.py`

| Name | Input | Description |
|---|---|---|
| `load_chat_model` | `streaming: bool = False` | Build the configured OpenAI, Azure, Anthropic, or Ollama chat model. |
| `get_azure_deployment_name` | None; reads model environment variables | Resolve the configured Azure deployment name. |
| `get_azure_v1_base_url` | None; reads `AZURE_OPENAI_ENDPOINT` | Normalize an Azure resource endpoint to the OpenAI-compatible v1 URL. |
| `get_env_variable` | `name: str`, `allow_empty: bool = False` | Return a required environment variable with consistent validation. |

### `src/robot_agent/harness.py`

| Name | Input | Description |
|---|---|---|
| `build_bounded_goal` | `goal: str`, `plan: list[dict]`, `robot_state` | Delimit user text and serialize the advisory plan and semantic state into the agent input. |
| `build_lead_agent_middleware` | `runtime: RobotAgentRuntime` | Assemble plan-completion and model-termination middleware in the required order. |
| `build_verified_final_response` | `runtime: RobotAgentRuntime`, `run_status: str` | Build the final user response from tool evidence and deterministic goal evaluation. |
| `RoboticsAgentHarness.__init__` | `settings: RobotAgentSettings`, `model: Any | None = None` | Store runtime configuration and an optional injected model. |
| `RoboticsAgentHarness.invoke` | `goal: str` | Create one run, plan, assemble the agent/tools, invoke LangChain, verify the result, persist state, and return a run summary. |

## Lead agent and planner

### `src/robot_agent/agents/lead_agent/agent.py`

| Name | Input | Description |
|---|---|---|
| `_filter_tools` | `tools`, `allowlist: set[str] | None`, `blocklist: set[str] | None` | Internal allow/block filtering for the final model-visible tool set. |
| `make_lead_agent` | `model`, `tools`, `known_locations`, `max_tool_calls`, optional `middleware`, `allowlist`, `blocklist` | Build the prompt and metadata, then compile the LangChain v1 tool-calling agent. |

### `src/robot_agent/agents/lead_agent/planner.py`

| Name | Input | Description |
|---|---|---|
| `TaskPlan.validate_requirements` | Current `TaskPlan` instance | Require `requires_perception=true` whenever requested colors or semantic labels are present. |
| `LeadTaskPlanner.__init__` | `model: Any` | Store the replaceable planner model. |
| `LeadTaskPlanner.plan` | `goal`, `robot_state`, `available_capabilities`, optional `known_locations` | Ask the model for a structured high-level plan and reject unavailable capabilities. |

Planner output contains `objective`, `assumptions`, up to eight `steps`,
`requires_perception`, `requested_colors`, and `requested_labels`. Each step
contains a description and one of `navigation`, `perception`, `behavior_tree`,
or `control`.

### `src/robot_agent/agents/lead_agent/prompt.py`

| Name | Input | Description |
|---|---|---|
| `build_lead_agent_prompt` | `known_locations`, `tool_names`, `max_tool_calls` | Render the lead-agent system prompt with tool policy, safety rules, and catalogs. |

## Tool layer

### `src/robot_agent/tools/contracts.py`

These Pydantic schemas validate model-generated tool arguments before execution.

| Schema | Input fields | Description |
|---|---|---|
| `NavigateToLocationInput` | `location: str` | Named navigation target. |
| `NavigateToPoseInput` | `x`, `y`, `yaw=0`, `frame_id="map"` | Explicit map-frame target pose. |
| `MoveRelativeInput` | `distance_m` in `[-2, 2]` | Signed motion along the current heading. |
| `FindObjectInput` | optional `color`, optional `label` | Semantic world-model query. |
| `InspectForColorInput` | `color` in red/green/blue | Single-frame color inspection. |
| `SearchForObjectInput` | `route` of 1-12 locations, optional `color`, optional `label` | Search-while-moving request; `require_target()` requires at least one target field. |
| `BehaviorTreeSkillInput` | `goal: str` | Behavior-tree generation goal. |
| `WaitInput` | `seconds: float` | Wait duration. |
| `ClarificationInput` | `question` (1-300 chars), `reason` (1-500 chars) | Structured user clarification. |

### `src/robot_agent/tools/registry.py`

| Name | Input | Description |
|---|---|---|
| `load_locations` | `path: Path` | Parse `{name: [x, y, yaw]}` YAML into `Pose2D` objects. |
| `RobotToolRegistry.__init__` | `runtime`, `ros`, `bt_skill`, optional `detector` | Assemble locations, world model, safety validator, optional detector, and loop detector for one run. |
| `_get_detector` | None | Lazily load the configured detector only when a perception tool needs it. |
| `_execute` | `tool_name`, `arguments`, `operation` | Enforce no-progress and loop limits, normalize exceptions/results, trace, and persist one tool call. |
| `_record` | `tool_name`, `arguments`, `operation` | Execute a tool and convert its `ToolResult` to a dictionary. |
| `_navigate_to_pose` | `pose: Pose2D`, optional `location` | Validate and execute navigation, then verify and store the actual final pose and error. |
| `_navigate_to_location` | `location: str` | Resolve a named location and delegate to `_navigate_to_pose`. |
| `_move_relative` | `distance_m: float` | Compute a map target from live pose and heading, then navigate to it. |
| `_search_for_object` | `route`, optional `color`, optional `label` | Coordinate watched Nav2 movement, confidence filtering, cancellation, bbox-driven alignment, and detection reporting. |
| `_search_for_object.detect_latest` | None; closes over query and tracking state | Acquire with the stop threshold, then track the nearest spatially consistent bbox with a lower confidence threshold during visual alignment. |
| `_wait_for` | `seconds: float` | Validate and execute a wait, or return `planned` in dry-run mode. |
| `build` | None | Return the 12 `StructuredTool` objects listed at the top of this document. |

`build()` also defines callbacks used by `run_behavior_tree`: `node_started`
updates the checkpoint, `navigate` executes `GoToPose`, `wait` executes `Wait`,
`stop` executes the final stop, and `abort` performs a safe stop after failure.

## ROS2 transport

### `src/robot_agent/ros/messages.py`

| Name | Input | Description |
|---|---|---|
| `yaw_to_quaternion` | `yaw: float` radians | Convert planar yaw into a ROS-compatible quaternion dictionary. |
| `quaternion_to_yaw` | `x`, `y`, `z`, `w` | Convert a ROS quaternion into planar yaw radians. |

### `src/robot_agent/ros/adapter.py`

#### Shared interface and timing

| Name | Input | Description |
|---|---|---|
| `_DetectionTicker.__init__` | `interval_sec`, optional monotonic `clock` | Configure a rate gate whose first detection is immediate. |
| `_DetectionTicker.ready` | None | Return true only when the next detection interval has elapsed. |
| `Ros2Adapter.navigate_to_pose` | `pose: Pose2D` | Abstract high-level navigation operation. |
| `Ros2Adapter.navigate_to_pose_with_watch` | `pose`, `on_tick`, `tick_interval_sec` | Watched-navigation interface; default implementation falls back to plain navigation. |
| `Ros2Adapter.stop_robot` | None | Abstract safe-stop operation. |
| `Ros2Adapter.get_pose` | None | Abstract current-pose operation. |
| `Ros2Adapter.cancel_navigation` | None | Abstract active-goal cancellation operation. |
| `Ros2Adapter.detect_color` | `color: str` | Abstract one-frame color detection operation. |
| `Ros2Adapter.get_camera_frame` | None | Return the latest BGR frame when supported. |
| `Ros2Adapter.update_detection_overlay` | `detections` | Update boxes shown on the annotated camera stream; base implementation is a no-op. |
| `Ros2Adapter.align_to_detection` | detection callback plus horizontal/box-size targets, speed bounds, gains, stability count, timeout | Base contract for staged bbox centering followed by visual distance regulation. |
| `Ros2Adapter.close` | None | Release backend resources. |

#### `Ros2CliAdapter`

| Name | Input | Description |
|---|---|---|
| `__init__` | `settings` | Configure transparent ROS2 CLI transport. |
| `_run` | `command: str`, `details: dict` | Return a planned command in dry-run mode or execute it with timeout and captured output. |
| `navigate_to_pose` | `pose: Pose2D` | Generate/execute `ros2 action send_goal` for Nav2. |
| `navigate_to_pose_with_watch` | `pose`, `on_tick`, `tick_interval_sec` | Fail explicitly because CLI cannot safely interleave navigation and camera processing. |
| `stop_robot` | None | Publish three zero `Twist` messages; does not own/cancel an unknown Nav2 goal. |
| `get_pose` | None | Read one odometry message and convert it to `Pose2D`. |
| `cancel_navigation` | None | Return unsupported because CLI does not retain a goal handle. |
| `detect_color` | `color: str` | Report that image interpretation requires the rclpy backend. |

#### `RclpyRos2Adapter`

| Name | Input | Description |
|---|---|---|
| `__init__` | `settings` | Create the rclpy node, Nav2 action client, TF buffer, subscribers, and velocity publisher. |
| `_on_odom` | ROS `Odometry` message | Cache the latest odometry pose. |
| `_on_image` | ROS `Image` message | Cache the latest camera image message. |
| `update_detection_overlay` | `detections` | Cache the latest search detections for short-lived visualization. |
| `_publish_annotated_image` | ROS `Image` message | Draw cached normalized boxes, labels, and confidence on a BGR frame and publish it with the source header. |
| `navigate_to_pose` | `pose: Pose2D` | Send a Nav2 action goal and wait for its terminal result. |
| `navigate_to_pose_with_watch` | `pose`, detection callback, interval | Run Nav2 while periodically detecting; cancel the action when a target is found. |
| `stop_robot` | None | Cancel this adapter's active goal, then publish zero velocity three times. |
| `get_pose` | None | Read the live `map -> base_link` TF and return a map-frame `Pose2D`. |
| `_on_navigation_feedback` | Nav2 feedback | Consume transport telemetry without sending raw feedback to the LLM. |
| `_publish_zero_velocity` | optional message count | Publish explicit zero `Twist` commands through the shared velocity publisher. |
| `_settle_after_navigation_cancel` | None | Hold zero velocity for the configured post-cancel handoff window so residual Nav2 output drains before visual control. |
| `cancel_navigation` | None | Cancel the retained Nav2 goal and verify terminal canceled status. |
| `detect_color` | `color: str` | Convert the latest ROS image to BGR and run HSV blob detection. |
| `get_camera_frame` | None | Convert and return the latest ROS image as a BGR array. |
| `align_to_detection` | detection callback plus horizontal/box-size targets, linear/angular bounds and gains, stability count, timeout | First rotate with zero linear velocity until horizontal alignment is stable, then translate with zero angular velocity until bbox height is stable; drift returns to rotation. |
| `align_to_detection.publish_stop` | None; nested helper | Publish three zero `Twist` messages whenever alignment stops, succeeds, fails, or times out. |
| `close` | None | Destroy the node and shut down an owned rclpy context. |
| `build_ros2_adapter` | `settings`, `backend="cli"` | Construct either `Ros2CliAdapter` or `RclpyRos2Adapter`. |

## Perception

### `src/robot_agent/perception/color_detection.py`

| Name | Input | Description |
|---|---|---|
| `detect_colored_blobs` | BGR `image`, `color: str` | Detect red, green, or blue HSV blobs and return confidence plus normalized image-plane boxes. |

### `src/robot_agent/perception/detector.py`

| Name | Input | Description |
|---|---|---|
| `Detector.validate_query` | optional `color`, optional `label` | Abstract pre-motion compatibility check for a detector query. |
| `Detector.detect` | BGR `image`, optional `color`, optional `label` | Abstract image-to-`Detection` conversion. |
| `ColorBlobDetector.validate_query` | `color`, optional `label` | Require a supported color and no class other than `colored_object`. |
| `ColorBlobDetector.detect` | BGR `image`, `color`, optional `label` | Delegate to HSV blob detection. |
| `YoloDetector.__init__` | `model_name="yolov8n.pt"`, `input_size=640` | Load an optional Ultralytics YOLO model. |
| `YoloDetector.validate_query` | optional `color`, required `label` | Require a class label and reject color filtering. |
| `YoloDetector.detect` | BGR `image`, required `label` | Run closed-set YOLO and return confidence plus normalized bounding boxes. |
| `YoloeDetector.__init__` | `model_name="yoloe-26s-seg.pt"`, `input_size=640` | Load pinned Ultralytics YOLOE-26 for open-vocabulary detection. |
| `YoloeDetector.validate_query` | optional `color`, required `label` | Require a concrete text-prompt label. |
| `YoloeDetector.detect` | BGR `image`, required `label` | Set the text vocabulary, run YOLOE, and return normalized bounding boxes. |
| `VlmDetector.validate_query` | optional `color`, optional `label` | Placeholder that currently raises `NotImplementedError`. |
| `VlmDetector.detect` | BGR `image`, optional `color`, optional `label` | Placeholder that currently raises `NotImplementedError`. |
| `build_detector` | `backend`, optional YOLO/YOLOE models and input size | Build `color_blob`, `yolo`, or default `yoloe`; reject the unimplemented `vlm` backend. |

## Behavior-tree skill

### `src/robot_agent/skills/behavior_tree.py`

| Name | Input | Description |
|---|---|---|
| `behavior_tree_to_xml` | `plan`, optional `tree_id`, `navigation_retry_count` | Export a validated plan as BehaviorTree.CPP-style XML. |
| `BehaviorTreeSkill._validate` | `plan: BehaviorTreePlan` | Enforce known locations, node limits, wait bounds, retries, and final `Stop`. |
| `BehaviorTreeSkill._generate_plan` | `goal: str` | Ask the model for a structured BT plan and validate it. |
| `BehaviorTreeSkill._persist` | `plan` | Write `behavior_tree.json` and `behavior_tree.xml` into the run directory. |
| `BehaviorTreeSkill.generate` | `goal: str` | Generate and persist a tree without commanding the robot. |
| `BehaviorTreeSkill.run` | `goal` plus `navigate`, `stop`, `wait`, `abort`, optional node callback | Generate once and execute validated `GoToPose`, `Wait`, and `Stop` nodes in sequence with bounded retries. |

## Runtime, persistence, and events

### `src/robot_agent/runtime/rviz_config.py`

| Name | Input | Description |
|---|---|---|
| `configure_rviz` | Parsed TurtleBot RViz config | Retain map/navigation displays, remove old auxiliary docks, and add raw plus YOLOE camera displays. |
| `write_robot_agent_config` | Source and destination paths | Materialize the focused RViz YAML used at GUI startup. |

### `src/robot_agent/runtime/runtime.py`

| Name | Input | Description |
|---|---|---|
| `RobotAgentRuntime.__init__` | `settings`, `goal` | Create a run ID/directory, restore semantic session state, and initialize journal/checkpoints. |
| `emit` | `event_type`, `payload`, optional `category` | Append a runtime event and optionally print its trace. |
| `record_tool_result` | `tool_name`, `arguments`, `result` | Update progress, plan, history, failures, events, and checkpoints after every tool. |
| `_update_plan` | `tool_name`, `result` | Match tool capability to the closest pending plan step and update completion state. |
| `no_progress_exhausted` | Property, no input | Indicate whether the semantic no-progress limit has been reached. |
| `save_checkpoint` | None | Atomically save full run state and persistent semantic session state. |
| `finish` | `status: str` | Mark the run terminal, save it, and emit `run_finished`. |

### `src/robot_agent/runtime/checkpoint.py`

| Name | Input | Description |
|---|---|---|
| `JsonCheckpointStore.__init__` | `path: Path` | Configure a JSON checkpoint file and ensure its parent directory exists. |
| `save` | `state: dict` | Atomically write state through a temporary file. |
| `load` | None | Load and validate a JSON-object checkpoint, or return `None`. |

### `src/robot_agent/runtime/events.py`

| Name | Input | Description |
|---|---|---|
| `RuntimeEvent.to_dict` | Current event | Serialize one immutable runtime event. |

### `src/robot_agent/runtime/journal.py`

| Name | Input | Description |
|---|---|---|
| `RunJournal.__init__` | `path`, `run_id` | Configure an append-only JSONL trace and restore its last sequence number. |
| `_load_last_sequence` | None | Scan an existing journal for the highest valid sequence. |
| `append` | `event: RuntimeEvent` | Assign the next sequence and append the event as one JSONL line. |

## Middleware and verification

### `src/robot_agent/middlewares/loop_detection.py`

| Name | Input | Description |
|---|---|---|
| `ToolLoopDetector.__init__` | `warn_threshold`, `hard_limit` | Configure identical-call warning and blocking thresholds. |
| `_key` | `tool_name`, `arguments` | Build a stable identity for one tool call. |
| `check` | `tool_name`, `arguments` | Count the call and return whether it is allowed or should warn. |

### `src/robot_agent/middlewares/model_termination.py`

| Name | Input | Description |
|---|---|---|
| `_finish_reason` | `AIMessage` | Normalize provider finish/stop reason metadata. |
| `_has_raw_tool_intent` | `AIMessage` | Detect complete, invalid, or partial tool-call intent. |
| `_append_explanation` | message `content`, `explanation` | Append a safe termination explanation to string or block content. |
| `ModelTerminationMiddleware.__init__` | `robot_runtime` | Attach termination handling to the current robot run. |
| `ModelTerminationMiddleware._clean_tool_calls` | `message`, `reason` | Remove incomplete/safety-blocked calls so they cannot execute. |
| `ModelTerminationMiddleware.after_model` | LangChain `state`, `runtime` | Record length/safety termination and repair the provider response. |

### `src/robot_agent/middlewares/plan_completion.py`

| Name | Input | Description |
|---|---|---|
| `_has_tool_intent` | `AIMessage` | Determine whether the model is attempting a tool call. |
| `PlanCompletionMiddleware.__init__` | `robot_runtime`, `max_reminders=2` | Configure bounded hidden continuation reminders for one run. |
| `PlanCompletionMiddleware._pending_steps` | None | Return advisory plan steps not marked completed. |
| `PlanCompletionMiddleware.after_model` | LangChain `state`, `runtime` | Request a bounded hidden continuation when the model stops with pending work. |
| `PlanCompletionMiddleware.wrap_model_call` | model `request`, next `handler` | Inject the pending hidden reminder into the next model call. |

### `src/robot_agent/goal_monitor/monitor.py`

| Name | Input | Description |
|---|---|---|
| `GoalMonitor.evaluate` | `state: RunState` | Deterministically classify success or blocker from tool, pose, plan, and perception evidence. |

### `src/robot_agent/guardrails/safety.py`

| Name | Input | Description |
|---|---|---|
| `SafetyValidator.__init__` | `settings: RobotAgentSettings` | Attach deterministic checks to the configured runtime safety limits. |
| `SafetyValidator.validate_pose` | `pose: Pose2D` | Require finite coordinates inside the configured workspace. |
| `SafetyValidator.validate_wait` | `seconds: float` | Require a positive wait no longer than the tool timeout. |

## Semantic state and world model

### `src/robot_agent/world_model/model.py`

| Name | Input | Description |
|---|---|---|
| `WorldModel.__init__` | `robot_state: RobotState` | Wrap the session's shared semantic robot state. |
| `WorldModel.update_pose` | `pose: Pose2D` | Store the latest confirmed robot pose. |
| `update_navigation_status` | `status`, `planned_pose` | Store navigation status and a dry-run planned pose when applicable. |
| `update_detections` | `detections: list[Detection]` | Replace visible objects and record perception time. |
| `find` | optional `color`, optional `label` | Filter currently visible semantic detections. |
| `has_perception_observation` | None | Report whether any perception update has occurred. |
| `context` | None | Return compact robot state for tools and prompts. |

### `src/robot_agent/state.py`

| Name | Input | Description |
|---|---|---|
| `utc_now` | None | Return the current UTC timestamp in ISO format. |
| `ToolResult.to_dict` | Current result | Serialize a normalized tool result and enum status. |
| `GoalEvaluation.to_dict` | Current evaluation | Serialize the deterministic goal verdict. |
| `Pose2D.to_dict` | Current pose | Serialize `x`, `y`, `yaw`, and `frame_id`. |
| `ImagePosition.to_dict` | Current image position | Serialize bbox center pixels plus normalized center, width, and height. |
| `Detection.to_dict` | Current detection | Serialize label, confidence, color, optional world pose, and image position. |
| `Detection.from_snapshot` | detection dictionary | Restore a `Detection` from persisted data. |
| `matching_goal_detections` | `goal_requirements`, `visible_objects` | Return detections matching every requested color and semantic label; either filter may be omitted. |
| `goal_requirements_satisfied` | `goal_requirements`, `visible_objects` | Provide the shared deterministic perception-goal verdict used by runtime plan updates and the goal monitor. |
| `RobotState.to_agent_context` | Current robot state | Return compact semantic state without raw ROS messages. |
| `RobotState.from_snapshot` | robot-state dictionary | Restore semantic robot state. |
| `SemanticSessionState.to_snapshot` | Current session state | Serialize robot facts and visited locations for cross-run persistence. |
| `SemanticSessionState.from_snapshot` | session dictionary | Validate and restore persistent semantic state. |
| `RunState.to_agent_context` | Current run | Return bounded plan, evidence, failures, progress, and status for the agent. |
| `RunState.to_snapshot` | Current run | Serialize complete recovery state including tool history. |
| `RunState.to_semantic_session_state` | Current run | Extract only robot-world facts that should survive future goals. |
| `RunState.progress_signature` | Current run | Return observable semantic fields used to detect no-progress loops. |

## Configuration

### `src/robot_agent/config/settings.py`

| Name | Input | Description |
|---|---|---|
| `_as_bool` | `value: str | None`, `default=False` | Parse common environment boolean strings. |
| `RobotAgentSettings.__post_init__` | Constructed settings | Validate backend names, timeouts, loop limits, perception controls, model size, and workspace bounds. |
| `RobotAgentSettings.from_env` | `project_root: Path` | Build complete runtime settings from environment variables and project defaults. |
