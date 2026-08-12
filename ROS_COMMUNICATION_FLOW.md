# ROS Communication Flow

This document summarizes how the current ROSA framework turns user input into ROS1/ROS2 communication, and how the TurtleSim demo ultimately turns tool calls into visible drawing behavior.

It covers:

- the LangChain/ROSA orchestration layer
- ROS1 communication in `src/rosa/tools/ros1.py`
- ROS2 communication in `src/rosa/tools/ros2.py`
- TurtleSim drawing communication in `src/turtle_agent/scripts/tools/turtle.py`

---

## Visual Overview

### High-Level Orchestration

```text
+------------------+
| User Natural     |
| Language Query   |
+------------------+
         |
         | query: str
         | via ROSA.invoke(query)
         v
+------------------+
| ROSA             |
| src/rosa/rosa.py |
+------------------+
         |
         | {"input": query, "chat_history": [...]}
         | via AgentExecutor.invoke(...)
         v
+-----------------------------+
| LangChain Tool-Calling      |
| Agent + AgentExecutor       |
+-----------------------------+
         |
         | tool selection
         | tool_name + tool_args
         | via create_tool_calling_agent(...)
         v
+-----------------------------+
| Python Tool Function        |
| @tool in ros1.py / ros2.py |
| or turtle.py               |
+-----------------------------+
         |
         | actual ROS communication
         | via ROS APIs or subprocess("ros2 ...")
         v
+-----------------------------+
| ROS Runtime / TurtleSim     |
+-----------------------------+
         |
         | tool result: dict / string / ROS message data
         v
+-----------------------------+
| LangChain Agent             |
| final natural-language      |
| answer                      |
+-----------------------------+
```

### TurtleSim Drawing Flow

```text
+------------------+
| User:            |
| "Draw a house"   |
+------------------+
         |
         | query string
         | ROSA.invoke(query)
         v
+----------------------+
| LangChain Agent      |
+----------------------+
         |
         | choose high-level tool
         | draw_rectangle(...)
         | draw_polyline(...)
         v
+----------------------+
| turtle.py high-level |
| drawing tools        |
+----------------------+
         |
         | nested tool calls
         | set_pen(...)
         | teleport_absolute(...)
         | publish_twist_to_cmd_vel(...)
         v
+----------------------+
| turtle.py low-level  |
| ROS comm layer       |
+----------------------+
   |            |             |
   | service    | service     | topic publish
   | request    | request     | Twist message
   | via        | via         | via publisher.publish(...)
   | call_      | call_       |
   | service()  | service()   |
   v            v             v
+-------------+ +----------------------+ +----------------------+
| /set_pen    | | /teleport_absolute   | | /turtle1/cmd_vel    |
| turtlesim   | | turtlesim service    | | geometry_msgs/Twist |
+-------------+ +----------------------+ +----------------------+
                                              |
                                              | turtlesim consumes Twist
                                              | turtle moves with pen on
                                              v
                                     +----------------------+
                                     | Visible drawing      |
                                     | on turtlesim canvas  |
                                     +----------------------+
```

---

## Core ROSA Execution Flow

Main file: [src/rosa/rosa.py](src/rosa/rosa.py)

### Construction

When you create:

```python
agent = ROSA(ros_version=1 or 2, llm=...)
```

the following internal setup happens:

```text
ROSA.__init__()
-> _get_tools()
-> _get_prompts()
-> _get_agent()
-> _get_executor()
```

### Runtime

When you call:

```python
agent.invoke("some query")
```

the actual invocation payload is:

```python
{
  "input": query,
  "chat_history": self.__chat_history,
}
```

This is passed into:

```python
self.__executor.invoke(...)
```

The executor is an `AgentExecutor`, and the agent is created with:

```python
create_tool_calling_agent(
    llm=self.__llm,
    tools=self.__tools.get_tools(),
    prompt=self.__prompts,
)
```

That means the tool-calling workflow is:

```text
query
-> prompt + tools + llm
-> agent decides tool name and args
-> executor calls Python tool function
-> tool returns result
-> agent decides whether to call another tool
-> final answer
```

---

## Tool Loading Flow

Main file: [src/rosa/tools/__init__.py](src/rosa/tools/__init__.py)

### Selection logic

```text
ROSA(ros_version=1)
-> load calculation + log + system + ros1

ROSA(ros_version=2)
-> load calculation + log + system + ros2
```

So:

- `src/rosa/tools/ros1.py` is used when `ros_version=1`
- `src/rosa/tools/ros2.py` is used when `ros_version=2`

All tool functions are ordinary Python functions decorated with `@tool`.

---

## ROS1 Communication Flow

Main file: [src/rosa/tools/ros1.py](src/rosa/tools/ros1.py)

### Communication style

ROS1 tools primarily use:

- `rosnode`
- `rostopic`
- `rosgraph`
- `rospy`
- `rosservice`
- `rosparam`

So the transport is usually direct ROS1 Python API access, not shell commands.

### ROS1 overview

```text
+-------------------+
| LangChain tool    |
| call              |
+-------------------+
         |
         | Python args
         | e.g. topics=["/turtle1/pose"]
         v
+-------------------+
| ros1.py tool      |
+-------------------+
         |
         | direct ROS1 API call
         | rostopic.* / rosnode.* / rospy.* / rosservice.*
         v
+-------------------+
| ROS1 Master /     |
| Topic / Service / |
| Param Server      |
+-------------------+
```

### ROS1 communication table

| Function | Transport type | Explicit transmission | Actual function used | Return format |
|---|---|---|---|---|
| `rosnode_list(...)` | ROS master introspection | query for node names | `rosnode.get_node_names()` | `dict` |
| `rostopic_list(...)` | ROS master/topic introspection | query for topic list | `rostopic.get_topic_list()` | `dict` |
| `rosgraph_get(...)` | ROS graph introspection | query full system state | `rosgraph.masterapi.Master(...).getSystemState()` | `dict` |
| `rostopic_info(topics)` | topic metadata introspection | topic name list | `rostopic.get_info_text(topic)` | `dict` |
| `rostopic_echo(topic, ...)` | ROS1 topic subscribe | topic name + msg type + timeout | `rostopic.get_topic_class(...)`, `rospy.wait_for_message(...)` | `dict` |
| `rosnode_info(nodes)` | node introspection | node name list | `rosnode.get_node_info_description(node)` | `dict` |
| `rosservice_list(...)` | service introspection | filters / namespace | `rosservice.*` functions | `dict` |
| `rosservice_call(...)` | ROS1 service request/response | service name + service type + request payload | ROS1 service proxy calls | `dict` or error |
| `rosparam_get/set/list` | ROS param server | param key/value | `rosparam.*` | `dict` |

### ROS1 explicit topic receive example

For:

```python
rostopic_echo(
    topic="/turtle1/pose",
    count=1,
    return_echoes=True,
    timeout=1.0
)
```

the transmission flow is:

```text
tool args:
{
  "topic": "/turtle1/pose",
  "count": 1,
  "return_echoes": True,
  "timeout": 1.0
}
         |
         | get message class
         | rostopic.get_topic_class("/turtle1/pose")
         v
message type inferred
         |
         | subscribe and block for one message
         | rospy.wait_for_message(...)
         v
actual ROS1 message object received
         |
         | wrap in dict
         v
{"echoes": [msg]}
```

---

## ROS2 Communication Flow

Main file: [src/rosa/tools/ros2.py](src/rosa/tools/ros2.py)

### Communication style

ROS2 tools in this file do not primarily use `rclpy`.

Instead, they wrap ROS2 CLI commands and execute them through:

```python
subprocess.check_output(command, shell=True).decode()
```

So the transport is:

```text
tool args
-> shell command string
-> subprocess.check_output(...)
-> ros2 CLI
-> stdout text
-> parse lines / wrap result
```

### ROS2 overview

```text
+-------------------+
| LangChain tool    |
| call              |
+-------------------+
         |
         | Python args
         | e.g. {"topic": "/turtle1/pose"}
         v
+-------------------+
| ros2.py tool      |
+-------------------+
         |
         | build command string
         | e.g. "ros2 topic echo /turtle1/pose --once"
         v
+-------------------+
| subprocess        |
| check_output(...) |
+-------------------+
         |
         | stdout text
         v
+-------------------+
| parsed result     |
+-------------------+
```

### ROS2 communication table

| Function | Transport type | Explicit transmission | Actual function used | Return format |
|---|---|---|---|---|
| `ros2_node_list(...)` | shell command | `"ros2 node list"` | `execute_ros_command()` -> `subprocess.check_output(...)` | `dict` |
| `ros2_topic_list(...)` | shell command | `"ros2 topic list"` | same | `dict` |
| `ros2_service_list(...)` | shell command | `"ros2 service list"` | same | `dict` |
| `ros2_topic_echo(...)` | shell command | `"ros2 topic echo {topic} --once --spin-time {timeout}"` | same | `dict` |
| `ros2_node_info(nodes)` | shell command | `"ros2 node info {node_name}"` | same | `dict` |
| `ros2_topic_info(topics)` | shell command | `"ros2 topic info {topic} --verbose"` | same | `dict` |
| `ros2_param_list(...)` | shell command | `"ros2 param list"` or `"ros2 param list {node}"` | same | `dict` |
| `ros2_param_get(...)` | shell command | `"ros2 param get {node} {param}"` | same | `dict` |
| `ros2_param_set(...)` | shell command | `"ros2 param set {node} {param} {value}"` | same | `dict` |
| `ros2_service_info(...)` | shell command | `"ros2 service type {service}"` | same | `dict` |
| `ros2_service_call(...)` | shell command | `"ros2 service call {service} {srv_type} \"{request}\""` | same | `dict` |
| `ros2_doctor()` | shell command | `"ros2 doctor"` | same | `dict` |

### ROS2 explicit service call example

For:

```python
ros2_service_call(
    service_name="/clear",
    srv_type="std_srvs/srv/Empty",
    request="{}"
)
```

the transmission flow is:

```text
tool args:
{
  "service_name": "/clear",
  "srv_type": "std_srvs/srv/Empty",
  "request": "{}"
}
         |
         | format shell command
         | via ros2_service_call(...)
         v
"ros2 service call /clear std_srvs/srv/Empty \"{}\""
         |
         | execute
         | execute_ros_command(...)
         v
subprocess.check_output(...)
         |
         | CLI stdout text
         v
{"response": "..."}
```

---

## TurtleSim Drawing Communication Flow

Main file: [src/turtle_agent/scripts/tools/turtle.py](src/turtle_agent/scripts/tools/turtle.py)

This is the file that actually turns tool calls into motion and drawing.

The current version uses ROS2 native communication through `rclpy`.

### TurtleSim overview

```text
+----------------------+
| High-level drawing   |
| tool                 |
| draw_rectangle(...)  |
+----------------------+
         |
         | nested Python tool calls
         | draw_rectangle -> set_pen -> teleport -> publish_twist
         v
+----------------------+
| Low-level turtle.py  |
| comm tools           |
+----------------------+
   |            |             |
   | service    | service     | topic publish
   | request    | request     | geometry_msgs/Twist
   v            v             v
/turtle1/set_pen  /turtle1/teleport_absolute  /turtle1/cmd_vel
   |            |             |
   | turtlesim node consumes requests/messages
   v            v             v
pen state     turtle moved   turtle moved continuously
                                  |
                                  | if pen is on
                                  v
                           visible line appears
```

### Low-level drawing communication table

| Function | Transport type | Explicit transmission | Actual function used | Destination |
|---|---|---|---|---|
| `set_pen(...)` | ROS2 service request | `SetPen.Request(r,g,b,width,off)` | `node.call_service(SetPen, f"/{name}/set_pen", request)` | `/{name}/set_pen` |
| `teleport_absolute(...)` | ROS2 service request | `TeleportAbsolute.Request(x,y,theta)` | `node.call_service(TeleportAbsolute, f"/{name}/teleport_absolute", request)` | `/{name}/teleport_absolute` |
| `teleport_relative(...)` | ROS2 service request | `TeleportRelative.Request(linear,angular)` | `node.call_service(TeleportRelative, f"/{name}/teleport_relative", request)` | `/{name}/teleport_relative` |
| `publish_twist_to_cmd_vel(...)` | ROS2 topic publish | `Twist(linear.x, linear.y, angular.z)` | `publisher.publish(vel)` | `/{name}/cmd_vel` |
| `get_turtle_pose(...)` | ROS2 topic subscribe | `Pose` message subscription | `create_subscription(Pose, f"/{name}/pose", callback, 10)` | `/{name}/pose` |
| `spawn_turtle(...)` | ROS2 service request | `Spawn.Request(x,y,theta,name)` | `node.call_service(Spawn, "/spawn", request)` | `/spawn` |
| `kill_turtle(...)` | ROS2 service request | `Kill.Request(name)` | `node.call_service(Kill, "/kill", request)` | `/kill` |
| `clear_turtlesim()` | ROS2 service request | `Empty.Request()` | `node.call_service(Empty, "/clear", request)` | `/clear` |
| `reset_turtlesim()` | ROS2 service request | `Empty.Request()` | `node.call_service(Empty, "/reset", request)` | `/reset` |

### Explicit drawing payloads

#### Twist publish

`publish_twist_to_cmd_vel(...)` builds:

```python
vel = Twist()
vel.linear.x = velocity
vel.linear.y = lateral
vel.linear.z = 0.0
vel.angular.x = 0.0
vel.angular.y = 0.0
vel.angular.z = angle
```

Then sends it through:

```python
pub.publish(vel)
```

to:

```text
topic: /turtle1/cmd_vel
type: geometry_msgs/msg/Twist
```

#### Teleport service

`teleport_absolute(...)` builds:

```python
request = TeleportAbsolute.Request()
request.x = float(x)
request.y = float(y)
request.theta = float(theta)
```

Then sends it through:

```python
node.call_service(
    TeleportAbsolute,
    "/turtle1/teleport_absolute",
    request
)
```

#### Pen service

`set_pen(...)` builds:

```python
request = SetPen.Request()
request.r = int(r)
request.g = int(g)
request.b = int(b)
request.width = int(width)
request.off = int(off)
```

Then sends it through:

```python
node.call_service(SetPen, "/turtle1/set_pen", request)
```

---

## End-to-End "Draw Something" Flow

Here is the full chain for a typical drawing request.

```text
+----------------------+
| User                 |
| "Draw a rectangle"   |
+----------------------+
         |
         | query string
         | ROSA.invoke(query)
         v
+----------------------+
| AgentExecutor        |
+----------------------+
         |
         | {"input": query, "chat_history": [...]}
         | invoke(...)
         v
+----------------------+
| Tool-calling agent   |
+----------------------+
         |
         | select tool
         | draw_rectangle(name="turtle1", x=2, y=2, width=3, height=3)
         v
+----------------------+
| draw_rectangle(...)  |
+----------------------+
         |
         | nested call
         | set_pen(... off=1)
         v
+----------------------+
| /turtle1/set_pen     |
| SetPen.Request       |
+----------------------+
         |
         | nested call
         | teleport_absolute(...)
         v
+----------------------+
| /turtle1/teleport_   |
| absolute             |
| TeleportAbsolute.    |
| Request              |
+----------------------+
         |
         | nested call
         | set_pen(... off=0)
         v
+----------------------+
| /turtle1/set_pen     |
+----------------------+
         |
         | nested call
         | publish_twist_to_cmd_vel(...)
         | publisher.publish(Twist)
         v
+----------------------+
| /turtle1/cmd_vel     |
| geometry_msgs/Twist  |
+----------------------+
         |
         | turtlesim updates turtle pose
         | pen is on, so path is drawn
         v
+----------------------+
| Canvas changes       |
| visible line         |
+----------------------+
         |
         | repeated for each edge
         v
+----------------------+
| Tool return string   |
+----------------------+
         |
         | fed back to agent
         v
+----------------------+
| Final natural-       |
| language answer      |
+----------------------+
```

---

## Simplified Example: "Draw Me a House"

This section gives a simplified, concrete example of what happens when the user asks:

```text
draw me a house
```

This example matches the current demo state:

- `ros_version=2`
- ROS2 runtime inspection tools come from `src/rosa/tools/ros2.py`
- TurtleSim drawing tools come from `src/turtle_agent/scripts/tools/turtle.py`

### Simplified overview

```text
+----------------------+
| User                 |
| "draw me a house"    |
+----------------------+
         |
         | query string
         | ROSA.invoke(...) / ROSA.astream(...)
         v
+----------------------+
| Prompt Assembly      |
| base ROSA prompts +  |
| TurtleSim prompts +  |
| chat history         |
+----------------------+
         |
         | built by
         | _get_prompts()
         v
+----------------------+
| LangChain Agent      |
| create_tool_calling_ |
| agent(...)           |
+----------------------+
         |
         | selects tool_name + tool_args
         | based on prompt + tool schemas
         v
+----------------------+
| AgentExecutor        |
+----------------------+
         |
         | executes chosen Python tool
         v
+----------------------+      +----------------------+
| ros2.py tools        |      | turtle.py tools      |
| inspection           |      | drawing              |
+----------------------+      +----------------------+
         |                              |
         | build CLI string             | build ROS2 request/message objects
         | subprocess("ros2 ...")       | call_service(...) / publish(Twist)
         v                              v
+----------------------+      +----------------------+
| ROS2 CLI             |      | turtlesim ROS2 APIs  |
+----------------------+      +----------------------+
         |                              |
         | parsed stdout                | turtle moves / draws
         v                              v
+----------------------+      +----------------------+
| tool result          |      | visible house        |
+----------------------+      +----------------------+
```

### What the prompt does

The model does not just see:

```text
draw me a house
```

It sees an assembled prompt made from:

- base ROSA tool-usage instructions in `src/rosa/prompts.py`
- TurtleSim drawing workflow instructions in `src/turtle_agent/scripts/prompts.py`
- prior `chat_history`
- the current user message
- agent scratchpad/tool traces

That means the model gets explicit guidance such as:

- verify runtime state with tools
- use sequential tool execution
- prefer high-level drawing tools like `draw_rectangle` and `draw_polyline`

### What tools are available

Because the demo currently sets:

```python
ros_version=2
```

the core ROSA tool loader includes:

- `src/rosa/tools/ros2.py`
- `src/rosa/tools/calculation.py`
- `src/rosa/tools/log.py`
- `src/rosa/tools/system.py`

and `TurtleAgent` also injects:

- `src/turtle_agent/scripts/tools/turtle.py`

So the model can choose both:

- ROS2 inspection tools
  - `ros2_node_list`
  - `ros2_topic_list`
  - `ros2_service_list`
- TurtleSim drawing tools
  - `draw_rectangle`
  - `draw_polyline`
  - `draw_circle`
  - `set_pen`
  - `teleport_absolute`
  - `publish_twist_to_cmd_vel`

### Likely tool sequence

For "draw me a house", a simplified likely sequence is:

```text
1. ros2_node_list({})
2. ros2_topic_list({})
3. draw_rectangle(base wall)
4. draw_rectangle(door)
5. draw_rectangle(window1)
6. draw_rectangle(window2)
7. draw_polyline(roof)
8. final natural-language answer
```

### Step-by-step table

| Step | Chosen tool | Input args | Internal behavior | Explicit transmission format | Destination |
|---|---|---|---|---|---|
| 1 | `ros2_node_list` | `{}` | builds CLI command | `"ros2 node list"` | ROS2 CLI |
| 2 | `ros2_topic_list` | `{}` | builds CLI command | `"ros2 topic list"` | ROS2 CLI |
| 3 | `draw_rectangle` | base rectangle args | high-level drawing helper | nested tool calls | turtle.py |
| 4 | `set_pen` | `{"name":"turtle1","r":0,"g":0,"b":0,"width":2,"off":1}` | build service request | `SetPen.Request(...)` | `/turtle1/set_pen` |
| 5 | `teleport_absolute` | start point and heading | build service request | `TeleportAbsolute.Request(x,y,theta)` | `/turtle1/teleport_absolute` |
| 6 | `set_pen` | pen-on args | build service request | `SetPen.Request(...)` | `/turtle1/set_pen` |
| 7 | `publish_twist_to_cmd_vel` | edge motion args | build topic message | `geometry_msgs/msg/Twist` | `/turtle1/cmd_vel` |
| 8 | repeat 4-7 | next edge args | draw all edges | service requests + Twist publishes | turtlesim |
| 9 | `draw_polyline` | roof point list | high-level helper | nested line drawing calls | turtle.py |
| 10 | final answer | none | summarize completed work | natural language | user |

### How ROS2 inspection tools become CLI calls

For inspection tools in `src/rosa/tools/ros2.py`, the path is:

```text
tool args
-> Python tool function
-> build shell command string
-> execute_ros_command(...)
-> subprocess.check_output(command, shell=True)
-> stdout text
-> parse into dict
```

Example:

```python
ros2_topic_list({})
```

becomes:

```bash
ros2 topic list
```

through:

```python
cmd = "ros2 topic list"
execute_ros_command(cmd)
```

### How TurtleSim drawing tools become ROS2 communication

The drawing part does not go through ROS2 CLI wrappers.

Instead, `src/turtle_agent/scripts/tools/turtle.py` builds typed ROS2 request/message objects.

#### Example: turning the pen off

Input:

```python
set_pen(name="turtle1", r=0, g=0, b=0, width=2, off=1)
```

Internal assembly:

```python
request = SetPen.Request()
request.r = 0
request.g = 0
request.b = 0
request.width = 2
request.off = 1
```

Transmission:

```python
node.call_service(SetPen, "/turtle1/set_pen", request)
```

#### Example: teleporting to the first corner

Input:

```python
teleport_absolute(name="turtle1", x=2, y=2, theta=0)
```

Internal assembly:

```python
request = TeleportAbsolute.Request()
request.x = 2.0
request.y = 2.0
request.theta = 0.0
```

Transmission:

```python
node.call_service(
    TeleportAbsolute,
    "/turtle1/teleport_absolute",
    request
)
```

#### Example: drawing one edge

Input:

```python
publish_twist_to_cmd_vel(
    name="turtle1",
    velocity=3.0,
    lateral=0.0,
    angle=0.0,
    steps=1
)
```

Internal message assembly:

```python
vel = Twist()
vel.linear.x = 3.0
vel.linear.y = 0.0
vel.linear.z = 0.0
vel.angular.x = 0.0
vel.angular.y = 0.0
vel.angular.z = 0.0
```

Transmission:

```python
pub.publish(vel)
```

Destination:

```text
/turtle1/cmd_vel
type: geometry_msgs/msg/Twist
```

### Full simplified chain

```text
draw me a house
-> query enters TurtleAgent
-> ROSA assembles prompt
-> LangChain agent sees tools + prompt
-> agent chooses ros2 inspection tools
-> ros2.py converts those into CLI commands like "ros2 node list"
-> agent chooses draw_rectangle / draw_polyline
-> turtle.py converts those into:
   - SetPen.Request(...)
   - TeleportAbsolute.Request(...)
   - Twist(...)
-> turtlesim receives service requests and topic messages
-> turtle moves with pen on
-> house appears on the canvas
-> agent returns final natural-language summary
```

---

## Short Summary

- `src/rosa/rosa.py` is the orchestration layer.
- `src/rosa/tools/ros1.py` communicates mainly through ROS1 Python APIs.
- `src/rosa/tools/ros2.py` communicates mainly through shelling out to `ros2` CLI.
- `src/turtle_agent/scripts/tools/turtle.py` is the layer that actually causes TurtleSim movement and visible drawing.
- The final visible drawing happens when `publish_twist_to_cmd_vel(...)` publishes a `geometry_msgs/Twist` message to `/{name}/cmd_vel` with the pen turned on.
