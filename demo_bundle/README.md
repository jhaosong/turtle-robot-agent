# ROSA Demo Bundle

This folder collects the main files involved in the TurtleSim demo so you can read the demo in isolation.

## Startup Path

1. `demo.sh`
   Builds and runs the Docker container.
2. `Dockerfile`
   Prepares the ROS Noetic + TurtleSim + Python environment inside Docker.
3. `src/turtle_agent/launch/agent.launch`
   Launches the TurtleSim demo agent node.
4. `src/turtle_agent/scripts/turtle_agent.py`
   Creates the `TurtleAgent` class and starts the CLI loop.
5. `src/turtle_agent/scripts/llm.py`
   Creates the LLM client from `.env`.
6. `src/turtle_agent/scripts/prompts.py`
   Provides TurtleSim-specific prompt instructions.
7. `src/rosa/rosa.py`
   Builds the LangChain agent, prompt, tools, and executor.
8. `src/rosa/tools/*.py`
   Provide the tools the agent can call.
9. `src/turtle_agent/scripts/tools/turtle.py`
   Adds TurtleSim-specific control and drawing tools.

## File Roles

### Root files

- `demo.sh`
  Entry point you run locally. Calls `docker build` and `docker run`.
- `Dockerfile`
  Defines the container image. Starts `roscore` and `turtlesim_node`, then tells you to run `start streaming:=true`.
- `.env`
  Holds model/provider configuration for the demo LLM.

### Turtle agent package

- `src/turtle_agent/package.xml`
  ROS package metadata for the `turtle_agent` package.
- `src/turtle_agent/CMakeLists.txt`
  Catkin package build file.
- `src/turtle_agent/launch/agent.launch`
  ROS launch file that runs `turtle_agent.py`.
- `src/turtle_agent/scripts/turtle_agent.py`
  Demo application layer.
  It:
  - subclasses `ROSA`
  - injects TurtleSim prompts and tools
  - runs the terminal chat loop
  - displays streaming tool events
- `src/turtle_agent/scripts/llm.py`
  Chooses the provider and creates `ChatOpenAI`, `AzureChatOpenAI`, `ChatAnthropic`, or `ChatOllama`.
- `src/turtle_agent/scripts/prompts.py`
  Defines TurtleSim-specific behavior, geometry rules, and drawing workflow.
- `src/turtle_agent/scripts/help.py`
  Generates the `help` command prompt text.
- `src/turtle_agent/scripts/tools/turtle.py`
  TurtleSim-specific tools such as drawing, teleporting, pen control, and pose checks.

### Core ROSA package

- `src/rosa/__init__.py`
  Export surface for `ROSA` and prompt types.
- `src/rosa/rosa.py`
  Core orchestrator.
  It:
  - loads tools
  - builds prompts
  - creates the LangChain tool-calling agent
  - wraps it in `AgentExecutor`
  - exposes `invoke()` and `astream()`
- `src/rosa/prompts.py`
  Base ROSA system prompts. These enforce tool usage rules and workflow expectations.
- `src/rosa/tools/__init__.py`
  Tool loader/collector. It gathers `@tool` functions and injects blacklist settings.
- `src/rosa/tools/ros1.py`
  ROS1 runtime tools used by this demo for nodes, topics, params, services, graph inspection, and logs.
- `src/rosa/tools/system.py`
  Generic helper tools like verbosity and wait.
- `src/rosa/tools/log.py`
  Log reading tool.
- `src/rosa/tools/calculation.py`
  Math tools the agent can call for geometry and reasoning support.

## How The Files Connect

### Launch flow

`demo.sh` -> `Dockerfile` -> `start streaming:=true` -> `agent.launch` -> `turtle_agent.py`

### Agent construction flow

`turtle_agent.py`
-> `get_llm()` from `llm.py`
-> `get_prompts()` from `prompts.py`
-> TurtleSim tools from `scripts/tools/turtle.py`
-> `ROSA(...)` in `src/rosa/rosa.py`

### ROSA internal flow

`ROSA.__init__`
-> `_get_tools()`
-> `src/rosa/tools/__init__.py`
-> load base tools + `ros1.py`

`ROSA.__init__`
-> `_get_prompts()`
-> base prompts from `src/rosa/prompts.py`
-> append TurtleSim prompts from `src/turtle_agent/scripts/prompts.py`

`ROSA.__init__`
-> `_get_agent()`
-> LangChain `create_tool_calling_agent(...)`

`ROSA.__init__`
-> `_get_executor()`
-> LangChain `AgentExecutor(...)`

### Runtime execution flow

User input
-> `TurtleAgent.run()`
-> `submit()`
-> `ROSA.invoke()` or `ROSA.astream()`
-> LangChain agent chooses a tool
-> `AgentExecutor` runs the selected Python tool
-> tool function executes ROS/TurtleSim logic
-> tool result returns to the agent
-> final answer is rendered in the CLI

## Most Important Files To Read First

If you only want the shortest path through the demo, read these in order:

1. `demo.sh`
2. `Dockerfile`
3. `src/turtle_agent/launch/agent.launch`
4. `src/turtle_agent/scripts/turtle_agent.py`
5. `src/turtle_agent/scripts/llm.py`
6. `src/turtle_agent/scripts/prompts.py`
7. `src/rosa/rosa.py`
8. `src/rosa/prompts.py`
9. `src/rosa/tools/__init__.py`
10. `src/turtle_agent/scripts/tools/turtle.py`
11. `src/rosa/tools/ros1.py`
