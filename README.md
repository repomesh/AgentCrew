<p align="center">
  <a href="https://github.com/saigontechnology/AgentCrew">
    <img src="https://saigontechnology.com/wp-content/uploads/2024/09/logo-black-1.svg" alt="AgentCrew Logo" width="300">
  </a>
</p>

<h1 align="center">AgentCrew: Your AI Team for Coding and Automation</h1>

<p align="center">
  Run specialized AI agents for coding, research, documentation, browser tasks,
  and automation from a GUI, terminal, job runner, or A2A server.
</p>

[![GitHub stars](https://img.shields.io/github/stars/saigontechnology/AgentCrew)](https://github.com/saigontechnology/AgentCrew/stargazers)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/agentcrew-ai?period=total&units=INTERNATIONAL_SYSTEM&left_color=LIGHTGREY&right_color=BRIGHTGREEN&left_text=Downloads)](https://pepy.tech/projects/agentcrew-ai)
[![Pylint](https://github.com/saigontechnology/AgentCrew/actions/workflows/pylint.yml/badge.svg)](https://github.com/saigontechnology/AgentCrew/actions/workflows/pylint.yml)
[![CodeQL](https://github.com/saigontechnology/AgentCrew/actions/workflows/codeql.yml/badge.svg)](https://github.com/saigontechnology/AgentCrew/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://github.com/saigontechnology/AgentCrew/blob/main/LICENSE)
[![Status: Actively Maintained](https://img.shields.io/badge/status-actively%20maintained-brightgreen)](https://github.com/saigontechnology/AgentCrew/releases)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://hub.docker.com/r/daltonnyx/agentcrew)

## Why try AgentCrew?

- **Use a team of specialists, not one generic assistant**: create agents for
  architecture, coding, reviewing, research, writing, QA, or browser work.
- **Run the same workflow in multiple modes**: use the desktop GUI, terminal,
  headless jobs, or an A2A-compatible server.
- **Stay provider-flexible**: work with Claude, OpenAI, Gemini, GitHub Copilot,
  DeepInfra, or custom OpenAI-compatible providers.
- **Do real work with tools**: combine MCP, web search, code analysis, file
  editing, browser automation, memory, and structured outputs.
- **Keep control**: review tool calls, limit permissions, and choose how much
  autonomy your agents get.

## What is AgentCrew?

AgentCrew is an AI assistant platform built around **specialized agents that can
collaborate**.

Instead of forcing one assistant to plan, research, code, review, and automate
everything, AgentCrew lets you define multiple focused agents with their own
instructions, tools, and behaviors. Those agents can either:

- **transfer** work to a specialist that takes over the conversation, or
- **delegate** tasks to multiple agents in parallel and merge the results

Think of it like giving yourself an AI architect, coder, reviewer, researcher,
and operator instead of one overloaded chatbot.

**Demo**

<https://github.com/user-attachments/assets/32876eac-b5e6-4608-bd5e-82d6fa4db80f>

## When AgentCrew shines

AgentCrew is especially useful when you need more than a simple coding chat:

- You want separate agents for planning, implementation, review, and docs
- You need GUI, CLI, automation jobs, and server mode in one tool
- You want to mix models and providers without changing your workflow
- You need MCP tools, browser automation, memory, or structured JSON output
- You want reusable agent configurations that can be shared across projects

## Try it in 2 minutes

### Quick installation

**Linux and macOS:**

```bash
curl -LsSf https://agentcrew.dev/install.sh | bash
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://agentcrew.dev/install.ps1 | iex"
```

**Docker:**

```bash
docker pull daltonnyx/agentcrew:latest
docker run -it --rm -e ANTHROPIC_API_KEY="your_key" daltonnyx/agentcrew chat
```

### First run

```bash
# Launch the desktop app
agentcrew chat

# Or use the terminal interface
agentcrew chat --console
```

Configure at least one provider API key, then try prompts like:

- `Create 3 specialized agents for architecture, coding, and code review.`
- `Review this repository structure and suggest a refactor plan.`
- `Analyze these files and return a JSON report with risks and recommendations.`
- `Use browser automation to inspect a page and summarize UX issues.`

## Core Capabilities

### 🤖 Multi-Agent Architecture

- Define multiple specialized agents with unique system prompts and tool access
- Agents automatically transfer tasks to teammates with better expertise
- Remote agent support for distributed team configurations
- A2A protocol for cross-system agent collaboration

### 🔄 Agent Interaction Modes

AgentCrew supports two mutually exclusive modes for agent-to-agent
collaboration, configured via `global_settings.agent_mode` in `config.json`:

**Transfer mode** (default): The active agent hands off full control to a
specialist. One agent is active at a time, and the full conversation history
carries over. Best for deep collaboration chains where context matters.

```
User → Agent A → [transfer] → Agent B takes over the conversation
```

**Delegate mode**: The active agent dispatches tasks to one or more agents in
parallel. Delegated agents run independently with fresh context and return
results as tool output. The orchestrating agent stays in control and synthesizes
the results.

```
User → Agent A → [delegate] → Agent B (task 1) ─┐
                            → Agent C (task 2) ─┤── all run concurrently
                            → Agent B (task 3) ─┘
               ← Agent A receives all results and responds
```

Key details:

- Switch modes at runtime with `/agent_mode [transfer|delegate|none]`
- Parallel fan-out: the LLM emits multiple delegate calls in one turn, all
  execute concurrently via `asyncio.gather`
- Each delegation gets a cloned agent with its own LLM service — no shared state
  between parallel delegates
- Delegated agents cannot delegate further (no recursion)
- Console shows live per-agent spinners with elapsed time during delegation
- GUI shows progress bars per delegate with running/completed status

### 🔌 AI Provider Support

- Anthropic Claude (all models including Claude 3.5 Sonnet)
- OpenAI GPT series with native and response modes
- OpenAI Codex via ChatGPT subscription (Plus/Pro) with OAuth
- Google Gemini with native API support
- GitHub Copilot integration with OAuth flow
- DeepInfra for alternative model hosting
- Custom OpenAI-compatible providers

### 🛠️ Tool Ecosystem

- **Web Capabilities**: Search current information, extract webpage content
- **Code Operations**: Analyze repository structure, read files with line
  ranges, understand project organization
- **File Editing**: Write or modify files using search-replace blocks with
  syntax validation, automatic backups, and safety checks
- **Browser Automation**: Navigate pages, click elements, fill forms, capture
  screenshots via Chrome DevTools Protocol
- **Command Execution**: Run shell commands with monitoring, rate limiting, and
  audit logging for safe system operations
- **System Integration**: Manage clipboard, generate images with DALL-E
- **Memory Systems**: Remember conversations, retrieve relevant context, forget
  by topic or date range, multiple embedding providers
- **Voice Interaction**: Speech-to-text input and text-to-speech responses via
  ElevenLabs or DeepInfra
- **MCP Integration**: Connect to external tools through Model Context Protocol
  with OAuth support

### ⚡ Parallel Tool Execution

When the LLM emits multiple tool calls in a single turn, AgentCrew executes them
concurrently where safe. Web searches, file reads, delegate calls, and MCP tools
all run in parallel via `asyncio.gather`. Sequential tools like transfer, ask,
and browser actions are handled one at a time. Results are always returned in the
order the LLM requested, and errors are isolated per-tool — one failure doesn't
block the others.

### 🎯 Adaptive Behavior System

Agents learn patterns using "when...do..." rules. Define behaviors like "when
analyzing code, provide security considerations" and agents automatically apply
these rules to future interactions. Behaviors persist across sessions and can be
updated anytime.

### 💻 User Interfaces

- **GUI**: Rich desktop interface with themes, syntax highlighting, conversation
  history, voice controls
- **Console**: Terminal interface with command completion, keyboard shortcuts,
  color output
- **Job Mode**: Non-interactive execution for automation and scripting
- **A2A Server**: HTTP API for programmatic agent access

### 📝 Conversation Management

- Rollback to any previous message and continue from there
- Consolidate message history to reduce token usage while preserving context
- Load and resume past conversations with full context restoration
- Smart paste detection for images and binary files
- File attachment support for text, images, PDFs, Office documents

## Installation and Authentication

If you already used the quickstart above, you can skip to the usage examples.
This section is for source installs and provider-specific authentication.

### Prerequisites

- Python 3.12 or later
- `uv` package manager: `pip install uv`
- API keys for at least one AI provider

### Standard Installation

```bash
git clone https://github.com/saigontechnology/AgentCrew.git
cd AgentCrew
uv sync
uv tool install .
```

After installing, launch AgentCrew with either:

```bash
agentcrew chat
agentcrew chat --console
```

Configure API keys and agents through the Settings menu in the GUI, or edit
configuration files in `~/.AgentCrew/`.

### Authentication

**GitHub Copilot:**

```bash
agentcrew copilot-auth
agentcrew chat --provider github_copilot
```

**ChatGPT Subscription (Plus/Pro):**

Use your existing ChatGPT Plus ($20/mo) or Pro ($200/mo) subscription for API
access — no separate API credits needed. This uses OpenAI's official Codex OAuth
flow.

```bash
agentcrew chatgpt-auth    # Opens browser to sign in with your ChatGPT account
agentcrew chat --provider openai_codex
```

Tokens are stored in `~/.codex/auth.json` (compatible with the official Codex
CLI) and refresh automatically. Models available include `gpt-5-codex`,
`gpt-5.1-codex`, `gpt-5.1-codex-mini`, and others.

> **Note:** ChatGPT subscription access is for personal development use. For
> production or multi-user applications, use the OpenAI Platform API with
> `--provider openai`.

## Usage Modes

### Interactive Chat

Start a conversation with your agent team. Switch between agents, add files, use
voice input, and approve tool usage as needed.

```bash
# GUI mode
agentcrew chat

# Console mode with specific provider
agentcrew chat --provider openai --console

# Custom agent configuration
agentcrew chat --agent-config ./my_agents.toml
```

### 🚀 Job Mode

Execute single-turn tasks without interaction. Perfect for automation pipelines
and batch processing.

```bash
# Analyze code files
agentcrew job --agent "CodeAssistant" \
  "Review for security issues" \
  ./src/**/*.py

# Generate documentation
agentcrew job --agent "TechnicalWriter" \
  "Create API documentation" \
  ./openapi.yaml

# Multiple files with specific provider
agentcrew job --agent "CodeAssistant" --provider claude \
  "Refactor these components for better performance" \
  ./src/component1.py ./src/component2.py ./src/utils.py

# With Docker
docker run --rm \
  -e ANTHROPIC_API_KEY="your_key" \
  -v $(pwd):/workspace \
  daltonnyx/agentcrew job \
  --agent "Architect" \
  "Design deployment architecture" \
  /workspace/requirements.md
```

#### Structured Output with JSON Schema

Job mode supports enforcing structured output using JSON Schema validation. The
agent will automatically retry if its response doesn't match the schema.

**Using Schema File:**

```bash
# Create schema file
cat > output_schema.json << 'EOF'
{
  "type": "object",
  "required": ["summary", "issues", "recommendations"],
  "properties": {
    "summary": {
      "type": "string",
      "description": "Brief summary of the analysis"
    },
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["severity", "description", "location"],
        "properties": {
          "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
          "description": {"type": "string"},
          "location": {"type": "string"},
          "suggestion": {"type": "string"}
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
EOF

# Use schema with job
agentcrew job --agent "CodeAssistant" \
  --output-schema output_schema.json \
  "Analyze this code for security issues" \
  ./src/authentication.py
```

**Using Inline Schema:**

```bash
agentcrew job --agent "DataAnalyst" \
  --output-schema '{"type":"object","required":["total","breakdown"],"properties":{"total":{"type":"number"},"breakdown":{"type":"array","items":{"type":"object"}}}}' \
  "Calculate project metrics" \
  ./project_data.csv
```

**Processing Structured Output:**

```bash
# Parse JSON output in shell scripts
output=$(agentcrew job --agent "CodeAssistant" \
  --output-schema schema.json \
  "Analyze code" ./main.py)

# Extract specific fields using jq
echo "$output" | jq -r '.issues[] | select(.severity == "critical") | .description'
```

The agent automatically receives instructions to format output according to your
schema and will retry up to 4 times if validation fails.

### 🌐 A2A Server Mode

Expose your agents as HTTP services for external integration or multi-instance
collaboration.

```bash
# Start A2A server
agentcrew a2a-server --host 0.0.0.0 --port 41241

# With Docker
docker run -d \
  --name agentcrew-server \
  -p 41241:41241 \
  -e ANTHROPIC_API_KEY="your_key" \
  daltonnyx/agentcrew a2a-server \
  --host 0.0.0.0 --port 41241
```

Agents expose their capabilities through agent cards at
`/.well-known/agent.json` and accept tasks via JSON-RPC. Other AgentCrew
instances or compatible clients can discover and invoke these agents.

## Configuration

Configuration is stored in `~/.AgentCrew/` and can be managed through the GUI or
by editing files directly.

**Key Configuration Files:**

- `config.json` - Global settings, API keys, UI preferences
- `agents.toml` - Agent definitions, tools, system prompts
- `mcp_servers.json` - Model Context Protocol server configurations
- `persistents/` - Saved conversations and learned behaviors

**Quick Configuration via GUI:**

1. Launch AgentCrew with `agentcrew chat`
2. Open Settings menu
3. Add API keys in Global Settings
4. Create or import agents in Agents tab
5. Configure MCP servers if needed

**Agent Definition Example:**

```toml
[[agents]]
name = "CodeAssistant"
description = "Specialized in code review and refactoring"
system_prompt = """You are an expert software engineer.
Focus on code quality, security, and maintainability.
Current date: {current_date}"""
tools = ["code_analysis", "file_editing", "web_search", "memory"]
temperature = 0.7
```

See `CONFIGURATION.md` for detailed configuration documentation.

## 📦 Agent Sharing & Reuse

Export and import agent configurations to share with teams or reuse across
projects.

### Export Agents

Export one or multiple agents to TOML or JSON format.

**Via Console Command:**

```bash
# Export specific agents
/export researcher,coder ~/my_agents.toml

# Export to JSON
/export analyst ~/agents/data_analyst.json
```

**Via GUI:**

- Open Settings → Agents
- Select agents to export
- Click "Export Selected" button
- Choose format and location

### Import Agents

Import agents from files or URLs with conflict resolution strategies.

**Via Console Command:**

```bash
# Import from local file
/import ~/downloaded_agents.toml

# Import from URL
/import https://example.com/agents/researcher.toml
```

**Via GUI:**

- Open Settings → Agents
- Click "Import Agents" button
- Choose file or enter URL
- Select merge strategy:
  - **Update**: Replace existing agents with same names
  - **Skip**: Keep existing agents, ignore imports with same names
  - **Add Only**: Only import agents with unique names

## Console Commands

Available in both GUI and console interfaces:

- `/clear` or `Ctrl+L` - Start new conversation
- `/copy` or `Ctrl+Shift+C` - Copy last response
- `/file <path>` - Attach file to message
- `/agent <name>` - Switch to different agent
- `/consolidate <n>` - Merge messages, preserve last n
- `/think <level>` - Enable reasoning mode (low/medium/high)
- `/agent_mode <mode>` - Switch agent interaction mode (transfer/delegate/none)
- `/voice` - Start voice recording
- `/end_voice` - Stop voice recording
- `/export <agents> <file>` - Export agent configurations
- `/import <file>` - Import agent configurations
- `exit` or `quit` - Close AgentCrew

## Development

AgentCrew is extensible by design. Add new capabilities without modifying core
code.

**Create Custom Tools:**

1. Add a module under `AgentCrew/modules/your_tool/`
2. Define the tool schema and handler in `tool.py`
3. Register the tool through the existing registration pattern
4. expose it through agent configuration

**Add AI Providers:**

- OpenAI-compatible APIs work through custom provider configuration
- Native providers should implement the `BaseLLMService` interface

**Contribute Safely:**

- Follow the existing service, tool, and config patterns
- Add tests for new behavior where practical
- Update user-facing docs when workflows change

See `CONTRIBUTING.md` for contribution workflow and development guidelines.

## Security Considerations

You control what your agents can do. Review these guidelines:

**Tool Permissions:**

- Enable only necessary tools for each agent
- Use tool approval mode to review actions before execution
- Configure file access restrictions and rate limits

**Sensitive Data:**

- Never include credentials in system prompts
- Use environment variables for API keys
- Be cautious with tools that access filesystem or network

**Audit and Monitoring:**

- Review command execution logs
- Monitor agent behavior for unexpected patterns
- Test agent configurations before production use

AgentCrew provides the framework. You are responsible for safe configuration and
usage.

## Documentation

- `README.md` - Product overview, install, usage modes, and examples
- `CONFIGURATION.md` - Detailed configuration guide
- `CONTRIBUTING.md` - Development and contribution guidelines
- `docker/DOCKER.md` - Docker-specific documentation

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=saigontechnology/AgentCrew&type=date&legend=top-left)](https://www.star-history.com/?repos=saigontechnology%2FAgentCrew&type=date&legend=top-left)

## Contributing

Contributions are welcome. Submit pull requests or open issues for bugs,
features, or improvements.

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.
