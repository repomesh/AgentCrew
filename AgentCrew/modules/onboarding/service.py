import asyncio
import os
import re
import sys
from typing import Any, Dict, Optional

import tomllib
from loguru import logger
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from AgentCrew.modules.config.agents_config import AgentsConfig

_ONBOARDING_SYSTEM_PROMPT = """
## Agent Identity
You are an expert **Meta Prompt Engineer & Agent Creator** specializing in designing and deploying production-ready AI agents.

## CRITICAL RULES - FOLLOW EXACTLY

### RULE 1: NEVER OUTPUT TOML ON FIRST RESPONSE
On your VERY FIRST response to the user, you MUST NOT generate any TOML, code blocks, or agent definitions.
Your first response MUST ONLY contain clarifying questions.

### RULE 2: ALWAYS ASK QUESTIONS FIRST
Before creating any agent, you MUST ask the user 1-3 short clarifying questions.
Only after the user answers may you proceed to generate the agent.

### RULE 3: NO PARTIAL OR PLACEHOLDER OUTPUT
Never output incomplete TOML, template placeholders like [AgentName], or "example" agent definitions.
Either ask questions OR output a complete, final agent. Nothing in between.

## Information Gathering Questions

Based on the user's brief name and description, ask about ANY of these that would help:
- **Domain specificity**: What programming language, industry, or tech stack?
- **Task complexity**: Simple Q&A, multi-step analysis, or tool-heavy workflows?
- **Output style**: Code, prose summaries, structured data, creative writing?
- **Tool needs**: Should it browse, code-analyze, edit files, run commands?
- **Constraints**: Any style guides, compliance, or guardrails?
- **Users**: Personal use, team, or client-facing?

Keep questions SHORT (1 sentence each), numbered, and friendly.

## Agent Structure Template (for Phase 2 only)

When generating the final agent, the system_prompt must use this XML structure:

```xml
<Agent_Instructions>
  <Identity>
    [Role definition]
    [Core competencies]
  </Identity>
  
  <Inputs>
    {$VARIABLE_1}  // Input type definitions
    {$VARIABLE_2}
  </Inputs>
  
  <Task_Patterns>
    <Pattern name="[name]">
      <Trigger>[When applies]</Trigger>
      <Structure>[Workflow template]</Structure>
      <Output>[Expected format]</Output>
    </Pattern>
  </Task_Patterns>
  
  <Reasoning_Framework>
    [Decision matrix]
    [Complexity guidelines]
  </Reasoning_Framework>
  
  <Output_Requirements>
    <Quality_Standards>
      - Concise: Each sentence serves one purpose
      - Dense: Maximum information, minimum words
      - Non-redundant: State once, reference as needed
      - Structured: Scannable sections, consistent formatting
    </Quality_Standards>
  </Output_Requirements>
  
  <Constraints>
    [Boundaries and limitations]
    [Error handling]
  </Constraints>
</Agent_Instructions>
```


### Technique Selection Matrix

```yaml
Simple_Tasks:
  Characteristics: [Well-defined, single-step, deterministic]
  Primary_Technique: Zero-shot with clear instructions
  Token_Budget: <500 tokens

Moderate_Complexity:
  Characteristics: [Multi-step, structured output, domain-specific]
  Primary_Technique: Few-shot (2-3 examples) + Meta patterns
  Token_Budget: 500-1500 tokens

Complex_Reasoning:
  Characteristics: [Multi-step logic, decision trees, tool orchestration]
  Primary_Technique: CoT + ReAct framework
  Token_Budget: 1500-3000 tokens

Expert_Systems:
  Characteristics: [Domain expertise, adaptive behavior, memory usage]
  Primary_Technique: Meta prompting + Reflexion + Behavioral adaptation
  Token_Budget: 3000+ tokens
```

### Technique Reference Guide

ALWAYS fetch the technique url to deeply understand the technique

1. Zero-Shot Prompting
- **What it is**: Directly instruct the model to perform a task without any examples or demonstrations. The model relies entirely on its pre-trained knowledge and instruction-following capabilities.
- **When to use**: Well-defined tasks with clear success criteria; tasks the model is likely already trained on; when token budget is constrained; for straightforward classification, extraction, or transformation tasks.
- **How to implement**: Provide a clear role definition, precise task specification, and explicit output format. Use strong action verbs and unambiguous language.
- **Optimization**: Include role definition, task specification, output format. If zero-shot fails, escalate to few-shot prompting rather than adding more instructions.

2. Few-Shot Prompting
- **What it is**: Provide demonstrations (examples) in the prompt to steer the model toward better performance through in-context learning. The demonstrations serve as conditioning for subsequent responses.
- **When to use**: Format consistency is critical; domain-specific tasks; example-driven learning scenarios; when zero-shot produces inconsistent results; tasks requiring specific output patterns or styles.
- **How to implement**: Include 2-5 diverse, representative examples that cover edge cases. Maintain consistent formatting across all examples. Ensure the label space and input distribution match the target task.
- **Optimization**: Diverse, representative examples; consistent formatting. The format matters more than label accuracy — even random labels with correct format outperform no examples. Increase shot count for more difficult tasks.

3. Chain-of-Thought (CoT)
- **What it is**: Enable complex reasoning by prompting the model to generate intermediate reasoning steps before arriving at the final answer. Combines naturally with few-shot prompting.
- **When to use**: Multi-step arithmetic or mathematical problems; commonsense reasoning; symbolic logic; logical deduction tasks; any problem where the path to the answer matters more than the answer alone.
- **How to implement**: For zero-shot CoT, append "Let's think step by step." For few-shot CoT, provide exemplars that show explicit reasoning traces leading to the answer. Break the problem into explicit intermediate steps.
- **Optimization**: "Let's think step by step" for zero-shot; explicit reasoning in examples for few-shot. CoT is an emergent ability — it works best with sufficiently capable models. Combine with self-consistency for maximum accuracy.

4. Meta Prompting
- **What it is**: Focus on the structural and syntactical aspects of tasks rather than specific content details. Uses abstract, reusable prompt structures that emphasize form and pattern over content.
- **When to use**: Token efficiency is critical; complex instructions that would otherwise be too long; bias reduction through abstraction; theoretical or mathematical problem-solving; coding challenges; when building reusable prompt templates.
- **How to implement**: Define the structure, syntax, and pattern of the expected response rather than giving content-specific examples. Use type-theory-inspired categorization and logical arrangement of components.
- **Optimization**: Abstract, reusable prompt structures; clear format definitions. Meta prompting is structure-oriented, not content-driven — provide the "shape" of the solution, not specific instances. Particularly effective for XML or schema-based outputs.

5. Self-Consistency
- **What it is**: Sample multiple diverse reasoning paths through CoT prompting, then select the most consistent answer across all paths. Replaces naive greedy decoding with majority voting.
- **When to use**: High-accuracy requirements; ambiguous problems where a single reasoning path might fail; arithmetic reasoning; commonsense reasoning; any task where multiple valid approaches exist and consensus improves reliability.
- **How to implement**: Generate the same CoT prompt multiple times (e.g., 3-10 samples) with temperature > 0 to encourage diversity. Extract the final answer from each path and select the answer that appears most frequently.
- **Optimization**: Multiple reasoning paths; majority voting on solutions. Combine with CoT for best results. More samples increase accuracy but increase cost — 3-5 samples often provides a good balance.

6. Prompt Chaining
- **What it is**: Break complex tasks into subtasks, then chain prompts where the output of one prompt becomes the input to the next. Each link performs a transformation or additional process.
- **When to use**: Complex projects that a single detailed prompt cannot handle reliably; multi-stage workflows; document question-answering; data processing pipelines; tasks requiring intermediate validation or cleanup.
- **How to implement**: Identify natural subtask boundaries. Design each prompt to accept the previous output as input. Include validation steps between chains. Make each prompt single-purpose and focused.
- **Optimization**: Clear handoff protocols; intermediate result validation. Example chain: (1) Extract relevant quotes → (2) Synthesize answer from quotes → (3) Format and validate final response. Debug each link independently.

7. Tree of Thought (ToT)
- **What it is**: Maintain a tree of intermediate reasoning paths (thoughts) and explore them systematically with search algorithms (BFS/DFS). Generalizes CoT by enabling exploration, lookahead, and backtracking.
- **When to use**: Creative problem-solving with multiple solution paths; strategic planning; game-playing; mathematical exploration; tasks requiring deliberate reasoning and evaluation of alternatives; problems where early wrong steps derail the entire answer.
- **How to implement**: Decompose the problem into discrete thought steps. At each step, generate multiple candidate continuations. Evaluate each candidate (e.g., sure/maybe/impossible). Prune poor candidates and continue exploring promising branches.
- **Optimization**: Structured exploration paths; evaluation criteria for branches. Define the number of candidates per step and the total number of steps based on task complexity. Use self-evaluation to prune branches early. For simple implementations, use the "panel of experts" approach.

8. ReAct (Reasoning + Acting)
- **What it is**: Interleave reasoning traces (Thought) with task-specific actions (Act) and observations (Observation). The model reasons about what to do, performs an action, observes the result, then reasons again.
- **When to use**: Tool usage and orchestration; research tasks requiring information retrieval; systematic investigation; tasks where the model must interact with external environments (databases, APIs, search engines); multi-hop question answering.
- **How to implement**: Structure the prompt with explicit Thought/Action/Observation steps. Define available actions and their expected formats. After each action, provide the observation from the environment. Continue the loop until the task is complete.
- **Optimization**: Clear tool descriptions; action-observation loops. Best results come from combining ReAct with CoT (internal knowledge + external information). Ensure actions have well-defined inputs and outputs. Handle failed or uninformative observations gracefully.

9. Reflexion
- **What it is**: A framework for reinforcing agents through linguistic feedback. Converts environmental feedback into self-reflection, which is stored in memory and provided as context for future trials.
- **When to use**: Learning from errors over multiple attempts; iterative improvement; sequential decision-making; programming tasks; reasoning tasks where trial-and-error learning is beneficial; when explicit memory of past mistakes improves performance.
- **How to implement**: Define the Actor (generates actions, can use CoT or ReAct), Evaluator (scores outputs with reward signals), and Self-Reflection (generates verbal reinforcement cues). Store reflections in episodic memory. On each new trial, provide relevant past reflections as context.
- **Optimization**: Explicit error analysis; improvement strategies. Start with a simple sliding-window memory of recent reflections. For complex tasks, consider vector-based retrieval of relevant past reflections. Combine with ReAct for state-of-the-art agent performance.

## TOML Output Format (Phase 2 ONLY)

When ready to generate after gathering answers, output ONLY:


```toml
[[agents]]
name = "ExactAgentName"
description = "Clear description"
system_prompt = '''
[Full XML system prompt]
'''
tools = ["memory", "code_analysis"]
temperature = 0.4
```

TOML block must always start with [[agents]].
Agent object must contains these mandatory fields: `name`, `description`, `tools`, `system_prompt`, `temperature`,
tools fields is an array that can be any value from ['memory', 'clipboard', 'code_analysis', 'web_search', 'browser', 'file_editing', 'command_execution']

"""


_ONBOARDING_STYLE = Style.from_dict(
    {
        "frame.border": "#3b82f6",
        "selected-option": "bold",
        "prompt": "#60a5fa bold",
        "hint": "#9ca3af",
        "title": "#f59e0b bold",
        "success": "#22c55e",
        "error": "#ef4444",
    }
)


class OnboardingService:
    """Guides new users through creating their first agent interactively."""

    def __init__(self, llm_service: Any, agents_config: Optional[AgentsConfig] = None):
        self.llm_service = llm_service
        self.agents_config = agents_config or AgentsConfig()
        self.console = Console()
        if self.llm_service:
            if self.llm_service.provider_name == "google":
                self.llm_service.model = "gemini-3.1-pro-preview"
            elif self.llm_service.provider_name == "claude":
                self.llm_service.model = "claude-sonnet-4-6"
            elif self.llm_service.provider_name == "openai":
                self.llm_service.model = "gpt-5.4"
            elif self.llm_service.provider_name == "deepinfra":
                self.llm_service.model = "zai-org/GLM-5.1"
            elif self.llm_service.provider_name == "github_copilot":
                self.llm_service.model = "claude-sonnet-4.6"
            elif self.llm_service.provider_name == "copilot_response":
                self.llm_service.model = "gpt-5.4"
            elif self.llm_service.provider_name == "openai_codex":
                self.llm_service.model = "gpt-5.4"
            elif self.llm_service.provider_name == "together":
                self.llm_service.model = "zai-org/GLM-5.1"
            elif self.llm_service.provider_name == "opencode_go":
                self.llm_service.model = "kimi-k2.6"

    def should_run(self, config_uri: Optional[str] = None) -> bool:
        """Check whether onboarding should run (no agents in config)."""
        config_path = config_uri or os.getenv("SW_AGENTS_CONFIG", "./agents.toml")
        config_path = os.path.expanduser(config_path)

        if not os.path.exists(config_path):
            return True

        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            agents = data.get("agents", [])
            remote_agents = data.get("remote_agents", [])
            enabled_local = [a for a in agents if a.get("enabled", True)]
            enabled_remote = [a for a in remote_agents if a.get("enabled", True)]
            return len(enabled_local) == 0 and len(enabled_remote) == 0
        except Exception:
            return True

    def run(self) -> bool:
        """Run the interactive onboarding flow. Returns True if an agent was created."""
        if not sys.stdin.isatty():
            logger.debug("Non-interactive environment detected; skipping onboarding.")
            return False

        self._print_header()

        if not self._confirm(
            "Would you like to create a custom agent now?", default=True
        ):
            self._print_skip()
            return False

        return self.create_agent()

    def create_agent(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> bool:
        """Create an agent interactively, skipping the welcome header and confirmation prompt.

        Args:
            name: Pre-filled agent name. If omitted, the user is prompted interactively.
            description: Pre-filled agent description. If omitted, the user is prompted interactively.

        Returns:
            True if an agent was created and saved successfully.
        """
        if not sys.stdin.isatty():
            logger.debug(
                "Non-interactive environment detected; skipping agent creation."
            )
            return False

        agent_name = name
        if not agent_name:
            agent_name = self._ask_text(
                "What would you like to name your agent?",
                hint="e.g., Engineer, Researcher, CodeAssistant",
            )
        agent_name = (agent_name or "").strip()
        if not agent_name:
            self._print_error("Agent name cannot be empty.")
            return False

        agent_description = description
        if not agent_description:
            agent_description = self._ask_text(
                "What should this agent do?",
                hint="e.g., Help me write Python code, Analyze financial data",
            )
        agent_description = (agent_description or "").strip()
        if not agent_description:
            self._print_error("Agent description cannot be empty.")
            return False

        self._print_status("Creating your agent '" + agent_name + "'...")

        try:
            toml_definition = asyncio.run(
                self._run_onboarding_chat(agent_name, agent_description)
            )
        except Exception as e:
            logger.warning("Onboarding LLM call failed: " + str(e))
            toml_definition = None

        if not toml_definition:
            self._print_error(
                "Failed to generate agent definition. You can create one manually in agents.toml."
            )
            return False

        if not self._save_agent(toml_definition):
            return False

        self._print_success("Agent '" + agent_name + "' has been saved to agents.toml!")
        return True

    def _print_header(self) -> None:
        self.console.print("")
        self.console.print(
            Panel(
                Text("Welcome to AgentCrew!", style="bold yellow", justify="center")
                + Text(
                    "\nLet's create your first personalized agent.",
                    style="dim",
                    justify="center",
                ),
                border_style="blue",
                padding=(1, 4),
            )
        )
        self.console.print("")

    def _print_skip(self) -> None:
        self.console.print(
            Panel(
                "Skipping onboarding. You can create agents later via the /agent command or by editing agents.toml.",
                border_style="grey58",
                padding=(0, 2),
            )
        )

    def _print_error(self, message: str) -> None:
        self.console.print(Panel(message, border_style="red", padding=(0, 2)))

    def _print_success(self, message: str) -> None:
        self.console.print(Panel(message, border_style="green", padding=(0, 2)))

    def _print_status(self, message: str) -> None:
        self.console.print("")
        self.console.print("  " + message, style="bold blue")
        self.console.print("")

    @staticmethod
    def _kb_skip() -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.ControlC)
        @kb.add("escape")
        def _(event):
            event.app.exit(result="")

        return kb

    def _confirm(self, message: str, default: bool = True) -> bool:
        kb = self._kb_skip()
        result = choice(
            message=HTML(f"<ansiyellow>{message}</ansiyellow> "),
            options=[(True, "Yes"), (False, "No")],
            default=default,
            style=_ONBOARDING_STYLE,
            key_bindings=kb,
            show_frame=True,
        )
        return bool(result)

    def _ask_text(self, message: str, hint: str = "") -> str:
        kb = self._kb_skip()
        if hint:
            prompt_msg = HTML(
                f"<ansiblue>{message}</ansiblue>\n<ansigrey>{hint}</ansigrey>\n"
            )
        else:
            prompt_msg = HTML(f"<ansiblue>{message}</ansiblue>")
        result = prompt(
            prompt_msg,
            key_bindings=kb,
            style=_ONBOARDING_STYLE,
        )
        return result.strip() if result else ""

    async def _run_onboarding_chat(
        self, agent_name: str, agent_description: str
    ) -> Optional[str]:
        """Run a multi-turn conversation with the LLM to gather info and generate an agent."""
        try:
            initial_message = (
                "I want to create a personalized agent.\n\n"
                "Proposed Name: " + agent_name + "\n"
                "Brief Description: " + agent_description + "\n\n"
                "If information user provided too vagued, DO NOT generate any TOML, code blocks, or agent definitions yet. "
                "Your ONLY job right now is to ask me 1-3 short clarifying questions "
                "so you can build the best possible agent."
            )

            original_system_prompt = getattr(self.llm_service, "system_prompt", None)
            self.llm_service.set_system_prompt(_ONBOARDING_SYSTEM_PROMPT)

            try:
                conversation: list = []
                max_turns = 5

                for _ in range(max_turns):
                    prompt = self._build_chat_prompt(initial_message, conversation)
                    response = await self.llm_service.process_message(
                        prompt, temperature=1.0
                    )

                    result_text = (
                        response[0] if isinstance(response, tuple) else response
                    )
                    if not isinstance(result_text, str):
                        return None

                    extracted = OnboardingService._extract_toml(result_text)
                    if extracted:
                        try:
                            parsed = tomllib.loads(extracted)
                            if OnboardingService._looks_like_agent_definition(parsed):
                                return result_text
                        except Exception:
                            pass

                    self._print_assistant_message(result_text)

                    user_answer = self._ask_onboarding_input()
                    if user_answer is None:
                        return None

                    conversation.append({"assistant": result_text, "user": user_answer})

                self._print_error(
                    "Reached maximum conversation turns without getting an agent definition."
                )
                return None

            finally:
                if original_system_prompt is not None:
                    self.llm_service.set_system_prompt(original_system_prompt)
                else:
                    self.llm_service.set_system_prompt("")

        except Exception as e:
            logger.warning("LLM agent generation failed: " + str(e))
            return None

    def _print_assistant_message(self, text: str) -> None:
        self.console.print("")
        self.console.print(
            Panel(
                Markdown(text),
                title="[bold cyan]Agent Creator[/bold cyan]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self.console.print("")

    def _ask_onboarding_input(self) -> Optional[str]:
        kb = self._kb_skip()
        result = prompt(
            HTML(
                "<ansigreen>Your answer</ansigreen> <ansigrey>(type 'skip' to cancel, Alt+Enter to submit)</ansigrey>\n"
            ),
            key_bindings=kb,
            multiline=True,
            style=_ONBOARDING_STYLE,
        )
        if result is None:
            return None
        result = result.strip()
        if result.lower() == "skip":
            return None
        self._print_status("Keep processing with your answers...")
        return result

    @staticmethod
    def _build_chat_prompt(initial_message: str, conversation: list) -> str:
        """Build a single prompt string from conversation history."""
        parts = [initial_message]
        for turn in conversation:
            parts.append("\n\nAssistant: " + turn["assistant"])
            parts.append("\n\nUser: " + turn["user"])
        return "".join(parts)

    @staticmethod
    def _extract_toml(text: str) -> Optional[str]:
        """Extract TOML content from a ```toml markdown block or raw [[agents]] text."""
        text = text.strip()
        pattern = r"```toml\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if text.startswith("[[agents]]"):
            return text
        return None

    @staticmethod
    def _looks_like_agent_definition(parsed: Dict[str, Any]) -> bool:
        """Check whether parsed TOML dict contains an agent definition."""
        agents = parsed.get("agents")
        if isinstance(agents, list) and len(agents) > 0:
            return bool(agents[0].get("name") and agents[0].get("system_prompt"))
        if isinstance(agents, list) and len(agents) == 0:
            return False
        if parsed.get("name") and parsed.get("system_prompt"):
            return True
        return False

    def _save_agent(self, toml_text: str) -> bool:
        """Parse and save the generated agent definition."""
        toml_content = OnboardingService._extract_toml(toml_text)
        if not toml_content:
            self._print_error("Could not parse agent definition from LLM response.")
            logger.debug("Raw LLM response:\n" + toml_text)
            return False

        try:
            parsed = tomllib.loads(toml_content)
        except Exception as e:
            self._print_error("Generated agent definition is not valid TOML: " + str(e))
            logger.debug("TOML content:\n" + toml_content)
            return False

        agents = parsed.get("agents")
        if isinstance(agents, list) and len(agents) > 0:
            agent_def = agents[0]
        elif parsed.get("name") and parsed.get("system_prompt"):
            agent_def = {
                "name": parsed.get("name"),
                "description": parsed.get("description", ""),
                "system_prompt": parsed.get("system_prompt", ""),
                "tools": parsed.get("tools", []),
            }
            if parsed.get("temperature") is not None:
                agent_def["temperature"] = parsed.get("temperature")
            if parsed.get("voice_enabled") is not None:
                agent_def["voice_enabled"] = parsed.get("voice_enabled")
            if parsed.get("voice_id") is not None:
                agent_def["voice_id"] = parsed.get("voice_id")
        else:
            self._print_error("Generated TOML does not contain an [[agents]] entry.")
            return False

        if not agent_def.get("name") or not agent_def.get("system_prompt"):
            self._print_error(
                "Generated agent is missing required fields (name or system_prompt)."
            )
            return False

        try:
            self._write_agent_directly(agent_def)
            return True
        except Exception as e:
            self._print_error("Failed to save agent to agents.toml: " + str(e))
            logger.error("Save agent error: " + str(e))
            return False

    def _write_agent_directly(self, agent_def: Dict[str, Any]) -> None:
        """Write agent directly to agents.toml, bypassing reload to avoid chicken-and-egg issues."""
        import tomli_w

        config_path = os.path.expanduser(os.getenv("SW_AGENTS_CONFIG", "./agents.toml"))
        config_dir = os.path.dirname(config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        existing: Dict[str, Any] = {"agents": []}
        if os.path.exists(config_path):
            try:
                with open(config_path, "rb") as f:
                    existing = tomllib.load(f)
            except Exception:
                pass

        existing_agents = existing.get("agents", [])
        if not isinstance(existing_agents, list):
            existing_agents = []

        existing_agents.append(agent_def)
        existing["agents"] = existing_agents

        with open(config_path, "wb") as f:
            tomli_w.dump(existing, f, multiline_strings=True)
