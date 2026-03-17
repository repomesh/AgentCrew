from typing import Dict, Any, Callable
from datetime import datetime as dt

from AgentCrew.modules.agents import AgentManager
from .base_service import BaseMemoryService


def get_memory_forget_tool_definition(provider="claude") -> Dict[str, Any]:
    """Optimized memory forgetting tool definition."""

    tool_description = """Removes memories using IDs from memory bank.

Use for clearing outdated information, removing sensitive data, resolving conflicting memories, or correcting errors.

Search for memories need to remote, use date filters to limit scope whenever posible, Eg: yesterday: from_date = current_date - 1 and use IDs for precise removal."""

    tool_arguments = {
        "ids": {
            "type": "array",
            "description": "Keywords describing what to forget. Use specific terms like 'project alpha 2024 credentials' or 'outdated api documentation v1'. Avoid broad terms like 'user' or 'project'.",
            "items": {"type": "string"},
        },
    }

    tool_required = ["ids"]

    if provider == "claude":
        return {
            "name": "forget_memory_topic",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "forget_memory_topic",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_memory_forget_tool_handler(memory_service: BaseMemoryService) -> Callable:
    """Optimized memory forgetting handler with concise feedback."""

    async def handle_memory_forget(**params) -> str:
        ids = params.get("ids", [])

        # Use provided agent_name or fallback to current agent
        current_agent = AgentManager.get_instance().get_current_agent()
        agent_name = current_agent.name if current_agent else "None"

        try:
            result = memory_service.forget_ids(ids, agent_name)
            return (
                f"Removed memories: {result.get('message', 'Success')}"
                if result.get("success")
                else f"Removal incomplete: {result.get('message', 'Not found')}"
            )
        except Exception as e:
            return f"Memories removal failed: {str(e)}"

    return handle_memory_forget


def get_memory_retrieve_tool_definition(provider="claude") -> Dict[str, Any]:
    """Optimized memory retrieval tool definition."""

    tool_description = """Search relevant information from conversation history using semantic search.
Use for gathering context, accessing user preferences, finding similar problems, and maintaining conversation continuity. 
Search with specific, descriptive queries for better results.
Use from_date and to_date to filter memories by time whenever posible, Eg: yesterday: from_date = current_date - 1"""

    tool_arguments = {
        "query": {
            "type": "string",
            "description": "A searching query for finding relevant memories. Use specific semantic phrases like 'project alpha database issues' or 'user preferences communication style' rather than keywords.",
        },
        "from_date": {
            "type": "string",
            "format": "date",
            "description": "Filter retrieving memories from this date (YYYY-MM-DD). Optional.",
        },
        "to_date": {
            "type": "string",
            "format": "date",
            "description": "Filter retrieving memories til this date (YYYY-MM-DD). Optional.",
        },
    }

    tool_required = ["query"]

    if provider == "claude":
        return {
            "name": "search_memory",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def memory_instruction_prompt():
    """Concise memory system instructions for system prompt."""
    return """<Memory_System>
  <Purpose>
    Extremely useful for gathering context through intelligent storage and retrieval of relevant information.
    Call search_memory when one of <Memory_Triggers> occur to provide better responses.
  </Purpose>
  <Usage_Guidelines>
    <Memory_Triggers>
      - When start a new conversation - gather relevant context from user request
      - When current topic changes - Get new topic-related memories context for better responses
      - When User references to past interactions
    </Memory_Triggers>
    <Search_Strategy>
      - Use specific, descriptive queries
      - Combine related concepts with spaces
      - Include temporal indicators when relevant
      - Include time filters when applicable
      - Balance specificity with breadth based on need
    </Search_Strategy>
    <Memory_Management>
      - Remove outdated/conflicting information when corrected
      - Clear sensitive data when requested
      - Search and Use ID-based removal for surgical precision
    </Memory_Management>
  </Usage_Guidelines>
</Memory_System>"""


def get_memory_retrieve_tool_handler(memory_service: BaseMemoryService) -> Callable:
    """Optimized memory retrieval handler with concise feedback."""

    async def handle_memory_retrieve(**params) -> str:
        query = params.get("query", "").strip()
        from_date = params.get("from_date", None)
        to_date = params.get("to_date", None)

        if not query:
            raise ValueError("Phrases required for memory search. Try again.")

        if len(query) < 3:
            raise ValueError(
                f"Search term '{query}' too short. Try again with more semantica and descriptive phrases."
            )

        # Use provided agent_name or fallback to current agent
        current_agent = AgentManager.get_instance().get_current_agent()
        agent_name = current_agent.name if current_agent else ""

        try:
            if from_date:
                from_date = int(dt.strptime(from_date, "%Y-%m-%d").timestamp())
            if to_date:
                to_date = int(dt.strptime(to_date, "%Y-%m-%d").timestamp())
            if from_date and to_date and from_date >= to_date:
                raise ValueError(
                    "from_date must be earlier than and not equal to to_date. Try again with valid dates."
                )

            result = memory_service.retrieve_memory(
                query, from_date, to_date, agent_name
            )

            if not result or result.strip() == "":
                return f"No memories found for '{query}'. Try broader phrases or related terms."

            # Count memories for user feedback
            return f"Found relevant memories:\n\n{result}"

        except Exception as e:
            return f"Memory search failed: {str(e)}"

    return handle_memory_retrieve


def get_learn_behavior_tool_definition(provider="claude") -> Dict[str, Any]:
    """Optimized adaptive behavior tool definition."""

    tool_description = """Stores behavioral patterns to personalize future interactions based on user preferences and successful approaches.

Use when you identify effective communication styles, task approaches, or user preferences that should be consistently applied.

All behaviors must follow 'when..., [action]...' format for automatic activation."""

    tool_arguments = {
        "id": {
            "type": "string",
            "description": "Unique identifier using format 'category_context' (e.g., 'communication_style_technical', 'task_execution_code_review'). Use existing ID to update behavior.",
        },
        "condition": {
            "type": "string",
            "description": "The triggering condition for the behavior. Example: 'user asks about debugging', 'working with python project', 'user mentions deadlines'.",
        },
        "action_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of action steps to execute when the condition is met. Each step should be a clear, actionable instruction. Example: ['provide step-by-step troubleshooting', 'include code examples', 'explain the root cause'].",
        },
        "scope": {
            "type": "string",
            "enum": ["global", "project"],
            "description": "Scope of the behavior. 'global' apply for all conversations, 'project' applys for current project only. Use project scope when behavior is project-specific, use global if behavior is general.",
        },
    }

    tool_required = ["id", "condition", "action_steps"]

    if provider == "claude":
        return {
            "name": "learn_behavior",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # provider == "groq"
        return {
            "type": "function",
            "function": {
                "name": "learn_behavior",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_learn_behavior_tool_handler(persistence_service: Any) -> Callable:
    """Optimized adaptive behavior handler with concise feedback."""

    async def handle_learn_behavior(**params) -> str:
        behavior_id = params.get("id", "").strip()
        condition = params.get("condition", "").strip()
        action_steps = params.get("action_steps", [])
        scope = params.get("scope", "global").strip().lower()

        if not behavior_id:
            return "Behavior ID required (e.g., 'communication_style_technical')."

        if not condition:
            return "Condition required (e.g., 'user asks about debugging')."

        if (
            not action_steps
            or not isinstance(action_steps, list)
            or len(action_steps) == 0
        ):
            return "Action steps required as a non-empty list of strings."

        action_steps = [
            step.strip()
            for step in action_steps
            if isinstance(step, str) and step.strip()
        ]
        if len(action_steps) == 0:
            return "At least one valid action step is required."

        if len(action_steps) == 1:
            behavior = f"when {condition}, do {action_steps[0]}"
        else:
            steps_joined = "; ".join(
                f"{i + 1}. {step}" for i, step in enumerate(action_steps)
            )
            behavior = f"when {condition}, do run following steps: {steps_joined}"

        current_agent = AgentManager.get_instance().get_current_agent()
        agent_name = current_agent.name if current_agent else "default"

        try:
            success = persistence_service.store_adaptive_behavior(
                agent_name, behavior_id, behavior, scope == "project"
            )
            return (
                f"Stored behavior '{behavior_id}': {behavior}"
                if success
                else "Storage completed but may need verification."
            )
        except ValueError as e:
            return f"Invalid format: {str(e)}"
        except Exception as e:
            return f"Storage failed: {str(e)}"

    return handle_learn_behavior


def adaptive_instruction_prompt():
    """Concise adaptive behavior instructions for system prompt."""
    return """<Adaptive_Behaviors>
  <Purpose>
    Learn and apply personalized interaction patterns to improve user experience over time.
  </Purpose>
  <Learn_Behavior_Triggers>
    - User expresses preferences for communication style
    - Positive feedback on specific approaches
    - Repeated requests indicating preferred workflows
    - Successful problem-solving patterns
    - Specific "when.[condition].do.[actions]." instructions from the user
    - Use `project` scope for behaviors relevant only to current project
  </Learn_Behavior_Triggers>
  <Behavior_Format>
    Use the `learn_behavior` tool with:
    - condition: The triggering condition (e.g., "user asks about debugging")
    - action_steps: Array of action steps to execute when condition is met
    
    Examples:
    - condition: "user asks about code"
      action_steps: ["provide complete examples", "include explanations"]
    - condition: "user mentions deadlines"
      action_steps: ["prioritize speed over detailed explanations", "focus on essential information"]
    - condition: "working with python project"
      action_steps: ["use uv as package manager", "follow PEP 8 style guidelines"]
  </Behavior_Format>
  <ID_Conventions>
    Use structured IDs: category_context
    • communication_style_[aspect]
    • task_execution_[domain] 
    • personalization_[area]
  </ID_Conventions>
</Adaptive_Behaviors>"""


def register(
    service_instance=None,
    persistence_service=None,
    agent=None,
):
    """Register optimized memory tools with comprehensive capabilities."""
    from AgentCrew.modules.tools.registration import register_tool

    # Register core memory management tools
    register_tool(
        get_memory_retrieve_tool_definition,
        get_memory_retrieve_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_memory_forget_tool_definition,
        get_memory_forget_tool_handler,
        service_instance,
        agent,
    )

    # Register adaptive behavior tool if persistence service is available
    if persistence_service is not None:
        register_tool(
            get_learn_behavior_tool_definition,
            get_learn_behavior_tool_handler,
            persistence_service,
            agent,
        )
