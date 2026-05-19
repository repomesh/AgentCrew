from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from loguru import logger

if TYPE_CHECKING:
    from .local_agent import LocalAgent

SHRINK_LENGTH_THRESHOLD = 10


class AgentContextManager:
    """
    Manages context preparation and token budget for a LocalAgent.

    Owns:
    - Injecting system prompt and adaptive context into message lists
    - Pruning tool results when approaching the model token limit
    """

    def __init__(self, agent: "LocalAgent") -> None:
        self._agent = agent

    def build_adaptive_context(self) -> dict[str, Any]:
        """Build the adaptive behavior dict (cwd, git branch, open files, etc.)."""
        from AgentCrew.modules.memory.context_persistent import (
            ContextPersistenceService,
        )

        agent = self._agent
        adaptive_messages: dict[str, Any] = {
            "role": "user",
            "content": [],
        }

        if "context_persistent" not in agent.services or not isinstance(
            agent.services["context_persistent"], ContextPersistenceService
        ):
            return adaptive_messages

        if (
            agent.services.get("agent_manager")
            and agent.services["agent_manager"].one_turn_process
        ):
            adaptive_messages["content"].append(
                {
                    "type": "text",
                    "text": """My next request is single-turn conversation.
You must analyze then execute it with your available tools and give answer without asking for confirmation or clarification.""",
                }
            )

        adaptive_text = []
        adaptive_behaviors = agent.services[
            "context_persistent"
        ].get_adaptive_behaviors(agent.name)

        if len(adaptive_behaviors.keys()) > 0:
            adaptive_text.extend(
                [
                    f"<Global_Behavior id='{key}'>{value}</Global_Behavior>"
                    for key, value in adaptive_behaviors.items()
                ]
            )

        local_adaptive_behaviors = agent.services[
            "context_persistent"
        ].get_adaptive_behaviors(agent.name, is_local=True)
        if len(local_adaptive_behaviors.keys()) > 0:
            adaptive_text.extend(
                [
                    f"<Project_Behavior id='{key}'>{value}</Project_Behavior>"
                    for key, value in local_adaptive_behaviors.items()
                ]
            )

        adaptive_text.extend(
            [
                "<Global_Behavior id='transfer'>When working on my request, consider whether if any other agents is more suitable, if yes, transfer to that agent.</Global_Behavior>",
                "<Global_Behavior id='good-to-say-no'>When encountering tasks that you have no data in the context and you don't know the anwser, say I don't know and ask user for helping you find the solution.</Global_Behavior>",
            ]
        )

        if len(adaptive_text) > 0:
            adaptive_messages["content"].append(
                {
                    "type": "text",
                    "text": f"""---
Apply matching behaviors from <Adaptive_Behaviors> immediately, overriding default instructions.
<Project_Behavior> has higher priority than <Global_Behavior>.
<Adaptive_Behaviors>
{"  \n".join(adaptive_text)}
</Adaptive_Behaviors>""",
                }
            )

        skills_service = agent.services.get("skills")
        if skills_service and skills_service.has_skills():
            catalog = skills_service.get_catalog()
            skills_xml = "\n".join(
                f"  <skill><name>{s['name']}</name><description>{s['description']}</description></skill>"
                for s in catalog
            )
            adaptive_messages["content"].append(
                {
                    "type": "text",
                    "text": (
                        f"<available_skills>\n{skills_xml}\n</available_skills>\n"
                        "When a task matches a skill's description, call the "
                        "`activate_skill` tool with the skill's name to load its "
                        "full instructions before proceeding."
                    ),
                }
            )

        return adaptive_messages

    def _get_directory_structure(self) -> str:
        try:
            cwd = os.getcwd()
            entries = []
            for entry in sorted(os.listdir(cwd)):
                full_path = os.path.join(cwd, entry)
                if os.path.isdir(full_path):
                    entries.append(f"{entry}/")
                else:
                    entries.append(entry)
            return "\n".join(entries) if entries else ""
        except Exception as e:
            logger.warning(f"Failed to get directory structure: {e}")
            return ""

    def enhance_messages(self, final_messages: list[dict[str, Any]]) -> None:
        """Inject system prompt and adaptive context into the message list in place."""
        agent = self._agent

        last_user_index = next(
            (
                i
                for i, msg in enumerate(reversed(final_messages))
                if msg.get("role") == "user"
            ),
            None,
        )
        if last_user_index is None:
            return
        last_user_index = len(final_messages) - 1 - last_user_index

        adaptive_messages = self.build_adaptive_context()

        if last_user_index == 0:
            dir_structure = self._get_directory_structure()
            if dir_structure:
                adaptive_messages["content"].append(
                    {
                        "type": "text",
                        "text": f"current directory `{os.getcwd()}` has structure:\n{dir_structure}",
                    }
                )

        if len(final_messages[last_user_index].get("content", [])) > 0 and (
            final_messages[last_user_index]["content"][0]
            .get("text", "")
            .find("<Transfer_Tool>")
            != 0
            and final_messages[last_user_index]["content"][0]
            .get("text", "")
            .find("<Transfer_Post_Action_Reminder>")
            != 0
        ):
            if agent.services.get("memory"):
                memory_headers = agent.services["memory"].list_memory_headers(
                    agent_name=agent.name
                )
                if memory_headers:
                    adaptive_messages["content"].append(
                        {
                            "type": "text",
                            "text": f"Our last recent conversations:\n- {'\n - '.join(memory_headers)}\n---\n If this is a new or different topic from our current conversation, call search_memory before responding.\n --- End of current context ---\n --- Start user request ---",
                        }
                    )

            if (
                agent.services.get("agent_manager")
                and agent.services["agent_manager"].agent_mode != "none"
            ):
                from AgentCrew.modules.agents.manager import AgentMode

                if agent._colaboration_mode == AgentMode.TRANSFER:
                    eval_text = """Before processing my request, quickly evaluate inside <agent_evaluation> tags:
- Plan out the tool call strategy for this request: which tools to call, in what order, and what inputs each needs.
- Is another agent better suited? If yes, transfer immediately.
Then execute your plan.
Skip evaluation for: simple one-sentence answers, or when the request matches "when [condition], [action]" — call `learn_behavior` directly instead."""
                elif agent._colaboration_mode == AgentMode.DELEGATE:
                    eval_text = """Before processing my request, quickly evaluate inside <agent_evaluation> tags:
- Plan out the tool call strategy for this request: which tools to call, in what order, and what inputs each needs.
- Can any sub-tasks be delegated to specialist agents? If yes, delegate them.
- Can multiple sub-tasks run in parallel? If yes, emit multiple delegate calls in one turn.
Then execute your plan.
Skip evaluation for: simple one-sentence answers, or when the request matches "when [condition], [action]" — call `learn_behavior` directly instead."""
                else:
                    eval_text = None

                if eval_text:
                    adaptive_messages["content"].insert(
                        0,
                        {
                            "type": "text",
                            "text": eval_text,
                        },
                    )

        if len(adaptive_messages["content"]) > 0:
            final_messages.insert(last_user_index, adaptive_messages)

    def shrink_tool_results(self, final_messages: list[dict[str, Any]]) -> None:
        """Prune large tool results when token usage exceeds 85% of model context."""
        from AgentCrew.modules.llm.model_registry import ModelRegistry

        agent = self._agent
        shrink_context_threshold = int(
            os.getenv(
                "AGENTCREW_DEFAULT_MAX_CONTEXT",
                ModelRegistry.get_model_limit(agent.get_model()) * 0.85,
            )
        )

        unique_tool_indices = []
        agent_manager = agent.services.get("agent_manager", None)

        is_shrinkable = (
            agent_manager.context_shrink_enabled if agent_manager else False
        ) and agent.token_usage.total_input_tokens > shrink_context_threshold
        shrink_threshold = len(final_messages) - int(
            os.getenv("AGENTCREW_CONTEXT_SHRINK_THRESHOLD", SHRINK_LENGTH_THRESHOLD)
        )
        shrink_excluded = set(
            agent_manager.shrink_excluded_list if agent_manager else []
        )
        shrink_excluded.add("activate_skill")
        shrink_excluded.add("search_memory")
        last_agent_tool_calls = -1
        tool_result_needed_rearrange: dict[int, list[int]] = {}
        tool_result_id_needed_rearrange: list[str] = []
        tool_result_id_needed_shrink: list[str] = []

        for i, msg in enumerate(final_messages):
            content = None

            if msg.get("role") == "assistant":
                if len(msg.get("tool_calls", [])) == 0:
                    continue

                if is_shrinkable and i < shrink_threshold:
                    last_agent_tool_calls = i

                    for tool_call in msg.get("tool_calls", []):
                        if tool_call.get("id", None):
                            if tool_call.get("name") in shrink_excluded:
                                tool_result_id_needed_rearrange.append(
                                    tool_call.get("id")
                                )
                                continue
                            tool_result_id_needed_shrink.append(tool_call.get("id"))
                    msg["tool_calls"] = [
                        t
                        for t in msg.get("tool_calls", [])
                        if t.get("name", "") in shrink_excluded
                    ]
                    if len(msg["tool_calls"]) == 0:
                        msg.pop("tool_calls", None)

            elif msg.get("role") == "tool":
                tool_name = msg.get("tool_name", "")

                content = msg.get("content", "")
                if (
                    content
                    and isinstance(content, str)
                    and content.startswith("[UNIQUE]")
                ):
                    unique_tool_indices.append(i)
                elif content and isinstance(content, list):
                    if (
                        len(
                            [
                                d.get("text", "")
                                for d in content
                                if isinstance(d, dict)
                                and d.get("text", "").startswith("[UNIQUE]")
                            ]
                        )
                        > 0
                    ):
                        unique_tool_indices.append(i)
                        continue

                if msg.get("tool_call_id", None) in tool_result_id_needed_rearrange:
                    if tool_result_needed_rearrange.get(last_agent_tool_calls, None):
                        tool_result_needed_rearrange[last_agent_tool_calls].append(i)
                    else:
                        tool_result_needed_rearrange[last_agent_tool_calls] = [i]
                    continue

                if msg.get("tool_call_id", None) in tool_result_id_needed_shrink:
                    msg["content"] = [
                        {
                            "text": f"[{msg.get('agent', 'Agent')} has used function_call `{tool_name}` but it has been truncated.]",
                            "type": "text",
                        }
                    ]
                    msg.pop("tool_name", None)
                    msg.pop("is_rejected", None)
                    msg["role"] = "user"
                    continue

        if len(unique_tool_indices) > 1:
            for i in unique_tool_indices[:-1]:
                msg = final_messages[i]

                if msg.get("role") == "tool" and "content" in msg:
                    msg["content"] = "[INVALIDATED]"

                elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for content_item in msg["content"]:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "tool_result"
                            and "content" in content_item
                        ):
                            content_item["content"] = "[INVALIDATED]"
                            break

        for assistant_idx, tool_result_indices in sorted(
            tool_result_needed_rearrange.items(), reverse=True
        ):
            if assistant_idx < 0 or not tool_result_indices:
                continue

            tool_results = [final_messages[idx] for idx in tool_result_indices]

            for idx in sorted(tool_result_indices, reverse=True):
                final_messages.pop(idx)

            for offset, tool_result in enumerate(tool_results):
                final_messages.insert(assistant_idx + 1 + offset, tool_result)
