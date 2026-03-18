from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional
from loguru import logger
from AgentCrew.modules.tools.utils import extract_tool_name

if TYPE_CHECKING:
    from .local_agent import LocalAgent


class AgentToolRegistrar:
    """
    Manages tool registration for a LocalAgent.

    Owns:
    - Adding tools to the agent's definition map
    - Syncing definitions to the LLM service
    - Clearing tools from the LLM service
    """

    def __init__(self, agent: "LocalAgent") -> None:
        self._agent = agent

    def register_tools(self) -> None:
        """
        Register all tools for the agent based on its services and tool list.
        """
        agent = self._agent

        from AgentCrew.modules.agents.tools.ask import (
            register as register_ask,
            ask_tool_prompt,
        )

        register_ask(agent)
        agent.tool_prompts.append(ask_tool_prompt())

        skills_service = agent.services.get("skills")
        if skills_service and skills_service.has_skills():
            from AgentCrew.modules.skills.tool import register as register_skills

            register_skills(skills_service, agent)

        if agent.services.get("agent_manager") and not agent.is_remoting_mode:
            from AgentCrew.modules.agents.manager import AgentMode

            mode = agent.services["agent_manager"].agent_mode
            if mode == AgentMode.TRANSFER:
                from AgentCrew.modules.agents.tools.transfer import (
                    register as register_transfer,
                    transfer_tool_prompt,
                )

                agent.tool_prompts.append(
                    agent.services["agent_manager"].get_agents_list_prompt()
                )
                register_transfer(agent.services["agent_manager"], agent)
                agent.tool_prompts.append(
                    transfer_tool_prompt(agent.services["agent_manager"])
                )
            elif mode == AgentMode.DELEGATE:
                from AgentCrew.modules.agents.tools.delegate import (
                    register as register_delegate,
                    delegate_tool_prompt,
                )

                agent.tool_prompts.append(
                    agent.services["agent_manager"].get_agents_list_prompt()
                )
                register_delegate(agent.services["agent_manager"], agent)
                agent.tool_prompts.append(
                    delegate_tool_prompt(agent.services["agent_manager"])
                )

        for tool_name in agent.tools:
            if agent.services and tool_name in agent.services:
                service = agent.services[tool_name]
                if service:
                    if tool_name == "memory" and not agent.is_remoting_mode:
                        from AgentCrew.modules.memory.tool import (
                            register as register_memory,
                            adaptive_instruction_prompt,
                            memory_instruction_prompt,
                        )

                        register_memory(
                            service,
                            agent.services.get("context_persistent", None),
                            agent,
                        )
                        agent.tool_prompts.append(memory_instruction_prompt())
                        agent.tool_prompts.append(adaptive_instruction_prompt())
                    elif tool_name == "clipboard":
                        from AgentCrew.modules.clipboard.tool import (
                            register as register_clipboard,
                        )

                        register_clipboard(service, agent)
                    elif tool_name == "code_analysis":
                        from AgentCrew.modules.code_analysis.tool import (
                            register as register_code_analysis,
                        )

                        register_code_analysis(service, agent)
                    elif tool_name == "web_search":
                        from AgentCrew.modules.web_search.tool import (
                            register as register_web_search,
                        )

                        register_web_search(service, agent)
                    elif tool_name == "image_generation":
                        from AgentCrew.modules.image_generation.tool import (
                            register as register_image_generation,
                        )

                        register_image_generation(service, agent)
                    elif tool_name == "browser":
                        from AgentCrew.modules.browser_automation.tool import (
                            register as register_browser,
                        )

                        register_browser(service, agent)
                    elif tool_name == "file_editing":
                        from AgentCrew.modules.file_editing.tool import (
                            register as register_file_editing,
                        )

                        register_file_editing(service, agent)
                    elif tool_name == "command_execution":
                        from AgentCrew.modules.command_execution.tool import (
                            register as register_command_execution,
                        )

                        register_command_execution(service, agent)
                    else:
                        logger.warning(f"⚠️ Tool {tool_name} not found in services")
            else:
                logger.warning(
                    f"⚠️ Service {tool_name} not available for tool registration"
                )

    def register_tool(
        self,
        definition_func: Callable,
        handler_factory: Callable,
        service_instance: Optional[Any] = None,
    ) -> None:
        """Add a single tool definition to the agent's local map."""
        tool_def = definition_func() if callable(definition_func) else definition_func
        tool_name = extract_tool_name(tool_def)
        self._agent.tool_definitions[tool_name] = (
            definition_func,
            handler_factory,
            service_instance,
        )

    def sync_to_llm(self) -> None:
        """Push all registered tool definitions to the LLM service."""
        agent = self._agent
        if not agent.llm:
            return

        self._clear_from_llm()

        provider = getattr(agent.llm, "provider_name", None)

        for tool_name, (
            definition_func,
            handler_factory,
            service_instance,
        ) in agent.tool_definitions.items():
            try:
                if callable(definition_func) and provider:
                    try:
                        tool_def = definition_func(provider)
                    except TypeError:
                        tool_def = definition_func()
                else:
                    tool_def = definition_func

                if callable(handler_factory):
                    handler = (
                        handler_factory(service_instance)
                        if service_instance
                        else handler_factory()
                    )
                else:
                    handler = handler_factory

                agent.llm.register_tool(tool_def, handler)
                agent.registered_tools.add(tool_name)
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {e}")

        agent._defer_tool_registration = False

    def _clear_from_llm(self) -> None:
        """Remove all tool registrations from the LLM service."""
        agent = self._agent
        if agent.llm:
            agent.llm.clear_tools()
            agent.registered_tools.clear()
