import os
import json
import time
import webbrowser
from typing import Optional, Dict, Any

import click
import requests

from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.config.global_config import GlobalConfig
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.service_manager import ServiceManager
from AgentCrew.modules.memory.chroma_service import ChromaMemoryService
from AgentCrew.modules.memory.context_persistent import ContextPersistenceService
from AgentCrew.modules.clipboard import ClipboardService
from AgentCrew.modules.web_search import TavilySearchService
from AgentCrew.modules.code_analysis import CodeAnalysisService
from AgentCrew.modules.image_generation import ImageGenerationService
from AgentCrew.modules.browser_automation import BrowserAutomationService
from AgentCrew.modules.agents.manager import AgentManager
from AgentCrew.modules.agents.local_agent import LocalAgent
from AgentCrew.modules.agents.remote_agent import RemoteAgent
from AgentCrew.modules.agents.example import (
    DEFAULT_NAME,
    DEFAULT_DESCRIPTION,
    DEFAULT_PROMPT,
)


PROVIDER_LIST = [
    "claude",
    "groq",
    "openai",
    "google",
    "deepinfra",
    "github_copilot",
    "copilot_response",
]


class ApplicationSetup:
    def __init__(self, config_manager: Optional[ConfigManagement] = None):
        self.config_manager = config_manager or ConfigManagement()
        self.services: Optional[Dict[str, Any]] = None
        self.agent_manager: Optional[AgentManager] = None

    def load_api_keys_from_config(self) -> None:
        config_file_path = os.getenv("AGENTCREW_CONFIG_PATH")
        if not config_file_path:
            config_file_path = "./config.json"
        config_file_path = os.path.expanduser(config_file_path)

        api_keys_config = {}
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    if isinstance(loaded_config, dict) and isinstance(
                        loaded_config.get("api_keys"), dict
                    ):
                        api_keys_config = loaded_config["api_keys"]
                    else:
                        click.echo(
                            f"\u26a0\ufe0f  API keys in {config_file_path} are not in the expected format.",
                            err=True,
                        )
            except json.JSONDecodeError:
                click.echo(
                    f"\u26a0\ufe0f  Error decoding API keys from {config_file_path}.",
                    err=True,
                )
            except Exception as e:
                click.echo(
                    f"\u26a0\ufe0f  Could not load API keys from {config_file_path}: {e}",
                    err=True,
                )

        keys_to_check = [
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "DEEPINFRA_API_KEY",
            "GITHUB_COPILOT_API_KEY",
            "TAVILY_API_KEY",
            "VOYAGE_API_KEY",
            "ELEVENLABS_API_KEY",
        ]

        for key_name in keys_to_check:
            if key_name in api_keys_config and api_keys_config[key_name]:
                os.environ[key_name] = str(api_keys_config[key_name]).strip()

    def detect_provider(self) -> Optional[str]:
        try:
            last_provider = GlobalConfig().get_last_used_provider()
            if last_provider:
                if last_provider in PROVIDER_LIST:
                    api_key_map = {
                        "claude": "ANTHROPIC_API_KEY",
                        "google": "GEMINI_API_KEY",
                        "openai": "OPENAI_API_KEY",
                        "groq": "GROQ_API_KEY",
                        "deepinfra": "DEEPINFRA_API_KEY",
                        "github_copilot": "GITHUB_COPILOT_API_KEY",
                        "copilot_response": "GITHUB_COPILOT_API_KEY",
                    }
                    if os.getenv(api_key_map.get(last_provider, "")):
                        return last_provider
                else:
                    custom_providers = GlobalConfig().read_custom_llm_providers_config()
                    if any(p["name"] == last_provider for p in custom_providers):
                        return last_provider
        except Exception as e:
            click.echo(f"\u26a0\ufe0f  Could not restore last used provider: {e}")

        if os.getenv("GITHUB_COPILOT_API_KEY"):
            return "github_copilot"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "claude"
        elif os.getenv("GEMINI_API_KEY"):
            return "google"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("GROQ_API_KEY"):
            return "groq"
        elif os.getenv("DEEPINFRA_API_KEY"):
            return "deepinfra"
        else:
            custom_providers = GlobalConfig().read_custom_llm_providers_config()
            if len(custom_providers) > 0:
                return custom_providers[0]["name"]

        return None

    def setup_services(
        self, provider: str, memory_llm: Optional[str] = None, need_memory: bool = True
    ) -> Dict[str, Any]:
        registry = ModelRegistry.get_instance()
        llm_manager = ServiceManager.get_instance()

        models = registry.get_models_by_provider(provider)
        if models:
            default_model = next((m for m in models if m.default), models[0])
            registry.set_current_model(f"{default_model.provider}/{default_model.id}")

        llm_service = llm_manager.get_service(provider)

        try:
            last_model = GlobalConfig().get_last_used_model()
            last_provider = GlobalConfig().get_last_used_provider()

            if last_model and last_provider:
                should_restore = False
                if provider == last_provider:
                    should_restore = True

                last_model_class = registry.get_model(last_model)
                if should_restore and last_model_class:
                    llm_service.model = last_model_class.id
        except Exception as e:
            click.echo(f"\u26a0\ufe0f  Could not restore last used model: {e}")

        memory_service = None
        context_service = None
        if need_memory:
            if memory_llm:
                memory_service = ChromaMemoryService(
                    llm_service=llm_manager.initialize_standalone_service(memory_llm)
                )
            else:
                memory_service = ChromaMemoryService(
                    llm_service=llm_manager.initialize_standalone_service(provider)
                )

            context_service = ContextPersistenceService()
        clipboard_service = ClipboardService()

        try:
            search_service = TavilySearchService()
        except Exception as e:
            click.echo(f"\u26a0\ufe0f Web search tools not available: {str(e)}")
            search_service = None

        try:
            code_analysis_llm = llm_manager.initialize_standalone_service(provider)
            code_analysis_service = CodeAnalysisService(llm_service=code_analysis_llm)
        except Exception as e:
            click.echo(f"\u26a0\ufe0f Code analysis tool not available: {str(e)}")
            code_analysis_service = None

        try:
            if os.getenv("OPENAI_API_KEY"):
                image_gen_service = ImageGenerationService()
            else:
                image_gen_service = None
                click.echo(
                    "\u26a0\ufe0f Image generation service not available: No API keys found."
                )
        except Exception as e:
            click.echo(f"\u26a0\ufe0f Image generation service not available: {str(e)}")
            image_gen_service = None

        try:
            browser_automation_service = BrowserAutomationService()
        except Exception as e:
            click.echo(
                f"\u26a0\ufe0f Browser automation service not available: {str(e)}"
            )
            browser_automation_service = None

        try:
            from AgentCrew.modules.file_editing import FileEditingService

            file_editing_service = FileEditingService()
        except Exception as e:
            click.echo(f"\u26a0\ufe0f File editing service not available: {str(e)}")
            file_editing_service = None

        try:
            from AgentCrew.modules.command_execution import CommandExecutionService

            command_execution_service = CommandExecutionService.get_instance()
        except Exception as e:
            click.echo(
                f"\u26a0\ufe0f Command execution service not available: {str(e)}"
            )
            command_execution_service = None

        try:
            from AgentCrew.modules.skills import SkillsService

            skills_service = SkillsService()
            if not skills_service.has_skills():
                skills_service = None
        except Exception as e:
            click.echo(f"\u26a0\ufe0f Skills service not available: {str(e)}")
            skills_service = None

        self.services = {
            "llm": llm_service,
            "memory": memory_service,
            "clipboard": clipboard_service,
            "code_analysis": code_analysis_service,
            "web_search": search_service,
            "context_persistent": context_service,
            "image_generation": image_gen_service,
            "browser": browser_automation_service,
            "file_editing": file_editing_service,
            "command_execution": command_execution_service,
            "skills": skills_service,
        }
        return self.services

    def setup_agents(
        self,
        services: Dict[str, Any],
        config_uri: Optional[str] = None,
        remoting_provider: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> AgentManager:
        self.agent_manager = AgentManager.get_instance()
        llm_manager = ServiceManager.get_instance()

        services["agent_manager"] = self.agent_manager

        global_config = GlobalConfig().read()
        self.agent_manager.context_shrink_enabled = global_config.get(
            "global_settings", {}
        ).get("auto_context_shrink", True)
        self.agent_manager.shrink_excluded_list = global_config.get(
            "global_settings", {}
        ).get("shrink_excluded", [])

        llm_service = services["llm"]

        if config_uri:
            os.environ["SW_AGENTS_CONFIG"] = config_uri
        else:
            config_uri = os.getenv("SW_AGENTS_CONFIG")
            if not config_uri:
                config_uri = "./agents.toml"
            if not os.path.exists(config_uri):
                click.echo(
                    f"Agent configuration not found at {config_uri}. Creating default configuration."
                )
                os.makedirs(os.path.dirname(config_uri), exist_ok=True)

                default_config = f"""
[[agents]]
name = "{DEFAULT_NAME}"
description = "{DEFAULT_DESCRIPTION}"
system_prompt = '''{DEFAULT_PROMPT}'''
tools = ["memory", "browser", "web_search", "code_analysis"]
"""

                with open(config_uri, "w+", encoding="utf-8") as f:
                    f.write(default_config)

                click.echo(f"Created default agent configuration at {config_uri}")

        agent_definitions = AgentManager.load_agents_from_config(config_uri)

        for agent_def in agent_definitions:
            if agent_def.get("base_url", ""):
                try:
                    agent = RemoteAgent(
                        agent_def["name"],
                        agent_def.get("base_url"),
                        headers=agent_def.get("headers", {}),
                    )
                except Exception:
                    print("Error: cannot connect to remote agent, skipping...")
                    continue
            else:
                if remoting_provider:
                    llm_service = llm_manager.initialize_standalone_service(
                        remoting_provider
                    )
                    if model_id:
                        llm_service.model = model_id
                agent = LocalAgent(
                    name=agent_def["name"],
                    description=agent_def["description"],
                    llm_service=llm_service,
                    services=services,
                    tools=agent_def["tools"],
                    temperature=agent_def.get("temperature", None),
                    voice_enabled=agent_def.get("voice_enabled", "disabled"),
                    voice_id=agent_def.get("voice_id", None),
                )
                agent.set_system_prompt(agent_def["system_prompt"])
                if remoting_provider:
                    agent.set_custom_system_prompt(
                        self.agent_manager.get_remote_system_prompt()
                    )
                    agent.is_remoting_mode = True
            self.agent_manager.register_agent(agent)

        from AgentCrew.modules.mcpclient.tool import register as mcp_register

        mcp_register()

        if remoting_provider:
            from AgentCrew.modules.mcpclient import MCPSessionManager

            mcp_manager = MCPSessionManager.get_instance()
            mcp_manager.initialize_for_agent()
            for agent in self.agent_manager.agents.values():
                agent.activate()

        return self.agent_manager

    def restore_last_agent(self) -> None:
        initial_agent_selected = False
        try:
            last_agent = GlobalConfig().get_last_used_agent()

            if (
                last_agent
                and self.agent_manager
                and last_agent in self.agent_manager.agents
            ):
                if self.agent_manager.select_agent(last_agent):
                    initial_agent_selected = True
        except Exception as e:
            click.echo(f"\u26a0\ufe0f  Could not restore last used agent: {e}")

        if not initial_agent_selected and self.agent_manager:
            first_agent_name = list(self.agent_manager.agents.keys())[0]
            if not self.agent_manager.select_agent(first_agent_name):
                available_agents = ", ".join(self.agent_manager.agents.keys())
                click.echo(
                    f"\u26a0\ufe0f Unknown agent: {first_agent_name}. Using default agent. Available agents: {available_agents}"
                )

    def login(self) -> bool:
        try:
            click.echo("\U0001f510 Starting GitHub Copilot authentication...")

            resp = requests.post(
                "https://github.com/login/device/code",
                headers={
                    "accept": "application/json",
                    "editor-version": "vscode/1.100.3",
                    "editor-plugin-version": "GitHub.copilot/1.330.0",
                    "content-type": "application/json",
                    "user-agent": "GithubCopilot/1.330.0",
                    "accept-encoding": "gzip,deflate,br",
                },
                data='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}',
            )

            if resp.status_code != 200:
                click.echo(
                    f"\u274c Failed to get device code: {resp.status_code}", err=True
                )
                return False

            resp_json = resp.json()
            device_code = resp_json.get("device_code")
            user_code = resp_json.get("user_code")
            verification_uri = resp_json.get("verification_uri")

            if not all([device_code, user_code, verification_uri]):
                click.echo("\u274c Invalid response from GitHub", err=True)
                return False

            click.echo(
                f"\U0001f4cb Please visit {verification_uri} and enter code: {user_code}"
            )
            click.echo("\u23f3 Waiting for authentication...")

            webbrowser.open(verification_uri)

            while True:
                time.sleep(5)

                resp = requests.post(
                    "https://github.com/login/oauth/access_token",
                    headers={
                        "accept": "application/json",
                        "editor-version": "vscode/1.100.3",
                        "editor-plugin-version": "GitHub.copilot/1.330.0",
                        "content-type": "application/json",
                        "user-agent": "GithubCopilot/1.330.0",
                        "accept-encoding": "gzip,deflate,br",
                    },
                    data=f'{{"client_id":"Iv1.b507a08c87ecfe98","device_code":"{device_code}","grant_type":"urn:ietf:params:oauth:grant-type:device_code"}}',
                )

                resp_json = resp.json()
                access_token = resp_json.get("access_token")
                error = resp_json.get("error")

                if access_token:
                    click.echo("\u2705 Authentication successful!")
                    break
                elif error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    time.sleep(5)
                    continue
                elif error == "expired_token":
                    click.echo(
                        "\u274c Authentication expired. Please try again.", err=True
                    )
                    return False
                elif error == "access_denied":
                    click.echo("\u274c Authentication denied by user.", err=True)
                    return False
                else:
                    click.echo(f"\u274c Authentication error: {error}", err=True)
                    return False

            global_config = GlobalConfig().read()

            if "api_keys" not in global_config:
                global_config["api_keys"] = {}

            global_config["api_keys"]["GITHUB_COPILOT_API_KEY"] = access_token
            GlobalConfig().write(global_config)

            click.echo("\U0001f4be GitHub Copilot API key saved to config file!")
            click.echo(
                "\U0001f680 You can now use GitHub Copilot with --provider github_copilot"
            )
            return True

        except ImportError:
            click.echo(
                "\u274c Error: 'requests' package is required for authentication",
                err=True,
            )
            click.echo("Install it with: pip install requests")
            return False
        except Exception as e:
            click.echo(f"\u274c Authentication failed: {str(e)}", err=True)
            return False
