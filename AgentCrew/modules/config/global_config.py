import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class GlobalConfig:
    """
    Manages the global config file (~/.agentcrew/config.json).
    Covers: last-used settings, API keys, auto-approval tools, custom LLM providers.
    """

    @property
    def _path(self) -> str:
        path = os.getenv("AGENTCREW_CONFIG_PATH")
        if not path:
            path = "./config.json"
        return os.path.expanduser(path)

    def read(self) -> Dict[str, Any]:
        """Reads data from the global config.json file."""
        config_path = self._path
        default_config = {
            "api_keys": {},
            "auto_approval_tools": [],
            "global_settings": {
                "theme": "dark",
                "swap_enter": False,
                "yolo_mode": False,
                "auto_context_shrink": True,
                "shrink_excluded": [],
            },
        }
        try:
            if not os.path.exists(config_path):
                return default_config
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(
                        f"Warning: Global config file {config_path} does not contain a valid JSON object. Returning default."
                    )
                    return default_config
                if "api_keys" not in data or not isinstance(data.get("api_keys"), dict):
                    data["api_keys"] = {}
                if "auto_approval_tools" not in data or not isinstance(
                    data.get("auto_approval_tools"), list
                ):
                    data["auto_approval_tools"] = []
                return data
        except json.JSONDecodeError:
            logger.warning(
                f"Warning: Error decoding global config file {config_path}. Returning default config."
            )
            return default_config
        except Exception as e:
            logger.warning(
                f"Warning: Could not read global config file {config_path}: {e}. Returning default config."
            )
            return default_config

    def write(self, config_data: Dict[str, Any]) -> None:
        """Writes data to the global config.json file."""
        from AgentCrew.modules.agents import AgentManager

        config_path = self._path
        try:
            dir_path = os.path.dirname(config_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            agent_manager = AgentManager.get_instance()
            agent_manager.context_shrink_enabled = config_data.get(
                "global_settings", {}
            ).get("auto_context_shrink", True)
            agent_manager.shrink_excluded_list = config_data.get(
                "global_settings", {}
            ).get("shrink_excluded", [])

            from AgentCrew.modules.agents.manager import AgentMode

            agent_mode_str = config_data.get("global_settings", {}).get(
                "agent_mode", "transfer"
            )
            try:
                agent_manager.agent_mode = AgentMode(agent_mode_str)
            except ValueError:
                agent_manager.agent_mode = AgentMode.TRANSFER
        except Exception as e:
            raise ValueError(
                f"Error writing global configuration to {config_path}: {str(e)}"
            )

    def get_last_used_settings(self) -> Dict[str, Any]:
        """Get the last used model and agent settings from the global config."""
        global_config = self.read()
        return global_config.get("last_used", {})

    def set_last_used_model(self, model_id: str, provider: str) -> None:
        """Save the last used model to global config."""
        try:
            global_config = self.read()
            if "last_used" not in global_config:
                global_config["last_used"] = {}
            global_config["last_used"]["model"] = model_id
            global_config["last_used"]["provider"] = provider
            global_config["last_used"]["timestamp"] = datetime.now().isoformat()
            self.write(global_config)
        except Exception as e:
            logger.warning(f"Warning: Failed to save last used model to config: {e}")

    def set_last_used_agent(self, agent_name: str) -> None:
        """Save the last used agent to global config."""
        try:
            global_config = self.read()
            if "last_used" not in global_config:
                global_config["last_used"] = {}
            global_config["last_used"]["agent"] = agent_name
            global_config["last_used"]["timestamp"] = datetime.now().isoformat()
            self.write(global_config)
        except Exception as e:
            logger.warning(f"Warning: Failed to save last used agent to config: {e}")

    def get_last_used_model(self) -> Optional[str]:
        """Get the last used model from global config."""
        return self.get_last_used_settings().get("model")

    def get_last_used_provider(self) -> Optional[str]:
        """Get the last used provider from global config."""
        return self.get_last_used_settings().get("provider")

    def get_last_used_agent(self) -> Optional[str]:
        """Get the last used agent name from global config."""
        return self.get_last_used_settings().get("agent")

    def get_auto_approval_tools(self) -> List[str]:
        """Get the list of auto-approved tools from global config."""
        global_config = self.read()
        return global_config.get("auto_approval_tools", [])

    def write_auto_approval_tools(self, tool_name: str, add: bool = True) -> None:
        """Add or remove a tool from the auto-approval list in global config."""
        try:
            global_config = self.read()
            auto_approval_tools = global_config.get("auto_approval_tools", [])

            if add and tool_name not in auto_approval_tools:
                auto_approval_tools.append(tool_name)
            elif not add and tool_name in auto_approval_tools:
                auto_approval_tools.remove(tool_name)

            global_config["auto_approval_tools"] = auto_approval_tools
            self.write(global_config)
        except Exception as e:
            action = "add" if add else "remove"
            logger.warning(
                f"Warning: Failed to {action} tool {tool_name} from auto-approval list: {e}"
            )

    def read_custom_llm_providers_config(self) -> List[Dict[str, Any]]:
        """Read the custom LLM providers configuration from the global config file."""
        global_config = self.read()
        providers = global_config.get("custom_llm_providers", [])
        for provider in providers:
            if "available_models" not in provider:
                provider["available_models"] = []
        return providers

    def write_custom_llm_providers_config(
        self, providers_data: List[Dict[str, Any]]
    ) -> None:
        """Write the custom LLM providers configuration to the global config file."""
        global_config = self.read()
        global_config["custom_llm_providers"] = providers_data
        self.write(global_config)
