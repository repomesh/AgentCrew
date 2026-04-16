import json
import os
from typing import Any, Dict, List

from loguru import logger
from pydantic import config
from tomli_w import dump as toml_dump


class AgentsConfig:
    """Manages agents.toml — CRUD, hot-reload, export, and import."""

    @property
    def _path(self) -> str:
        return os.getenv("SW_AGENTS_CONFIG", os.path.expanduser("./agents.toml"))

    def read(self) -> Dict[str, Any]:
        """Return the full agents config dict, or {"agents": []} on error."""
        try:
            from AgentCrew.modules.config.config_management import ConfigManagement

            config = ConfigManagement(self._path)
            return config.get_config()
        except Exception:
            return {"agents": []}

    def write(self, config_data: Dict[str, Any]) -> None:
        """Persist config_data to agents.toml and hot-reload live agents."""
        from AgentCrew.modules.config.config_management import ConfigManagement

        try:
            config = ConfigManagement(self._path)
            config.update_config(config_data, merge=False)
            config.save_config()
            self.reload()
        except FileNotFoundError:
            dir_path = os.path.dirname(self._path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self._path, "wb") as f:
                toml_dump(config_data, f, multiline_strings=True)
            self.reload()

    def reload(self) -> None:
        """Hot-reload all agents from the current agents.toml without restarting."""
        from AgentCrew.modules.agents import RemoteAgent, LocalAgent, AgentManager

        agent_manager = AgentManager.get_instance()
        new_agents_config = agent_manager.load_agents_from_config(self._path)
        for agent_cfg in new_agents_config:
            if agent_cfg.get("base_url"):
                try:
                    agent_manager.agents[agent_cfg["name"]] = RemoteAgent(
                        agent_cfg["name"],
                        agent_cfg["base_url"],
                        headers=agent_cfg.get("headers", {}),
                    )
                except Exception as e:
                    logger.error(str(e))
                    continue

            existing_agent = agent_manager.get_local_agent(agent_cfg["name"])
            system_prompt = agent_cfg.get("system_prompt", "")
            if existing_agent:
                existing_agent.tools = agent_cfg.get("tools", [])
                existing_agent.set_system_prompt(system_prompt)
                existing_agent.temperature = agent_cfg.get("temperature", 0.4)
                existing_agent.voice_enabled = (
                    "enabled"
                    if agent_cfg.get("voice_enabled", "disabled") == "enabled"
                    else "disabled"
                )
                existing_agent.voice_id = agent_cfg.get("voice_id", None)
            else:
                clone_agent = agent_manager.get_current_agent()
                if not isinstance(clone_agent, LocalAgent):
                    clone_agent = [
                        agent
                        for agent in agent_manager.agents.values()
                        if isinstance(agent, LocalAgent)
                    ][0]

                voice_enabled = (
                    "enabled"
                    if agent_cfg.get("voice_enabled", "disabled") == "enabled"
                    else "disabled"
                )
                voice_id = agent_cfg.get("voice_id", None)

                new_agent = LocalAgent(
                    name=agent_cfg["name"],
                    description=agent_cfg["description"],
                    llm_service=clone_agent.llm,
                    services=clone_agent.services,
                    tools=agent_cfg["tools"],
                    temperature=agent_cfg.get("temperature", None),
                    voice_enabled=voice_enabled,
                    voice_id=voice_id,
                )
                new_agent.set_system_prompt(system_prompt)
                agent_manager.register_agent(new_agent)

        new_agent_names = [a["name"] for a in new_agents_config]
        old_agent_names = [
            n for n in agent_manager.agents.keys() if n not in new_agent_names
        ]
        for agent_name in old_agent_names:
            old_agent = agent_manager.get_agent(agent_name)
            if old_agent and old_agent.is_active:
                agent_manager.select_agent(new_agent_names[0])
            agent_manager.deregister_agent(agent_name)

        for _, agent in agent_manager.agents.items():
            was_active = False
            if agent.is_active:
                was_active = True
                agent.deactivate()
            if isinstance(agent, LocalAgent):
                agent.custom_system_prompt = None
            if was_active:
                agent_manager.select_agent(agent.name)

    def update_agent_system_prompt(self, agent_name: str, new_prompt: str) -> bool:
        config_data = self.read()
        agents = config_data.get("agents", [])
        if not isinstance(agents, list):
            return False

        updated = False
        for agent in agents:
            if agent.get("name") == agent_name:
                agent["system_prompt"] = new_prompt
                updated = True
                break

        if not updated:
            return False

        self.write(config_data)
        return True

    def export(
        self, agent_names: List[str], output_file: str, file_format: str = "toml"
    ) -> Dict[str, Any]:
        """Export selected agents to a portable file."""
        result = {
            "success": False,
            "exported_count": 0,
            "local_count": 0,
            "remote_count": 0,
            "missing_agents": [],
            "output_file": output_file,
        }

        try:
            agents_config = self.read()
            local_agents = agents_config.get("agents", [])
            remote_agents = agents_config.get("remote_agents", [])

            selected_local_agents = []
            selected_remote_agents = []
            found_names = set()

            for agent in local_agents:
                if agent.get("name") in agent_names:
                    export_data = {k: v for k, v in agent.items() if k != "agent_type"}
                    selected_local_agents.append(export_data)
                    found_names.add(agent.get("name"))

            for agent in remote_agents:
                if agent.get("name") in agent_names:
                    export_data = {k: v for k, v in agent.items() if k != "agent_type"}
                    selected_remote_agents.append(export_data)
                    found_names.add(agent.get("name"))

            missing_names = set(agent_names) - found_names
            result["missing_agents"] = list(missing_names)

            if not selected_local_agents and not selected_remote_agents:
                result["error"] = "No matching agents found to export"
                return result

            export_config = {}
            if selected_local_agents:
                export_config["agents"] = selected_local_agents
                result["local_count"] = len(selected_local_agents)
            if selected_remote_agents:
                export_config["remote_agents"] = selected_remote_agents
                result["remote_count"] = len(selected_remote_agents)

            result["exported_count"] = len(selected_local_agents) + len(
                selected_remote_agents
            )

            if file_format == "toml" and not output_file.endswith(".toml"):
                output_file += ".toml"
            elif file_format == "json" and not output_file.endswith(".json"):
                output_file += ".json"

            output_file = os.path.expanduser(output_file)
            result["output_file"] = output_file

            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if file_format == "toml":
                with open(output_file, "wb") as f:
                    toml_dump(export_config, f, multiline_strings=True)
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(export_config, f, indent=2, ensure_ascii=False)

            result["success"] = True
            logger.info(f"Exported {result['exported_count']} agents to {output_file}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Export agents error: {str(e)}", exc_info=True)

        return result

    def import_agents(
        self,
        import_file_path: str,
        merge_strategy: str = "update",
        skip_conflicts: bool = False,
    ) -> Dict[str, Any]:
        """Import agents from a previously exported file."""
        result = {
            "success": False,
            "added_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "conflicts": [],
            "imported_agents": [],
        }

        temp_file = None

        try:
            if import_file_path.startswith(("http://", "https://")):
                import requests
                import tempfile

                response = requests.get(import_file_path, timeout=30)
                response.raise_for_status()

                suffix = (
                    ".json"
                    if "json" in response.headers.get("content-type", "")
                    else ".toml"
                )
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False, encoding="utf-8"
                )
                temp_file.write(response.text)
                temp_file.close()
                import_file_path = temp_file.name

            import_file_path = os.path.expanduser(import_file_path)

            if not os.path.exists(import_file_path):
                result["error"] = f"File not found: {import_file_path}"
                return result

            from AgentCrew.modules.config.config_management import ConfigManagement

            temp_config = ConfigManagement(import_file_path)
            imported_config = temp_config.get_config()

            imported_local_agents = imported_config.get("agents", [])
            imported_remote_agents = imported_config.get("remote_agents", [])

            if not imported_local_agents and not imported_remote_agents:
                result["error"] = "No agent configurations found in the file"
                return result

            current_config = self.read()
            current_local_agents = current_config.get("agents", [])
            current_remote_agents = current_config.get("remote_agents", [])

            local_agent_map = {
                agent.get("name"): agent for agent in current_local_agents
            }
            remote_agent_map = {
                agent.get("name"): agent for agent in current_remote_agents
            }

            existing_names = set(local_agent_map.keys()) | set(remote_agent_map.keys())

            for agent in imported_local_agents:
                agent_name = agent.get("name")
                if not agent_name:
                    continue

                is_conflict = agent_name in existing_names

                if is_conflict:
                    result["conflicts"].append(agent_name)

                    if skip_conflicts:
                        result["skipped_count"] += 1
                        continue

                    if merge_strategy == "skip":
                        result["skipped_count"] += 1
                        continue

                if "enabled" not in agent:
                    agent["enabled"] = True

                local_agent_map[agent_name] = agent
                if is_conflict:
                    result["updated_count"] += 1
                else:
                    result["added_count"] += 1
                result["imported_agents"].append(agent_name)

            for agent in imported_remote_agents:
                agent_name = agent.get("name")
                if not agent_name:
                    continue

                is_conflict = agent_name in existing_names

                if is_conflict:
                    if agent_name not in result["conflicts"]:
                        result["conflicts"].append(agent_name)

                    if skip_conflicts:
                        result["skipped_count"] += 1
                        continue

                    if merge_strategy == "skip":
                        result["skipped_count"] += 1
                        continue

                if "enabled" not in agent:
                    agent["enabled"] = True

                remote_agent_map[agent_name] = agent
                if is_conflict:
                    result["updated_count"] += 1
                else:
                    result["added_count"] += 1
                result["imported_agents"].append(agent_name)

            final_config = {}
            if local_agent_map:
                final_config["agents"] = list(local_agent_map.values())
            if remote_agent_map:
                final_config["remote_agents"] = list(remote_agent_map.values())

            self.write(final_config)

            result["success"] = True
            logger.info(
                f"Imported agents: added={result['added_count']}, "
                f"updated={result['updated_count']}, skipped={result['skipped_count']}"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Import agents error: {str(e)}", exc_info=True)

        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

        return result
