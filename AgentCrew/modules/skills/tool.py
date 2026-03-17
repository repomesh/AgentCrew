import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import SkillsService


def get_activate_skill_tool_definition(
    skills_service: "SkillsService", provider="claude"
):
    skill_names = skills_service.get_skill_names()
    tool_description = (
        "Load the full instructions for a named skill. "
        "Call this tool when the current task matches a skill's description. "
        "Returns the skill body, its directory path, and a list of bundled resource files."
    )
    tool_arguments = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": skill_names,
                "description": "The name of the skill to activate.",
            }
        },
        "required": ["name"],
    }

    if provider == "claude":
        return {
            "name": "activate_skill",
            "description": tool_description,
            "input_schema": tool_arguments,
        }
    return {
        "type": "function",
        "function": {
            "name": "activate_skill",
            "description": tool_description,
            "parameters": tool_arguments,
        },
    }


def get_activate_skill_tool_handler(skills_service: "SkillsService"):
    activated: set = set()

    async def handler(**params):
        name = params.get("name", "")
        if not name:
            raise ValueError("Error: No skill name provided")

        if name in activated:
            return f"Skill '{name}' is already loaded in this session."

        skill = skills_service.get_skill(name)
        if not skill:
            available = ", ".join(skills_service.get_skill_names())
            return f"Skill '{name}' not found. Available skills: {available}"

        body = skill["body"]
        skill_dir = os.path.dirname(skill["location"])
        resources = skills_service.list_resources(skill_dir)

        resources_xml = ""
        if resources:
            files_xml = "\n".join(f"  <file>{r}</file>" for r in resources)
            resources_xml = f"\n<skill_resources>\n{files_xml}\n</skill_resources>"

        activated.add(name)

        return (
            f'<skill_content name="{name}">\n'
            f"{body}\n\n"
            f"Skill directory: {skill_dir}\n"
            f"Relative paths in this skill are relative to the skill directory."
            f"{resources_xml}\n"
            f"</skill_content>"
        )

    return handler


def register(service_instance: "SkillsService", agent=None):
    from AgentCrew.modules.tools.registration import register_tool

    def definition_func(provider="claude"):
        return get_activate_skill_tool_definition(service_instance, provider)

    register_tool(
        definition_func,
        get_activate_skill_tool_handler,
        service_instance,
        agent,
    )
