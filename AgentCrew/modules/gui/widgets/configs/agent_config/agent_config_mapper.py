def populate_local_agent_form(editor, agent_data: dict) -> None:
    """Populate local agent editor fields from config data.

    Accepts either a LocalAgentEditor or the legacy AgentsConfigTab.
    """
    editor.name_input.setText(agent_data.get("name", ""))
    editor.description_input.setText(agent_data.get("description", ""))
    editor.temperature_input.setText(str(agent_data.get("temperature", "0.5")))
    editor.enabled_checkbox.setChecked(agent_data.get("enabled", True))

    voice_state = agent_data.get("voice_enabled", "disabled")
    editor.voice_enabled_checkbox.setChecked(voice_state == "enabled")
    editor.voice_id_input.setText(agent_data.get("voice_id", ""))

    tools = agent_data.get("tools", [])
    for tool, checkbox in editor.tool_checkboxes.items():
        checkbox.setChecked(tool in tools)

    editor.system_prompt_input.set_markdown(agent_data.get("system_prompt", ""))


def clear_local_agent_form(editor) -> None:
    """Clear local agent editor fields.

    Accepts either a LocalAgentEditor or the legacy AgentsConfigTab.
    """
    editor.name_input.clear()
    editor.description_input.clear()
    editor.temperature_input.clear()
    editor.system_prompt_input.clear()
    editor.enabled_checkbox.setChecked(True)
    editor.voice_enabled_checkbox.setChecked(False)
    editor.voice_id_input.clear()

    for checkbox in editor.tool_checkboxes.values():
        checkbox.setChecked(False)

    editor.behavior_editor.clear()


def collect_local_agent_form(editor) -> dict:
    """Collect local agent editor fields into a config dict.

    Accepts either a LocalAgentEditor or the legacy AgentsConfigTab.
    """
    try:
        temperature = float(editor.temperature_input.text().strip() or "0.5")
        temperature = max(0.0, min(2.0, temperature))
    except ValueError:
        temperature = 0.5

    voice_state = "enabled" if editor.voice_enabled_checkbox.isChecked() else "disabled"

    return {
        "name": editor.name_input.text().strip(),
        "description": editor.description_input.text().strip(),
        "temperature": temperature,
        "tools": [
            tool
            for tool, checkbox in editor.tool_checkboxes.items()
            if checkbox.isChecked()
        ],
        "system_prompt": editor.system_prompt_input.get_markdown().strip(),
        "enabled": editor.enabled_checkbox.isChecked(),
        "voice_enabled": voice_state,
        "voice_id": editor.voice_id_input.text().strip(),
        "agent_type": "local",
    }


def populate_remote_agent_form(editor, agent_data: dict) -> None:
    """Populate remote agent editor fields from config data.

    Accepts either a RemoteAgentEditor or the legacy AgentsConfigTab.
    """
    editor.remote_name_input.setText(agent_data.get("name", ""))
    editor.remote_base_url_input.setText(agent_data.get("base_url", ""))
    editor.remote_enabled_checkbox.setChecked(agent_data.get("enabled", True))


def clear_remote_agent_form(editor) -> None:
    """Clear remote agent editor fields.

    Accepts either a RemoteAgentEditor or the legacy AgentsConfigTab.
    """
    editor.remote_name_input.clear()
    editor.remote_base_url_input.clear()
    editor.remote_enabled_checkbox.setChecked(True)


def collect_remote_headers(remote_header_inputs: list[dict]) -> dict:
    """Collect remote header key-value pairs from header input list."""
    headers = {}
    for header_data in remote_header_inputs:
        key = header_data["key_input"].text().strip()
        value = header_data["value_input"].text().strip()
        if key:
            headers[key] = value
    return headers


def collect_remote_agent_form(editor, headers: dict) -> dict:
    """Collect remote agent editor fields into a config dict.

    Accepts either a RemoteAgentEditor or the legacy AgentsConfigTab.
    """
    return {
        "name": editor.remote_name_input.text().strip(),
        "base_url": editor.remote_base_url_input.text().strip(),
        "enabled": editor.remote_enabled_checkbox.isChecked(),
        "headers": headers,
        "agent_type": "remote",
    }
