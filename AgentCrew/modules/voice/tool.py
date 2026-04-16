from typing import Dict, Any, Callable


SPEAK_MAX_LENGTH = 280


def get_speak_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "Speak to the user using voice. "
        "When voice is available, use this as the primary way to communicate with the user. "
        "Keep spoken language concise, natural, and suitable for speech. "
        "Use light fillers like 'um', 'uh', or 'so like' naturally when helpful. "
        "Use normal chat text only as supporting detail when needed."
    )
    tool_arguments = {
        "text": {
            "type": "string",
            "description": "The spoken message for the user. Prefer short, natural, voice-friendly phrasing.",
        }
    }
    tool_required = ["text"]

    if provider == "claude":
        return {
            "name": "speak",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "speak",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        },
    }


def speak_tool_prompt() -> str:
    return """
<Speak_Tool_Instruction>
  <Purpose>
    Use the `speak` tool as the primary way to communicate with the user when voice is available.
  </Purpose>

  <When_To_Use>
    - use it for normal replies to the user
    - use it for acknowledgements and progress updates
    - use it for guidance, answers, and conversational feedback
    - use normal chat text only when extra detail or structured content is needed
  </When_To_Use>

  <How_To_Speak>
    - prefer the `speak` tool over plain text responses when available
    - keep spoken language concise and natural
    - use natural spoken phrasing rather than formal writing
    - you may use light fillers like "um", "uh", or "so like" naturally
    - if more detail is needed, keep the voice message short and put the extra detail in normal chat text
    - avoid long, dense, overly written speech
  </How_To_Speak>
</Speak_Tool_Instruction>
"""


def get_speak_tool_handler(voice_service, agent=None) -> Callable:
    async def handle_speak(**params) -> str:
        text = (params.get("text") or "").strip()
        if not text:
            raise ValueError("Error: No text provided")

        if len(text) > SPEAK_MAX_LENGTH:
            text = text[:SPEAK_MAX_LENGTH].rstrip()

        voice_id = getattr(agent, "voice_id", None) if agent is not None else None
        voice_service.text_to_speech_stream(text, voice_id=voice_id)
        return "Queued speech output."

    return handle_speak


def register(service_instance=None, agent=None):
    from AgentCrew.modules.tools.registration import register_tool

    if agent is None:
        raise ValueError("register() requires an agent for the speak tool")

    def handler_factory(_service_instance=None):
        service = _service_instance or service_instance
        return get_speak_tool_handler(service, agent)

    register_tool(get_speak_tool_definition, handler_factory, service_instance, agent)
