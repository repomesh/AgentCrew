"""Ask for guided clarification from the user or a delegate's source agent."""

from collections.abc import Awaitable, Callable
from typing import Any


def get_ask_tool_definition(recipient: str = "user") -> dict[str, Any]:
    """
    Get the definition for the ask tool.

    Returns:
        The tool definition
    """
    tool_description = (
        "PRIMARY tool for gathering missing information — use when the request is "
        "ambiguous, details are unclear, you hit a blocker, or need direction.\n"
        "When the user presents a plan/design/proposal, interview relentlessly: "
        "walk down each branch of the design tree, resolve dependencies one-by-one, "
        "and provide your recommended answer as the first guided option.\n"
        "Bundle up to 10 related, independent questions in the `questions` array."
    )

    if recipient != "user":
        tool_description = (
            f"Ask the {recipient} for clarification on the delegated task. "
            "The answer is returned directly to this delegate; do not ask the user. "
            "Bundle up to 10 related, independent questions in the `questions` array."
        )

    tool_arguments = {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": f"The question to ask the {recipient}. Be clear, specific, and concise.",
                    },
                    "guided_answers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-10 suggested answers. Put your recommended answer first. "
                        ),
                        "minItems": 1,
                        "maxItems": 10,
                    },
                },
                "required": ["question", "guided_answers"],
            },
            "minItems": 1,
            "maxItems": 10,
            "description": "One or more questions to ask (up to 10). Use multiple entries to bundle related questions and reduce round trips.",
        },
    }

    tool_required = ["questions"]

    return {
        "type": "function",
        "function": {
            "name": "ask",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        },
    }


def ask_tool_prompt(recipient: str = "user") -> str:
    """
    Get the system prompt instructions for the ask tool.

    Returns:
        System prompt describing when and how to use the ask tool
    """
    if recipient != "user":
        return f"""
<Ask_Tool_Instruction>
  <Purpose>
    Ask the {recipient} for clarification when the delegated task is unclear or blocked.
    The answer is returned as the ask tool result. Do not ask the user or escalate.
    If the {recipient} lacks the information, use that limitation in your response.
  </Purpose>
  <Best_Practices>
    - Ask focused questions with 2-6 guided answers each
    - Bundle related independent questions in one call
  </Best_Practices>
</Ask_Tool_Instruction>
"""
    return """
<Ask_Tool_Instruction>
  <Purpose>
    Your PRIMARY tool when anything is unclear, incomplete, or blocked.
    Use it to:
    - Resolve ambiguity or blockers before guessing
    - Interview the user on plans/designs/proposals: walk each branch of the design tree,
      resolve dependencies one-by-one, and provide your recommended answer as the first option
    - Clarify specs, constraints, or choices where context is insufficient

    Bundle up to 10 related questions in one call. 
  </Purpose>

  <Best_Practices>
    - Default to `ask` when unclear — do not guess or assume
    - Bundle related questions (up to 10), resolve one dependency at a time
    - If questions depend on other question, split into different ask tool calls
    - Put your recommended answer first; provide 2-6 clear, mutually exclusive options
    - Ask the smallest useful question — don't bundle unrelated decisions
  </Best_Practices>
</Ask_Tool_Instruction>
"""


def get_ask_tool_handler(
    responder: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None,
) -> Callable:
    """
    Get the handler function for the ask tool.

    Without a responder, the UI handles user interaction and this handler returns
    a placeholder. Delegated clones await their injected source-agent responder.

    Returns:
        The handler function
    """

    async def handler(**params) -> str:
        """
        Handle an ask request.

        Validate guided questions before either the UI confirmation flow or
        the delegate's source-agent clarification.

        Args:
            questions: list of {question, guided_answers} objects

        Returns:
            The source-agent answer, or a placeholder for UI confirmation.
        """
        questions = params.get("questions", [])

        if not questions or not isinstance(questions, list):
            raise ValueError(
                "Error: `questions` array is required with at least one {question, guided_answers} entry."
            )

        if len(questions) > 10:
            raise ValueError(
                "Error: Cannot ask more than 10 questions in a single call. "
                "Limit to 10 questions."
            )

        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                raise TypeError(f"Error: Question at index {i} must be an object")
            question_text = q.get("question", "").strip()
            guided_answers = q.get("guided_answers", [])

            if not question_text:
                raise ValueError(f"Error: Question at index {i} has no question text")

            if not isinstance(guided_answers, list) or len(guided_answers) < 2:
                raise ValueError(
                    f"Error: Question at index {i} must have at least 2 guided answers"
                )

            if len(guided_answers) > 6:
                raise ValueError(
                    f"Error: Question at index {i} cannot have more than 6 guided answers"
                )

        if responder is not None:
            return await responder(questions)

        # This return value is a placeholder - the actual response
        # will be injected by the UI layer after user interaction
        count = len(questions)
        return f"Asking user {count} question{'s' if count > 1 else ''}"

    return handler


def register(
    agent=None,
    responder: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None,
):
    """
    Register the ask tool with a specific agent.

    Args:
        agent: Specific agent to register with (optional)
    """
    from AgentCrew.modules.tools.registration import register_tool

    if responder is not None:
        from functools import partial

        definition = partial(get_ask_tool_definition, "source/main agent")
        handler_factory = partial(get_ask_tool_handler, responder)
    else:
        definition = get_ask_tool_definition
        handler_factory = get_ask_tool_handler
    register_tool(definition, handler_factory, None, agent)
