from AgentCrew.modules.openai import OpenAIService
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from dotenv import load_dotenv
from AgentCrew.modules.llm.model_registry import ModelRegistry
from AgentCrew.modules.llm.token_usage import TokenUsage


class GoogleAIService(OpenAIService):
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = "gemini-2.0-flash-001"

    def format_tool_result(
        self, tool_use: Dict, tool_result: Any, is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Format a tool result for OpenAI API.

        Args:
            tool_use: The tool use details
            tool_result: The result from the tool execution
            is_error: Whether the result is an error

        Returns:
            A formatted message for tool response
        """
        # OpenAI format for tool responses
        message = {
            "name": tool_use["name"],
            "role": "user",
            "tool_call_id": tool_use["id"],
            "content": str(tool_result),
        }

        # Add error indication if needed
        if is_error:
            message["content"] = f"ERROR: {message['content']}"

        return message

    def stream_assistant_response(self, messages):
        """Stream the assistant's response with tool support."""
        stream_params = {
            "model": self.model,
            "messages": messages,
            "stream_options": {"include_usage": True},
            "max_tokens": 8192,
        }
        stream_params["temperature"] = 0.6
        stream_params["top_p"] = 0.95

        # Add system message if provided
        if self.system_prompt:
            stream_params["messages"] = [
                {"role": "system", "content": self.system_prompt}
            ] + messages

        # Add tools if available
        if self.tools and "tool_use" in ModelRegistry.get_model_capabilities(
            f"{self._provider_name}/{self.model}"
        ):
            stream_params["tools"] = self.tools

        return self.client.chat.completions.create(**stream_params, stream=True)

    def process_stream_chunk(
        self, chunk, assistant_response: str, tool_uses: List[Dict]
    ) -> Tuple[str, List[Dict], TokenUsage, Optional[str], Optional[tuple]]:
        """
        Process a single chunk from the streaming response.

        Args:
            chunk: The chunk from the stream
            assistant_response: Current accumulated assistant response
            tool_uses: Current tool use information

        Returns:
            tuple: (
                updated_assistant_response,
                updated_tool_uses,
                input_tokens,
                output_tokens,
                chunk_text,
                thinking_data
            )
        """
        chunk_text = None
        input_tokens = 0
        output_tokens = 0
        # thinking_content = None  # OpenAI doesn't support thinking mode

        # Handle tool call chunks
        if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, "tool_calls"):
            delta_tool_calls = chunk.choices[0].delta.tool_calls
            if delta_tool_calls:
                # Process each tool call in the delta
                for tool_call_delta in delta_tool_calls:
                    tool_call_index = tool_call_delta.index or 0

                    # Check if this is a new tool call
                    if tool_call_index >= len(tool_uses):
                        # Create a new tool call entry
                        tool_uses.append(
                            {
                                "id": tool_call_delta.id
                                if hasattr(tool_call_delta, "id")
                                else None,
                                "name": getattr(tool_call_delta.function, "name", "")
                                if hasattr(tool_call_delta, "function")
                                else "",
                                "input": {},
                                "type": "function",
                                "response": "",
                            }
                        )

                    # Update existing tool call with new data
                    if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                        tool_uses[tool_call_index]["id"] = tool_call_delta.id

                    if hasattr(tool_call_delta, "function"):
                        if (
                            hasattr(tool_call_delta.function, "name")
                            and tool_call_delta.function.name
                        ):
                            tool_uses[tool_call_index]["name"] = (
                                tool_call_delta.function.name
                            )

                        if (
                            hasattr(tool_call_delta.function, "arguments")
                            and tool_call_delta.function.arguments
                        ):
                            # Accumulate arguments as they come in chunks
                            current_args = tool_uses[tool_call_index].get(
                                "args_json", ""
                            )
                            tool_uses[tool_call_index]["args_json"] = (
                                current_args + tool_call_delta.function.arguments
                            )

                            # Try to parse JSON if it seems complete
                            try:
                                args_json = tool_uses[tool_call_index]["args_json"]
                                tool_uses[tool_call_index]["input"] = json.loads(
                                    args_json
                                )
                                # Keep args_json for accumulation but use input for execution
                            except json.JSONDecodeError:
                                # Arguments JSON is still incomplete, keep accumulating
                                pass

                # For tool calls, we don't append to assistant_response as it's handled separately
                return (
                    assistant_response or " ",
                    tool_uses,
                    TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    "",
                    None,
                )

        # Handle regular content chunks
        if (
            len(chunk.choices) > 0
            and hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            chunk_text = chunk.choices[0].delta.content
            assistant_response += chunk_text

        # Handle final chunk with usage information
        cached_tokens = 0
        if hasattr(chunk, "usage"):
            if hasattr(chunk.usage, "prompt_tokens"):
                input_tokens = chunk.usage.prompt_tokens
            if hasattr(chunk.usage, "completion_tokens"):
                output_tokens = chunk.usage.completion_tokens
            if (
                hasattr(chunk.usage, "prompt_tokens_details")
                and chunk.usage.prompt_tokens_details
            ):
                if hasattr(chunk.usage.prompt_tokens_details, "cached_tokens"):
                    cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens
                    input_tokens = input_tokens - cached_tokens

        return (
            assistant_response or " ",
            tool_uses,
            TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            ),
            chunk_text,
            None,
        )
