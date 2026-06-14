from __future__ import annotations

import json
from typing import Any

from AgentCrew.modules.utils.file_handler import read_optimized_image_file

from .service import ImageGenerationService


def get_generate_image_tool_definition() -> dict[str, Any]:
    """Return the tool definition for image generation."""
    tool_description = (
        "Generate an image from a structured JSON meta prompt. "
        "The tool constructs a detailed image prompt from the provided "
        "subject, environment, camera, lighting, and style parameters, "
        "then generates an image using the best available provider "
        "(OpenAI gpt-image-2 > Gemini > DeepInfra). "
        "Returns the file path of the generated image."
    )

    tool_arguments = {
        "meta_prompt": {
            "type": "object",
            "description": ("Structured JSON describing the image to generate."),
            "properties": {
                "subject": {
                    "type": "object",
                    "description": "The main subject of the image.",
                    "properties": {
                        "main_character": {
                            "type": "string",
                            "description": (
                                "The primary subject or character in the image."
                            ),
                        },
                        "expression": {
                            "type": "string",
                            "description": (
                                "Facial expression or emotional state of the subject."
                            ),
                        },
                        "clothing": {
                            "type": "string",
                            "description": ("Clothing or attire of the subject."),
                        },
                    },
                    "required": ["main_character"],
                },
                "environment": {
                    "type": "object",
                    "description": ("The setting and background of the image."),
                    "properties": {
                        "setting": {
                            "type": "string",
                            "description": (
                                "The overall setting or scene description."
                            ),
                        },
                        "background": {
                            "type": "string",
                            "description": (
                                "Specific background elements or description."
                            ),
                        },
                    },
                    "required": ["background"],
                },
                "camera": {
                    "type": "object",
                    "description": ("Camera and framing parameters."),
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": (
                                "Camera type (e.g., 'Cinema camera', 'DSLR')."
                            ),
                        },
                        "lens": {
                            "type": "string",
                            "description": ("Lens specification (e.g., '50mm f/1.4')."),
                        },
                        "angle": {
                            "type": "string",
                            "description": (
                                "Camera angle (e.g., 'Low-angle, looking up')."
                            ),
                        },
                        "shot_type": {
                            "type": "string",
                            "description": (
                                "Shot type (e.g., 'Medium full shot', 'Close-up')."
                            ),
                        },
                    },
                },
                "lighting": {
                    "type": "object",
                    "description": ("Lighting configuration."),
                    "properties": {
                        "key_light": {
                            "type": "string",
                            "description": ("Primary light source description."),
                        },
                        "fill_light": {
                            "type": "string",
                            "description": ("Secondary/fill light description."),
                        },
                        "mood": {
                            "type": "string",
                            "description": (
                                "Overall lighting mood (e.g., 'Cinematic, moody')."
                            ),
                        },
                    },
                },
                "style": {
                    "type": "object",
                    "description": "Visual style parameters.",
                    "properties": {
                        "art_form": {
                            "type": "string",
                            "description": (
                                "Art style (e.g., 'Photorealistic', 'Oil painting')."
                            ),
                        },
                        "color_palette": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Hex color codes for the palette"
                                " (e.g., ['#00FFFF', '#FF007F'])."
                            ),
                        },
                        "textures": {
                            "type": "string",
                            "description": (
                                "Texture descriptions (e.g.,"
                                " 'Wet surfaces, metallic sheen')."
                            ),
                        },
                    },
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional file paths to reference images"
                        " for style guidance or in-painting."
                    ),
                },
            },
            "required": ["subject", "environment"],
        },
        "size": {
            "type": "string",
            "description": (
                "Image dimensions. Supported: '1024x1024', '1536x1024', '1024x1536'."
            ),
            "default": "1024x1024",
        },
    }

    return {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": tool_arguments,
                "required": ["meta_prompt"],
            },
        },
    }


def get_generate_image_tool_handler(
    image_service: ImageGenerationService,
):
    """Return a handler function for the image generation tool."""

    async def handle_generate_image(**params) -> Any:
        meta_prompt = params.get("meta_prompt")
        size = params.get("size", "1024x1024")

        if not meta_prompt:
            raise ValueError("No meta_prompt provided.")

        if isinstance(meta_prompt, str):
            try:
                meta_prompt = json.loads(meta_prompt)
            except json.JSONDecodeError:
                return ValueError("meta_prompt is not valid JSON.")

        result = await image_service.generate_image(
            meta_prompt=meta_prompt,
            size=size,
        )

        if result["success"]:
            file_path = result["file_path"]

            optimized_image = read_optimized_image_file(file_path)
            if optimized_image:
                mime_type, image_data, _ = optimized_image
            else:
                image_data = None
                mime_type = None

            text_summary = f"Image generated: {file_path}"

            structured_result: list[dict[str, Any]] = [
                {"type": "text", "text": text_summary},
            ]

            if image_data:
                data_uri = f"data:{mime_type};base64,{image_data}"
                structured_result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

            return structured_result
        else:
            return f"Error generating image: {result['error']}"

    return handle_generate_image


def register(service_instance=None, agent=None):
    """Register the image generation tool with an agent."""
    from AgentCrew.modules.tools.registration import register_tool

    register_tool(
        get_generate_image_tool_definition,
        get_generate_image_tool_handler,
        service_instance,
        agent,
    )
