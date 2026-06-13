from __future__ import annotations

from typing import Any


class MetaPromptBuilder:
    """Converts structured JSON meta prompt into a text prompt for image generation.

    Sections are composed in visual priority order:
    subject → environment → camera → lighting → style
    """

    def validate(self, meta_prompt: dict[str, Any]) -> tuple[bool, str]:
        """Validate the meta prompt has required fields.

        Returns:
            Tuple of (is_valid, error_message)
        """
        subject = meta_prompt.get("subject")
        if not subject or not isinstance(subject, dict):
            return False, "subject is required and must be an object"
        if not subject.get("main_character"):
            return False, "subject.main_character is required"

        environment = meta_prompt.get("environment")
        if not environment or not isinstance(environment, dict):
            return False, "environment is required and must be an object"
        if not environment.get("background"):
            return False, "environment.background is required"

        images = meta_prompt.get("images")
        if images is not None:
            if not isinstance(images, list):
                return False, "images must be an array of file paths"
            for img in images:
                if not isinstance(img, str):
                    return False, "each image path must be a string"

        return True, ""

    def build(self, meta_prompt: dict[str, Any]) -> str:
        """Build a text prompt from the JSON meta prompt structure.

        Each section is only included if present and non-empty.
        """
        sections: list[str] = []

        subject = meta_prompt.get("subject", {})
        if subject:
            parts = [subject.get("main_character", "")]
            if subject.get("expression"):
                parts.append(f"with {subject['expression']} expression")
            if subject.get("clothing"):
                parts.append(f"wearing {subject['clothing']}")
            section = " ".join(p for p in parts if p)
            if section:
                sections.append(section)

        environment = meta_prompt.get("environment", {})
        if environment:
            env_parts = []
            if environment.get("setting"):
                env_parts.append(f"Set in {environment['setting']}")
            if environment.get("background"):
                env_parts.append(f"Background: {environment['background']}")
            if env_parts:
                sections.append(". ".join(env_parts))

        camera = meta_prompt.get("camera")
        if camera:
            cam_parts = []
            if camera.get("type"):
                cam_parts.append(f"Shot on {camera['type']}")
            if camera.get("lens"):
                cam_parts.append(f"with {camera['lens']} lens")
            if camera.get("angle"):
                cam_parts.append(camera["angle"])
            if camera.get("shot_type"):
                cam_parts.append(camera["shot_type"])
            if cam_parts:
                sections.append(", ".join(cam_parts))

        lighting = meta_prompt.get("lighting")
        if lighting:
            light_parts = []
            if lighting.get("key_light"):
                light_parts.append(f"Key light: {lighting['key_light']}")
            if lighting.get("fill_light"):
                light_parts.append(f"Fill light: {lighting['fill_light']}")
            if lighting.get("mood"):
                light_parts.append(f"Mood: {lighting['mood']}")
            if light_parts:
                sections.append(". ".join(light_parts))

        style = meta_prompt.get("style")
        if style:
            style_parts = []
            if style.get("art_form"):
                style_parts.append(f"Style: {style['art_form']}")
            if style.get("color_palette"):
                palette = ", ".join(style["color_palette"])
                style_parts.append(f"Color palette: {palette}")
            if style.get("textures"):
                style_parts.append(f"Textures: {style['textures']}")
            if style_parts:
                sections.append(". ".join(style_parts))

        return ". ".join(sections) if sections else ""
