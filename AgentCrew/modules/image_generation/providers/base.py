from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageGenerationResult:
    """Standardized result from any image generation provider."""

    image_data: bytes | None = None
    base64_data: str | None = None
    mime_type: str = "image/png"
    file_path: str | None = None
    provider: str = ""
    model: str = ""
    revised_prompt: str | None = None


class BaseImageProvider(ABC):
    """Abstract base class for image generation providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'gemini', 'deepinfra')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier for this provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider has the required API key configured."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        reference_images: list[str] | None = None,
    ) -> ImageGenerationResult:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt for image generation
            size: Image dimensions (e.g., "1024x1024")
            reference_images: Optional list of file paths for reference images

        Returns:
            ImageGenerationResult with the generated image data
        """
        ...

    def supports_reference_images(self) -> bool:
        """Whether this provider supports reference images."""
        return False
