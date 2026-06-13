from __future__ import annotations

import os
import mimetypes
from openai import AsyncOpenAI

from .base import BaseImageProvider, ImageGenerationResult


class OpenAIImageProvider(BaseImageProvider):
    """OpenAI gpt-image-2 image generation provider."""

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._client: AsyncOpenAI | None = None
        if self._api_key:
            self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return "gpt-image-2"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def supports_reference_images(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        reference_images: list[str] | None = None,
    ) -> ImageGenerationResult:

        if reference_images:
            return await self._generate_with_reference(prompt, size, reference_images)
        return await self._generate_text_only(prompt, size)

    async def _generate_text_only(
        self, prompt: str, size: str
    ) -> ImageGenerationResult:
        if not self._client:
            raise ValueError("Image Generation provider is not available")

        response = await self._client.images.generate(
            model=self.model_id,
            prompt=prompt,
            size=size,
            response_format="b64_json",
        )
        if not response or not response.data:
            raise SystemError("Image generated failed.")

        image_b64 = response.data[0].b64_json
        revised_prompt = getattr(response.data[0], "revised_prompt", None)

        return ImageGenerationResult(
            base64_data=image_b64,
            mime_type="image/png",
            provider=self.name,
            model=self.model_id,
            revised_prompt=revised_prompt,
        )

    async def _generate_with_reference(
        self,
        prompt: str,
        size: str,
        reference_images: list[str],
    ) -> ImageGenerationResult:
        if not self._client:
            raise ValueError("Image Generation provider is not available")
        image_file = reference_images[0]
        mime_type, _ = mimetypes.guess_type(image_file)
        if not mime_type:
            mime_type = "image/png"

        with open(image_file, "rb") as f:
            image_data = f.read()

        response = await self._client.images.edit(
            model=self.model_id,
            image=image_data,
            prompt=prompt,
            response_format="b64_json",
        )

        if not response or not response.data:
            raise SystemError("Image generated failed.")

        image_b64 = response.data[0].b64_json
        revised_prompt = getattr(response.data[0], "revised_prompt", None)

        return ImageGenerationResult(
            base64_data=image_b64,
            mime_type="image/png",
            provider=self.name,
            model=self.model_id,
            revised_prompt=revised_prompt,
        )
