from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from .base import BaseImageProvider, ImageGenerationResult

MAX_REFERENCE_IMAGES = 4


class DeepInfraImageProvider(BaseImageProvider):
    """DeepInfra FLUX-2-klein-9b image generation provider.

    Uses the OpenAI-compatible API at api.deepinfra.com/v1.
    Supports reference images via the input_images array (up to 4 images).
    Reference images are passed as base64 data URLs in the request body.
    """

    DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1"

    def __init__(self):
        self._api_key = os.getenv("DEEPINFRA_API_KEY")
        self._client: AsyncOpenAI | None = None
        if self._api_key:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.DEEPINFRA_BASE_URL,
            )

    @property
    def name(self) -> str:
        return "deepinfra"

    @property
    def model_id(self) -> str:
        return "black-forest-labs/FLUX-2-klein-9b"

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
        if not self._client:
            raise RuntimeError("DeepInfra API key not configured")

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
            raise SystemError("Image generation failed.")

        image_b64 = response.data[0].b64_json

        return ImageGenerationResult(
            base64_data=image_b64,
            mime_type="image/png",
            provider=self.name,
            model=self.model_id,
        )

    async def _generate_with_reference(
        self,
        prompt: str,
        size: str,
        reference_images: list[str],
    ) -> ImageGenerationResult:
        if not self._client:
            raise ValueError("Image Generation provider is not available")

        images = reference_images[:MAX_REFERENCE_IMAGES]
        if len(reference_images) > MAX_REFERENCE_IMAGES:
            logger.warning(
                f"DeepInfra FLUX-2 klein supports up to {MAX_REFERENCE_IMAGES} "
                f"reference images; only the first {MAX_REFERENCE_IMAGES} will be used."
            )

        encoded_images = []
        for image_path in images:
            data_url = self._encode_image_as_data_url(image_path)
            encoded_images.append(data_url)

        response = await self._client.images.generate(
            model=self.model_id,
            prompt=prompt,
            size=size,
            response_format="b64_json",
            extra_body={"input_images": encoded_images},
        )

        if not response or not response.data:
            raise SystemError("Image generation failed.")

        image_b64 = response.data[0].b64_json

        return ImageGenerationResult(
            base64_data=image_b64,
            mime_type="image/png",
            provider=self.name,
            model=self.model_id,
        )

    @staticmethod
    def _encode_image_as_data_url(file_path: str) -> str:
        """Encode a local image file as a data URL for the DeepInfra API."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/png"

        image_bytes = path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
