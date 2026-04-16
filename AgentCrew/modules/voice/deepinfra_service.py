import os
import tempfile
import threading
from typing import Dict, Any, Optional, Callable
import queue

import numpy as np
import soundfile as sf
import sounddevice as sd
from openai import OpenAI

from .text_cleaner import TextCleaner
from .audio_handler import AudioHandler
from .base import BaseVoiceService

from loguru import logger


DEEPINFRA_OPENAI_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEEPINFRA_TTS_RESPONSE_FORMAT = "pcm"
DEEPINFRA_PCM_SAMPLE_RATE = 24000
DEEPINFRA_FALLBACK_VOICES = [
    {"voice_id": "alloy", "name": "Alloy", "category": "standard"},
    {"voice_id": "echo", "name": "Echo", "category": "standard"},
    {"voice_id": "fable", "name": "Fable", "category": "standard"},
    {"voice_id": "onyx", "name": "Onyx", "category": "standard"},
    {"voice_id": "nova", "name": "Nova", "category": "standard"},
    {"voice_id": "shimmer", "name": "Shimmer", "category": "standard"},
    {"voice_id": "tara", "name": "Tara", "category": "standard"},
]


class DeepInfraVoiceService(BaseVoiceService):
    """DeepInfra voice service using OpenAI-compatible STT and TTS."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()

        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepInfra API key not found. Set DEEPINFRA_API_KEY environment variable."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=DEEPINFRA_OPENAI_BASE_URL,
        )
        self.stt_client = self.client

        self.audio_handler = AudioHandler()
        self.text_cleaner = TextCleaner()

        self.stt_model = "openai/whisper-large-v3-turbo"
        self.default_voice_id = "None"
        self.default_model = "ResembleAI/chatterbox-turbo"

        self._start_tts_thread()

    def start_voice_recording(
        self, sample_rate: int = 44100, voice_completed_cb: Optional[Callable] = None
    ) -> Dict[str, Any]:
        try:
            self.audio_handler.start_recording(sample_rate, voice_completed_cb)
            return {"success": True, "message": "Recording started."}
        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}")
            return {"success": False, "error": f"Failed to start recording: {str(e)}"}

    def stop_voice_recording(self) -> Dict[str, Any]:
        try:
            audio_data, sample_rate = self.audio_handler.stop_recording()

            if audio_data is None:
                return {"success": False, "error": "No audio data captured"}

            duration = len(audio_data) / sample_rate
            return {
                "success": True,
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duration": duration,
                "message": f"Recording stopped. Duration: {duration:.2f} seconds",
            }
        except Exception as e:
            logger.error(f"Failed to stop recording: {str(e)}")
            return {"success": False, "error": f"Failed to stop recording: {str(e)}"}

    def is_recording(self) -> bool:
        return self.audio_handler.is_recording()

    async def speech_to_text(self, audio_data: Any, sample_rate: int) -> Dict[str, Any]:
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, "rb") as audio_file:
                transcript = self.stt_client.audio.transcriptions.create(
                    model=self.stt_model,
                    file=audio_file,
                    language="en",
                    response_format="verbose_json",
                    temperature=0.2,
                    timestamp_granularities=["segment"],
                )

            text = transcript.text if hasattr(transcript, "text") else ""
            language = transcript.language if hasattr(transcript, "language") else "en"

            return {
                "success": True,
                "text": text,
                "language": language,
                "confidence": 1.0,
                "words": [],
            }
        except Exception as e:
            logger.error(f"Speech-to-text failed: {str(e)}")
            return {"success": False, "error": f"Failed to transcribe audio: {str(e)}"}
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary STT file {tmp_file_path}: {cleanup_error}"
                    )

    def clean_text_for_speech(self, text: str) -> str:
        return self.text_cleaner.clean_for_speech(text)

    def _start_tts_thread(self):
        with self.tts_lock:
            if not self.tts_thread_running:
                self.tts_thread_running = True
                self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
                self.tts_thread.start()
                logger.debug("TTS worker thread started (DeepInfra)")

    def _tts_worker(self):
        while self.tts_thread_running:
            try:
                tts_request = self.tts_queue.get(timeout=1.0)
                if tts_request is None:
                    break

                text, voice_id, model_id = tts_request
                self._process_tts_request(text, voice_id, model_id)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {str(e)}")

        logger.debug("TTS worker thread stopped (DeepInfra)")

    def _build_tts_request_kwargs(
        self,
        text: str,
        voice_id: Optional[str],
        model_id: Optional[str],
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        cleaned_text = self.clean_text_for_speech(text)
        if not cleaned_text.strip():
            raise ValueError("No speakable text after cleaning")

        return {
            "input": cleaned_text,
            "model": model_id or self.default_model,
            "voice": voice_id or self.default_voice_id,
            "response_format": response_format or DEEPINFRA_TTS_RESPONSE_FORMAT,
        }

    def _create_tts_stream(
        self,
        text: str,
        voice_id: Optional[str],
        model_id: Optional[str],
        response_format: Optional[str] = None,
    ):
        request_kwargs = self._build_tts_request_kwargs(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            response_format=response_format,
        )
        return self.client.audio.speech.with_streaming_response.create(**request_kwargs)

    def _process_tts_request(
        self, text: str, voice_id: Optional[str], model_id: Optional[str]
    ):
        try:
            cleaned_text = self.clean_text_for_speech(text)
            if not cleaned_text.strip():
                logger.warning("No speakable text after cleaning")
                return

            logger.debug(f"Processing TTS for text: {cleaned_text[:50]}...")

            self.audio_handler.is_host_playing = True
            try:
                with self._create_tts_stream(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    response_format="pcm",
                ) as response:
                    pcm_bytes = response.read()

                if not pcm_bytes:
                    logger.warning("DeepInfra PCM TTS returned empty audio")
                    return

                audio_data = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                sd.play(audio_data, DEEPINFRA_PCM_SAMPLE_RATE)
                sd.wait()
            finally:
                self.audio_handler.is_host_playing = False

            logger.debug("TTS streaming completed (DeepInfra)")
        except Exception as e:
            self.audio_handler.is_host_playing = False
            logger.error(f"Text-to-speech processing failed: {str(e)}")

    def text_to_speech_stream(
        self, text: str, voice_id: Optional[str] = None, model_id: Optional[str] = None
    ):
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return

            if not self.tts_thread_running:
                self._start_tts_thread()

            try:
                self.tts_queue.put((text, voice_id, model_id), block=False)
                logger.debug(f"TTS request queued for text: {text[:50]}...")
            except queue.Full:
                logger.warning(
                    f"TTS queue is full (size: {self.tts_queue.qsize()}), dropping request"
                )
        except Exception as e:
            logger.error(f"Failed to queue TTS request: {str(e)}")

    def list_voices(self) -> Dict[str, Any]:
        return {"success": True, "voices": DEEPINFRA_FALLBACK_VOICES}

    def set_voice(self, voice_id: str):
        self.default_voice_id = voice_id
        logger.info(f"Default DeepInfra voice set to: {voice_id}")

    def get_configured_voice_id(self) -> str:
        logger.warning(
            "get_configured_voice_id is deprecated. Voice ID should be managed by MessageHandler."
        )
        return self.default_voice_id

    def set_voice_settings(self, **kwargs):
        return None

    def stop_tts_thread(self):
        with self.tts_lock:
            if self.tts_thread_running:
                self.tts_thread_running = False

                try:
                    while not self.tts_queue.empty():
                        self.tts_queue.get_nowait()
                except queue.Empty:
                    pass

                self.tts_queue.put(None)

                if self.tts_thread and self.tts_thread.is_alive():
                    self.tts_thread.join(timeout=2.0)

                logger.debug("TTS thread stopped (DeepInfra)")

    def clear_tts_queue(self):
        try:
            while not self.tts_queue.empty():
                self.tts_queue.get_nowait()
            logger.debug("TTS queue cleared (DeepInfra)")
        except queue.Empty:
            pass

    def __del__(self):
        try:
            self.stop_tts_thread()
        except Exception:
            pass
