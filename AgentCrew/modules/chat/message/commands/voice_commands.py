from __future__ import annotations

from typing import Any, TYPE_CHECKING

from AgentCrew.modules.chat.message.commands.base import CommandResult

if TYPE_CHECKING:
    from AgentCrew.modules.chat.message import MessageHandler


class VoiceCommands:
    """Handles voice-related slash commands."""

    def __init__(self, message_handler: MessageHandler):
        self.message_handler = message_handler

    async def handle_voice(self, command: str) -> CommandResult:
        """Handle /voice command to start voice recording."""
        try:
            # Check if already recording
            if self.message_handler.voice_service is None:
                self.message_handler._notify(
                    "error",
                    "Voice service not available. Start AgentCrew with --with-voice and set ELEVENLABS_API_KEY or DEEPINFRA_API_KEY.",
                )
                return CommandResult(handled=True, clear_flag=True)

            if self.message_handler.voice_service.is_recording():
                self.message_handler._notify(
                    "error",
                    "Already recording. Use /end_voice to stop current recording.",
                )
                return CommandResult(handled=True, clear_flag=True)

            if self.message_handler.has_active_stream():
                self.message_handler._notify(
                    "system_message",
                    "🎤 Voice input is unavailable while the assistant is still responding.",
                )
                return CommandResult(handled=True, clear_flag=True)

            async def submit_active_voice(audio_data: Any, sample_rate: int):
                if self.message_handler.has_active_stream():
                    return
                transcript = await self._voice_transcript(audio_data, sample_rate)
                if self.message_handler.has_active_stream():
                    return
                self.message_handler._notify("voice_activate", transcript)

            # Start recording
            result = self.message_handler.voice_service.start_voice_recording(
                voice_completed_cb=submit_active_voice
            )

            if result["success"]:
                self.message_handler._notify("voice_recording_started", None)
                self.message_handler._notify(
                    "system_message",
                    "🎤 Recording started. Press Enter to stop.",
                )
            else:
                self.message_handler._notify("error", result["error"])

            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify("error", f"Voice command failed: {str(e)}")
            return CommandResult(handled=True, clear_flag=True)

    async def _voice_transcript(self, audio_data: Any, sample_rate: int):
        transcribed_text = None
        if audio_data is not None and self.message_handler.voice_service:
            transcribe_result = await self.message_handler.voice_service.speech_to_text(
                audio_data, sample_rate
            )
            if transcribe_result["success"]:
                transcribed_text = transcribe_result["text"]
                confidence = transcribe_result.get("confidence", 1.0)

                # Notify about transcription
                self.message_handler._notify(
                    "system_message",
                    f"✅ Transcribed (confidence: {confidence:.0%}): {transcribed_text}",
                )

            else:
                self.message_handler._notify("error", transcribe_result["error"])

            return transcribed_text

    async def handle_end_voice(self, command: str) -> CommandResult:
        """Handle /end_voice command to stop recording and transcribe."""
        try:
            # Check if voice service exists and is recording
            if self.message_handler.voice_service is None:
                # self.message_handler._notify(
                #     "error",
                #     "No voice service initialized. Use /voice to start recording.",
                # )
                return CommandResult(handled=True, clear_flag=True)

            if not self.message_handler.voice_service.is_recording():
                # self.message_handler._notify("error", "No recording in progress.")
                return CommandResult(handled=True, clear_flag=True)

            # Stop recording
            self.message_handler._notify("voice_recording_stopping", None)
            stop_result = self.message_handler.voice_service.stop_voice_recording()

            if not stop_result["success"]:
                # self.message_handler._notify("error", stop_result["error"])
                return CommandResult(handled=True, clear_flag=True)

            # Transcribe
            # self.message_handler._notify("system_message", "🔄 Transcribing audio...")

            # transcribed_text = await self._voice_transcript(
            #     stop_result["audio_data"], stop_result["sample_rate"]
            # )

            self.message_handler._notify("voice_recording_completed", None)
            return CommandResult(handled=True, clear_flag=True)

        except Exception as e:
            self.message_handler._notify("error", f"End voice command failed: {str(e)}")
            self.message_handler._notify("voice_recording_completed", None)
            return CommandResult(handled=True, clear_flag=True)
