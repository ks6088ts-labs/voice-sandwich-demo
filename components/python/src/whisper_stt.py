"""
Whisper Local STT Provider

A local Speech-to-Text provider using OpenAI's Whisper model.
This provider runs entirely locally without requiring API keys or internet connectivity.

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_output for final transcripts)

Note: This provider does not produce streaming partial transcripts (stt_chunk)
as Whisper processes the entire audio buffer at once.
"""

import asyncio
from typing import AsyncIterator, Optional

import numpy as np
import whisper

from events import STTEvent, STTOutputEvent
from stt_base import STTProvider

# PCM 16-bit audio normalization constant
INT16_MAX = 32768.0


class WhisperSTT(STTProvider):
    """
    Local Whisper STT provider.

    This implementation buffers audio chunks and processes them using
    OpenAI's Whisper model running locally. It does not provide streaming
    partial transcripts (stt_chunk), only final outputs (stt_output).
    """

    def __init__(
        self,
        model_name: str = "base",
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ):
        """
        Initialize the Whisper STT provider.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Sample rate of input audio (default: 16000)
            language: Optional language code (e.g., 'en', 'ja')
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language
        self._model: Optional[whisper.Whisper] = None
        self._audio_buffer: list[bytes] = []
        self._queue: asyncio.Queue[Optional[STTEvent]] = asyncio.Queue()
        self._transcription_task: Optional[asyncio.Task] = None
        self._closed = False

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """
        Receive transcription events from Whisper.

        Yields:
            STTEvent: Final transcription events (stt_output)
        """
        # Load model lazily on first use
        if self._model is None:
            await self._load_model()

        while True:
            event = await self._queue.get()
            if event is None:  # Sentinel value indicating completion
                break
            yield event

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Buffer audio chunks for transcription.

        Args:
            audio_chunk: PCM audio bytes (16-bit, mono, 16kHz)
        """
        if self._closed:
            return
        self._audio_buffer.append(audio_chunk)

    async def close(self) -> None:
        """
        Process buffered audio and cleanup resources.
        """
        if self._closed:
            return

        self._closed = True

        # Trigger transcription of buffered audio
        if self._audio_buffer:
            await self._transcribe_buffer()

        # Send sentinel to stop the receive_events iterator
        await self._queue.put(None)

        # Wait for transcription task to complete if running
        if self._transcription_task and not self._transcription_task.done():
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

    async def _load_model(self) -> None:
        """
        Load the Whisper model.

        This is done lazily to avoid loading the model until needed.
        The loading happens in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, whisper.load_model, self.model_name
        )

    async def _transcribe_buffer(self) -> None:
        """
        Transcribe all buffered audio chunks.

        This method combines buffered audio chunks, converts them to the format
        expected by Whisper, and performs transcription in a thread pool executor.
        """
        if not self._audio_buffer or self._model is None:
            return

        # Combine all audio chunks
        audio_data = b"".join(self._audio_buffer)
        self._audio_buffer.clear()

        # Convert bytes to numpy array
        # Audio is 16-bit PCM, so convert to float32 normalized to [-1, 1]
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            / INT16_MAX
        )

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_np,
                    language=self.language,
                    fp16=False,  # Use fp32 for better compatibility
                ),
            )

            transcript = result.get("text", "").strip()
            if transcript:
                event = STTOutputEvent.create(transcript)
                await self._queue.put(event)
        except Exception as e:
            print(f"Whisper transcription error: {e}")
