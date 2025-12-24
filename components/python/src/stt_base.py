"""
Abstract base class for STT (Speech-to-Text) providers.

This module defines the interface that all STT providers must implement,
ensuring consistency and interchangeability across different STT services.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from events import STTEvent


class STTProvider(ABC):
    """
    Abstract base class for Speech-to-Text providers.

    All STT providers should inherit from this class and implement
    the required methods. This ensures a consistent interface for
    streaming audio input and receiving transcription events.
    """

    @abstractmethod
    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """
        Receive transcription events from the STT provider.

        This method should yield STTChunkEvent for partial transcriptions
        and STTOutputEvent for final transcriptions.

        Yields:
            STTEvent: Transcription events (stt_chunk or stt_output)
        """
        pass

    @abstractmethod
    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send an audio chunk to the STT provider for transcription.

        Args:
            audio_chunk: PCM audio bytes (16-bit, mono, 16kHz)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the connection to the STT provider and cleanup resources.

        This method should ensure all resources are properly released,
        including WebSocket connections, file handles, etc.
        """
        pass
