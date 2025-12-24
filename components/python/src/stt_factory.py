"""
STT Provider Factory

Factory module for creating STT provider instances based on configuration.
Supports switching between different STT providers via environment variables.
"""

import os
from typing import Optional

from assemblyai_stt import AssemblyAISTT
from stt_base import STTProvider
from whisper_stt import WhisperSTT


def create_stt_provider(
    provider: Optional[str] = None,
    sample_rate: int = 16000,
) -> STTProvider:
    """
    Create an STT provider instance based on the provider name.

    Args:
        provider: Name of the STT provider ('assemblyai' or 'whisper').
                 If None, reads from STT_PROVIDER environment variable.
                 Defaults to 'assemblyai' if not specified.
        sample_rate: Sample rate for audio input (default: 16000)

    Returns:
        STTProvider: An instance of the requested STT provider

    Raises:
        ValueError: If an unknown provider is requested
    """
    if provider is None:
        provider = os.getenv("STT_PROVIDER", "assemblyai").lower()

    if provider == "assemblyai":
        return AssemblyAISTT(sample_rate=sample_rate)
    elif provider == "whisper":
        # Read Whisper-specific configuration from environment
        model_name = os.getenv("WHISPER_MODEL", "base")
        language = os.getenv("WHISPER_LANGUAGE")  # Optional
        return WhisperSTT(
            model_name=model_name,
            sample_rate=sample_rate,
            language=language,
        )
    else:
        raise ValueError(
            f"Unknown STT provider: {provider}. "
            f"Supported providers: 'assemblyai', 'whisper'"
        )
