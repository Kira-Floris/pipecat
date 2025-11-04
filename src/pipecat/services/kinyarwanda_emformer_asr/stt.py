#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kinyarwanda Emformer speech-to-text services with locally-downloaded models.

This module implements Kinyarwanda Emformer transcription using locally-downloaded models.
"""

import asyncio
from enum import Enum
from typing import AsyncGenerator, Optional, List

import numpy as np
from loguru import logger
from typing_extensions import TYPE_CHECKING, override

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

import os

if TYPE_CHECKING:
    try:
        import sentencepiece as spm
        from huggingface_hub import snapshot_download
        import torch
    except ModuleNotFoundError as e:
        logger.error(f"Exception: {e}")
        logger.error("In order to use Kinyarwanda Emformer, you need to `pip install sentencepiece torch transformers`.")
        raise Exception(f"Missing module: {e}")
    

class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context
    
    def reset(self):
        """Reset the context cache"""
        self.context = torch.zeros([self.context_length])


class KinyarwandaEmformerSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded Kinyarwanda Emformer model.
    """

    SAMPLE_RATE = 16000
    HOP_LENGTH = 160
    SEGMENT_LENGTH = 16
    RIGHT_CONTEXT_LENGTH = 4

    # Calculate frame lengths
    SEGMENT_LENGTH_FRAMES = SEGMENT_LENGTH * HOP_LENGTH
    CONTEXT_LENGTH_FRAMES = RIGHT_CONTEXT_LENGTH * HOP_LENGTH

    state = None
    hypothesis = None

    def __init__(
        self,
        *,
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float = 0.4,
        **kwargs,
    ):
        """Initialize the Kinyarwanda Emformer STT service.

        Args:
            device: The device to run inference on ('cpu', 'cuda', or 'auto').
            compute_type: The compute type for inference ('default', 'int8', 'int8_float16', etc.).
            no_speech_prob: Probability threshold for filtering out non-speech segments.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(**kwargs)
        self._device: str = device
        self._compute_type = compute_type
        self._no_speech_prob = no_speech_prob
        self._model: Optional[torch.jit.load] = None
        self.model_name = "DigitalUmuganda/Emformer_afrivoice"
        self.destination_path = "./Emformer_afrivoice"

        self._settings = {
            "device": self._device,
            "compute_type": self._compute_type,
            "no_speech_prob": self._no_speech_prob,
        }

        self._load()

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return False

    def _load(self):
        """Loads the Whisper model.

        Note:
            If this is the first time this model is being run,
            it will take time to download from the Hugging Face model hub.
        """
        try:
            logger.debug("Loading Kinyarwanda Emformer model...")
            snapshot_download(
                self.model_name, local_dir=self.destination_path
            )
            model_path = os.path.join(self.destination_path, "scripted_bundle.pt")
            self._model = torch.jit.load(model_path)
            self._model = self._model.to(self._device)
            self._model.eval()
            logger.debug("Loaded Kinyarwanda Emformer model")

            tokenizer_path = os.path.join(self.destination_path, "tokenizer.model")
            self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            logger.debug("Loaded Kinyarwanda Emformer Tokenizer")

        except ModuleNotFoundError as e:
            logger.error(f"Exception: {e}")
            # logger.error("In order to use Kinyarwanda Emformer, you need to `pip install pipecat-ai[whisper]`.")
            self._model = None

    def _post_process_hypos(self, tokens: List[int]) -> str:
        """Post-process hypothesis tokens to text"""
        hypotheses = [x for x in tokens if x > 0 and x < 128]
        pred_texts = self.sp.decode(hypotheses)
        return pred_texts
    
    @torch.inference_mode()
    async def transcribe_segment(self, segment:torch.Tensor, cacher):
        try:
            segment = segment.to(self._device)

            segment_with_context = cacher(segment)

            self.hypothesis, self.state = self._model.stream(segment_with_context, self.state, self.hypothesis)
            transcription = self._post_process_hypos(self.hypothesis[0][0])
            return transcription
        except Exception as e:
            logger.error(f"Exception: {e}")
            self._model = None


    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using Whisper.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.

        Note:
            The audio is expected to be 16-bit signed PCM data.
            The service will normalize it to float32 in the range [-1, 1].
        """
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            yield ErrorFrame("Whisper model not available")
            return

        cacher = ContextCacher(self.SEGMENT_LENGTH_FRAMES, self.CONTEXT_LENGTH_FRAMES)
        
        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments = await asyncio.to_thread(
            self.transcribe_segment, audio_float, cacher
        )
        text: str = ""
        for segment in segments:
            text += f"{segment} "

        if text:
            await self._handle_transcription(text, True, self._settings["language"])
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601()
            )

