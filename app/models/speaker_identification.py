from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None


class SpeakerIdentificationRequest(BaseModel):
    transcription_data: Dict[str, Any]
    known_speakers: Optional[Dict[str, List[float]]] = None


class SpeakerIdentificationResponse(BaseModel):
    transcription_data: Dict[str, Any]
    formatted_transcription: str
    speaker_embeddings: Dict[str, List[float]]
    detected_speakers: int
    processing_metadata: Dict[str, Any]
