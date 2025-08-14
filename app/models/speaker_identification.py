from pydantic import BaseModel
from typing import Dict, Any


class SpeakerIdentificationRequest(BaseModel):
    transcription_data: Dict[str, Any]


class SpeakerIdentificationResponse(BaseModel):
    transcription_data: Dict[str, Any]
