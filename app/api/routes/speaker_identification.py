from fastapi import APIRouter, File, Form, UploadFile
from app.models.speaker_identification import SpeakerIdentificationResponse
from app.services.speaker_service import SpeakerService
import json
from typing import Dict, Any

router = APIRouter()
speaker_service = SpeakerService()


@router.post("/speaker-identification", response_model=SpeakerIdentificationResponse)
async def speaker_identification(
    audio_file: UploadFile = File(...), transcription_data: str = Form(...)
):
    transcription_dict: Dict[str, Any] = json.loads(transcription_data)

    enhanced_data = await speaker_service.process_speaker_identification(
        audio_file, transcription_dict
    )

    return SpeakerIdentificationResponse(transcription_data=enhanced_data)
