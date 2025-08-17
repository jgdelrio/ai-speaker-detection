from fastapi import APIRouter, File, Form, UploadFile
from app.services.speaker_service import SpeakerService
import json
from typing import Dict, Any

router = APIRouter()
speaker_service = SpeakerService()


@router.post("/speaker-identification")
async def speaker_identification(
    audio: UploadFile = File(...), 
    transcription_data: str = Form(...)
):
    transcription_dict: Dict[str, Any] = json.loads(transcription_data)

    speaker_response = await speaker_service.process_speaker_identification(
        audio, transcription_dict
    )

    return speaker_response
