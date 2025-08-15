from fastapi import APIRouter, File, Form, UploadFile
from app.services.speaker_service import SpeakerService
import json
from typing import Dict, Any, Optional

router = APIRouter()
speaker_service = SpeakerService()


@router.post("/speaker-identification", response_model=str)
async def speaker_identification(
    audio_file: UploadFile = File(...), 
    transcription_data: str = Form(...),
    known_speakers: Optional[str] = Form(None)
):
    transcription_dict: Dict[str, Any] = json.loads(transcription_data)
    known_speakers_dict = json.loads(known_speakers) if known_speakers else None

    formatted_transcription = await speaker_service.process_speaker_identification(
        audio_file, transcription_dict, known_speakers_dict
    )

    return formatted_transcription
