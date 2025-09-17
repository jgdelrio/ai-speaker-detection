"""
AWS Lambda handler for the AI Speaker Detection service.
"""
import os
from mangum import Mangum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import traceback
import httpx
from typing import Dict, Any

# Import warnings filter to suppress ML library warnings
from app.core.warnings_filter import suppress_known_warnings

# Apply warnings filter early
suppress_known_warnings()


async def send_webhook_callback(webhook_url: str, webhook_auth: str, job_id: str,
                               audio_file_id: int, transcription_id: int,
                               result: Dict[str, Any], status: str):
    """Send webhook callback to main server with speaker detection results."""
    try:
        # Prepare callback payload (matching transcription service format)
        callback_payload = {
            "job_id": job_id,
            "audio_file_id": audio_file_id,
            "transcription_id": transcription_id,
            "status": status,
            "service": "speaker-detection",
            "result": result,
            "error_message": None if status == "completed" else "Speaker detection failed"
        }

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ai-speaker-detection-lambda/1.0.0"
        }

        # Add authentication if provided
        if webhook_auth:
            headers["Authorization"] = f"Bearer {webhook_auth}"

        # Send HTTP POST request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                webhook_url,
                json=callback_payload,
                headers=headers
            )

            if response.status_code == 200:
                print(f"✅ Webhook callback sent successfully for job {job_id}")
            else:
                print(f"⚠️ Webhook callback failed for job {job_id}: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Webhook callback error for job {job_id}: {str(e)}")
        # Don't raise the exception - webhook failures shouldn't crash the lambda


# Create FastAPI app (this replaces the missing app.main)
app = FastAPI(
    title="AI Speaker Detection API",
    description="Identify different speakers in audio and update transcription information",
    version="1.0.0"
)

# Health check endpoint
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint for Lambda."""
    return {
        "status": "healthy",
        "service": "ai-speaker-detection",
        "version": "1.0.0"
    }

# Speaker detection endpoint - main entry point
@app.post("/detect-speakers")
async def detect_speakers(request: Dict[str, Any]):
    """
    Main Lambda handler endpoint.
    Called directly by AWS Lambda when payload is sent to the function.
    """
    return await process_speaker_detection(request)


# Speaker detection processing
async def process_speaker_detection(request: Dict[str, Any]):
    """
    Detect speakers in audio file.
    
    Expected request format:
    {
        "audio_file": "path/to/audio.mp3",
        "transcription": "text to update with speaker info",
        "expected_speakers": 2  # optional
    }
    """
    try:
        # Validate request
        if "audio_file" not in request:
            raise HTTPException(status_code=400, detail="audio_file is required")
        
        # TODO: Implement actual speaker detection logic
        # This is a placeholder implementation
        audio_file = request.get("audio_file")
        transcription = request.get("transcription", "")
        expected_speakers = request.get("expected_speakers", 1)
        
        # Mock response based on test data structure
        result = {
            "file": os.path.basename(audio_file),
            "expected_speakers": expected_speakers,
            "detected_speakers": 1,  # Placeholder
            "correct": True,
            "processing_time": 2.5,
            "text_length": len(transcription),
            "word_count": len(transcription.split()) if transcription else 0,
            "speakers": [
                {
                    "speaker_id": "Speaker1",
                    "speaker_label": "Speaker1",
                    "confidence": 0.95,
                    "total_speaking_time": 10.0,
                    "word_count": len(transcription.split()) if transcription else 0
                }
            ]
        }

        # Send webhook callback if configured
        webhook_url = request.get("webhook_url")
        webhook_auth = request.get("webhook_auth")

        if webhook_url:
            await send_webhook_callback(
                webhook_url=webhook_url,
                webhook_auth=webhook_auth,
                job_id=request.get("job_id"),
                audio_file_id=request.get("audio_file_id"),
                transcription_id=request.get("transcription_id"),
                result=result,
                status="completed"
            )

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error for debugging
        error_msg = f"Error processing speaker detection: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {traceback.format_exc()}")

        # Send error webhook callback if configured
        webhook_url = request.get("webhook_url")
        webhook_auth = request.get("webhook_auth")

        if webhook_url:
            try:
                await send_webhook_callback(
                    webhook_url=webhook_url,
                    webhook_auth=webhook_auth,
                    job_id=request.get("job_id"),
                    audio_file_id=request.get("audio_file_id"),
                    transcription_id=request.get("transcription_id"),
                    result={"error": error_msg},
                    status="failed"
                )
            except Exception as webhook_error:
                print(f"Additional webhook error: {webhook_error}")

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )

# Error handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for Lambda."""
    error_msg = f"Unhandled error: {str(exc)}"
    print(f"GLOBAL ERROR: {error_msg}")
    print(f"TRACEBACK: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": error_msg,
            "path": str(request.url)
        }
    )

# Create Mangum handler for Lambda
handler = Mangum(app, lifespan="off")

# For local testing (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)