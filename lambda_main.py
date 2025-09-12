"""
AWS Lambda handler for the AI Speaker Detection service.
"""
import os
from mangum import Mangum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import traceback
from typing import Dict, Any

# Import warnings filter to suppress ML library warnings
from app.core.warnings_filter import suppress_known_warnings

# Apply warnings filter early
suppress_known_warnings()

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

# Speaker detection endpoint
@app.post("/detect-speakers")
async def detect_speakers(request: Dict[str, Any]):
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
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error for debugging
        error_msg = f"Error processing speaker detection: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        
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